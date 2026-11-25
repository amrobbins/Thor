#include "Utilities/TensorOperations/Sort/RadixSortOneBlock.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Sort/RadixSort.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ThorImplementation {
namespace {

constexpr uint32_t kMaxBins = 1U << RadixSortOneBlockPlan::kMaxDigitBits;
constexpr uint32_t kMaxPasses = RadixSortOneBlockPlan::kMaxPasses;
constexpr uint32_t kChunkBins = 16;  // the bin starts are summed in chunks of this many bins
constexpr uint32_t kItems = 4;       // a thread's pairs
constexpr uint32_t kPrefixWarps = 8;  // the warps whose counts of a digit are read at once
constexpr uint32_t kWarpPairs = RadixSortOneBlockPlan::kWarpPairs;
constexpr uint32_t kNoPlace = 0xffffffffU;

static_assert(kWarpPairs == 32U * kItems);
static_assert(kMaxBins / kChunkBins <= 32U);  // one warp is enough threads for the chunk sums
static_assert(kMaxBins % kChunkBins == 0U);

struct OneBlockParams {
    void* keys;
    uint32_t* values;
    const void* activeCount;  // null: the capacity
    uint32_t capacity;
    uint32_t activeCountBytes;  // 0, 4 or 8
    uint32_t warps;
    uint32_t passes;
    uint8_t digitBits[kMaxPasses];
    uint8_t digitShift[kMaxPasses];
};

__device__ __forceinline__ uint32_t activePairs(const void* activeCount, uint32_t activeCountBytes, uint32_t capacity) {
    if (activeCountBytes == 0) return capacity;
    const uint64_t active = activeCountBytes == 8 ? *static_cast<const uint64_t*>(activeCount)
                                                  : static_cast<uint64_t>(*static_cast<const uint32_t*>(activeCount));
    return active < capacity ? static_cast<uint32_t>(active) : capacity;
}

template <typename KeyT>
__device__ __forceinline__ uint32_t digitOf(KeyT key, uint32_t shift, uint32_t mask) {
    return static_cast<uint32_t>(static_cast<uint64_t>(key) >> shift) & mask;
}

// Where place p lies in the exchange memory: a one-to-one map of [0, 2^k) onto itself that carries the bits above a
// place's low five down into them. Shared memory has 32 banks, a word apart, and a warp's accesses to one bank go one
// after another. When the keys come in order (consecutive rows, as an embedding of positions has them) a warp's 32
// lanes hold 32 consecutive digits, each digit with the same number of pairs in the block, a multiple of 32: placed
// straight, the lanes' stores are a multiple of 32 words apart, all in one bank, 32 turns for each store (1634: two of
// the three passes of 4,096 consecutive 21-bit keys, 15,872 of the kernel's 17,947 store turns, the kernel twice as
// long as on random keys). Folded, places any power of two apart take at most two turns; random places are as before.
__device__ __forceinline__ uint32_t exchangeSlot(uint32_t place) { return place ^ (place >> 5) ^ (place >> 10); }

// Every pass of the sort over one block of p.warps x 32 threads, the pairs in registers between the passes. A warp
// holds 128 consecutive pairs: pair i is item (i / 32) % 4 of lane i % 32 of warp i / 128, so warps, a warp's items and
// an item's lanes are all in input order. In a pass a pair's place, sorted by digit, is
//   the block's pairs of smaller digits (binStart)
// + those of its digit in the warps before its own (warpCounts, summed over the warps in place)
// + those before it in its warp: the warp's count of the digit so far, kept as the warp goes through its items, plus
//   the lanes before it with the digit in its own item (ballots).
// The keys go to shared memory at their places (at exchangeSlot of them) and each thread reads back the keys now at
// its own pairs' indices; then the values the same way through the same memory. Keys of 16 bits take a word each
// there, as the values do. A place past the item count holds a reread of the last key: it
// ballots and reads a count like any lane, adds nothing, and is never written or read back.
// The sums over shared memory (a digit's counts over the warps, the bins of a chunk, the chunks and bins before a
// digit) are straight lines: a fixed number of loads, all issued before the first is used, and what lies past the
// pass's warps, bins or chunks is read (it is inside the arrays) and left out of the sum. As loops of one load a
// step, each waited for, a block of fewer threads than bins lost 0.3 to 0.5 us a pass for every further digit a
// thread took (1630's times).
// The launch bounds hold the kernel to 64 registers, as the pass kernels of RadixSort.cu are held.
template <typename KeyT, int MaxWarps>
__global__ void __launch_bounds__(MaxWarps * 32, 1024 / (MaxWarps * 32))
    radixSortOneBlockKernel(const __grid_constant__ OneBlockParams p) {
    constexpr uint32_t kMaxPairs = MaxWarps * kWarpPairs;
    constexpr uint32_t kCountBytes = MaxWarps * kMaxBins * sizeof(uint16_t);
    constexpr uint32_t kExchangeBytes = kMaxPairs * (sizeof(KeyT) > sizeof(uint32_t) ? sizeof(KeyT) : sizeof(uint32_t));
    // The warps' digit counts: 256 two-byte counts a warp, which is one 16-byte word a thread.
    __shared__ uint4 warpCountWords[(kCountBytes + 15U) / 16U];
    __shared__ uint4 exchange[(kExchangeBytes + 15U) / 16U];
    __shared__ uint32_t binCounts[kMaxBins];
    __shared__ uint32_t chunkSums[kMaxBins / kChunkBins];
    __shared__ uint32_t binStart[kMaxBins];
    uint16_t* warpCounts = reinterpret_cast<uint16_t*>(warpCountWords);
    static_assert((kMaxPairs & (kMaxPairs - 1U)) == 0U);  // exchangeSlot stays inside the exchange memory
    static_assert(MaxWarps % kPrefixWarps == 0);  // a batch of warps' counts stays inside warpCounts

    const uint32_t count = activePairs(p.activeCount, p.activeCountBytes, p.capacity);
    if (count == 0U) return;  // the whole block
    const uint32_t warps = p.warps;
    const uint32_t threads = warps * 32U;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31U;
    const uint32_t warp = tid >> 5;
    const uint32_t first = warp * kWarpPairs + lane;  // the thread's first pair; its items are 32 pairs apart

    // Every pair, all loads issued before anything else.
    KeyT* keys = static_cast<KeyT*>(p.keys);
    KeyT key[kItems];
    uint32_t value[kItems];
    for (uint32_t r = 0; r < kItems; ++r) {
        const uint32_t i = first + r * 32U;
        key[r] = keys[i < count ? i : count - 1U];
        value[r] = i < count ? p.values[i] : 0U;
    }
    warpCountWords[tid] = uint4{0U, 0U, 0U, 0U};
    __syncthreads();

    using ExchangeKeyT = std::conditional_t<(sizeof(KeyT) > sizeof(uint32_t)), uint64_t, uint32_t>;
    ExchangeKeyT* blockKeys = reinterpret_cast<ExchangeKeyT*>(exchange);
    uint32_t* blockValues = reinterpret_cast<uint32_t*>(exchange);
    for (uint32_t pass = 0; pass < p.passes; ++pass) {
        const uint32_t shift = p.digitShift[pass];
        const uint32_t bins = 1U << p.digitBits[pass];
        const uint32_t mask = bins - 1U;

        // A pair's rank among its warp's pairs of the same digit, as in RadixSort.cu's pass: for each item, every lane
        // reads the warp's count of its digit, then the first lane of each digit adds the item's pairs of it; the warp
        // is held together between the two, and before the next item reads. The lanes with the same digit are found
        // with one ballot a digit bit, the bits above the pass's being 0 in every lane.
        uint16_t* countsOfWarp = warpCounts + warp * bins;
        uint32_t rank[kItems];
        for (uint32_t r = 0; r < kItems; ++r) {
            const bool held = first + r * 32U < count;
            const uint32_t digit = digitOf(key[r], shift, mask);
            uint32_t same = __ballot_sync(0xffffffffU, held);
#pragma unroll
            for (uint32_t b = 0; b < RadixSortOneBlockPlan::kMaxDigitBits; ++b) {
                const bool one = (digit & (1U << b)) != 0U;
                const uint32_t ones = __ballot_sync(0xffffffffU, one);
                same &= one ? ones : ~ones;
            }
            const uint32_t earlier = __popc(same & ((1U << lane) - 1U));
            const uint32_t soFar = countsOfWarp[digit];
            __syncwarp();
            if (held && earlier == 0U) countsOfWarp[digit] = static_cast<uint16_t>(soFar + __popc(same));
            __syncwarp();
            rank[r] = soFar + earlier;
        }
        __syncthreads();

        // Digit g's pairs per warp become those before each warp. A block of fewer threads than bins gives a thread
        // several digits.
        for (uint32_t g = tid; g < bins; g += threads) {
            uint32_t inBlock = 0U;
            for (uint32_t w0 = 0; w0 < warps; w0 += kPrefixWarps) {
                uint16_t* const ofDigit = warpCounts + w0 * bins + g;
                uint32_t inWarp[kPrefixWarps];
#pragma unroll
                for (uint32_t k = 0; k < kPrefixWarps; ++k) inWarp[k] = ofDigit[k * bins];
#pragma unroll
                for (uint32_t k = 0; k < kPrefixWarps; ++k) {
                    const bool used = w0 + k < warps;
                    if (used) ofDigit[k * bins] = static_cast<uint16_t>(inBlock);
                    inBlock += used ? inWarp[k] : 0U;
                }
            }
            binCounts[g] = inBlock;
        }
        __syncthreads();
        const uint32_t chunks = (bins + kChunkBins - 1U) / kChunkBins;
        // Only a pass of fewer than 16 bins has a chunk that reaches past its bins, and that chunk is the last one,
        // whose sum no digit uses.
        if (tid < chunks) {
            uint32_t sum = 0U;
#pragma unroll
            for (uint32_t b = 0; b < kChunkBins; ++b) sum += binCounts[tid * kChunkBins + b];
            chunkSums[tid] = sum;
        }
        __syncthreads();
        for (uint32_t g = tid; g < bins; g += threads) {
            const uint32_t chunk = g / kChunkBins;
            const uint32_t inChunk = g % kChunkBins;
            uint32_t start = 0U;
#pragma unroll
            for (uint32_t c = 0; c < kMaxBins / kChunkBins; ++c) {
                const uint32_t ofChunk = chunkSums[c];
                start += c < chunk ? ofChunk : 0U;
            }
#pragma unroll
            for (uint32_t b = 0; b < kChunkBins; ++b) {
                const uint32_t inBin = binCounts[chunk * kChunkBins + b];
                start += b < inChunk ? inBin : 0U;
            }
            binStart[g] = start;
        }
        __syncthreads();

        // Each pair's place, and the keys go to shared memory in that order (place[] holds the places' slots).
        uint32_t place[kItems];
        for (uint32_t r = 0; r < kItems; ++r) {
            const uint32_t digit = digitOf(key[r], shift, mask);
            const uint32_t sorted = binStart[digit] + countsOfWarp[digit] + rank[r];
            place[r] = first + r * 32U < count ? exchangeSlot(sorted) : kNoPlace;
        }
        for (uint32_t r = 0; r < kItems; ++r) {
            if (place[r] != kNoPlace) blockKeys[place[r]] = key[r];
        }
        __syncthreads();

        // The keys now at the thread's own pairs; the counts are cleared for the next pass, whose rank loop comes
        // after the next barrier.
        for (uint32_t r = 0; r < kItems; ++r) {
            const uint32_t i = first + r * 32U;
            if (i < count) key[r] = static_cast<KeyT>(blockKeys[exchangeSlot(i)]);
        }
        warpCountWords[tid] = uint4{0U, 0U, 0U, 0U};
        __syncthreads();

        for (uint32_t r = 0; r < kItems; ++r) {
            if (place[r] != kNoPlace) blockValues[place[r]] = value[r];
        }
        __syncthreads();
        // No barrier after these reads: nothing writes the exchange memory again until four barriers into the next
        // pass.
        for (uint32_t r = 0; r < kItems; ++r) {
            const uint32_t i = first + r * 32U;
            if (i < count) value[r] = blockValues[exchangeSlot(i)];
        }
    }

    // In place: every load was issued before the first barrier. A warp's store is 32 consecutive pairs.
    for (uint32_t r = 0; r < kItems; ++r) {
        const uint32_t i = first + r * 32U;
        if (i < count) {
            keys[i] = key[r];
            p.values[i] = value[r];
        }
    }
}

template <typename KeyT, int MaxWarps>
void launchTyped(const OneBlockParams& p, cudaStream_t stream) {
    const uint32_t threads = p.warps * 32U;
    radixSortOneBlockKernel<KeyT, MaxWarps><<<1, threads, 0, stream>>>(p);
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace

uint32_t radixSortOneBlockMaxPairs(DataType keyDataType) {
    switch (keyDataType) {
        case DataType::UINT16:
        case DataType::UINT32: return 32U * kWarpPairs;
        case DataType::UINT64: return 16U * kWarpPairs;
        default: throw std::invalid_argument("Radix sort keys are UINT16, UINT32 or UINT64.");
    }
}

RadixSortOneBlockPlan planRadixSortPairsOneBlock(DataType keyDataType, uint64_t capacity, uint32_t keyBits) {
    const uint32_t maxPairs = radixSortOneBlockMaxPairs(keyDataType);
    if (capacity == 0 || capacity > maxPairs) {
        throw std::invalid_argument("One-block radix sort capacity must be 1 to " + std::to_string(maxPairs) +
                                    " pairs for this key type.");
    }
    // The key bits' check and the digits are the general sort's.
    const RadixSortPairsPlan general = planRadixSortPairs(keyDataType, capacity, keyBits);

    RadixSortOneBlockPlan plan;
    plan.keyDataType = keyDataType;
    plan.capacity = static_cast<uint32_t>(capacity);
    plan.keyBits = keyBits;
    plan.passes = general.passes;
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
        plan.digitBits[pass] = general.digitBits[pass];
        plan.digitShift[pass] = general.digitShift[pass];
    }
    plan.warps = (plan.capacity + kWarpPairs - 1U) / kWarpPairs;
    return plan;
}

void launchRadixSortPairsOneBlock(const RadixSortOneBlockPlan& plan,
                                  void* keys,
                                  uint32_t* values,
                                  const void* activeCount,
                                  DataType activeCountDataType,
                                  Stream stream) {
    if (plan.capacity == 0 || plan != planRadixSortPairsOneBlock(plan.keyDataType, plan.capacity, plan.keyBits)) {
        throw std::invalid_argument("One-block radix sort: the plan is not one planRadixSortPairsOneBlock made.");
    }
    if (keys == nullptr || values == nullptr) {
        throw std::invalid_argument("One-block radix sort: the keys and the values are required.");
    }
    uint32_t activeCountBytes = 0;
    if (activeCount != nullptr) {
        if (activeCountDataType != DataType::UINT32 && activeCountDataType != DataType::UINT64) {
            throw std::invalid_argument("One-block radix sort: the device item count is UINT32 or UINT64.");
        }
        activeCountBytes = activeCountDataType == DataType::UINT64 ? 8U : 4U;
    }

    OneBlockParams p{};
    p.keys = keys;
    p.values = values;
    p.activeCount = activeCount;
    p.capacity = plan.capacity;
    p.activeCountBytes = activeCountBytes;
    p.warps = plan.warps;
    p.passes = plan.passes;
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
        p.digitBits[pass] = plan.digitBits[pass];
        p.digitShift[pass] = plan.digitShift[pass];
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    switch (plan.keyDataType) {
        case DataType::UINT16: launchTyped<uint16_t, 32>(p, stream.getStream()); break;
        case DataType::UINT32: launchTyped<uint32_t, 32>(p, stream.getStream()); break;
        case DataType::UINT64: launchTyped<uint64_t, 16>(p, stream.getStream()); break;
        default: throw std::invalid_argument("Radix sort keys are UINT16, UINT32 or UINT64.");
    }
}

}  // namespace ThorImplementation
