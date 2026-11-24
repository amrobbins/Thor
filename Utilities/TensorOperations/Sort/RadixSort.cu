#include "Utilities/TensorOperations/Sort/RadixSort.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ThorImplementation {
namespace {

constexpr uint32_t kThreads = 256;  // the histogram kernel's block
constexpr uint32_t kMaxBins = 1U << RadixSortPairsPlan::kMaxDigitBits;
constexpr uint32_t kMaxPasses = RadixSortPairsPlan::kMaxPasses;
constexpr uint32_t kChunkBins = 16;                // the bin starts are summed in chunks of this many bins
constexpr uint32_t kHistogramItemsPerThread = 16;  // a histogram tile is 4,096 keys
constexpr uint64_t kMaxItems = (uint64_t{1} << 30) - 1;

// A lookback word: a count in the low 30 bits and what the count is. 0 is a row not yet written (the launch clears
// the scratch), so a tile with no pair of a digit still publishes a non-zero word.
constexpr uint32_t kLocalFlag = 1U << 30;   // the tile's own pairs of the digit
constexpr uint32_t kPrefixFlag = 1U << 31;  // those of the tile and of every tile before it
constexpr uint32_t kCountMask = kLocalFlag - 1U;
constexpr uint32_t kLookbackRows = 8;  // the rows a lookback reads at once

struct HistogramParams {
    const void* keys;
    uint32_t* histogram;      // [passes][kMaxBins], zero at launch
    const void* activeCount;  // null: the capacity
    uint64_t capacity;
    uint32_t activeCountBytes;  // 0, 4 or 8
    uint32_t passes;
    uint8_t digitBits[kMaxPasses];
    uint8_t digitShift[kMaxPasses];
};

struct PassParams {
    const void* keysIn;
    void* keysOut;
    const uint32_t* valuesIn;
    uint32_t* valuesOut;
    const uint32_t* histogram;  // this pass's [bins]
    uint32_t* lookback;         // this pass's [tiles][bins], zero at launch
    const void* activeCount;
    uint64_t capacity;
    uint32_t activeCountBytes;
    uint32_t shift;
    uint32_t bins;
};

__device__ __forceinline__ uint64_t activeItems(const void* activeCount, uint32_t activeCountBytes, uint64_t capacity) {
    if (activeCountBytes == 0) return capacity;
    const uint64_t active = activeCountBytes == 8 ? *static_cast<const uint64_t*>(activeCount)
                                                  : static_cast<uint64_t>(*static_cast<const uint32_t*>(activeCount));
    return active < capacity ? active : capacity;
}

// Lookback words are published and polled with relaxed atomics at device scope, as the scan kernel's carries are
// (SegmentedScanKernel.cuh): a word is a whole message written by one 32-bit store, and a relaxed device-scope load
// reads L2, so a poll sees the store once it lands.
__device__ __forceinline__ void publish(uint32_t* slot, uint32_t word) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ref(*slot);
    ref.store(word, cuda::memory_order_relaxed);
}

__device__ __forceinline__ uint32_t peek(uint32_t* slot) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ref(*slot);
    return ref.load(cuda::memory_order_relaxed);
}

template <typename KeyT>
__device__ __forceinline__ uint32_t digitOf(KeyT key, uint32_t shift, uint32_t mask) {
    return static_cast<uint32_t>(static_cast<uint64_t>(key) >> shift) & mask;
}

// Every pass's digit counts of the active keys, in one read of them: a tile counts into shared memory, then adds its
// non-zero bins to the launch's histogram. A thread issues the loads of all its keys before it counts any, so it
// waits for memory once rather than once a key (read one at a time, the waits were two thirds of the kernel's time).
// It counts a pass at a time, the pass's shift and mask in registers: a shift, a mask and an add a key (counted a key
// at a time with every pass's shift and bits read per key, the counting was 70% of the kernel's instructions, 13 a
// digit). A place past the tile's count rereads the tile's last key and is counted like any other; the last thread,
// whose last place is such a reread, takes the surplus back off that key's bins.
template <typename KeyT>
__global__ void __launch_bounds__(kThreads) radixSortHistogramKernel(const __grid_constant__ HistogramParams p) {
    constexpr uint32_t kTile = kThreads * kHistogramItemsPerThread;
    __shared__ uint32_t tileCounts[kMaxPasses * kMaxBins];

    const uint64_t items = activeItems(p.activeCount, p.activeCountBytes, p.capacity);
    const uint64_t tileBegin = static_cast<uint64_t>(blockIdx.x) * kTile;
    if (tileBegin >= items) return;
    const uint32_t count = items - tileBegin < kTile ? static_cast<uint32_t>(items - tileBegin) : kTile;
    const uint32_t tid = threadIdx.x;

    const KeyT* keys = static_cast<const KeyT*>(p.keys) + tileBegin;
    KeyT key[kHistogramItemsPerThread];
#pragma unroll
    for (uint32_t r = 0; r < kHistogramItemsPerThread; ++r) {
        const uint32_t i = r * kThreads + tid;
        key[r] = keys[i < count ? i : count - 1U];
    }

    const uint32_t words = p.passes * kMaxBins;
    for (uint32_t w = tid; w < words; w += kThreads) tileCounts[w] = 0U;
    __syncthreads();

    const uint32_t surplus = kTile - count;  // the places past the count
    const bool takesSurplus = surplus != 0U && tid == kThreads - 1U;
    for (uint32_t pass = 0; pass < p.passes; ++pass) {
        const uint32_t shift = p.digitShift[pass];
        const uint32_t mask = (1U << p.digitBits[pass]) - 1U;
        uint32_t* const counts = tileCounts + pass * kMaxBins;
#pragma unroll
        for (uint32_t r = 0; r < kHistogramItemsPerThread; ++r) {
            atomicAdd(&counts[static_cast<uint32_t>(key[r] >> shift) & mask], 1U);
        }
        if (takesSurplus) {
            atomicAdd(&counts[static_cast<uint32_t>(key[kHistogramItemsPerThread - 1U] >> shift) & mask], 0U - surplus);
        }
    }
    __syncthreads();

    for (uint32_t w = tid; w < words; w += kThreads) {
        const uint32_t count = tileCounts[w];
        if (count != 0U) atomicAdd(&p.histogram[w], count);
    }
}

// One pass over one tile of Warps x 32 x ItemsPerThread pairs. A warp holds 32 x ItemsPerThread consecutive pairs:
// pair i of the tile is item (i / 32) % ItemsPerThread of lane i % 32 of warp i / (32 x ItemsPerThread), so warps, a
// warp's items and an item's lanes are all in tile order. A pair's place in the tile, sorted by digit, is
//   the tile's pairs of smaller digits (tileStart)
// + those of its digit in the warps before its own (warpCounts, summed over the warps in place)
// + those before it in its warp: the warp's count of the digit so far, kept as the warp goes through its items, plus
//   the lanes before it with the digit in its own item (a warp match).
// The tile is put in that order in shared memory, keys then values, and written out from there: place j of the tile
// goes to j + binOffset[its digit], where binOffset is the pairs of smaller digits (the launch's histogram) plus those
// of the digit in the tiles before this one (the lookback) less tileStart. Consecutive threads write consecutive
// places of a digit's run, so the stores are whole sectors rather than one pair each.
// The launch bounds hold the kernel to 64 registers (1024 / threads blocks of that many threads fill an SM's 64K):
// without them the 8-item kernels take 75 to 84 and a block fewer fits on an SM, which measured slower at every size.
template <typename KeyT, int ItemsPerThread, int Warps>
__global__ void __launch_bounds__(Warps * 32, 1024 / (Warps * 32))
    radixSortPassKernel(const __grid_constant__ PassParams p) {
    constexpr uint32_t kBlockThreads = Warps * 32;
    constexpr uint32_t kWarpPairs = ItemsPerThread * 32;
    constexpr uint32_t kTile = ItemsPerThread * kBlockThreads;
    constexpr uint32_t kCountBytes = Warps * kMaxBins * sizeof(uint16_t);
    constexpr uint32_t kExchangeBytes = kTile * (sizeof(KeyT) > sizeof(uint32_t) ? sizeof(KeyT) : sizeof(uint32_t));
    // The warps' digit counts, and apart from them the tile's keys, then its values, in tile order: the keys go in
    // while the lookback runs (sharing the memory, they had to wait for it).
    __shared__ uint4 warpCountWords[(kCountBytes + 15U) / 16U];
    __shared__ uint4 exchange[(kExchangeBytes + 15U) / 16U];
    __shared__ uint32_t binCounts[kMaxBins];
    __shared__ uint32_t chunkSums[kMaxBins / kChunkBins];
    __shared__ uint32_t tileCounts[kMaxBins];
    __shared__ uint32_t tileChunkSums[kMaxBins / kChunkBins];
    __shared__ uint32_t tileStart[kMaxBins];
    __shared__ uint32_t binOffset[kMaxBins];
    uint16_t* warpCounts = reinterpret_cast<uint16_t*>(warpCountWords);

    const uint64_t items = activeItems(p.activeCount, p.activeCountBytes, p.capacity);
    const uint64_t tileBegin = static_cast<uint64_t>(blockIdx.x) * kTile;
    if (tileBegin >= items) return;  // the whole block: no tile after this one looks back past its own index
    const uint32_t count = items - tileBegin < kTile ? static_cast<uint32_t>(items - tileBegin) : kTile;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31U;
    const uint32_t warp = tid >> 5;
    const uint32_t bins = p.bins;
    const uint32_t shift = p.shift;
    const uint32_t mask = bins - 1U;
    const uint32_t first = warp * kWarpPairs + lane;  // the thread's first pair; its items are 32 pairs apart

    // The tile's keys, all loads issued before anything else. A place past the tile's count rereads its last key.
    // The values are loaded once the keys are in shared memory, so their wait overlaps the lookback: held from the
    // start, they pushed the 8-item kernels past 64 registers into spills, whose stores waited on the loads.
    const KeyT* keysIn = static_cast<const KeyT*>(p.keysIn) + tileBegin;
    KeyT key[ItemsPerThread];
    for (int r = 0; r < ItemsPerThread; ++r) {
        const uint32_t i = first + static_cast<uint32_t>(r) * 32U;
        key[r] = keysIn[i < count ? i : count - 1U];
    }

    const uint32_t zeroVectors = (static_cast<uint32_t>(Warps) * bins * 2U + 15U) / 16U;
    for (uint32_t v = tid; v < zeroVectors; v += kBlockThreads) warpCountWords[v] = uint4{0U, 0U, 0U, 0U};
    if (tid < bins) binCounts[tid] = p.histogram[tid];
    __syncthreads();

    // A pair's rank among its warp's pairs of the same digit. For each item, every lane reads the warp's count of its
    // digit, then the first lane of each digit adds the item's pairs of it; the warp is held together between the
    // two, and before the next item reads.
    // The lanes with the same digit are found with one ballot a digit bit, as CUB does: the hardware match
    // (MATCH.ANY) was the largest stall of this loop. The item's code is a straight line: a place past the tile's
    // count holds a reread key, so its digit is a bin like any other and it reads a count like any other; the first
    // ballot (the lanes within the count) keeps it out of the others' ranks and it adds nothing. Every digit bit a
    // digit can have is balloted, the ones above the pass's being 0 in every lane. (With a test of the count around
    // the digit and a test of the pass's bits around each ballot, the compiler at times rebuilt the digit from the
    // key inside every ballot's branch: twice the instructions.)
    // A digit is two instructions from its key, so nothing but the rank is kept for later.
    uint16_t* countsOfWarp = warpCounts + warp * bins;
    uint32_t rank[ItemsPerThread];  // under 32 x ItemsPerThread
    for (int r = 0; r < ItemsPerThread; ++r) {
        const bool held = first + static_cast<uint32_t>(r) * 32U < count;
        const uint32_t digit = digitOf(key[r], shift, mask);
        uint32_t same = __ballot_sync(0xffffffffU, held);
#pragma unroll
        for (uint32_t b = 0; b < RadixSortPairsPlan::kMaxDigitBits; ++b) {
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
    const uint32_t chunks = (bins + kChunkBins - 1U) / kChunkBins;
    if (tid < chunks) {
        uint32_t sum = 0U;
        for (uint32_t b = tid * kChunkBins; b < bins && b < (tid + 1U) * kChunkBins; ++b) sum += binCounts[b];
        chunkSums[tid] = sum;
    }
    __syncthreads();

    // Thread g owns digit g. Its pairs per warp become those before each warp, and the tile publishes its own count
    // of the digit at once, so a later tile's lookback waits as little as it can.
    uint32_t* own = p.lookback + static_cast<uint64_t>(blockIdx.x) * bins + tid;
    if (tid < bins) {
        const uint32_t g = tid;
        uint32_t inTile = 0U;
        for (uint32_t w = 0; w < static_cast<uint32_t>(Warps); ++w) {
            const uint32_t inWarp = warpCounts[w * bins + g];
            warpCounts[w * bins + g] = static_cast<uint16_t>(inTile);
            inTile += inWarp;
        }
        tileCounts[g] = inTile;
        publish(own, inTile | (blockIdx.x == 0U ? kPrefixFlag : kLocalFlag));
    }
    __syncthreads();
    if (tid < chunks) {
        uint32_t sum = 0U;
        for (uint32_t b = tid * kChunkBins; b < bins && b < (tid + 1U) * kChunkBins; ++b) sum += tileCounts[b];
        tileChunkSums[tid] = sum;
    }
    __syncthreads();

    // Where digit g starts in the output and in the tile.
    uint32_t start = 0U;
    uint32_t startInTile = 0U;
    if (tid < bins) {
        const uint32_t g = tid;
        for (uint32_t c = 0; c < g / kChunkBins; ++c) {
            start += chunkSums[c];
            startInTile += tileChunkSums[c];
        }
        for (uint32_t b = g - g % kChunkBins; b < g; ++b) {
            start += binCounts[b];
            startInTile += tileCounts[b];
        }
        tileStart[g] = startInTile;
    }
    __syncthreads();

    // Each pair's place in the tile (a place past the count takes kTile), and the tile's keys go to shared memory in
    // that order; then the values are loaded. None of it waits for the tiles before this one, so all of it is done
    // before the lookback: the other warps' share of it runs while the digits' threads walk, and the walk starts
    // later, when more of the tiles behind have published their prefixes.
    uint32_t place[ItemsPerThread];
    for (int r = 0; r < ItemsPerThread; ++r) {
        const uint32_t digit = digitOf(key[r], shift, mask);
        const uint32_t inTile = tileStart[digit] + countsOfWarp[digit] + rank[r];
        place[r] = first + static_cast<uint32_t>(r) * 32U < count ? inTile : kTile;
    }
    KeyT* tileKeys = reinterpret_cast<KeyT*>(exchange);
    for (int r = 0; r < ItemsPerThread; ++r) {
        if (place[r] != kTile) tileKeys[place[r]] = key[r];
    }
    const uint32_t* valuesIn = p.valuesIn + tileBegin;
    uint32_t value[ItemsPerThread];
    for (int r = 0; r < ItemsPerThread; ++r) {
        const uint32_t i = first + static_cast<uint32_t>(r) * 32U;
        value[r] = i < count ? valuesIn[i] : 0U;
    }

    // The tiles before this one, for digit g: the tile adds the words behind it until it meets a prefix, and
    // publishes its own prefix, so no tile waits for more than a tile's first store.
    if (tid < bins) {
        const uint32_t g = tid;
        // The rows behind are read kLookbackRows at a time, nearest first, all the loads in flight together, so a
        // walk back over n rows waits for about n / kLookbackRows round trips to L2 rather than n. A row not yet
        // written ends the batch at it, and the next batch starts from it.
        uint32_t before = 0U;
        if (blockIdx.x != 0U) {
            uint32_t* next = own;  // the row after the nearest one not yet added
            uint32_t rowsBehind = blockIdx.x;
            bool found = false;
            while (!found) {
                const uint32_t batch = rowsBehind < kLookbackRows ? rowsBehind : kLookbackRows;
                uint32_t word[kLookbackRows];
#pragma unroll
                for (uint32_t k = 0; k < kLookbackRows; ++k) {
                    word[k] = k < batch ? peek(next - (k + 1U) * bins) : 0U;
                }
                uint32_t added = 0U;
                bool stop = false;
#pragma unroll
                for (uint32_t k = 0; k < kLookbackRows; ++k) {
                    if (stop || word[k] == 0U) {
                        stop = true;
                        continue;
                    }
                    before += word[k] & kCountMask;
                    added = k + 1U;
                    if ((word[k] & kPrefixFlag) != 0U) {
                        found = true;
                        stop = true;
                    }
                }
                if (!found) {
                    next -= static_cast<uint64_t>(added) * bins;
                    rowsBehind -= added;
                    if (added < batch) __nanosleep(32);
                }
            }
            publish(own, (before + tileCounts[g]) | kPrefixFlag);
        }
        binOffset[g] = start + before - startInTile;  // not below 0: the output's pairs before g include the tile's
    }
    __syncthreads();

    KeyT* keysOut = static_cast<KeyT*>(p.keysOut);
    uint32_t to[ItemsPerThread];
    for (int r = 0; r < ItemsPerThread; ++r) {
        const uint32_t j = tid + static_cast<uint32_t>(r) * kBlockThreads;
        if (j >= count) continue;
        const KeyT k = tileKeys[j];
        to[r] = j + binOffset[digitOf(k, shift, mask)];
        keysOut[to[r]] = k;
    }
    __syncthreads();

    uint32_t* tileValues = reinterpret_cast<uint32_t*>(exchange);
    for (int r = 0; r < ItemsPerThread; ++r) {
        if (place[r] != kTile) tileValues[place[r]] = value[r];
    }
    __syncthreads();

    for (int r = 0; r < ItemsPerThread; ++r) {
        const uint32_t j = tid + static_cast<uint32_t>(r) * kBlockThreads;
        if (j < count) p.valuesOut[to[r]] = tileValues[j];
    }
}

template <typename KeyT>
void launchPass(const RadixSortPairsPlan& plan, const PassParams& p, cudaStream_t stream) {
    switch (plan.tilePairs) {
        case 512: radixSortPassKernel<KeyT, 2, 8><<<plan.tiles, 256, 0, stream>>>(p); break;
        case 4096: radixSortPassKernel<KeyT, 8, 16><<<plan.tiles, 512, 0, stream>>>(p); break;
        default: throw std::invalid_argument("Radix sort: a tile holds 512 or 4096 pairs.");
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename KeyT>
void launchTyped(const RadixSortPairsPlan& plan,
                 void* keys,
                 void* keysAlternate,
                 uint32_t* values,
                 uint32_t* valuesAlternate,
                 uint32_t* scratch,
                 const void* activeCount,
                 uint32_t activeCountBytes,
                 cudaStream_t stream) {
    HistogramParams h{};
    h.keys = keys;
    h.histogram = scratch;
    h.activeCount = activeCount;
    h.capacity = plan.capacity;
    h.activeCountBytes = activeCountBytes;
    h.passes = plan.passes;
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
        h.digitBits[pass] = plan.digitBits[pass];
        h.digitShift[pass] = plan.digitShift[pass];
    }
    radixSortHistogramKernel<KeyT><<<plan.histogramTiles, kThreads, 0, stream>>>(h);
    CUDA_CHECK(cudaPeekAtLastError());

    uint32_t* lookback = scratch + plan.passes * kMaxBins;
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
        const bool fromFirst = pass % 2U == 0U;
        PassParams p{};
        p.keysIn = fromFirst ? keys : keysAlternate;
        p.keysOut = fromFirst ? keysAlternate : keys;
        p.valuesIn = fromFirst ? values : valuesAlternate;
        p.valuesOut = fromFirst ? valuesAlternate : values;
        p.histogram = scratch + pass * kMaxBins;
        p.lookback = lookback;
        p.activeCount = activeCount;
        p.capacity = plan.capacity;
        p.activeCountBytes = activeCountBytes;
        p.shift = plan.digitShift[pass];
        p.bins = 1U << plan.digitBits[pass];
        launchPass<KeyT>(plan, p, stream);
        lookback += static_cast<uint64_t>(plan.tiles) * p.bins;
    }
}

uint32_t keyTypeBits(DataType keyDataType) {
    switch (keyDataType) {
        case DataType::UINT16: return 16;
        case DataType::UINT32: return 32;
        case DataType::UINT64: return 64;
        default: throw std::invalid_argument("Radix sort keys are UINT16, UINT32 or UINT64.");
    }
}

}  // namespace

RadixSortPairsPlan planRadixSortPairs(DataType keyDataType, uint64_t capacity, uint32_t keyBits, uint32_t tilePairs) {
    const uint32_t typeBits = keyTypeBits(keyDataType);
    if (capacity == 0 || capacity > kMaxItems) {
        throw std::invalid_argument("Radix sort capacity must be 1 to 2^30 - 1 pairs.");
    }
    if (keyBits == 0 || keyBits > typeBits) {
        throw std::invalid_argument("Radix sort key bits must be 1 to the key type's " + std::to_string(typeBits) +
                                    ".");
    }
    if (tilePairs != 0 && tilePairs != 512 && tilePairs != 4096) {
        throw std::invalid_argument("Radix sort: a tile holds 512 or 4096 pairs (0 chooses).");
    }

    RadixSortPairsPlan plan;
    plan.keyDataType = keyDataType;
    plan.capacity = capacity;
    plan.keyBits = keyBits;
    plan.passes = (keyBits + RadixSortPairsPlan::kMaxDigitBits - 1U) / RadixSortPairsPlan::kMaxDigitBits;
    // Digits as even as the bits allow, the wider ones first: 21 bits are 7, 7, 7; 17 are 6, 6, 5.
    uint32_t shift = 0;
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
        const uint32_t bits = keyBits / plan.passes + (pass < keyBits % plan.passes ? 1U : 0U);
        plan.digitBits[pass] = static_cast<uint8_t>(bits);
        plan.digitShift[pass] = static_cast<uint8_t>(shift);
        shift += bits;
    }

    // Measured on 1609 (sm_120): the 512 tile is fastest up to 262144 pairs with digits of up to 7 bits; with 8-bit
    // digits (256 bins, 2 pairs a bin in a 512 tile) the 4096 tile is faster from 262144; above, the 4096 tile.
    if (tilePairs == 0) {
        const bool wideDigits = plan.digitBits[0] == RadixSortPairsPlan::kMaxDigitBits;
        tilePairs = capacity <= (wideDigits ? 131072U : 262144U) ? 512U : 4096U;
    }
    plan.tilePairs = tilePairs;
    plan.tiles = static_cast<uint32_t>((capacity + tilePairs - 1) / tilePairs);
    const uint64_t histogramTile = static_cast<uint64_t>(kHistogramItemsPerThread) * kThreads;
    plan.histogramTiles = static_cast<uint32_t>((capacity + histogramTile - 1) / histogramTile);
    plan.scratchWords = static_cast<uint64_t>(plan.passes) * kMaxBins;
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
        plan.scratchWords += static_cast<uint64_t>(plan.tiles) << plan.digitBits[pass];
    }
    plan.sortedInAlternate = plan.passes % 2U == 1U;
    return plan;
}

void launchRadixSortPairs(const RadixSortPairsPlan& plan,
                          void* keys,
                          void* keysAlternate,
                          uint32_t* values,
                          uint32_t* valuesAlternate,
                          uint32_t* scratch,
                          const void* activeCount,
                          DataType activeCountDataType,
                          Stream stream) {
    if (plan != planRadixSortPairs(plan.keyDataType, plan.capacity, plan.keyBits, plan.tilePairs)) {
        throw std::invalid_argument("Radix sort: the plan is not one planRadixSortPairs made.");
    }
    if (keys == nullptr || keysAlternate == nullptr || values == nullptr || valuesAlternate == nullptr ||
        scratch == nullptr) {
        throw std::invalid_argument("Radix sort: the keys, the values, their alternates and the scratch are required.");
    }
    if (keys == keysAlternate || values == valuesAlternate) {
        throw std::invalid_argument("Radix sort: an array and its alternate must be two arrays.");
    }
    uint32_t activeCountBytes = 0;
    if (activeCount != nullptr) {
        if (activeCountDataType != DataType::UINT32 && activeCountDataType != DataType::UINT64) {
            throw std::invalid_argument("Radix sort: the device item count is UINT32 or UINT64.");
        }
        activeCountBytes = activeCountDataType == DataType::UINT64 ? 8U : 4U;
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    CUDA_CHECK(cudaMemsetAsync(scratch, 0, plan.scratchWords * sizeof(uint32_t), stream.getStream()));
    switch (plan.keyDataType) {
        case DataType::UINT16:
            launchTyped<uint16_t>(plan,
                                  keys,
                                  keysAlternate,
                                  values,
                                  valuesAlternate,
                                  scratch,
                                  activeCount,
                                  activeCountBytes,
                                  stream.getStream());
            break;
        case DataType::UINT32:
            launchTyped<uint32_t>(plan,
                                  keys,
                                  keysAlternate,
                                  values,
                                  valuesAlternate,
                                  scratch,
                                  activeCount,
                                  activeCountBytes,
                                  stream.getStream());
            break;
        case DataType::UINT64:
            launchTyped<uint64_t>(plan,
                                  keys,
                                  keysAlternate,
                                  values,
                                  valuesAlternate,
                                  scratch,
                                  activeCount,
                                  activeCountBytes,
                                  stream.getStream());
            break;
        default: throw std::invalid_argument("Radix sort keys are UINT16, UINT32 or UINT64.");
    }
}

}  // namespace ThorImplementation
