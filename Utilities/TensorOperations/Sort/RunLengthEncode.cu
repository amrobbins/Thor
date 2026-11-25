#include "Utilities/TensorOperations/Sort/RunLengthEncode.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ThorImplementation {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kItemsPerThread = 8;
constexpr uint32_t kTile = kThreads * kItemsPerThread;  // 2,048 keys
constexpr uint32_t kPaddedTile = kTile + kTile / 32;    // a word of padding every 32 keys, see paddedIndex
constexpr uint64_t kMaxItems = (uint64_t{1} << 30) - 1;
constexpr uint32_t kLookbackRows = 8;

// A lookback word: what the tile publishes, (heads, last). heads is the run heads (bits 32 to 61), last the position
// of the latest of them plus one (bits 0 to 31; 0 when there is none), then what the word covers. 0 is a row not yet
// written (the launch clears the scratch); a flag is always set in a written one, so a tile with no head still
// publishes a non-zero word. Words combine as (h1, l1) then (h2, l2) = (h1 + h2, l2 != 0 ? l2 : l1).
constexpr uint64_t kLocalFlag = uint64_t{1} << 62;   // the tile's own heads
constexpr uint64_t kPrefixFlag = uint64_t{1} << 63;  // those of the tile and of every tile before it
constexpr uint32_t kHeadsMask = (1U << 30) - 1U;

struct Params {
    const void* keys;
    void* uniqueKeys;
    uint32_t* runCounts;
    uint32_t* runOffsets;
    uint32_t* numRuns;
    uint64_t* lookback;
    const void* activeCount;
    uint64_t capacity;
    uint32_t activeCountBytes;  // 0, 4 or 8
};

__device__ __forceinline__ uint64_t activeItems(const void* activeCount, uint32_t activeCountBytes, uint64_t capacity) {
    if (activeCountBytes == 0) return capacity;
    const uint64_t active = activeCountBytes == 8 ? *static_cast<const uint64_t*>(activeCount)
                                                  : static_cast<uint64_t>(*static_cast<const uint32_t*>(activeCount));
    return active < capacity ? active : capacity;
}

// Relaxed device-scope atomics, as the radix sort's lookback: a word is a whole message written by one 64-bit store.
__device__ __forceinline__ void publish(uint64_t* slot, uint64_t word) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(*slot);
    ref.store(word, cuda::memory_order_relaxed);
}

__device__ __forceinline__ uint64_t peek(uint64_t* slot) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(*slot);
    return ref.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ uint64_t lookbackWord(uint64_t flag, uint32_t heads, uint32_t last) {
    return flag | static_cast<uint64_t>(heads) << 32 | last;
}

// Thread t holds keys 8t to 8t + 7 of the tile. Stored with a word of padding every 32 keys, a warp's reads of its
// threads' k-th keys fall in 32 different banks (4-byte keys) instead of 4.
__device__ __forceinline__ uint32_t paddedIndex(uint32_t i) { return i + (i >> 5); }

// One tile of 2,048 keys. A key is a run head when it is the first key or differs from the key before it. The block
// counts its heads and finds the latest (a scan over its threads in shared memory), publishes that, adds what the
// tiles before it published (nearest first, until a prefix), and publishes its prefix. Then the tile's runs go to
// shared memory in order and out from there (see below); the last key's tile writes the length of the last run and
// the number of runs. Integer counts, so the result is the same whatever the lookback's timing.
template <typename KeyT>
__global__ void __launch_bounds__(kThreads) runLengthEncodeKernel(const __grid_constant__ Params p) {
    __shared__ KeyT tileKeys[kPaddedTile];
    __shared__ uint32_t scanHeads[kThreads];
    __shared__ uint32_t scanLast[kThreads];
    __shared__ uint32_t warpHeads[kWarps];
    __shared__ uint32_t warpLast[kWarps];
    __shared__ uint32_t beforeHeads;
    __shared__ uint32_t beforeLast;
    __shared__ uint32_t ownHeads;          // the tile's runs that start in it
    __shared__ KeyT runKeys[kTile];        // the tile's runs in order: their keys
    __shared__ uint32_t runStarts[kTile];  // and their first positions
    __shared__ KeyT keyBefore;

    const uint64_t items = activeItems(p.activeCount, p.activeCountBytes, p.capacity);
    const uint32_t tid = threadIdx.x;
    if (items == 0) {
        if (blockIdx.x == 0 && tid == 0) *p.numRuns = 0U;
        return;
    }
    const uint64_t tileBegin = static_cast<uint64_t>(blockIdx.x) * kTile;
    if (tileBegin >= items) return;  // the whole block: no tile after this one looks back past its own index
    const uint32_t count = items - tileBegin < kTile ? static_cast<uint32_t>(items - tileBegin) : kTile;
    const uint32_t lane = tid & 31U;
    const uint32_t warp = tid >> 5;

    const KeyT* keys = static_cast<const KeyT*>(p.keys) + tileBegin;
    for (uint32_t r = 0; r < kItemsPerThread; ++r) {
        const uint32_t i = r * kThreads + tid;
        if (i < count) tileKeys[paddedIndex(i)] = keys[i];
    }
    if (tid == 0 && tileBegin != 0) keyBefore = keys[-1];
    __syncthreads();

    const uint32_t first = tid * kItemsPerThread;
    const uint32_t tileBegin32 = static_cast<uint32_t>(tileBegin);  // under 2^30
    uint32_t heads = 0U;
    uint32_t last = 0U;
    uint32_t headBits = 0U;
    for (uint32_t k = 0; k < kItemsPerThread; ++k) {
        const uint32_t j = first + k;
        if (j >= count) break;
        const KeyT key = tileKeys[paddedIndex(j)];
        const bool head = j == 0 ? tileBegin == 0 || key != keyBefore : key != tileKeys[paddedIndex(j - 1U)];
        if (head) {
            ++heads;
            last = tileBegin32 + j + 1U;
            headBits |= 1U << k;
        }
    }

    // Inclusive scan of (heads, last) over the warp's threads, then over the warps.
    scanHeads[tid] = heads;
    scanLast[tid] = last;
    __syncwarp();
    for (uint32_t offset = 1; offset < 32U; offset <<= 1) {
        uint32_t leftHeads = 0U;
        uint32_t leftLast = 0U;
        if (lane >= offset) {
            leftHeads = scanHeads[tid - offset];
            leftLast = scanLast[tid - offset];
        }
        __syncwarp();
        if (lane >= offset) {
            scanHeads[tid] += leftHeads;
            if (scanLast[tid] == 0U) scanLast[tid] = leftLast;
        }
        __syncwarp();
    }
    if (lane == 31U) {
        warpHeads[warp] = scanHeads[tid];
        warpLast[warp] = scanLast[tid];
    }
    __syncthreads();

    if (tid == 0) {
        // The warps' exclusive prefixes in place, and the tile's own (heads, last).
        uint32_t tileHeads = 0U;
        uint32_t tileLast = 0U;
        for (uint32_t w = 0; w < kWarps; ++w) {
            const uint32_t h = warpHeads[w];
            const uint32_t l = warpLast[w];
            warpHeads[w] = tileHeads;
            warpLast[w] = tileLast;
            tileHeads += h;
            if (l != 0U) tileLast = l;
        }

        uint64_t* own = p.lookback + blockIdx.x;
        uint32_t heads0 = 0U;
        uint32_t last0 = 0U;
        if (blockIdx.x == 0U) {
            publish(own, lookbackWord(kPrefixFlag, tileHeads, tileLast));
        } else {
            publish(own, lookbackWord(kLocalFlag, tileHeads, tileLast));
            // The rows behind are read kLookbackRows at a time, nearest first, all the loads in flight together. A
            // row not yet written ends the batch at it, and the next batch starts from it.
            uint64_t* next = own;  // the row after the nearest one not yet added
            uint32_t rowsBehind = blockIdx.x;
            bool found = false;
            while (!found) {
                const uint32_t batch = rowsBehind < kLookbackRows ? rowsBehind : kLookbackRows;
                uint64_t word[kLookbackRows];
#pragma unroll
                for (uint32_t k = 0; k < kLookbackRows; ++k) word[k] = k < batch ? peek(next - (k + 1U)) : 0U;
                uint32_t added = 0U;
                bool stop = false;
#pragma unroll
                for (uint32_t k = 0; k < kLookbackRows; ++k) {
                    if (stop || word[k] == 0U) {
                        stop = true;
                        continue;
                    }
                    heads0 += static_cast<uint32_t>(word[k] >> 32) & kHeadsMask;
                    if (last0 == 0U) last0 = static_cast<uint32_t>(word[k]);  // the nearest head is the latest
                    added = k + 1U;
                    if ((word[k] & kPrefixFlag) != 0U) {
                        found = true;
                        stop = true;
                    }
                }
                if (!found) {
                    next -= added;
                    rowsBehind -= added;
                    if (added < batch) __nanosleep(32);
                }
            }
            publish(own, lookbackWord(kPrefixFlag, heads0 + tileHeads, tileLast != 0U ? tileLast : last0));
        }
        beforeHeads = heads0;
        beforeLast = last0;
        ownHeads = tileHeads;
    }
    __syncthreads();

    // The tile's runs, in order, into shared memory: each head at its run's index in the tile.
    uint32_t local = warpHeads[warp] + (lane != 0U ? scanHeads[tid - 1U] : 0U);
    for (uint32_t k = 0; k < kItemsPerThread; ++k) {
        if ((headBits >> k) & 1U) {
            const uint32_t j = first + k;
            runKeys[local] = tileKeys[paddedIndex(j)];
            runStarts[local] = tileBegin32 + j;
            ++local;
        }
    }
    __syncthreads();

    // Then out to the runs' places, consecutive threads writing consecutive runs (written by its head, each run's
    // output was one sector a run). A run's count is the next run's start less its own; the count of the tile's last
    // run is written by the first head of a later tile, or below by the last key's tile; that of the run in progress
    // when the tile begins (one that started in an earlier tile) by this tile's first head.
    const uint32_t tileHeads = ownHeads;
    const uint32_t base = beforeHeads;
    KeyT* uniqueKeys = static_cast<KeyT*>(p.uniqueKeys);
    for (uint32_t i = tid; i < tileHeads; i += kThreads) {
        uniqueKeys[base + i] = runKeys[i];
        p.runOffsets[base + i] = runStarts[i];
        if (i + 1U < tileHeads) p.runCounts[base + i] = runStarts[i + 1U] - runStarts[i];
    }
    if (tid == 0) {
        if (tileHeads != 0U && base != 0U) p.runCounts[base - 1U] = runStarts[0] - (beforeLast - 1U);
        if (tileBegin + count == items) {  // the last key: position 0 is a head, so there is a run
            const uint32_t runs = base + tileHeads;
            const uint32_t lastStart = tileHeads != 0U ? runStarts[tileHeads - 1U] : beforeLast - 1U;
            p.runCounts[runs - 1U] = static_cast<uint32_t>(items) - lastStart;
            *p.numRuns = runs;
        }
    }
}

uint32_t keyTypeBytes(DataType keyDataType) {
    switch (keyDataType) {
        case DataType::UINT16: return 2;
        case DataType::UINT32: return 4;
        case DataType::UINT64: return 8;
        default: throw std::invalid_argument("Run-length encode keys are UINT16, UINT32 or UINT64.");
    }
}

template <typename KeyT>
void launchTyped(const RunLengthEncodePlan& plan, const Params& p, cudaStream_t stream) {
    runLengthEncodeKernel<KeyT><<<plan.tiles, kThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace

RunLengthEncodePlan planRunLengthEncode(DataType keyDataType, uint64_t capacity) {
    (void)keyTypeBytes(keyDataType);
    if (capacity == 0 || capacity > kMaxItems) {
        throw std::invalid_argument("Run-length encode capacity must be 1 to 2^30 - 1 keys.");
    }
    RunLengthEncodePlan plan;
    plan.keyDataType = keyDataType;
    plan.capacity = capacity;
    plan.tiles = static_cast<uint32_t>((capacity + kTile - 1) / kTile);
    plan.scratchWords = plan.tiles;
    return plan;
}

void launchRunLengthEncode(const RunLengthEncodePlan& plan,
                           const void* keys,
                           void* uniqueKeys,
                           uint32_t* runCounts,
                           uint32_t* runOffsets,
                           uint32_t* numRuns,
                           uint64_t* scratch,
                           const void* activeCount,
                           DataType activeCountDataType,
                           Stream stream,
                           bool clearScratch) {
    if (plan != planRunLengthEncode(plan.keyDataType, plan.capacity)) {
        throw std::invalid_argument("Run-length encode: the plan is not one planRunLengthEncode made.");
    }
    if (keys == nullptr || uniqueKeys == nullptr || runCounts == nullptr || runOffsets == nullptr ||
        numRuns == nullptr || scratch == nullptr) {
        throw std::invalid_argument("Run-length encode: the keys, the outputs and the scratch are required.");
    }
    if (keys == uniqueKeys) {
        throw std::invalid_argument("Run-length encode: the unique keys must not be written over the keys.");
    }
    uint32_t activeCountBytes = 0;
    if (activeCount != nullptr) {
        if (activeCountDataType != DataType::UINT32 && activeCountDataType != DataType::UINT64) {
            throw std::invalid_argument("Run-length encode: the device item count is UINT32 or UINT64.");
        }
        activeCountBytes = activeCountDataType == DataType::UINT64 ? 8U : 4U;
    }

    Params p{};
    p.keys = keys;
    p.uniqueKeys = uniqueKeys;
    p.runCounts = runCounts;
    p.runOffsets = runOffsets;
    p.numRuns = numRuns;
    p.lookback = scratch;
    p.activeCount = activeCount;
    p.capacity = plan.capacity;
    p.activeCountBytes = activeCountBytes;

    ScopedGpu scopedGpu(stream.getGpuNum());
    if (clearScratch) {
        CUDA_CHECK(cudaMemsetAsync(scratch, 0, plan.scratchWords * sizeof(uint64_t), stream.getStream()));
    }
    switch (plan.keyDataType) {
        case DataType::UINT16: launchTyped<uint16_t>(plan, p, stream.getStream()); break;
        case DataType::UINT32: launchTyped<uint32_t>(plan, p, stream.getStream()); break;
        case DataType::UINT64: launchTyped<uint64_t>(plan, p, stream.getStream()); break;
        default: throw std::invalid_argument("Run-length encode keys are UINT16, UINT32 or UINT64.");
    }
}

}  // namespace ThorImplementation
