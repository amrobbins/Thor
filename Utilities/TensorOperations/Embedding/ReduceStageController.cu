#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ThorImplementation {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;
static_assert(kWarps >= 4, "The four grid updates go to four warps (writeTotals).");
constexpr uint32_t kItemsPerThread = 8;
constexpr uint32_t kTile = kThreads * kItemsPerThread;  // 2,048 keys
constexpr uint32_t kPaddedTile = kTile + kTile / 32;    // a word of padding every 32 entries, see paddedIndex
constexpr uint64_t kMaxItems = (uint64_t{1} << 30) - 1;
constexpr uint32_t kLookbackRows = 8;

// Lookback words: what a tile publishes, then what the word covers. 0 is a row not yet written (the scratch is cleared
// before a launch); a flag is always set in a written one. Three lookbacks, a word a tile in each:
// - runs (thread 0): the tile's run heads (bits 32 to 61) and the position of the latest of them plus one (bits 0 to
//   31; 0 when there is none). Words combine as (h1, l1) then (h2, l2) = (h1 + h2, l2 != 0 ? l2 : l1).
// - rows (thread 0): the low runs the tile completes (bits 0 to 30) and its high runs (31 to 61).
// - ultra (thread 32): its ultra-high runs' partials (bits 0 to 31) and its ultra-high runs (32 to 61).
// Counts are at most the keys, under 2^30, so adding words never carries from one count into the next.
constexpr uint64_t kLocalFlag = uint64_t{1} << 62;   // the tile's own
constexpr uint64_t kPrefixFlag = uint64_t{1} << 63;  // the tile's and every tile's before it
constexpr uint64_t kCountsMask = kLocalFlag - 1U;
constexpr uint32_t kHeadsMask = (1U << 30) - 1U;
constexpr uint32_t kHighShift = 31;
constexpr uint32_t kUltraShift = 32;

struct RunsParams {
    const void* keys;  // the sorted row keys
    void* rows;        // each run's key
    uint32_t* runCounts;
    uint32_t* runOffsets;
    uint32_t* numRuns;
    void* numRows;  // the valid runs: all but a last run of the sentinel row
    uint4* lowRunEntries;  // (run, first position, length, key's low 32 bits)
    uint32_t* highRunRows;
    uint32_t* ultraRunRows;
    uint32_t* ultraRunPartialCounts;
    uint32_t* ultraRunPartialOffsets;
    uint32_t* numUltraPartials;
    uint32_t* numLowRunRows;
    uint32_t* numHighRunRows;
    uint32_t* numUltraRunRows;
    uint64_t* runsLookback;
    uint64_t* rowsLookback;
    uint64_t* ultraLookback;
    const void* activeCount;
    uint64_t capacity;
    uint32_t activeCountBytes;  // 0, 4 or 8
    uint64_t vocabularySize;
    uint32_t lowRunMax;
    uint32_t ultraRunMin;
    uint32_t ultraTokensPerPartial;
    const cudaGraphDeviceNode_t* lowReduceNode;
    const cudaGraphDeviceNode_t* highReduceNode;
    const cudaGraphDeviceNode_t* ultraPartialReduceNode;
    const cudaGraphDeviceNode_t* ultraReduceNode;
    EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGrid;
    EmbeddingSparseGradientReduceGridUpdateConfig highReduceGrid;
    EmbeddingSparseGradientReduceGridUpdateConfig ultraPartialReduceGrid;
    EmbeddingSparseGradientReduceGridUpdateConfig ultraReduceGrid;
};

void validateReduceGridUpdateConfig(const DeviceUpdatableKernelNodeDeviceHandle* nodeHandle,
                                    EmbeddingSparseGradientReduceGridUpdateConfig config,
                                    Stream stream,
                                    const char* label) {
    if (nodeHandle == nullptr) {
        return;
    }
    if (!nodeHandle->isInitialized()) {
        throw std::invalid_argument(std::string("Embedding sparse-gradient ") + label + " node handle is not initialized.");
    }
    if (nodeHandle->getGpuNum() != stream.getGpuNum()) {
        throw std::invalid_argument(std::string("Embedding sparse-gradient ") + label + " node handle GPU does not match the launch stream.");
    }
    if (config.reduceRowsPerGridX == 0U || config.reduceGridDimY == 0U || config.minReduceGridDimX == 0U ||
        config.maxReduceGridDimX == 0U || config.minReduceGridDimX > config.maxReduceGridDimX ||
        config.maxReduceGridDimY == 0U || config.reduceGridDimY > config.maxReduceGridDimY) {
        throw std::invalid_argument(std::string("Embedding sparse-gradient ") + label + " grid update configuration is invalid.");
    }
}

__device__ __forceinline__ cudaGraphDeviceNode_t loadEmbeddingSparseGradientTargetNode(const cudaGraphDeviceNode_t* targetNode) {
    return *targetNode;
}

__device__ __forceinline__ uint32_t embeddingSparseGradientGridDim(uint64_t value, uint32_t minGrid, uint32_t maxGrid) {
    uint64_t clamped = value;
    if (clamped < static_cast<uint64_t>(minGrid)) {
        clamped = static_cast<uint64_t>(minGrid);
    } else if (clamped > static_cast<uint64_t>(maxGrid)) {
        clamped = static_cast<uint64_t>(maxGrid);
    }
    return static_cast<uint32_t>(clamped);
}

__device__ __forceinline__ void setEmbeddingSparseGradientKernelNodeGridDim(const cudaGraphDeviceNode_t* node, dim3 gridDim) {
    if (node == nullptr) {
        return;
    }
    (void)cudaGraphKernelNodeSetGridDim(loadEmbeddingSparseGradientTargetNode(node), gridDim);
}

__device__ __forceinline__ void updateEmbeddingSparseGradientBucketReduceGrid(const cudaGraphDeviceNode_t* reduceNode,
                                                                              uint64_t rows,
                                                                              EmbeddingSparseGradientReduceGridUpdateConfig config) {
    if (reduceNode == nullptr) {
        return;
    }
    const uint64_t packedGridX =
        (rows + static_cast<uint64_t>(config.reduceRowsPerGridX) - 1ULL) / static_cast<uint64_t>(config.reduceRowsPerGridX);
    const uint32_t gridX = embeddingSparseGradientGridDim(packedGridX, config.minReduceGridDimX, config.maxReduceGridDimX);
    setEmbeddingSparseGradientKernelNodeGridDim(reduceNode, dim3(gridX, config.reduceGridDimY, 1U));
}

__device__ __forceinline__ uint32_t ceilDivU32(uint32_t x, uint32_t y) { return (x + y - 1U) / y; }

// Relaxed device-scope atomics, as the radix sort's lookback: a word is a whole message written by one 64-bit store.
__device__ __forceinline__ void publish(uint64_t* slot, uint64_t word) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(*slot);
    ref.store(word, cuda::memory_order_relaxed);
}

__device__ __forceinline__ uint64_t peek(uint64_t* slot) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(*slot);
    return ref.load(cuda::memory_order_relaxed);
}

// A tile's own counts go into its row as soon as it has them, so the tiles after it can add them. Tile 0's are a
// prefix already.
__device__ __forceinline__ void publishOwnCounts(uint64_t* lookback, uint32_t tile, uint64_t counts) {
    publish(lookback + tile, (tile == 0U ? kPrefixFlag : kLocalFlag) | counts);
}

// The counts of the tiles before this one: the rows behind are read kLookbackRows at a time, nearest first, all the
// loads in flight together, and added until a prefix. A row not yet written ends the batch at it, and the next batch
// starts from it. Then the tile's row becomes a prefix.
__device__ __forceinline__ uint64_t countsBeforeTile(uint64_t* lookback, uint32_t tile, uint64_t counts) {
    if (tile == 0U) return 0U;
    uint64_t before = 0U;
    uint64_t* next = lookback + tile;  // the row after the nearest one not yet added
    uint32_t rowsBehind = tile;
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
            before += word[k] & kCountsMask;
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
    publish(lookback + tile, kPrefixFlag | (before + counts));
    return before;
}

// Stored with a word of padding every 32 entries, a warp's accesses to its threads' k-th entries (thread t holds
// entries 8t to 8t + 7) fall in 32 different banks instead of 4.
__device__ __forceinline__ uint32_t paddedIndex(uint32_t i) { return i + (i >> 5); }

__device__ __forceinline__ uint64_t activeItems(const void* activeCount, uint32_t activeCountBytes, uint64_t capacity) {
    if (activeCountBytes == 0) return capacity;
    const uint64_t active = activeCountBytes == 8 ? *static_cast<const uint64_t*>(activeCount)
                                                  : static_cast<uint64_t>(*static_cast<const uint32_t*>(activeCount));
    return active < capacity ? active : capacity;
}

__device__ __forceinline__ uint64_t runsWord(uint64_t flag, uint32_t heads, uint32_t last) {
    return flag | static_cast<uint64_t>(heads) << 32 | last;
}

// The bucket of a run of `tokens` tokens: 0 low (at most lowRunMax), 1 high, 2 ultra-high (at least ultraRunMin).
__device__ __forceinline__ uint32_t bucketOf(const RunsParams& p, uint32_t tokens) {
    return tokens >= p.ultraRunMin ? 2U : tokens > p.lowRunMax ? 1U : 0U;
}

// What the reducers read: each bucket's size, and their grids; with the runs and the valid runs. Called by every
// thread of one block. A grid update is a call into the device runtime that takes a few µs on one thread (ncu on
// 1620), so the four go to the first thread of four warps, which make them at the same time (1621: 2 µs saved).
template <typename KeyT>
__device__ __forceinline__ void writeTotals(const RunsParams& p,
                                            uint32_t tid,
                                            uint32_t runs,
                                            uint32_t validRuns,
                                            uint32_t low,
                                            uint32_t high,
                                            uint32_t ultra,
                                            uint32_t ultraPartials) {
    switch (tid) {
        case 0U:
            *p.numRuns = runs;
            *static_cast<KeyT*>(p.numRows) = static_cast<KeyT>(validRuns);
            *p.numLowRunRows = low;
            *p.numHighRunRows = high;
            *p.numUltraRunRows = ultra;
            *p.numUltraPartials = ultraPartials;
            updateEmbeddingSparseGradientBucketReduceGrid(p.lowReduceNode, low, p.lowReduceGrid);
            break;
        case 32U: updateEmbeddingSparseGradientBucketReduceGrid(p.highReduceNode, high, p.highReduceGrid); break;
        case 64U:
            updateEmbeddingSparseGradientBucketReduceGrid(p.ultraPartialReduceNode, ultraPartials, p.ultraPartialReduceGrid);
            break;
        case 96U: updateEmbeddingSparseGradientBucketReduceGrid(p.ultraReduceNode, ultra, p.ultraReduceGrid); break;
        default: break;
    }
}

// One tile of 2,048 sorted keys; the run-length encode and the bucketing of the runs in one pass.
//
// The runs: a key is a run head when it is the first key or differs from the key before it. The block counts its heads
// and finds the latest (a scan over its threads in shared memory), publishes that, adds what the tiles before it
// published (nearest first, until a prefix) and publishes its prefix. Its runs go to shared memory in order and out
// from there, consecutive threads writing consecutive runs' keys, first positions and counts (a run's count is the
// next run's start less its own).
//
// The buckets: a run's length is known in the tile that holds the next run's head (the last run's, in the last key's
// tile), so a tile buckets the runs it completes: the one in progress when the tile begins (it started in an earlier
// tile; its start is the lookback's latest head) and its own runs but the last. Runs are completed in tile order, so
// each bucket lists its runs in run order: the block counts its runs of each kind and the ultra-high runs' partials
// (a second scan), publishes those in two more lookbacks, stages its low then high runs in shared memory and writes
// them out from there (a low run as an entry: its index, start, length and key, for its reducer); the ultra-high runs,
// few, are written by their threads. The last key's tile buckets the last run itself (unless it is the sentinel
// row's: the valid runs are all but a last run of row vocabularySize) and writes the totals and the reducers'
// grids. Integer counts, so the result is the same whatever the lookbacks' timing.
template <typename KeyT>
__global__ void __launch_bounds__(kThreads)
    encodeAndBucketizeEmbeddingSparseGradientRunsKernel(const __grid_constant__ RunsParams p) {
    // The tile's keys, then (once they are in runKeys) its staged low and high runs.
    __shared__ uint4 keyWords[(kPaddedTile * (sizeof(KeyT) > 4 ? sizeof(KeyT) : 4) + 15) / 16];
    __shared__ uint32_t scanA[kThreads];  // heads, then low runs (bits 0 to 15) and high runs (16 to 31)
    __shared__ uint32_t scanB[kThreads];  // latest head + 1, then ultra-high partials
    __shared__ uint32_t warpA[kWarps];
    __shared__ uint32_t warpB[kWarps];
    __shared__ uint32_t beforeHeads;
    __shared__ uint32_t beforeLast;
    __shared__ uint32_t ownHeads;          // the tile's runs that start in it
    __shared__ KeyT runKeys[kTile];        // the tile's runs in order: their keys
    __shared__ uint32_t runStarts[kPaddedTile];  // and their first positions (padded)
    __shared__ KeyT keyBefore;
    __shared__ uint64_t rowsBeforeTile;  // the rows and ultra lookbacks' counts of the tiles before this one
    __shared__ uint64_t ultraBeforeTile;
    KeyT* tileKeys = reinterpret_cast<KeyT*>(keyWords);
    uint32_t* staged = reinterpret_cast<uint32_t*>(keyWords);

    const uint64_t items = activeItems(p.activeCount, p.activeCountBytes, p.capacity);
    const uint32_t tid = threadIdx.x;
    if (items == 0) {
        if (blockIdx.x == 0) writeTotals<KeyT>(p, tid, 0U, 0U, 0U, 0U, 0U, 0U);
        return;
    }
    const uint64_t tileBegin = static_cast<uint64_t>(blockIdx.x) * kTile;
    if (tileBegin >= items) return;  // the whole block: no tile with keys looks back at this one
    const uint32_t tileItems = items - tileBegin < kTile ? static_cast<uint32_t>(items - tileBegin) : kTile;
    const bool lastTile = tileBegin + tileItems == items;
    const uint32_t lane = tid & 31U;
    const uint32_t warp = tid >> 5;
    const uint32_t first = tid * kItemsPerThread;
    const uint32_t tileBegin32 = static_cast<uint32_t>(tileBegin);  // under 2^30

    const KeyT* keys = static_cast<const KeyT*>(p.keys) + tileBegin;
    for (uint32_t r = 0; r < kItemsPerThread; ++r) {
        const uint32_t i = r * kThreads + tid;
        if (i < tileItems) tileKeys[paddedIndex(i)] = keys[i];
    }
    if (tid == 0 && tileBegin != 0) keyBefore = keys[-1];
    __syncthreads();

    uint32_t heads = 0U;
    uint32_t last = 0U;
    uint32_t headBits = 0U;
    for (uint32_t k = 0; k < kItemsPerThread; ++k) {
        const uint32_t j = first + k;
        if (j >= tileItems) break;
        const KeyT key = tileKeys[paddedIndex(j)];
        const bool head = j == 0 ? tileBegin == 0 || key != keyBefore : key != tileKeys[paddedIndex(j - 1U)];
        if (head) {
            ++heads;
            last = tileBegin32 + j + 1U;
            headBits |= 1U << k;
        }
    }

    // Inclusive scan of (heads, last) over the warp's threads, then over the warps.
    scanA[tid] = heads;
    scanB[tid] = last;
    __syncwarp();
    for (uint32_t offset = 1; offset < 32U; offset <<= 1) {
        uint32_t leftHeads = 0U;
        uint32_t leftLast = 0U;
        if (lane >= offset) {
            leftHeads = scanA[tid - offset];
            leftLast = scanB[tid - offset];
        }
        __syncwarp();
        if (lane >= offset) {
            scanA[tid] += leftHeads;
            if (scanB[tid] == 0U) scanB[tid] = leftLast;
        }
        __syncwarp();
    }
    if (lane == 31U) {
        warpA[warp] = scanA[tid];
        warpB[warp] = scanB[tid];
    }
    __syncthreads();

    if (tid == 0) {
        // The warps' exclusive prefixes in place, and the tile's own (heads, last).
        uint32_t tileHeads = 0U;
        uint32_t tileLast = 0U;
        for (uint32_t w = 0; w < kWarps; ++w) {
            const uint32_t h = warpA[w];
            const uint32_t l = warpB[w];
            warpA[w] = tileHeads;
            warpB[w] = tileLast;
            tileHeads += h;
            if (l != 0U) tileLast = l;
        }

        uint64_t* own = p.runsLookback + blockIdx.x;
        uint32_t heads0 = 0U;
        uint32_t last0 = 0U;
        if (blockIdx.x == 0U) {
            publish(own, runsWord(kPrefixFlag, tileHeads, tileLast));
        } else {
            publish(own, runsWord(kLocalFlag, tileHeads, tileLast));
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
            publish(own, runsWord(kPrefixFlag, heads0 + tileHeads, tileLast != 0U ? tileLast : last0));
        }
        beforeHeads = heads0;
        beforeLast = last0;
        ownHeads = tileHeads;
    }
    __syncthreads();

    // The tile's runs, in order, into shared memory: each head at its run's index in the tile.
    uint32_t local = warpA[warp] + (lane != 0U ? scanA[tid - 1U] : 0U);
    for (uint32_t k = 0; k < kItemsPerThread; ++k) {
        if ((headBits >> k) & 1U) {
            const uint32_t j = first + k;
            runKeys[local] = tileKeys[paddedIndex(j)];
            runStarts[paddedIndex(local)] = tileBegin32 + j;
            ++local;
        }
    }
    __syncthreads();

    // The runs out to their places, consecutive threads writing consecutive runs. The count of the tile's last run is
    // written by the first head of a later tile, or below by the last key's tile; that of the run in progress when
    // the tile begins by this tile's thread 0.
    const uint32_t tileHeads = ownHeads;
    const uint32_t base = beforeHeads;
    const uint32_t startOfRunInProgress = beforeLast - 1U;  // when base != 0
    KeyT* rows = static_cast<KeyT*>(p.rows);
    for (uint32_t i = tid; i < tileHeads; i += kThreads) {
        rows[base + i] = runKeys[i];
        const uint32_t start = runStarts[paddedIndex(i)];
        p.runOffsets[base + i] = start;
        if (i + 1U < tileHeads) p.runCounts[base + i] = runStarts[paddedIndex(i + 1U)] - start;
    }
    if (tid == 0 && tileHeads != 0U && base != 0U) p.runCounts[base - 1U] = runStarts[0] - startOfRunInProgress;

    // The runs the tile completes, the last run aside: completed run k ends at the tile's head e = k + 1 - shift and
    // starts at head e - 1 (or, for e = 0, where the run in progress began). shift = 1 when a run is in progress.
    const uint32_t shift = base != 0U ? 1U : 0U;
    const uint32_t completed = tileHeads + shift - 1U;  // tile 0 holds a head at its first key
    const uint32_t firstRun = base - shift;             // completed run 0's index
    uint32_t tokens[kItemsPerThread];
    uint32_t lowBits = 0U;  // bit k: the thread's k-th completed run is a low one
    uint32_t highBits = 0U;
    uint32_t ultraBits = 0U;
    uint32_t partials = 0U;
#pragma unroll
    for (uint32_t k = 0; k < kItemsPerThread; ++k) {
        const uint32_t run = first + k;
        tokens[k] = 0U;
        if (run < completed) {
            const uint32_t e = run + 1U - shift;
            const uint32_t start = e == 0U ? startOfRunInProgress : runStarts[paddedIndex(e - 1U)];
            tokens[k] = runStarts[paddedIndex(e)] - start;
            const uint32_t bucket = bucketOf(p, tokens[k]);
            lowBits |= (bucket == 0U ? 1U : 0U) << k;
            highBits |= (bucket == 1U ? 1U : 0U) << k;
            ultraBits |= (bucket == 2U ? 1U : 0U) << k;
            if (bucket == 2U) partials += ceilDivU32(tokens[k], p.ultraTokensPerPartial);
        }
    }

    // Inclusive scan of (low and high runs, partials) over the warp's threads; the warps' totals for every thread. The
    // scan arrays were last read before the barrier above.
    scanA[tid] = static_cast<uint32_t>(__popc(lowBits)) | static_cast<uint32_t>(__popc(highBits)) << 16;
    scanB[tid] = partials;
    __syncwarp();
    for (uint32_t offset = 1; offset < 32U; offset <<= 1) {
        uint32_t leftRows = 0U;
        uint32_t leftPartials = 0U;
        if (lane >= offset) {
            leftRows = scanA[tid - offset];
            leftPartials = scanB[tid - offset];
        }
        __syncwarp();
        if (lane >= offset) {
            scanA[tid] += leftRows;
            scanB[tid] += leftPartials;
        }
        __syncwarp();
    }
    if (lane == 31U) {
        warpA[warp] = scanA[tid];
        warpB[warp] = scanB[tid];
    }
    __syncthreads();

    // The tile's counts, and those of the tile's runs before this thread's.
    uint32_t rowsHere = lane != 0U ? scanA[tid - 1U] : 0U;
    uint32_t partialsHere = lane != 0U ? scanB[tid - 1U] : 0U;
    uint32_t tileRows = 0U;
    uint32_t tilePartials = 0U;
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w) {
        const uint32_t rowsOfWarp = warpA[w];
        const uint32_t partialsOfWarp = warpB[w];
        if (w < warp) {
            rowsHere += rowsOfWarp;
            partialsHere += partialsOfWarp;
        }
        tileRows += rowsOfWarp;
        tilePartials += partialsOfWarp;
    }
    const uint32_t tileLow = tileRows & 0xffffU;
    const uint32_t tileHigh = tileRows >> 16;
    const uint32_t tileUltra = completed - tileLow - tileHigh;
    const uint32_t lowHere = rowsHere & 0xffffU;
    const uint32_t highHere = rowsHere >> 16;
    // Used by a thread with an ultra-high run, so every completed run before the thread's is one of the counted.
    const uint32_t ultraHere = first - lowHere - highHere;

    // Thread 0 keeps the rows lookback and thread 32 the ultra one. Both publish before they stage their own runs,
    // and look back after: the tiles behind have had that long to publish, and the other threads stage meanwhile.
    const uint64_t rowsCounts = static_cast<uint64_t>(tileHigh) << kHighShift | tileLow;
    const uint64_t ultraCounts = static_cast<uint64_t>(tileUltra) << kUltraShift | tilePartials;
    if (tid == 0U) publishOwnCounts(p.rowsLookback, blockIdx.x, rowsCounts);
    if (tid == 32U) publishOwnCounts(p.ultraLookback, blockIdx.x, ultraCounts);

    // The tile's keys were last read before the barrier after the runs' staging, so their memory takes the bucket
    // entries now.
    uint32_t lowAt = lowHere;
    uint32_t highAt = tileLow + highHere;
#pragma unroll
    for (uint32_t k = 0; k < kItemsPerThread; ++k) {
        const uint32_t run = firstRun + first + k;
        if ((lowBits >> k) & 1U) {
            staged[paddedIndex(lowAt)] = run;
            ++lowAt;
        } else if ((highBits >> k) & 1U) {
            staged[paddedIndex(highAt)] = run;
            ++highAt;
        }
    }

    if (tid == 0U) rowsBeforeTile = countsBeforeTile(p.rowsLookback, blockIdx.x, rowsCounts);
    if (tid == 32U) ultraBeforeTile = countsBeforeTile(p.ultraLookback, blockIdx.x, ultraCounts);
    __syncthreads();

    const uint32_t lowBefore = static_cast<uint32_t>(rowsBeforeTile) & 0x7fffffffU;
    const uint32_t highBefore = static_cast<uint32_t>(rowsBeforeTile >> kHighShift);
    const uint32_t partialsBefore = static_cast<uint32_t>(ultraBeforeTile);
    const uint32_t ultraBefore = static_cast<uint32_t>(ultraBeforeTile >> kUltraShift);

    const uint32_t stagedRuns = tileLow + tileHigh;
    for (uint32_t i = tid; i < stagedRuns; i += kThreads) {
        const uint32_t run = staged[paddedIndex(i)];
        if (i < tileLow) {
            // A low run's entry, from the tile's runs in shared memory: completed run k = run - firstRun ends at head
            // e = k + 1 - shift and starts at head e - 1, or, for e = 0, where the run in progress began (its key is
            // the one before the tile).
            const uint32_t e = run - firstRun + 1U - shift;
            const uint32_t start = e == 0U ? startOfRunInProgress : runStarts[paddedIndex(e - 1U)];
            const KeyT key = e == 0U ? keyBefore : runKeys[e - 1U];
            p.lowRunEntries[lowBefore + i] = make_uint4(run, start, runStarts[paddedIndex(e)] - start, static_cast<uint32_t>(key));
        } else {
            p.highRunRows[highBefore + (i - tileLow)] = run;
        }
    }
    if (ultraBits != 0U) {
        uint32_t at = ultraBefore + ultraHere;
        uint32_t partialOffset = partialsBefore + partialsHere;
#pragma unroll
        for (uint32_t k = 0; k < kItemsPerThread; ++k) {
            if ((ultraBits >> k) & 1U) {
                const uint32_t runPartials = ceilDivU32(tokens[k], p.ultraTokensPerPartial);
                p.ultraRunRows[at] = firstRun + first + k;
                p.ultraRunPartialCounts[at] = runPartials;
                p.ultraRunPartialOffsets[at] = partialOffset;
                ++at;
                partialOffset += runPartials;
            }
        }
    }

    if (lastTile) {
        // The last run: its count, and its bucket unless it is the sentinel row's. Every thread that writes the totals
        // works it out (a few instructions); thread 0 writes it.
        const uint32_t runs = base + tileHeads;
        const uint32_t lastRun = runs - 1U;
        const uint32_t lastStart = tileHeads != 0U ? runStarts[paddedIndex(tileHeads - 1U)] : startOfRunInProgress;
        const uint32_t lastTokens = static_cast<uint32_t>(items) - lastStart;
        const KeyT lastKey = tileHeads != 0U ? runKeys[tileHeads - 1U] : keyBefore;
        const bool lastValid = static_cast<uint64_t>(lastKey) != p.vocabularySize;
        const uint32_t lastBucket = bucketOf(p, lastTokens);
        const uint32_t lastPartials = lastBucket == 2U ? ceilDivU32(lastTokens, p.ultraTokensPerPartial) : 0U;
        uint32_t low = lowBefore + tileLow;
        uint32_t high = highBefore + tileHigh;
        uint32_t ultra = ultraBefore + tileUltra;
        uint32_t ultraPartials = partialsBefore + tilePartials;
        if (tid == 0U) {
            p.runCounts[lastRun] = lastTokens;
            if (lastValid) {
                if (lastBucket == 0U) {
                    p.lowRunEntries[low] = make_uint4(lastRun, lastStart, lastTokens, static_cast<uint32_t>(lastKey));
                }
                if (lastBucket == 1U) p.highRunRows[high] = lastRun;
                if (lastBucket == 2U) {
                    p.ultraRunRows[ultra] = lastRun;
                    p.ultraRunPartialCounts[ultra] = lastPartials;
                    p.ultraRunPartialOffsets[ultra] = ultraPartials;
                }
            }
        }
        if (lastValid) {
            low += lastBucket == 0U ? 1U : 0U;
            high += lastBucket == 1U ? 1U : 0U;
            ultra += lastBucket == 2U ? 1U : 0U;
            ultraPartials += lastPartials;
        }
        writeTotals<KeyT>(p, tid, runs, lastValid ? runs : lastRun, low, high, ultra, ultraPartials);
    }
}

uint32_t runsTiles(uint64_t capacity) {
    if (capacity == 0 || capacity > kMaxItems) {
        throw std::invalid_argument("Embedding sparse-gradient runs: the capacity is 1 to 2^30 - 1 keys.");
    }
    return static_cast<uint32_t>((capacity + kTile - 1) / kTile);
}

template <typename KeyT>
void launchTyped(uint32_t tiles, const RunsParams& p, cudaStream_t stream) {
    encodeAndBucketizeEmbeddingSparseGradientRunsKernel<KeyT><<<tiles, kThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace

uint64_t embeddingSparseGradientRunsScratchWords(uint64_t capacity) { return 3ULL * runsTiles(capacity); }

void launchEncodeAndBucketizeEmbeddingSparseGradientRuns(const void* sortedKeys,
                                                         DataType keyDataType,
                                                         uint64_t capacity,
                                                         const void* activeCount,
                                                         DataType activeCountDataType,
                                                         void* rows,
                                                         uint32_t* runCounts,
                                                         uint32_t* runOffsets,
                                                         uint32_t* numRuns,
                                                         void* numRows,
                                                         uint32_t* lowRunEntries,
                                                         uint32_t* highRunRowIndices,
                                                         uint32_t* ultraHighRunRowIndices,
                                                         uint32_t* ultraHighRunPartialCounts,
                                                         uint32_t* ultraHighRunPartialOffsets,
                                                         uint32_t* numUltraHighPartials,
                                                         uint32_t* numLowRunRows,
                                                         uint32_t* numHighRunRows,
                                                         uint32_t* numUltraHighRunRows,
                                                         uint64_t* scratch,
                                                         uint64_t vocabularySize,
                                                         uint32_t lowRunMax,
                                                         uint32_t ultraHighRunMin,
                                                         uint32_t ultraHighTokensPerPartial,
                                                         const DeviceUpdatableKernelNodeDeviceHandle* lowReduceNodeHandle,
                                                         const DeviceUpdatableKernelNodeDeviceHandle* highReduceNodeHandle,
                                                         const DeviceUpdatableKernelNodeDeviceHandle* ultraHighPartialReduceNodeHandle,
                                                         const DeviceUpdatableKernelNodeDeviceHandle* ultraHighReduceNodeHandle,
                                                         EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
                                                         EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
                                                         EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
                                                         EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig,
                                                         Stream stream,
                                                         bool clearScratch) {
    const uint32_t tiles = runsTiles(capacity);
    if (keyDataType != DataType::UINT16 && keyDataType != DataType::UINT32 && keyDataType != DataType::UINT64) {
        throw std::invalid_argument("Embedding sparse-gradient runs: the row keys are UINT16, UINT32 or UINT64.");
    }
    if (sortedKeys == nullptr || rows == nullptr || runCounts == nullptr || runOffsets == nullptr || numRuns == nullptr ||
        numRows == nullptr || lowRunEntries == nullptr || highRunRowIndices == nullptr || ultraHighRunRowIndices == nullptr ||
        ultraHighRunPartialCounts == nullptr || ultraHighRunPartialOffsets == nullptr || numUltraHighPartials == nullptr ||
        numLowRunRows == nullptr || numHighRunRows == nullptr || numUltraHighRunRows == nullptr || scratch == nullptr) {
        throw std::invalid_argument("Embedding sparse-gradient runs: the keys, the outputs and the scratch are required.");
    }
    if (sortedKeys == rows) {
        throw std::invalid_argument("Embedding sparse-gradient runs: the rows must not be written over the keys.");
    }
    if (reinterpret_cast<uintptr_t>(lowRunEntries) % 16U != 0U) {
        throw std::invalid_argument("Embedding sparse-gradient runs: the low bucket's entries must be 16-byte aligned.");
    }
    uint32_t activeCountBytes = 0;
    if (activeCount != nullptr) {
        if (activeCountDataType != DataType::UINT32 && activeCountDataType != DataType::UINT64) {
            throw std::invalid_argument("Embedding sparse-gradient runs: the device item count is UINT32 or UINT64.");
        }
        activeCountBytes = activeCountDataType == DataType::UINT64 ? 8U : 4U;
    }
    if (ultraHighTokensPerPartial == 0U) {
        throw std::invalid_argument("Embedding sparse-gradient runs: an ultra-high partial holds at least one token.");
    }
    validateReduceGridUpdateConfig(lowReduceNodeHandle, lowReduceGridConfig, stream, "low-reduction");
    validateReduceGridUpdateConfig(highReduceNodeHandle, highReduceGridConfig, stream, "high-reduction");
    validateReduceGridUpdateConfig(ultraHighPartialReduceNodeHandle, ultraHighPartialReduceGridConfig, stream, "ultra-partial-reduction");
    validateReduceGridUpdateConfig(ultraHighReduceNodeHandle, ultraHighReduceGridConfig, stream, "ultra-reduction");

    RunsParams p{};
    p.keys = sortedKeys;
    p.rows = rows;
    p.runCounts = runCounts;
    p.runOffsets = runOffsets;
    p.numRuns = numRuns;
    p.numRows = numRows;
    p.lowRunEntries = reinterpret_cast<uint4*>(lowRunEntries);
    p.highRunRows = highRunRowIndices;
    p.ultraRunRows = ultraHighRunRowIndices;
    p.ultraRunPartialCounts = ultraHighRunPartialCounts;
    p.ultraRunPartialOffsets = ultraHighRunPartialOffsets;
    p.numUltraPartials = numUltraHighPartials;
    p.numLowRunRows = numLowRunRows;
    p.numHighRunRows = numHighRunRows;
    p.numUltraRunRows = numUltraHighRunRows;
    p.runsLookback = scratch;
    p.rowsLookback = scratch + tiles;
    p.ultraLookback = scratch + 2ULL * tiles;
    p.activeCount = activeCount;
    p.capacity = capacity;
    p.activeCountBytes = activeCountBytes;
    p.vocabularySize = vocabularySize;
    p.lowRunMax = lowRunMax;
    p.ultraRunMin = ultraHighRunMin;
    p.ultraTokensPerPartial = ultraHighTokensPerPartial;
    p.lowReduceNode = lowReduceNodeHandle != nullptr ? lowReduceNodeHandle->devicePtr() : nullptr;
    p.highReduceNode = highReduceNodeHandle != nullptr ? highReduceNodeHandle->devicePtr() : nullptr;
    p.ultraPartialReduceNode = ultraHighPartialReduceNodeHandle != nullptr ? ultraHighPartialReduceNodeHandle->devicePtr() : nullptr;
    p.ultraReduceNode = ultraHighReduceNodeHandle != nullptr ? ultraHighReduceNodeHandle->devicePtr() : nullptr;
    p.lowReduceGrid = lowReduceGridConfig;
    p.highReduceGrid = highReduceGridConfig;
    p.ultraPartialReduceGrid = ultraHighPartialReduceGridConfig;
    p.ultraReduceGrid = ultraHighReduceGridConfig;

    ScopedGpu scopedGpu(stream.getGpuNum());
    if (clearScratch) {
        CUDA_CHECK(cudaMemsetAsync(scratch, 0, 3ULL * tiles * sizeof(uint64_t), stream.getStream()));
    }
    switch (keyDataType) {
        case DataType::UINT16: launchTyped<uint16_t>(tiles, p, stream.getStream()); break;
        case DataType::UINT32: launchTyped<uint32_t>(tiles, p, stream.getStream()); break;
        default: launchTyped<uint64_t>(tiles, p, stream.getStream()); break;
    }
}

}  // namespace ThorImplementation
