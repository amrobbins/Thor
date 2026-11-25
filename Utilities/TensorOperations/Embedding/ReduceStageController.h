#pragma once

#include "DeepLearning/Api/DataType.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"

#include <cstdint>

namespace ThorImplementation {

struct EmbeddingSparseGradientReduceGridUpdateConfig {
    uint32_t reduceRowsPerGridX = 1U;
    uint32_t reduceGridDimY = 1U;
    uint32_t minReduceGridDimX = 1U;
    uint32_t maxReduceGridDimX = 1U;
    uint32_t maxReduceGridDimY = 1U;
};

/**
 * UINT64 words of scratch the runs need for up to `capacity` keys (1 to 2^30 - 1; throws std::invalid_argument
 * otherwise).
 */
[[nodiscard]] uint64_t embeddingSparseGradientRunsScratchWords(uint64_t capacity);

/**
 * The step between the sort of the row keys and the reducers, one kernel: the runs of the first n sorted keys
 * (n = capacity when activeCount is null, else min(*activeCount, capacity) read on the device; activeCountDataType
 * UINT32 or UINT64) and their buckets.
 *
 * The runs: with R runs of equal keys, *numRuns = R and for r < R, rows[r] is run r's key, runOffsets[r] its first
 * position and runCounts[r] its length (keyDataType UINT16, UINT32 or UINT64; rows of the same type). n = 0 gives
 * R = 0.
 *
 * The buckets: the valid runs are all but a last run of the sentinel row `vocabularySize`; their number goes to
 * numRows (of the rows' type). Each valid run goes into one bucket by its length: low (at most lowRunMax), ultra-high
 * (at least ultraHighRunMin) or high (between). A bucket lists its runs in run order and has its size written
 * (numLowRunRows, numHighRunRows, numUltraHighRunRows). The high and ultra-high buckets list the runs' indices
 * (highRunRowIndices, ultraHighRunRowIndices). The low bucket lists an entry of four words a run (lowRunEntries,
 * 16-byte aligned): the run's index, its first position, its length and the low 32 bits of its key, so its reducer
 * needs no second load to find them. An ultra-high run is reduced in partials of ultraHighTokensPerPartial tokens:
 * ultraHighRunPartialCounts holds each run's partials, ultraHighRunPartialOffsets the partials before it, and
 * numUltraHighPartials their total. Where a reducer's node handle is given, its grid is set on the device for its
 * bucket's size (rows / reduceRowsPerGridX rounded up, within its limits).
 *
 * Nothing past R runs or a bucket's size is written. The grid is sized for the capacity and the count is read on the
 * device, so a launch can be captured in a CUDA graph. `scratch` holds embeddingSparseGradientRunsScratchWords(capacity)
 * words, which the launch clears first; with clearScratch false it does not, and they must already be 0 (cleared by
 * an earlier kernel of the same stream or graph). Throws std::invalid_argument for a missing pointer, rows over the
 * keys, low-bucket entries not 16-byte aligned, a capacity out of range, keys that are not UINT16 / UINT32 / UINT64, a
 * count type that is not UINT32 / UINT64, a partial of 0 tokens, or a node handle that is not initialized, is on
 * another GPU or has an invalid grid configuration.
 */
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
                                                         bool clearScratch = true);

}  // namespace ThorImplementation
