#pragma once

#include "DeepLearning/Api/DataType.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"

#include <cstdint>

namespace ThorImplementation {

#ifndef THOR_EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD
#define THOR_EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD 32768ULL
#endif

#ifndef THOR_EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK
#define THOR_EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK 4096ULL
#endif

constexpr uint64_t EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD =
    THOR_EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD;
constexpr uint64_t EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK =
    THOR_EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK;
static_assert(EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK != 0ULL,
              "THOR_EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK must be non-zero.");

inline bool useTwoStageEmbeddingSparseGradientFinalize(uint64_t maxPossibleRuns) {
    return EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD != 0ULL &&
           maxPossibleRuns >= EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD;
}

inline uint64_t twoStageEmbeddingSparseGradientFinalizeBlockCount(uint64_t maxPossibleRuns) {
    return (maxPossibleRuns + EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK - 1ULL) /
           EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUNS_PER_BLOCK;
}

inline uint32_t runtimeTwoStageEmbeddingSparseGradientFinalizeRunThreshold() {
    if (EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD == 0ULL) {
        return 0U;
    }
    return EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD > 0xffffffffULL
               ? 0xffffffffU
               : static_cast<uint32_t>(EMBEDDING_SPARSE_GRADIENT_TWO_STAGE_FINALIZE_RUN_THRESHOLD);
}

struct EmbeddingSparseGradientReduceGridUpdateConfig {
    uint32_t reduceRowsPerGridX = 1U;
    uint32_t reduceGridDimY = 1U;
    uint32_t minReduceGridDimX = 1U;
    uint32_t maxReduceGridDimX = 1U;
    uint32_t maxReduceGridDimY = 1U;
};

struct EmbeddingSparseGradientTwoStageFinalizeCapturedNodes {
    DeviceUpdatableKernelNode classifyNode;
    DeviceUpdatableKernelNode accumulateNode;
};

void initializeEmbeddingKernelsSharedAttributes();

void launchFinalizeAndBucketizeEmbeddingSparseGradientRows(const void* outputRows,
                                                           const uint32_t* numRuns,
                                                           void* outputNumRows,
                                                           const uint32_t* runCounts,
                                                           uint32_t* lowRunRowIndices,
                                                           uint32_t* highRunRowIndices,
                                                           uint32_t* ultraHighRunRowIndices,
                                                           uint32_t* ultraHighRunPartialCounts,
                                                           uint32_t* ultraHighRunPartialOffsets,
                                                           uint32_t* numUltraHighPartials,
                                                           uint32_t* numLowRunRows,
                                                           uint32_t* numHighRunRows,
                                                           uint32_t* numUltraHighRunRows,
                                                           uint64_t vocabularySize,
                                                           uint32_t maxPossibleRuns,
                                                           DataType rowDataType,
                                                           uint32_t lowRunMax,
                                                           uint32_t ultraHighRunMin,
                                                           uint32_t ultraHighTokensPerPartial,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* lowReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* highReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* ultraHighPartialReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* ultraHighReduceNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* twoStageClassifyNodeHandle,
                                                           const DeviceUpdatableKernelNodeDeviceHandle* twoStageAccumulateNodeHandle,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig,
                                                           bool runtimeTwoStageDelegate,
                                                           uint32_t runtimeTwoStageRunThreshold,
                                                           Stream stream);

EmbeddingSparseGradientTwoStageFinalizeCapturedNodes captureTwoStageFinalizeAndBucketizeEmbeddingSparseGradientRows(
    const void* outputRows,
    const uint32_t* numRuns,
    void* outputNumRows,
    const uint32_t* runCounts,
    uint32_t* lowRunRowIndices,
    uint32_t* highRunRowIndices,
    uint32_t* ultraHighRunRowIndices,
    uint32_t* ultraHighRunPartialCounts,
    uint32_t* ultraHighRunPartialOffsets,
    uint32_t* numUltraHighPartials,
    uint32_t* numLowRunRows,
    uint32_t* numHighRunRows,
    uint32_t* numUltraHighRunRows,
    uint32_t* twoStageLowRunRowsScratch,
    uint32_t* twoStageHighRunRowsScratch,
    uint32_t* twoStageUltraHighRunRowsScratch,
    uint32_t* twoStageUltraHighRunPartialCountsScratch,
    uint32_t* twoStageUltraHighRunPartialOffsetsScratch,
    uint32_t* twoStageLowRunRowCounts,
    uint32_t* twoStageHighRunRowCounts,
    uint32_t* twoStageUltraHighRunRowCounts,
    uint32_t* twoStageUltraHighPartialCounts,
    uint64_t vocabularySize,
    uint32_t maxPossibleRuns,
    DataType rowDataType,
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
    Stream stream);

}  // namespace ThorImplementation
