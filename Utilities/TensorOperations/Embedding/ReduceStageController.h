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
                                                           EmbeddingSparseGradientReduceGridUpdateConfig lowReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig highReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig ultraHighPartialReduceGridConfig,
                                                           EmbeddingSparseGradientReduceGridUpdateConfig ultraHighReduceGridConfig,
                                                           Stream stream);

}  // namespace ThorImplementation
