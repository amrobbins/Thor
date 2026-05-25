#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/SparseRowGradient.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"
#include "Utilities/Expression/SparseRowUpdate.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace ThorImplementation {

struct PreparedEmbeddingSparseGradient;

struct EmbeddingSparseGradientRunBucketConfig {
    uint32_t lowRunMax = 32U;
    uint32_t ultraHighRunMin = 4096U;
    uint32_t ultraHighTokensPerPartial = 1024U;
};

EmbeddingSparseGradientRunBucketConfig defaultEmbeddingSparseGradientRunBucketConfig();
void setEmbeddingSparseGradientRunBucketConfigOverrideForTesting(std::optional<EmbeddingSparseGradientRunBucketConfig> config);

struct EmbeddingSparseGradientProfileResult {
    uint64_t numTokens = 0;
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    uint64_t capacity = 0;
    DataType indexDataType = DataType::UINT32;
    DataType gradientDataType = DataType::FP32;
    DataType rowDataType = DataType::UINT64;
    size_t sortTempBytes = 0;
    size_t rleTempBytes = 0;
    size_t scanTempBytes = 0;

    uint64_t activeRows = 0;
    uint64_t singletonRows = 0;
    uint64_t duplicateRows = 0;
    uint64_t lowRunRows = 0;
    uint64_t highRunRows = 0;
    uint64_t ultraHighRunRows = 0;
    uint64_t lowRunTokens = 0;
    uint64_t highRunTokens = 0;
    uint64_t ultraHighRunTokens = 0;
    uint32_t maxRunCount = 0;

    float materializeSortPairsMs = 0.0f;
    float cubSortMs = 0.0f;
    float clearRunCountsMs = 0.0f;
    float cubRleMs = 0.0f;
    float finalizeRowsMs = 0.0f;
    float cubScanOffsetsMs = 0.0f;
    float reduceValuesMs = 0.0f;
    float totalMs = 0.0f;
};

struct CapturedEmbeddingSparseGradient {
    CapturedEmbeddingSparseGradient() = default;
    explicit CapturedEmbeddingSparseGradient(int deviceNum)
        : lowReduceNodeHandle(deviceNum),
          highReduceNodeHandle(deviceNum),
          ultraHighPartialReduceNodeHandle(deviceNum),
          ultraHighReduceNodeHandle(deviceNum),
          highRunCaptureStream(deviceNum),
          ultraHighRunCaptureStream(deviceNum) {}

    CapturedEmbeddingSparseGradient(const CapturedEmbeddingSparseGradient&) = delete;
    CapturedEmbeddingSparseGradient& operator=(const CapturedEmbeddingSparseGradient&) = delete;
    CapturedEmbeddingSparseGradient(CapturedEmbeddingSparseGradient&&) noexcept = default;
    CapturedEmbeddingSparseGradient& operator=(CapturedEmbeddingSparseGradient&&) noexcept = default;

    void uploadTargetNodes(Stream stream) const {
        lowReduceNodeHandle.upload(lowReduceNode, stream);
        highReduceNodeHandle.upload(highReduceNode, stream);
        if (ultraHighPartialReduceNode) {
            ultraHighPartialReduceNodeHandle.upload(ultraHighPartialReduceNode, stream);
        }
        ultraHighReduceNodeHandle.upload(ultraHighReduceNode, stream);
    }

    DeviceUpdatableKernelNodeDeviceHandle lowReduceNodeHandle;
    DeviceUpdatableKernelNodeDeviceHandle highReduceNodeHandle;
    DeviceUpdatableKernelNodeDeviceHandle ultraHighPartialReduceNodeHandle;
    DeviceUpdatableKernelNodeDeviceHandle ultraHighReduceNodeHandle;
    DeviceUpdatableKernelNode lowReduceNode;
    DeviceUpdatableKernelNode highReduceNode;
    DeviceUpdatableKernelNode ultraHighPartialReduceNode;
    DeviceUpdatableKernelNode ultraHighReduceNode;
    Stream highRunCaptureStream;
    Stream ultraHighRunCaptureStream;
};

std::shared_ptr<PreparedEmbeddingSparseGradient> prepareEmbeddingSparseGradient(const Tensor& indices,
                                                                               const Tensor& upstreamGradient,
                                                                               SparseRowGradient& outputGradient,
                                                                               std::optional<uint64_t> paddingIndex);

std::shared_ptr<PreparedEmbeddingSparseGradient> prepareEmbeddingSparseGradientWithSparseRowUpdate(
    const Tensor& indices,
    const Tensor& upstreamGradient,
    SparseRowGradient& outputGradient,
    PhysicalOutputs updateOutputs,
    const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& updateInputs,
    const std::unordered_map<std::string, Tensor>& indexedUpdateOutputs,
    std::optional<uint64_t> paddingIndex);

bool preparedEmbeddingSparseGradientHasSparseRowUpdate(const PreparedEmbeddingSparseGradient& prepared);

void launchPreparedEmbeddingSparseGradient(PreparedEmbeddingSparseGradient& prepared,
                                           const Tensor& indices,
                                           const Tensor& upstreamGradient,
                                           SparseRowGradient& outputGradient,
                                           Stream stream);

void launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(PreparedEmbeddingSparseGradient& prepared,
                                                              const Tensor& indices,
                                                              const Tensor& upstreamGradient,
                                                              SparseRowGradient& outputGradient,
                                                              const std::unordered_map<std::string, float>& runtimeScalars,
                                                              Stream stream);

EmbeddingSparseGradientProfileResult profilePreparedEmbeddingSparseGradient(PreparedEmbeddingSparseGradient& prepared,
                                                                           const Tensor& indices,
                                                                           const Tensor& upstreamGradient,
                                                                           SparseRowGradient& outputGradient,
                                                                           Stream stream);

EmbeddingSparseGradientProfileResult profilePreparedEmbeddingSparseGradientWithSparseRowUpdate(
    PreparedEmbeddingSparseGradient& prepared,
    const Tensor& indices,
    const Tensor& upstreamGradient,
    SparseRowGradient& outputGradient,
    const std::unordered_map<std::string, float>& runtimeScalars,
    Stream stream);

void capturePreparedEmbeddingSparseGradient(CudaGraphCaptureBuilder& builder,
                                            PreparedEmbeddingSparseGradient& prepared,
                                            const Tensor& indices,
                                            const Tensor& upstreamGradient,
                                            SparseRowGradient& outputGradient,
                                            CapturedEmbeddingSparseGradient& captured);

void capturePreparedEmbeddingSparseGradientWithSparseRowUpdate(
    CudaGraphCaptureBuilder& builder,
    PreparedEmbeddingSparseGradient& prepared,
    const Tensor& indices,
    const Tensor& upstreamGradient,
    SparseRowGradient& outputGradient,
    const std::unordered_map<std::string, float>& runtimeScalars,
    CapturedEmbeddingSparseGradient& captured);


}  // namespace ThorImplementation
