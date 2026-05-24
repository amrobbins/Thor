#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/SparseRowGradient.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace ThorImplementation {

struct PreparedEmbeddingSparseGradient;

struct CapturedEmbeddingSparseGradient {
    CapturedEmbeddingSparseGradient() = default;
    explicit CapturedEmbeddingSparseGradient(int deviceNum) : reduceNodeHandle(deviceNum) {}

    CapturedEmbeddingSparseGradient(const CapturedEmbeddingSparseGradient&) = delete;
    CapturedEmbeddingSparseGradient& operator=(const CapturedEmbeddingSparseGradient&) = delete;
    CapturedEmbeddingSparseGradient(CapturedEmbeddingSparseGradient&&) noexcept = default;
    CapturedEmbeddingSparseGradient& operator=(CapturedEmbeddingSparseGradient&&) noexcept = default;

    void uploadTargetNodes(Stream stream) const { reduceNodeHandle.upload(reduceNode, stream); }

    DeviceUpdatableKernelNodeDeviceHandle reduceNodeHandle;
    DeviceUpdatableKernelNode reduceNode;
};

std::shared_ptr<PreparedEmbeddingSparseGradient> prepareEmbeddingSparseGradient(const Tensor& indices,
                                                                               const Tensor& upstreamGradient,
                                                                               SparseRowGradient& outputGradient,
                                                                               std::optional<uint64_t> paddingIndex);

void launchPreparedEmbeddingSparseGradient(PreparedEmbeddingSparseGradient& prepared,
                                           const Tensor& indices,
                                           const Tensor& upstreamGradient,
                                           SparseRowGradient& outputGradient,
                                           Stream stream);

void capturePreparedEmbeddingSparseGradient(CudaGraphCaptureBuilder& builder,
                                            PreparedEmbeddingSparseGradient& prepared,
                                            const Tensor& indices,
                                            const Tensor& upstreamGradient,
                                            SparseRowGradient& outputGradient,
                                            CapturedEmbeddingSparseGradient& captured);

void launchEmbeddingSparseGradient(const Tensor& indices,
                                   const Tensor& upstreamGradient,
                                   SparseRowGradient& outputGradient,
                                   std::optional<uint64_t> paddingIndex,
                                   Stream stream);

}  // namespace ThorImplementation
