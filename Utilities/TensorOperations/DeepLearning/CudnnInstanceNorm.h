#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ThorImplementation {

/**
 * cuDNN Frontend InstanceNorm wrapper.
 *
 * Thor API tensors do not include the batch dimension.  Physical tensors do.  This wrapper treats any packed physical
 * tensor with dimensions [N, C, spatial...] as NVIDIA's canonical 4D instance-normalization view:
 *
 *   x/y:         [N, C, S, 1]
 *   scale/bias:  [1, C, 1, 1]
 *   stats:       [N, C, 1, 1]
 *
 * where S is the product of all spatial dimensions.  This is a metadata-only view over Thor's contiguous tensor storage.
 */
struct CudnnInstanceNormDescriptor {
    uint64_t batchSize = 0;
    uint64_t channelCount = 0;
    uint64_t spatialElementCount = 0;

    TensorDescriptor::DataType inputDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType outputDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType parameterDataType = TensorDescriptor::DataType::FP32;
    TensorDescriptor::DataType computeDataType = TensorDescriptor::DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    std::string debugName = "thor_instance_norm";

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnInstanceNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor bias;
    Tensor y;

    // Required when descriptor.training is true.  FP32 tensors with batchSize * channelCount elements.
    std::optional<Tensor> mean;
    std::optional<Tensor> invVariance;
};

struct CudnnInstanceNormBackwardArgs {
    Tensor dy;
    Tensor x;
    Tensor scale;
    Tensor mean;
    Tensor invVariance;
    Tensor dx;
    Tensor dscale;
    Tensor dbias;
};

class CudnnInstanceNorm {
   public:
    static CudnnInstanceNorm& instance();

    void forward(const CudnnInstanceNormDescriptor& descriptor, const CudnnInstanceNormForwardArgs& args, Stream stream);
    void backward(const CudnnInstanceNormDescriptor& descriptor, const CudnnInstanceNormBackwardArgs& args, Stream stream);

    void warmForward(const CudnnInstanceNormDescriptor& descriptor, int gpuNum);
    void warmBackward(const CudnnInstanceNormDescriptor& descriptor, int gpuNum);

    void clearCache();
    size_t cachedGraphCount() const;

    static bool frontendAvailable();

   private:
    CudnnInstanceNorm() = default;
};

}  // namespace ThorImplementation
