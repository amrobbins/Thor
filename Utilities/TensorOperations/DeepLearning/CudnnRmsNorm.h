#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ThorImplementation {

enum class CudnnRmsNormFusedActivation { NONE, SWISH };

const char* toString(CudnnRmsNormFusedActivation activation);
CudnnRmsNormFusedActivation cudnnRmsNormFusedActivationFromString(std::string_view value);

/**
 * cuDNN Frontend RMSNorm wrapper.
 *
 * Thor exposes RMSNorm over an arbitrary contiguous trailing normalized shape. The frontend graph is built
 * with NVIDIA's canonical 4D view:
 *
 *   x/y:   [outer, hidden, 1, 1]
 *   scale: [1, hidden, 1, 1]
 *   stats: [outer, 1, 1, 1]
 *
 * where hidden is the product of the normalized trailing dimensions and outer is the product of the remaining
 * leading dimensions. No tensor materialization is required for this view; cuDNN only sees packed dimensions
 * and strides for the contiguous allocation already owned by Thor.
 */
struct CudnnRmsNormDescriptor {
    uint64_t outerSize = 0;
    uint64_t normalizedFeatureCount = 0;

    TensorDescriptor::DataType inputDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType outputDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType parameterDataType = TensorDescriptor::DataType::FP32;
    TensorDescriptor::DataType computeDataType = TensorDescriptor::DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    CudnnRmsNormFusedActivation fusedActivation = CudnnRmsNormFusedActivation::NONE;
    std::string debugName = "thor_rms_norm";

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnRmsNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor y;

    // Required when descriptor.training is true. FP32 tensor with outerSize elements.
    std::optional<Tensor> invVariance;
};

struct CudnnRmsNormBackwardArgs {
    Tensor dy;
    Tensor x;
    Tensor scale;
    Tensor invVariance;
    Tensor dx;
    Tensor dscale;
};

class CudnnRmsNorm {
   public:
    static CudnnRmsNorm& instance();

    void forward(const CudnnRmsNormDescriptor& descriptor, const CudnnRmsNormForwardArgs& args, Stream stream);
    void backward(const CudnnRmsNormDescriptor& descriptor, const CudnnRmsNormBackwardArgs& args, Stream stream);

    void warmForward(const CudnnRmsNormDescriptor& descriptor, int gpuNum);
    void warmBackward(const CudnnRmsNormDescriptor& descriptor, int gpuNum);

    void clearCache();
    size_t cachedGraphCount() const;

    static bool frontendAvailable();

   private:
    CudnnRmsNorm() = default;
};

}  // namespace ThorImplementation
