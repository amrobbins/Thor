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
 * cuDNN Frontend LayerNorm wrapper.
 *
 * Thor exposes LayerNorm over an arbitrary contiguous trailing normalized shape.  The frontend graph is built
 * with NVIDIA's canonical 4D view:
 *
 *   x/y:        [outer, hidden, 1, 1]
 *   scale/bias: [1, hidden, 1, 1]
 *   stats:      [outer, 1, 1, 1]
 *
 * where hidden is the product of the normalized trailing dimensions and outer is the product of the remaining
 * leading dimensions.  No tensor materialization is required for this view; cuDNN only sees packed dimensions
 * and strides for the contiguous allocation already owned by Thor.
 */
struct CudnnLayerNormDescriptor {
    uint64_t outerSize = 0;
    uint64_t normalizedFeatureCount = 0;

    DataType inputDataType = DataType::FP16;
    DataType outputDataType = DataType::FP16;
    DataType parameterDataType = DataType::FP32;
    DataType computeDataType = DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    std::string debugName = "thor_layer_norm";

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnLayerNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor bias;
    Tensor y;

    // Required when descriptor.training is true.  FP32 tensors with outerSize elements.
    std::optional<Tensor> mean;
    std::optional<Tensor> invVariance;
};

struct CudnnLayerNormBackwardArgs {
    Tensor dy;
    Tensor x;
    Tensor scale;
    Tensor mean;
    Tensor invVariance;
    Tensor dx;
    Tensor dscale;
    Tensor dbias;
};

class CudnnLayerNorm {
   public:
    static CudnnLayerNorm& instance();

    void forward(const CudnnLayerNormDescriptor& descriptor, const CudnnLayerNormForwardArgs& args, Stream stream);
    void backward(const CudnnLayerNormDescriptor& descriptor, const CudnnLayerNormBackwardArgs& args, Stream stream);

    void warmForward(const CudnnLayerNormDescriptor& descriptor, int gpuNum);
    void warmBackward(const CudnnLayerNormDescriptor& descriptor, int gpuNum);

    void clearCache();
    size_t cachedGraphCount() const;

    static bool frontendAvailable();

   private:
    CudnnLayerNorm() = default;
};

}  // namespace ThorImplementation
