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
 * cuDNN Frontend Adaptive LayerNorm wrapper.
 *
 * Adaptive LayerNorm is LayerNorm with per-sample scale and bias tensors. Thor exposes it over an
 * arbitrary contiguous trailing normalized shape. The frontend graph is built with NVIDIA's canonical 4D view:
 *
 *   x/y/scale/bias: [outer, hidden, 1, 1]
 *   stats:          [outer, 1, 1, 1]
 *
 * where hidden is the product of the normalized trailing dimensions and outer is the product of the remaining
 * leading dimensions, including the physical batch dimension. No tensor materialization is required for this view.
 */
struct CudnnAdaptiveLayerNormDescriptor {
    uint64_t outerSize = 0;
    uint64_t normalizedFeatureCount = 0;

    TensorDescriptor::DataType inputDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType outputDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType scaleBiasDataType = TensorDescriptor::DataType::FP32;
    TensorDescriptor::DataType computeDataType = TensorDescriptor::DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    std::string debugName = "thor_adaptive_layer_norm";

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnAdaptiveLayerNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor bias;
    Tensor y;

    // Required when descriptor.training is true. FP32 tensors with outerSize elements.
    std::optional<Tensor> mean;
    std::optional<Tensor> invVariance;
};

struct CudnnAdaptiveLayerNormBackwardArgs {
    Tensor dy;
    Tensor x;
    Tensor scale;
    Tensor mean;
    Tensor invVariance;
    Tensor dx;
    Tensor dscale;
    Tensor dbias;
};

class CudnnAdaptiveLayerNorm {
   public:
    static CudnnAdaptiveLayerNorm& instance();

    void forward(const CudnnAdaptiveLayerNormDescriptor& descriptor, const CudnnAdaptiveLayerNormForwardArgs& args, Stream stream);
    void backward(const CudnnAdaptiveLayerNormDescriptor& descriptor, const CudnnAdaptiveLayerNormBackwardArgs& args, Stream stream);

    void warmForward(const CudnnAdaptiveLayerNormDescriptor& descriptor, int gpuNum);
    void warmBackward(const CudnnAdaptiveLayerNormDescriptor& descriptor, int gpuNum);

    void clearCache();
    size_t cachedGraphCount() const;

    static bool frontendAvailable();

   private:
    CudnnAdaptiveLayerNorm() = default;
};

}  // namespace ThorImplementation
