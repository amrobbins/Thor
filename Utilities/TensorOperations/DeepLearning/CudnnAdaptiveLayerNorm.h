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
 * cuDNN's Adaptive LayerNorm shape contract is different from ordinary LayerNorm:
 * scale and bias vary per batch sample, but are broadcast across the non-normalized
 * leading feature dimensions. Thor exposes this as:
 *
 *   data feature shape:       [leading..., normalized...]
 *   scale/bias feature shape: [normalized...]
 *
 * Once the network batch dimension is included, the frontend graph is built with
 * NVIDIA's canonical 3D view:
 *
 *   x/y:        [batch, leading, hidden]
 *   scale/bias: [batch, 1, hidden]
 *   stats:      [batch, leading, 1]
 *
 * where hidden is the product of the normalized trailing dimensions and leading is
 * the product of the non-normalized feature dimensions. No tensor materialization is
 * required for this view.
 */
struct CudnnAdaptiveLayerNormDescriptor {
    uint64_t batchSize = 0;
    uint64_t leadingFeatureCount = 0;
    uint64_t normalizedFeatureCount = 0;

    DataType inputDataType = DataType::FP16;
    DataType outputDataType = DataType::FP16;
    DataType scaleBiasDataType = DataType::FP32;
    DataType computeDataType = DataType::FP32;

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

    // Required when descriptor.training is true. FP32 tensors with batchSize * leadingFeatureCount elements.
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
