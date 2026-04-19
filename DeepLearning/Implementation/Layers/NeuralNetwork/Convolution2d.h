#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"

#include <memory>

namespace ThorImplementation {

class Convolution2d : public CustomLayer {
   public:
    ~Convolution2d() override = default;

    Convolution2d(uint32_t filterWidth,
                  uint32_t filterHeight,
                  uint32_t filterHorizontalStride,
                  uint32_t filterVerticalStride,
                  uint32_t leftAndRightPadWidth,
                  uint32_t topAndBottomPadHeight,
                  uint32_t numOutputChannels,
                  bool hasBias,
                  Optional<TensorDescriptor::DataType> weightsDataType,
                  const TensorPlacement& placement,
                  bool inferenceOnly,
                  int64_t stampedId = -1);

    std::string getLayerType() override { return "Convolution2d"; }

    Tensor getWeights() { return getParameterStorage("weights"); }
    Optional<Tensor> getBiases() {
        const auto params = getParameters();
        if (!params.contains("biases")) {
            return Optional<Tensor>::empty();
        }
        return params.at("biases")->getStorage();
    }

   private:
    static DynamicExpression buildExpression(
        bool hasBias, uint32_t strideH, uint32_t strideW, uint32_t padH, uint32_t padW, const TensorPlacement& placement);

    std::vector<std::shared_ptr<Parameter>> defineParameters(uint32_t numOutputChannels,
                                                             bool hasBias,
                                                             uint32_t filterWidth,
                                                             uint32_t filterHeight,
                                                             Optional<TensorDescriptor::DataType> weightsDataType);
};

}  // namespace ThorImplementation
