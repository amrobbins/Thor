#pragma once

#include <optional>
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "Utilities/Expression/ConvolutionSpatial.h"

#include <memory>

namespace ThorImplementation {

class Convolution2d : public CustomLayer {
   public:
    ~Convolution2d() override = default;

    Convolution2d(uint32_t filterWidth,
                  uint32_t filterHeight,
                  ConvolutionSpatial2d spatial,
                  uint32_t numOutputChannels,
                  bool hasBias,
                  std::optional<DataType> weightsDataType,
                  const TensorPlacement& placement,
                  bool inferenceOnly,
                  int64_t stampedId = -1,
                  uint32_t groups = 1);

    std::string getLayerType() override { return "Convolution2d"; }

    Tensor getWeights() { return getParameterStorage("weights"); }
    std::optional<Tensor> getBiases() {
        const auto params = getParameters();
        if (!params.contains("biases")) {
            return std::nullopt;
        }
        return params.at("biases")->getStorage();
    }

    static std::vector<std::shared_ptr<PhysicalParameter>> defineParameters(uint32_t numOutputChannels,
                                                                             bool hasBias,
                                                                             uint32_t filterWidth,
                                                                             uint32_t filterHeight,
                                                                             std::optional<DataType> weightsDataType,
                                                                             uint32_t groups);

   private:
    static DynamicExpression buildExpression(
        bool hasBias, uint32_t groups, ConvolutionSpatial2d spatial, const TensorPlacement& placement);
};

}  // namespace ThorImplementation
