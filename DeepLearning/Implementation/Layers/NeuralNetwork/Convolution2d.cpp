#include <optional>
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2d.h"

#include <stdexcept>

#include "DeepLearning/Implementation/ThorError.h"
namespace ThorImplementation {

namespace {
class ConvWeightsParameter : public PhysicalParameter {
   public:
    ConvWeightsParameter(std::string name,
                         std::optional<DataType> storageDataType,
                         bool trainable,
                         bool trainingEnabled,
                         uint32_t numOutputChannels,
                         uint32_t filterWidth,
                         uint32_t filterHeight,
                         uint32_t groups)
        : PhysicalParameter(name, trainable),
          numOutputChannels(numOutputChannels),
          groups(groups),
          filterWidth(filterWidth),
          filterHeight(filterHeight),
          storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        DataType resolvedDataType = storageDataType.has_value() ? storageDataType.value() : inputTensor.getDataType();

        const auto& inputDims = inputTensor.getDimensions();
        if (inputDims.size() != 4) {
            throw std::runtime_error("Convolution2d weights require 4D NCHW feature input tensor.");
        }

        if (groups == 0 || inputDims[1] % groups != 0 || numOutputChannels % groups != 0)
            throw std::runtime_error(
                "Convolution2d parameter storage requires input/output channels divisible by groups.");
        TensorDescriptor descriptor(
            resolvedDataType, {numOutputChannels, inputDims[1] / groups, filterHeight, filterWidth});
        storage = Tensor(inputTensor.getPlacement(), descriptor);
    }

   private:
    const uint32_t numOutputChannels;
    const uint32_t groups;
    const uint32_t filterWidth;
    const uint32_t filterHeight;
    const std::optional<DataType> storageDataType;
};

class ConvBiasesParameter : public PhysicalParameter {
   public:
    ConvBiasesParameter(std::string name,
                        std::optional<DataType> storageDataType,
                        bool trainable,
                        bool trainingEnabled,
                        uint32_t numOutputChannels)
        : PhysicalParameter(name, trainable), numOutputChannels(numOutputChannels), storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        DataType resolvedDataType = storageDataType.has_value() ? storageDataType.value() : inputTensor.getDataType();
        storage = Tensor(inputTensor.getPlacement(), TensorDescriptor(resolvedDataType, {numOutputChannels}));
    }

   private:
    const uint32_t numOutputChannels;
    const std::optional<DataType> storageDataType;
};
}  // namespace


Convolution2d::Convolution2d(uint32_t filterWidth,
                             uint32_t filterHeight,
                             ConvolutionSpatial2d spatial,
                             uint32_t numOutputChannels,
                             bool hasBias,
                             std::optional<DataType> weightsDataType,
                             const TensorPlacement& placement,
                             bool inferenceOnly,
                             int64_t stampedId,
                             uint32_t groups)
    : CustomLayer(
          buildExpression(hasBias, groups, spatial, placement),
          placement,
          defineParameters(numOutputChannels, hasBias, filterWidth, filterHeight, weightsDataType, groups),
          inferenceOnly,
          stampedId) {}

DynamicExpression Convolution2d::buildExpression(
    bool hasBias, uint32_t groups, ConvolutionSpatial2d spatial, const TensorPlacement& placement) {
    return DynamicExpression([hasBias, groups, spatial, placement](const DynamicExpression::TensorMap& inputs,
                                                           const DynamicExpression::TensorMap& outputs,
                                                           Stream& stream) -> DynamicExpressionBuild {
        (void)stream;

        const Tensor& featureInputTensor = inputs.at("feature_input");
        const Tensor& wTensor = inputs.at("weights");
        THOR_THROW_IF_FALSE(wTensor.getPlacement() == placement);

        if (featureInputTensor.getDimensions().size() != 4) {
            throw std::runtime_error("Convolution2d expects feature_input to be 4D NCHW.");
        }
        if (wTensor.getDimensions().size() != 4) {
            throw std::runtime_error("Convolution2d expects weights to be 4D KCRS.");
        }
        if (groups == 0 || featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1] * groups ||
            wTensor.getDimensions()[0] % groups != 0) {
            throw std::runtime_error("Convolution2d grouped channel geometry is invalid.");
        }
        THOR_THROW_IF_FALSE(featureInputTensor.getPlacement() == placement);

        const uint64_t effectiveFilterRows =
            static_cast<uint64_t>(spatial.dilation_h) * (wTensor.getDimensions()[2] - 1ULL) + 1ULL;
        const uint64_t effectiveFilterCols =
            static_cast<uint64_t>(spatial.dilation_w) * (wTensor.getDimensions()[3] - 1ULL) + 1ULL;
        const uint64_t expectedOutputRows =
            (featureInputTensor.getDimensions()[2] + spatial.pre_padding_h + spatial.post_padding_h - effectiveFilterRows) /
                spatial.stride_h +
            1;
        const uint64_t expectedOutputCols =
            (featureInputTensor.getDimensions()[3] + spatial.pre_padding_w + spatial.post_padding_w - effectiveFilterCols) /
                spatial.stride_w +
            1;

        if (outputs.contains("feature_output")) {
            const Tensor& featureOutputTensor = outputs.at("feature_output");
            if (featureOutputTensor.getDimensions().size() != 4) {
                throw std::runtime_error("Convolution2d expects feature_output to be 4D NCHW.");
            }
            if (featureOutputTensor.getDimensions()[0] != featureInputTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[1] != wTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[2] != expectedOutputRows ||
                featureOutputTensor.getDimensions()[3] != expectedOutputCols) {
                throw std::runtime_error("Convolution2d feature_output shape does not match the implied convolution output shape.");
            }
            THOR_THROW_IF_FALSE(featureOutputTensor.getPlacement() == placement);
        }

        const DataType weightsDType = wTensor.getDescriptor().getDataType();

        auto fin = Expression::input("feature_input");
        auto w = Expression::input("weights", weightsDType, weightsDType);

        Expression fout = Expression::conv2d(fin, w, spatial, DataType::FP32, std::nullopt, groups);

        if (hasBias) {
            const Tensor& bTensor = inputs.at("biases");
            if (bTensor.getDimensions().size() != 1) {
                throw std::runtime_error("Convolution2d expects biases to be 1D [K].");
            }
            if (bTensor.getDimensions()[0] != wTensor.getDimensions()[0]) {
                throw std::runtime_error("Convolution2d bias size must match number of output channels.");
            }

            const DataType biasDType = bTensor.getDescriptor().getDataType();
            auto b = Expression::input("biases", biasDType, biasDType).unsqueeze({0, 2, 3});
            fout = fout + b;
        }

        auto expressionOutputs = Expression::outputs({{"feature_output", fout}});

        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            {outputs},
            {}};
    });
}

std::vector<std::shared_ptr<PhysicalParameter>> Convolution2d::defineParameters(uint32_t numOutputChannels,
                                                                                bool hasBias,
                                                                                uint32_t filterWidth,
                                                                                uint32_t filterHeight,
                                                                                std::optional<DataType> weightsDataType,
                                                                                uint32_t groups) {
    std::vector<std::shared_ptr<PhysicalParameter>> parameters;
    parameters.push_back(
        std::make_shared<ConvWeightsParameter>(
            "weights", weightsDataType, true, true, numOutputChannels, filterWidth, filterHeight, groups));
    if (hasBias) {
        parameters.push_back(std::make_shared<ConvBiasesParameter>("biases", weightsDataType, true, true, numOutputChannels));
    }
    return parameters;
}

}  // namespace ThorImplementation
