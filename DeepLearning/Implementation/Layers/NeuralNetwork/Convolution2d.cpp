#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2d.h"

#include <stdexcept>

namespace ThorImplementation {

namespace {
class ConvWeightsParameter : public Parameter {
   public:
    ConvWeightsParameter(std::string name,
                         Optional<TensorDescriptor::DataType> storageDataType,
                         bool trainable,
                         bool trainingEnabled,
                         uint32_t numOutputChannels,
                         uint32_t filterWidth,
                         uint32_t filterHeight)
        : Parameter(name, trainable),
          numOutputChannels(numOutputChannels),
          filterWidth(filterWidth),
          filterHeight(filterHeight),
          storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        TensorDescriptor::DataType resolvedDataType = storageDataType.isPresent() ? storageDataType.get() : inputTensor.getDataType();

        const auto& inputDims = inputTensor.getDimensions();
        if (inputDims.size() != 4) {
            throw std::runtime_error("Convolution2d weights require 4D NCHW feature input tensor.");
        }

        TensorDescriptor descriptor(resolvedDataType, {numOutputChannels, inputDims[1], filterHeight, filterWidth});
        storage = Tensor(inputTensor.getPlacement(), descriptor);
    }

   private:
    const uint32_t numOutputChannels;
    const uint32_t filterWidth;
    const uint32_t filterHeight;
    const Optional<TensorDescriptor::DataType> storageDataType;
};

class ConvBiasesParameter : public Parameter {
   public:
    ConvBiasesParameter(std::string name,
                        Optional<TensorDescriptor::DataType> storageDataType,
                        bool trainable,
                        bool trainingEnabled,
                        uint32_t numOutputChannels)
        : Parameter(name, trainable), numOutputChannels(numOutputChannels), storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        TensorDescriptor::DataType resolvedDataType = storageDataType.isPresent() ? storageDataType.get() : inputTensor.getDataType();
        storage = Tensor(inputTensor.getPlacement(), TensorDescriptor(resolvedDataType, {numOutputChannels}));
    }

   private:
    const uint32_t numOutputChannels;
    const Optional<TensorDescriptor::DataType> storageDataType;
};
}  // namespace

using DataType = TensorDescriptor::DataType;

Convolution2d::Convolution2d(uint32_t filterWidth,
                             uint32_t filterHeight,
                             uint32_t filterHorizontalStride,
                             uint32_t filterVerticalStride,
                             uint32_t leftAndRightPadWidth,
                             uint32_t topAndBottomPadHeight,
                             uint32_t numOutputChannels,
                             bool hasBias,
                             Optional<DataType> weightsDataType,
                             const TensorPlacement& placement,
                             bool inferenceOnly,
                             int64_t stampedId)
    : CustomLayer(
          buildExpression(hasBias, filterVerticalStride, filterHorizontalStride, topAndBottomPadHeight, leftAndRightPadWidth, placement),
          placement,
          defineParameters(numOutputChannels, hasBias, filterWidth, filterHeight, weightsDataType),
          inferenceOnly,
          stampedId,
          false) {}

DynamicExpression Convolution2d::buildExpression(
    bool hasBias, uint32_t strideH, uint32_t strideW, uint32_t padH, uint32_t padW, const TensorPlacement& placement) {
    return DynamicExpression([hasBias, strideH, strideW, padH, padW, placement](const DynamicExpression::TensorMap& inputs,
                                                                                const DynamicExpression::TensorMap& outputs,
                                                                                Stream& stream) -> DynamicExpressionBuild {
        (void)stream;

        const Tensor& featureInputTensor = inputs.at("feature_input");
        const Tensor& wTensor = inputs.at("weights");

        if (featureInputTensor.getDimensions().size() != 4) {
            throw std::runtime_error("Convolution2d expects feature_input to be 4D NCHW.");
        }
        if (wTensor.getDimensions().size() != 4) {
            throw std::runtime_error("Convolution2d expects weights to be 4D KCRS.");
        }
        if (featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1]) {
            throw std::runtime_error("Convolution2d input channels must match weight channels.");
        }

        const uint64_t expectedOutputRows = (featureInputTensor.getDimensions()[2] + 2 * padH - wTensor.getDimensions()[2]) / strideH + 1;
        const uint64_t expectedOutputCols = (featureInputTensor.getDimensions()[3] + 2 * padW - wTensor.getDimensions()[3]) / strideW + 1;

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
        }

        const DataType weightsDType = wTensor.getDescriptor().getDataType();

        auto fin = Expression::input("feature_input");
        auto w = Expression::input("weights", weightsDType, weightsDType);

        Expression fout = Expression::conv2d(fin, w, strideH, strideW, padH, padW);

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

std::vector<std::shared_ptr<Parameter>> Convolution2d::defineParameters(uint32_t numOutputChannels,
                                                                        bool hasBias,
                                                                        uint32_t filterWidth,
                                                                        uint32_t filterHeight,
                                                                        Optional<TensorDescriptor::DataType> weightsDataType) {
    std::vector<std::shared_ptr<Parameter>> parameters;
    parameters.push_back(
        std::make_shared<ConvWeightsParameter>("weights", weightsDataType, true, true, numOutputChannels, filterWidth, filterHeight));
    if (hasBias) {
        parameters.push_back(std::make_shared<ConvBiasesParameter>("biases", weightsDataType, true, true, numOutputChannels));
    }
    return parameters;
}

}  // namespace ThorImplementation
