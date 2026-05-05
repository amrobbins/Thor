#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution3d.h"

#include <stdexcept>

namespace ThorImplementation {

namespace {
class Conv3dWeightsParameter : public PhysicalParameter {
   public:
    Conv3dWeightsParameter(std::string name,
                           Optional<TensorDescriptor::DataType> storageDataType,
                           bool trainable,
                           bool trainingEnabled,
                           uint32_t numOutputChannels,
                           uint32_t filterWidth,
                           uint32_t filterHeight,
                           uint32_t filterDepth)
        : PhysicalParameter(name, trainable),
          numOutputChannels(numOutputChannels),
          filterWidth(filterWidth),
          filterHeight(filterHeight),
          filterDepth(filterDepth),
          storageDataType(storageDataType) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& inputTensor = context.getFeatureInput();
        TensorDescriptor::DataType resolvedDataType = storageDataType.isPresent() ? storageDataType.get() : inputTensor.getDataType();

        const auto& inputDims = inputTensor.getDimensions();
        if (inputDims.size() != 5) {
            throw std::runtime_error("Convolution3d weights require 5D NCDHW feature input tensor.");
        }

        TensorDescriptor descriptor(resolvedDataType, {numOutputChannels, inputDims[1], filterDepth, filterHeight, filterWidth});
        storage = Tensor(inputTensor.getPlacement(), descriptor);
    }

   private:
    const uint32_t numOutputChannels;
    const uint32_t filterWidth;
    const uint32_t filterHeight;
    const uint32_t filterDepth;
    const Optional<TensorDescriptor::DataType> storageDataType;
};

class Conv3dBiasesParameter : public PhysicalParameter {
   public:
    Conv3dBiasesParameter(std::string name,
                          Optional<TensorDescriptor::DataType> storageDataType,
                          bool trainable,
                          bool trainingEnabled,
                          uint32_t numOutputChannels)
        : PhysicalParameter(name, trainable), numOutputChannels(numOutputChannels), storageDataType(storageDataType) {}

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

Convolution3d::Convolution3d(uint32_t filterWidth,
                             uint32_t filterHeight,
                             uint32_t filterDepth,
                             uint32_t filterHorizontalStride,
                             uint32_t filterVerticalStride,
                             uint32_t filterDepthStride,
                             uint32_t leftAndRightPadWidth,
                             uint32_t topAndBottomPadHeight,
                             uint32_t frontAndBackPadDepth,
                             uint32_t numOutputChannels,
                             bool hasBias,
                             Optional<DataType> weightsDataType,
                             const TensorPlacement& placement,
                             bool inferenceOnly,
                             int64_t stampedId)
    : CustomLayer(buildExpression(hasBias,
                                  filterDepthStride,
                                  filterVerticalStride,
                                  filterHorizontalStride,
                                  frontAndBackPadDepth,
                                  topAndBottomPadHeight,
                                  leftAndRightPadWidth,
                                  placement),
                  placement,
                  defineParameters(numOutputChannels, hasBias, filterWidth, filterHeight, filterDepth, weightsDataType),
                  inferenceOnly,
                  stampedId,
                  false) {}

DynamicExpression Convolution3d::buildExpression(bool hasBias,
                                                 uint32_t strideD,
                                                 uint32_t strideH,
                                                 uint32_t strideW,
                                                 uint32_t padD,
                                                 uint32_t padH,
                                                 uint32_t padW,
                                                 const TensorPlacement& placement) {
    return DynamicExpression([hasBias, strideD, strideH, strideW, padD, padH, padW, placement](
                                 const DynamicExpression::TensorMap& inputs,
                                 const DynamicExpression::TensorMap& outputs,
                                 Stream& stream) -> DynamicExpressionBuild {
        (void)stream;

        const Tensor& featureInputTensor = inputs.at("feature_input");
        const Tensor& wTensor = inputs.at("weights");
        assert(wTensor.getPlacement() == placement);

        if (featureInputTensor.getDimensions().size() != 5) {
            throw std::runtime_error("Convolution3d expects feature_input to be 5D NCDHW.");
        }
        if (wTensor.getDimensions().size() != 5) {
            throw std::runtime_error("Convolution3d expects weights to be 5D KCDHW.");
        }
        if (featureInputTensor.getDimensions()[1] != wTensor.getDimensions()[1]) {
            throw std::runtime_error("Convolution3d input channels must match weight channels.");
        }
        assert(featureInputTensor.getPlacement() == placement);

        const uint64_t expectedOutputDepth =
            (featureInputTensor.getDimensions()[2] + 2 * padD - wTensor.getDimensions()[2]) / strideD + 1;
        const uint64_t expectedOutputRows =
            (featureInputTensor.getDimensions()[3] + 2 * padH - wTensor.getDimensions()[3]) / strideH + 1;
        const uint64_t expectedOutputCols =
            (featureInputTensor.getDimensions()[4] + 2 * padW - wTensor.getDimensions()[4]) / strideW + 1;

        if (outputs.contains("feature_output")) {
            const Tensor& featureOutputTensor = outputs.at("feature_output");
            if (featureOutputTensor.getDimensions().size() != 5) {
                throw std::runtime_error("Convolution3d expects feature_output to be 5D NCDHW.");
            }
            if (featureOutputTensor.getDimensions()[0] != featureInputTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[1] != wTensor.getDimensions()[0] ||
                featureOutputTensor.getDimensions()[2] != expectedOutputDepth ||
                featureOutputTensor.getDimensions()[3] != expectedOutputRows ||
                featureOutputTensor.getDimensions()[4] != expectedOutputCols) {
                throw std::runtime_error("Convolution3d feature_output shape does not match the implied convolution output shape.");
            }
            assert(featureOutputTensor.getPlacement() == placement);
        }

        const DataType weightsDType = wTensor.getDescriptor().getDataType();

        auto fin = Expression::input("feature_input");
        auto w = Expression::input("weights", weightsDType, weightsDType);

        Expression fout = Expression::conv3d(fin, w, strideD, strideH, strideW, padD, padH, padW);

        if (hasBias) {
            const Tensor& bTensor = inputs.at("biases");
            if (bTensor.getDimensions().size() != 1) {
                throw std::runtime_error("Convolution3d expects biases to be 1D [K].");
            }
            if (bTensor.getDimensions()[0] != wTensor.getDimensions()[0]) {
                throw std::runtime_error("Convolution3d bias size must match number of output channels.");
            }

            const DataType biasDType = bTensor.getDescriptor().getDataType();
            auto b = Expression::input("biases", biasDType, biasDType).unsqueeze({0, 2, 3, 4});
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

std::vector<std::shared_ptr<PhysicalParameter>> Convolution3d::defineParameters(uint32_t numOutputChannels,
                                                                                bool hasBias,
                                                                                uint32_t filterWidth,
                                                                                uint32_t filterHeight,
                                                                                uint32_t filterDepth,
                                                                                Optional<TensorDescriptor::DataType> weightsDataType) {
    std::vector<std::shared_ptr<PhysicalParameter>> parameters;
    parameters.push_back(std::make_shared<Conv3dWeightsParameter>(
        "weights", weightsDataType, true, true, numOutputChannels, filterWidth, filterHeight, filterDepth));
    if (hasBias) {
        parameters.push_back(std::make_shared<Conv3dBiasesParameter>("biases", weightsDataType, true, true, numOutputChannels));
    }
    return parameters;
}

}  // namespace ThorImplementation
