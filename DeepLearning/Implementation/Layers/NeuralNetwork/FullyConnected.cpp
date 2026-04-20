#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"

namespace ThorImplementation {

namespace {
class FCWeightsParameter : public Parameter {
   public:
    FCWeightsParameter(Optional<TensorDescriptor::DataType> storageDataType, uint32_t numOutputFeatures)
        : Parameter("weights", true), numOutputFeatures(numOutputFeatures), storageDataType(storageDataType) {}

    void createStorage(const Tensor& inputTensor) override {
        TensorDescriptor::DataType resolvedDataType;
        if (storageDataType.isPresent())
            resolvedDataType = storageDataType.get();
        else
            resolvedDataType = inputTensor.getDataType();

        const uint64_t batchSize = inputTensor.getDimensions()[0];
        TensorDescriptor descriptor(resolvedDataType, {inputTensor.getTotalNumElements() / batchSize, numOutputFeatures});
        storage = Tensor(inputTensor.getPlacement(), descriptor);
    }

   private:
    const uint32_t numOutputFeatures;
    const Optional<TensorDescriptor::DataType> storageDataType;
};

class FCBiasesParameter : public Parameter {
   public:
    FCBiasesParameter(Optional<TensorDescriptor::DataType> storageDataType, uint32_t numOutputFeatures)
        : Parameter("biases", true), numOutputFeatures(numOutputFeatures), storageDataType(storageDataType) {}

    void createStorage(const Tensor& inputTensor) override {
        TensorDescriptor::DataType resolvedDataType;
        if (storageDataType.isPresent())
            resolvedDataType = storageDataType.get();
        else
            resolvedDataType = inputTensor.getDataType();

        TensorDescriptor descriptor(resolvedDataType, {numOutputFeatures});
        storage = Tensor(inputTensor.getPlacement(), descriptor);
    }

   private:
    const uint32_t numOutputFeatures;
    const Optional<TensorDescriptor::DataType> storageDataType;
};
}  // namespace

using DataType = TensorDescriptor::DataType;

FullyConnected::FullyConnected(const uint32_t numOutputFeatures,
                               const bool hasBias,
                               Optional<DataType> weightsDataType,
                               const TensorPlacement& placement,
                               bool inferenceOnly,
                               int64_t stampedId)
    : CustomLayer(buildExpression(hasBias, placement),
                  placement,
                  defineParameters(numOutputFeatures, hasBias, weightsDataType),
                  inferenceOnly,
                  stampedId,
                  false) {}

DynamicExpression FullyConnected::buildExpression(bool hasBias, TensorPlacement placement) {
    return DynamicExpression([hasBias, placement](const DynamicExpression::TensorMap& inputs,
                                                  const DynamicExpression::TensorMap& outputs,
                                                  Stream& stream) -> DynamicExpressionBuild {
        (void)stream;

        // This is just validation. CustomLayer connects the proper tensors with the contracted names
        // (feature_input, feature_output, parameter names)
        const Tensor& featureInputTensor = inputs.at("feature_input");
        const Tensor& wTensor = inputs.at("weights");
        assert(wTensor.getDimensions().size() == 2);
        assert(featureInputTensor.getDimensions()[1] == wTensor.getDimensions()[0]);
        if (outputs.contains("feature_output")) {
            const Tensor& featureOutputTensor = outputs.at("feature_output");
            assert(featureOutputTensor.getDimensions().size() == 2);
            assert(featureOutputTensor.getDimensions()[1] == wTensor.getDimensions()[1]);
        }

        const DataType weightsDType = wTensor.getDescriptor().getDataType();

        auto fin = Expression::input("feature_input");
        auto w = Expression::input("weights", weightsDType, weightsDType);

        // [batch, in_features] @ [in_features, out_features]
        Expression fout = Expression::matmul(fin, w);

        if (hasBias) {
            const Tensor& bTensor = inputs.at("biases");
            assert(bTensor.getDimensions().size() == 1);
            assert(bTensor.getDimensions()[0] == wTensor.getDimensions()[1]);

            const DataType biasDType = bTensor.getDescriptor().getDataType();
            auto b = Expression::input("biases", biasDType, biasDType);

            // Broadcast [out_features] over batch
            fout = fout + b;
        }

        auto expressionOutputs = Expression::outputs({{"feature_output", fout}});

        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,     // stamp_inputs
            {},         // tensor_scalar_inputs
            {outputs},  // preallocated_outputs
            {}          // requested_output_shapes
        };
    });
}

std::vector<std::shared_ptr<Parameter>> FullyConnected::defineParameters(uint32_t numOutputFeatures,
                                                                         bool hasBias,
                                                                         Optional<TensorDescriptor::DataType> weightsDataType) {
    std::vector<std::shared_ptr<Parameter>> parameters;
    parameters.push_back(std::make_shared<FCWeightsParameter>(weightsDataType, numOutputFeatures));
    if (hasBias)
        parameters.push_back(std::make_shared<FCBiasesParameter>(weightsDataType, numOutputFeatures));
    return parameters;
}

}  // namespace ThorImplementation
