#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected2.h"

namespace ThorImplementation {

namespace {
class FCWeightsParameter : public Parameter {
   public:
    FCWeightsParameter(std::string name, bool trainable, bool trainingEnabled, uint32_t numOutputFeatures)
        : Parameter(name, trainable, trainingEnabled), numOutputFeatures(numOutputFeatures) {}

    void createStorage(const Tensor& inputTensor) override {
        const uint64_t batchSize = inputTensor.getDimensions()[0];
        TensorDescriptor descriptor(inputTensor.getDataType(), {inputTensor.getTotalNumElements() / batchSize, numOutputFeatures});
        storage = Tensor(inputTensor.getPlacement(), descriptor);
    }

   private:
    const uint32_t numOutputFeatures;
};

class FCBiasesParameter : public Parameter {
   public:
    FCBiasesParameter(std::string name, bool trainable, bool trainingEnabled, uint32_t numOutputFeatures)
        : Parameter(name, trainable, trainingEnabled), numOutputFeatures(numOutputFeatures) {}

    void createStorage(const Tensor& inputTensor) override {
        TensorDescriptor descriptor(inputTensor.getDataType(), {numOutputFeatures});
        storage = Tensor(inputTensor.getPlacement(), descriptor);
    }

   private:
    const uint32_t numOutputFeatures;
};
}  // namespace

FullyConnected2::FullyConnected2(const uint32_t numOutputFeatures,
                                 const bool hasBias,
                                 Optional<DataType> weightsDataType,
                                 const TensorPlacement& placement,
                                 bool inferenceOnly,
                                 int64_t stampedId)
    : CustomLayer(
          buildExpression(hasBias, placement), placement, defineParameters(numOutputFeatures, hasBias), inferenceOnly, stampedId, false) {
    printf("%d", placement.getDeviceNum());
}

DynamicExpression FullyConnected2::buildExpression(bool hasBias, TensorPlacement placement) {
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

std::vector<std::shared_ptr<Parameter>> FullyConnected2::defineParameters(uint32_t numOutputFeatures, bool hasBias) {
    std::vector<std::shared_ptr<Parameter>> parameters;
    parameters.push_back(std::make_shared<FCWeightsParameter>("weights", true, true, numOutputFeatures));
    if (hasBias)
        parameters.push_back(std::make_shared<FCBiasesParameter>("biases", true, true, numOutputFeatures));
    return parameters;
}

}  // namespace ThorImplementation
