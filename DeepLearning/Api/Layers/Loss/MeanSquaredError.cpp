#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"

#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";

void validateLabelsDType(DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported MSE label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported MSE predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("MSE example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw runtime_error("Unsupported MSE example_weights dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error("MSE example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

ThorImplementation::DynamicExpression makeMSELossExpression(DataType lossDataType) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression loss = (diff * diff).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMSELossExpression(DataType lossDataType) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression loss = ((diff * diff) * exampleWeights).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeMSEGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression gradient =
        (diff * ThorImplementation::Expression(2.0f * ThorImplementation::Loss::getLossScalingFactor())).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMSEGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression gradient =
        (diff * exampleWeights * ThorImplementation::Expression(2.0f * ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void MSE::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);

    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawMSE = MultiInputCustomLoss::Builder()
                                          .network(*network)
                                          .lossExpression(makeWeightedMSELossExpression(lossDataType))
                                          .gradientExpression(makeWeightedMSEGradientExpression(predictionsTensor.getDataType()))
                                          .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                                          .auxiliaryInput(kLabelsName, labelsTensor)
                                          .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                                          .lossName(kLossName)
                                          .lossDataType(lossDataType)
                                          .lossWeight(lossWeight.value_or(1.0f))
                                          .reportsRawLoss()
                                          .build();
        lossShaperInput = rawMSE.getLoss();
    } else {
        CustomLoss rawMSE = CustomLoss::Builder()
                                           .network(*network)
                                           .lossExpression(makeMSELossExpression(lossDataType))
                                           .gradientExpression(makeMSEGradientExpression(predictionsTensor.getDataType()))
                                           .predictions(predictionsTensor)
                                           .labels(labelsTensor)
                                           .predictionsName(kPredictionsName)
                                           .labelsName(kLabelsName)
                                           .lossName(kLossName)
                                           .gradientName(kGradientName)
                                           .lossDataType(lossDataType)
                                           .lossWeight(lossWeight.value_or(1.0f))
                                           .reportsRawLoss()
                                           .build();

        lossShaperInput = rawMSE.getLoss();
    }

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json MSE::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "mse";
    return j;
}

void MSE::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MSE::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mse")
        throw runtime_error("Layer type mismatch in MSE::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    MSE meanSquaredError;
    meanSquaredError.lossShape = j.at("loss_shape").get<LossShape>();
    meanSquaredError.lossDataType = j.at("loss_data_type").get<DataType>();

    meanSquaredError.lossWeight = ThorImplementation::lossWeightFromJson(j);
    meanSquaredError.predictionsTensor = predictions;
    meanSquaredError.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        meanSquaredError.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    meanSquaredError.network = network;
    meanSquaredError.initialized = true;
    meanSquaredError.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mse", &Thor::MSE::deserialize);
    return true;
}();
}  // namespace
