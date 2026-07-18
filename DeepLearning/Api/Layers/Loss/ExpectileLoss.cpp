#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/ExpectileLoss.h"

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
    ThorImplementation::RegressionLossDType::validateLabelsDType("ExpectileLoss", dtype);
}

void validatePredictionsDType(DataType dtype) {
    ThorImplementation::RegressionLossDType::validatePredictionsDType("ExpectileLoss", dtype);
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("ExpectileLoss example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    ThorImplementation::RegressionLossDType::validateExampleWeightDType("ExpectileLoss", dtype);
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error("ExpectileLoss example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

ThorImplementation::DynamicExpression makeExpectileLossExpression(DataType lossDataType, float expectile) {
    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression squaredError = error * error;
    ThorImplementation::Expression underPredictionWeight(2.0f * expectile);
    ThorImplementation::Expression overPredictionWeight(2.0f * (1.0f - expectile));
    ThorImplementation::Expression asymmetricWeight =
        ThorImplementation::Expression::where(error > zero, underPredictionWeight, overPredictionWeight);
    ThorImplementation::Expression loss = (asymmetricWeight * squaredError).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedExpectileLossExpression(DataType lossDataType, float expectile) {
    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression squaredError = error * error;
    ThorImplementation::Expression underPredictionWeight(2.0f * expectile);
    ThorImplementation::Expression overPredictionWeight(2.0f * (1.0f - expectile));
    ThorImplementation::Expression asymmetricWeight =
        ThorImplementation::Expression::where(error > zero, underPredictionWeight, overPredictionWeight);
    ThorImplementation::Expression loss =
        (asymmetricWeight * squaredError * exampleWeights).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeExpectileGradientExpression(DataType predictionsDataType, float expectile) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression predictionError = predictions - labels;
    ThorImplementation::Expression underPredictionScale(4.0f * expectile);
    ThorImplementation::Expression overPredictionScale(4.0f * (1.0f - expectile));
    ThorImplementation::Expression asymmetricScale =
        ThorImplementation::Expression::where(error > zero, underPredictionScale, overPredictionScale);
    ThorImplementation::Expression gradient =
        (asymmetricScale * predictionError *
         ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedExpectileGradientExpression(DataType predictionsDataType, float expectile) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression predictionError = predictions - labels;
    ThorImplementation::Expression underPredictionScale(4.0f * expectile);
    ThorImplementation::Expression overPredictionScale(4.0f * (1.0f - expectile));
    ThorImplementation::Expression asymmetricScale =
        ThorImplementation::Expression::where(error > zero, underPredictionScale, overPredictionScale);
    ThorImplementation::Expression gradient =
        (asymmetricScale * predictionError * exampleWeights *
         ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void ExpectileLoss::buildSupportLayersAndAddToNetwork() {
    ThorImplementation::RegressionLossDType::validateLossDType("ExpectileLoss", lossDataType);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(expectile > 0.0f && expectile < 1.0f);

    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawExpectileLoss = MultiInputCustomLoss::Builder()
                                                       .network(*network)
                                                       .lossExpression(makeWeightedExpectileLossExpression(lossDataType, expectile))
                                                       .gradientExpression(makeWeightedExpectileGradientExpression(
                                                           predictionsTensor.getDataType(), expectile))
                                                       .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                                                       .auxiliaryInput(kLabelsName, labelsTensor)
                                                       .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                                                       .lossName(kLossName)
                                                       .lossDataType(lossDataType)
                                                       .lossWeight(lossWeight.value_or(1.0f))
                                                       .reportsRawLoss()
                                                       .build();
        lossShaperInput = rawExpectileLoss.getLoss();
    } else {
        CustomLoss rawExpectileLoss = CustomLoss::Builder()
                                                 .network(*network)
                                                 .lossExpression(makeExpectileLossExpression(lossDataType, expectile))
                                                 .gradientExpression(
                                                     makeExpectileGradientExpression(predictionsTensor.getDataType(), expectile))
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

        lossShaperInput = rawExpectileLoss.getLoss();
    }

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json ExpectileLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["expectile"] = expectile;
    return j;
}

void ExpectileLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in ExpectileLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "expectile_loss")
        throw runtime_error("Layer type mismatch in ExpectileLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    ExpectileLoss expectileLoss;
    expectileLoss.lossShape = j.at("loss_shape").get<LossShape>();
    expectileLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    expectileLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    expectileLoss.expectile = j.value("expectile", 0.5f);
    THOR_THROW_IF_FALSE(expectileLoss.expectile > 0.0f && expectileLoss.expectile < 1.0f);
    expectileLoss.predictionsTensor = predictions;
    expectileLoss.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        expectileLoss.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    expectileLoss.network = network;
    expectileLoss.initialized = true;
    expectileLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("expectile_loss", &Thor::ExpectileLoss::deserialize);
    return true;
}();
}  // namespace
