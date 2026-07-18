#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"

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
    ThorImplementation::RegressionLossDType::validateLabelsDType("QuantileLoss", dtype);
}

void validatePredictionsDType(DataType dtype) {
    ThorImplementation::RegressionLossDType::validatePredictionsDType("QuantileLoss", dtype);
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("QuantileLoss example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    ThorImplementation::RegressionLossDType::validateExampleWeightDType("QuantileLoss", dtype);
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error("QuantileLoss example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

ThorImplementation::DynamicExpression makeQuantileLossExpression(DataType lossDataType, float quantile) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression q(quantile);
    ThorImplementation::Expression qMinusOne(quantile - 1.0f);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression overPredictionLoss = qMinusOne * error;
    ThorImplementation::Expression underPredictionLoss = q * error;
    ThorImplementation::Expression loss =
        ThorImplementation::Expression::where(error > zero, underPredictionLoss, overPredictionLoss).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedQuantileLossExpression(DataType lossDataType, float quantile) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression q(quantile);
    ThorImplementation::Expression qMinusOne(quantile - 1.0f);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression overPredictionLoss = qMinusOne * error;
    ThorImplementation::Expression underPredictionLoss = q * error;
    ThorImplementation::Expression loss =
        (ThorImplementation::Expression::where(error > zero, underPredictionLoss, overPredictionLoss) * exampleWeights)
            .withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeQuantileGradientExpression(DataType predictionsDataType, float quantile) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression underPredictionGradient(-quantile);
    ThorImplementation::Expression overPredictionGradient(1.0f - quantile);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression subgradient = ThorImplementation::Expression::where(
        error > zero,
        underPredictionGradient,
        ThorImplementation::Expression::where(error < zero, overPredictionGradient, zero));
    ThorImplementation::Expression gradient =
        (subgradient * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedQuantileGradientExpression(DataType predictionsDataType, float quantile) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression negativeQ(-quantile);
    ThorImplementation::Expression oneMinusQ(1.0f - quantile);
    ThorImplementation::Expression error = labels - predictions;
    ThorImplementation::Expression subgradient =
        ThorImplementation::Expression::where(error > zero,
                                             negativeQ,
                                             ThorImplementation::Expression::where(error < zero, oneMinusQ, zero));
    ThorImplementation::Expression gradient =
        (subgradient * exampleWeights * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void QuantileLoss::buildSupportLayersAndAddToNetwork() {
    ThorImplementation::RegressionLossDType::validateLossDType("QuantileLoss", lossDataType);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);
    THOR_THROW_IF_FALSE(quantile > 0.0f && quantile < 1.0f);

    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawQuantileLoss = MultiInputCustomLoss::Builder()
                                                 .network(*network)
                                                 .lossExpression(makeWeightedQuantileLossExpression(lossDataType, quantile))
                                                 .gradientExpression(
                                                     makeWeightedQuantileGradientExpression(predictionsTensor.getDataType(), quantile))
                                                 .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                                                 .auxiliaryInput(kLabelsName, labelsTensor)
                                                 .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                                                 .lossName(kLossName)
                                                 .lossDataType(lossDataType)
                                                 .lossWeight(lossWeight.value_or(1.0f))
                                                 .reportsRawLoss()
                                                 .build();
        lossShaperInput = rawQuantileLoss.getLoss();
    } else {
        CustomLoss rawQuantileLoss = CustomLoss::Builder()
                                          .network(*network)
                                          .lossExpression(makeQuantileLossExpression(lossDataType, quantile))
                                          .gradientExpression(makeQuantileGradientExpression(predictionsTensor.getDataType(), quantile))
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

        lossShaperInput = rawQuantileLoss.getLoss();
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

json QuantileLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["quantile"] = quantile;
    return j;
}

void QuantileLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in QuantileLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "quantile_loss")
        throw runtime_error("Layer type mismatch in QuantileLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    QuantileLoss quantileLoss;
    quantileLoss.lossShape = j.at("loss_shape").get<LossShape>();
    quantileLoss.lossDataType = j.at("loss_data_type").get<DataType>();

    quantileLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    quantileLoss.quantile = j.value("quantile", 0.5f);
    quantileLoss.predictionsTensor = predictions;
    quantileLoss.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        quantileLoss.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    quantileLoss.network = network;
    quantileLoss.initialized = true;
    quantileLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("quantile_loss", &Thor::QuantileLoss::deserialize);
    return true;
}();
}  // namespace
