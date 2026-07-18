#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"

#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kExampleWeightsName = "example_weights";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";

void validateExponent(float exponent) {
    if (!std::isfinite(exponent) || exponent < 1.0f) {
        throw runtime_error("MeanPowerError exponent must be finite and greater than or equal to 1.0.");
    }
}

void validateLabelsDType(DataType dtype) {
    ThorImplementation::RegressionLossDType::validateLabelsDType("MeanPowerError", dtype);
}

void validatePredictionsDType(DataType dtype) {
    ThorImplementation::RegressionLossDType::validatePredictionsDType("MeanPowerError", dtype);
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("MeanPowerError example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    ThorImplementation::RegressionLossDType::validateExampleWeightDType("MeanPowerError", dtype);
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error("MeanPowerError example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

ThorImplementation::Expression meanPowerLossExpression(float exponent) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = predictions - labels;
    if (exponent == 1.0f) {
        return diff.abs();
    }
    if (exponent == 2.0f) {
        return diff * diff;
    }
    ThorImplementation::Expression exponentExpr(exponent);
    return diff.abs().pow(exponentExpr);
}

ThorImplementation::DynamicExpression makeMeanPowerLossExpression(float exponent, DataType lossDataType) {
    validateExponent(exponent);
    ThorImplementation::Expression loss = meanPowerLossExpression(exponent).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMeanPowerLossExpression(float exponent, DataType lossDataType) {
    validateExponent(exponent);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = (meanPowerLossExpression(exponent) * exampleWeights).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::Expression signOf(const ThorImplementation::Expression& diff) {
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression positive(1.0);
    ThorImplementation::Expression negative(-1.0);
    return ThorImplementation::Expression::where(diff > zero,
                                                 positive,
                                                 ThorImplementation::Expression::where(diff < zero, negative, zero));
}

ThorImplementation::Expression meanPowerGradientExpression(float exponent) {
    validateExponent(exponent);
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression scale(exponent * ThorImplementation::Loss::getLossScalingFactor());
    if (exponent == 1.0f) {
        return signOf(diff) * scale;
    }
    ThorImplementation::Expression absDiff = diff.abs();
    ThorImplementation::Expression power(exponent - 1.0f);
    return signOf(diff) * absDiff.pow(power) * scale;
}

ThorImplementation::DynamicExpression makeMeanPowerGradientExpression(float exponent, DataType predictionsDataType) {
    validateExponent(exponent);
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression gradient = meanPowerGradientExpression(exponent).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMeanPowerGradientExpression(float exponent, DataType predictionsDataType) {
    validateExponent(exponent);
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression gradient = (meanPowerGradientExpression(exponent) * exampleWeights).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void MeanPowerError::buildSupportLayersAndAddToNetwork() {
    ThorImplementation::RegressionLossDType::validateLossDType("MeanPowerError", lossDataType);
    validateExponent(exponent);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);

    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawMeanPowerError =
            MultiInputCustomLoss::Builder()
                .network(*network)
                .lossExpression(makeWeightedMeanPowerLossExpression(exponent, lossDataType))
                .gradientExpression(makeWeightedMeanPowerGradientExpression(exponent, predictionsTensor.getDataType()))
                .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                .auxiliaryInput(kLabelsName, labelsTensor)
                .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                .lossName(kLossName)
                .lossDataType(lossDataType)
                .lossWeight(lossWeight.value_or(1.0f))
                .reportsRawLoss()
                .build();
        lossShaperInput = rawMeanPowerError.getLoss();
    } else {
        CustomLoss rawMeanPowerError = CustomLoss::Builder()
                                           .network(*network)
                                           .lossExpression(makeMeanPowerLossExpression(exponent, lossDataType))
                                           .gradientExpression(makeMeanPowerGradientExpression(exponent, predictionsTensor.getDataType()))
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

        lossShaperInput = rawMeanPowerError.getLoss();
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

json MeanPowerError::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "mean_power_error";
    j["exponent"] = exponent;
    return j;
}

void MeanPowerError::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MeanPowerError::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mean_power_error")
        throw runtime_error("Layer type mismatch in MeanPowerError::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    MeanPowerError meanPowerError;
    meanPowerError.lossShape = j.at("loss_shape").get<LossShape>();
    meanPowerError.lossDataType = j.at("loss_data_type").get<DataType>();

    meanPowerError.lossWeight = ThorImplementation::lossWeightFromJson(j);
    meanPowerError.exponent = j.value("exponent", 1.5f);
    validateExponent(meanPowerError.exponent);
    meanPowerError.predictionsTensor = predictions;
    meanPowerError.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        meanPowerError.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    meanPowerError.network = network;
    meanPowerError.initialized = true;
    meanPowerError.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mean_power_error", &Thor::MeanPowerError::deserialize);
    return true;
}();
}  // namespace
