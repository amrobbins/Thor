#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"

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

void validateLevel(float level) {
    if (!std::isfinite(level) || level <= 0.0f || level >= 1.0f)
        throw runtime_error("AsymmetricPowerLoss level must be finite, greater than zero, and less than one.");
}

void validateExponent(float exponent) {
    if (!std::isfinite(exponent) || exponent < 1.0f)
        throw runtime_error("AsymmetricPowerLoss exponent must be finite and greater than or equal to 1.0.");
}

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
            throw runtime_error("Unsupported AsymmetricPowerLoss label dtype: " +
                                ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported AsymmetricPowerLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("AsymmetricPowerLoss example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported AsymmetricPowerLoss example_weights dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error(
            "AsymmetricPowerLoss example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

ThorImplementation::Expression absolutePower(const ThorImplementation::Expression& value, float exponent) {
    if (exponent == 1.0f)
        return value.abs();
    if (exponent == 2.0f)
        return value * value;
    return value.abs().pow(ThorImplementation::Expression(exponent));
}

ThorImplementation::Expression asymmetricWeight(const ThorImplementation::Expression& labelMinusPrediction, float level) {
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression underPredictionWeight(2.0f * level);
    ThorImplementation::Expression overPredictionWeight(2.0f * (1.0f - level));
    return ThorImplementation::Expression::where(labelMinusPrediction > zero, underPredictionWeight, overPredictionWeight);
}

ThorImplementation::Expression asymmetricPowerLossExpression(float level, float exponent) {
    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression error = labels - predictions;
    return asymmetricWeight(error, level) * absolutePower(error, exponent);
}

ThorImplementation::DynamicExpression makeAsymmetricPowerLossExpression(DataType lossDataType, float level, float exponent) {
    validateLevel(level);
    validateExponent(exponent);
    ThorImplementation::Expression loss = asymmetricPowerLossExpression(level, exponent).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedAsymmetricPowerLossExpression(DataType lossDataType,
                                                                                 float level,
                                                                                 float exponent) {
    validateLevel(level);
    validateExponent(exponent);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss =
        (asymmetricPowerLossExpression(level, exponent) * exampleWeights).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::Expression signOf(const ThorImplementation::Expression& value) {
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression positive(1.0f);
    ThorImplementation::Expression negative(-1.0f);
    return ThorImplementation::Expression::where(
        value > zero, positive, ThorImplementation::Expression::where(value < zero, negative, zero));
}

ThorImplementation::Expression meanPowerGradientTerm(const ThorImplementation::Expression& predictionMinusLabel,
                                                      float exponent) {
    ThorImplementation::Expression scale(exponent * ThorImplementation::Loss::getLossScalingFactor());
    if (exponent == 1.0f)
        return signOf(predictionMinusLabel) * scale;
    if (exponent == 2.0f)
        return predictionMinusLabel * scale;
    ThorImplementation::Expression power(exponent - 1.0f);
    return signOf(predictionMinusLabel) * predictionMinusLabel.abs().pow(power) * scale;
}

ThorImplementation::Expression asymmetricPowerGradientExpression(float level, float exponent) {
    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression predictionMinusLabel = predictions - labels;
    ThorImplementation::Expression labelMinusPrediction = labels - predictions;
    return asymmetricWeight(labelMinusPrediction, level) * meanPowerGradientTerm(predictionMinusLabel, exponent);
}

ThorImplementation::DynamicExpression makeAsymmetricPowerGradientExpression(DataType predictionsDataType,
                                                                             float level,
                                                                             float exponent) {
    validatePredictionsDType(predictionsDataType);
    validateLevel(level);
    validateExponent(exponent);
    ThorImplementation::Expression gradient =
        asymmetricPowerGradientExpression(level, exponent).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedAsymmetricPowerGradientExpression(DataType predictionsDataType,
                                                                                     float level,
                                                                                     float exponent) {
    validatePredictionsDType(predictionsDataType);
    validateLevel(level);
    validateExponent(exponent);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression gradient =
        (asymmetricPowerGradientExpression(level, exponent) * exampleWeights).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void AsymmetricPowerLoss::buildSupportLayersAndAddToNetwork() {
    validateLevel(level);
    validateExponent(exponent);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);

    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawAsymmetricPowerLoss =
            MultiInputCustomLoss::Builder()
                .network(*network)
                .lossExpression(makeWeightedAsymmetricPowerLossExpression(lossDataType, level, exponent))
                .gradientExpression(
                    makeWeightedAsymmetricPowerGradientExpression(predictionsTensor.getDataType(), level, exponent))
                .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                .auxiliaryInput(kLabelsName, labelsTensor)
                .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                .lossName(kLossName)
                .lossDataType(lossDataType)
                .lossWeight(lossWeight.value_or(1.0f))
                .reportsRawLoss()
                .build();
        lossShaperInput = rawAsymmetricPowerLoss.getLoss();
    } else {
        CustomLoss rawAsymmetricPowerLoss =
            CustomLoss::Builder()
                .network(*network)
                .lossExpression(makeAsymmetricPowerLossExpression(lossDataType, level, exponent))
                .gradientExpression(makeAsymmetricPowerGradientExpression(predictionsTensor.getDataType(), level, exponent))
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
        lossShaperInput = rawAsymmetricPowerLoss.getLoss();
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

json AsymmetricPowerLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["level"] = level;
    j["exponent"] = exponent;
    return j;
}

void AsymmetricPowerLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in AsymmetricPowerLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "asymmetric_power_loss")
        throw runtime_error("Layer type mismatch in AsymmetricPowerLoss::deserialize: " +
                            j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    AsymmetricPowerLoss asymmetricPowerLoss;
    asymmetricPowerLoss.lossShape = j.at("loss_shape").get<LossShape>();
    asymmetricPowerLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    asymmetricPowerLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    asymmetricPowerLoss.level = j.value("level", 0.5f);
    asymmetricPowerLoss.exponent = j.value("exponent", 1.5f);
    validateLevel(asymmetricPowerLoss.level);
    validateExponent(asymmetricPowerLoss.exponent);
    asymmetricPowerLoss.predictionsTensor = predictions;
    asymmetricPowerLoss.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        asymmetricPowerLoss.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    asymmetricPowerLoss.network = network;
    asymmetricPowerLoss.initialized = true;
    asymmetricPowerLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("asymmetric_power_loss", &Thor::AsymmetricPowerLoss::deserialize);
    return true;
}();
}  // namespace
