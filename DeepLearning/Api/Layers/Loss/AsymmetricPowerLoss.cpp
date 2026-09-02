#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"

#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"

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
    ThorImplementation::RegressionLossDType::validateLabelsDType("AsymmetricPowerLoss", dtype);
}

void validatePredictionsDType(DataType dtype) {
    ThorImplementation::RegressionLossDType::validatePredictionsDType("AsymmetricPowerLoss", dtype);
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("AsymmetricPowerLoss example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    ThorImplementation::RegressionLossDType::validateExampleWeightDType("AsymmetricPowerLoss", dtype);
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

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& predictionValueDimensions) {
    if (predictionValueDimensions.empty())
        throw invalid_argument("AsymmetricPowerLoss ragged prediction values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(predictionValueDimensions.size(), 1);
    dimensions.front() = predictionValueDimensions.front();
    return dimensions;
}

ThorImplementation::DynamicExpression makeAsymmetricPowerLossExpression(DataType lossDataType, float level, float exponent) {
    validateLevel(level);
    validateExponent(exponent);
    ThorImplementation::Expression loss = asymmetricPowerLossExpression(level, exponent).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedAsymmetricPowerLossExpression(
    DataType lossDataType, float level, float exponent, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validateLevel(level);
    validateExponent(exponent);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
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

ThorImplementation::DynamicExpression makeWeightedAsymmetricPowerGradientExpression(
    DataType predictionsDataType, float level, float exponent, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validatePredictionsDType(predictionsDataType);
    validateLevel(level);
    validateExponent(exponent);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
    ThorImplementation::Expression gradient =
        (asymmetricPowerGradientExpression(level, exponent) * exampleWeights).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void AsymmetricPowerLoss::buildSupportLayersAndAddToNetwork() {
    ThorImplementation::RegressionLossDType::validateLossDType("AsymmetricPowerLoss", lossDataType);
    validateLevel(level);
    validateExponent(exponent);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("AsymmetricPowerLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");

        RaggedCustomLoss::Builder rawBuilder;
        rawBuilder.network(*network)
            .predictions(raggedPredictionsTensor.value())
            .labels(raggedLabelsTensor.value())
            .predictionsName(kPredictionsName)
            .labelsName(kLabelsName)
            .lossName(kLossName)
            .gradientName(kGradientName)
            .lossDataType(lossDataType)
            .lossWeight(lossWeight.value_or(1.0f));

        if (exampleWeightsTensor.has_value()) {
            validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);
            if (exampleWeightsTensor->getDimensions() != vector<uint64_t>{1})
                throw invalid_argument("AsymmetricPowerLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
            TypeConverter weightConverter = TypeConverter::Builder()
                                                .network(*network)
                                                .featureInput(exampleWeightsTensor.value())
                                                .newDataType(DataType::FP32)
                                                .build();
            SegmentedBroadcast weightBroadcast = SegmentedBroadcast::Builder()
                                                     .network(*network)
                                                     .featureInput(weightConverter.getFeatureOutput().value())
                                                     .partitionInput(raggedPredictionsTensor.value())
                                                     .build();
            const vector<uint64_t> predictionValueDimensions = raggedPredictionsTensor->getValuesDimensions();
            rawBuilder.lossExpression(makeWeightedAsymmetricPowerLossExpression(lossDataType, level, exponent, predictionValueDimensions))
                .gradientExpression(makeWeightedAsymmetricPowerGradientExpression(predictionsTensor.getDataType(), level, exponent, predictionValueDimensions))
                .exampleWeights(weightBroadcast.getRaggedFeatureOutput())
                .exampleWeightsName(kExampleWeightsName);
        } else {
            rawBuilder.lossExpression(makeAsymmetricPowerLossExpression(lossDataType, level, exponent))
                .gradientExpression(makeAsymmetricPowerGradientExpression(predictionsTensor.getDataType(), level, exponent));
        }

        RaggedCustomLoss rawLoss = rawBuilder.build();
        raggedRawLossTensor = rawLoss.getRaggedRawLoss();
        lossShaperInput = raggedRawLossTensor->getValues();

        if (lossShape == LossShape::NONE) {
            lossTensor = lossShaperInput;
            Stub::Builder().network(*network).inputTensor(lossShaperInput).build();
        } else if (lossShape == LossShape::RAW) {
            lossTensor = lossShaperInput;
        } else if (lossShape == LossShape::PER_EXAMPLE) {
            RaggedLossShaper shaper = RaggedLossShaper::Builder()
                                          .network(*network)
                                          .lossInput(raggedRawLossTensor.value())
                                          .reportsPerExampleLoss()
                                          .build();
            lossTensor = shaper.getLossOutput();
        } else if (lossShape == LossShape::BATCH) {
            RaggedLossShaper shaper = RaggedLossShaper::Builder()
                                          .network(*network)
                                          .lossInput(raggedRawLossTensor.value())
                                          .reportsBatchLoss()
                                          .build();
            lossTensor = shaper.getLossOutput();
        } else {
            THOR_UNREACHABLE();
        }
        return;
    }

    validateExampleWeights(predictionsTensor, labelsTensor, exampleWeightsTensor);
    if (exampleWeightsTensor.has_value()) {
        MultiInputCustomLoss rawAsymmetricPowerLoss = MultiInputCustomLoss::Builder()
            .network(*network)
            .lossExpression(makeWeightedAsymmetricPowerLossExpression(lossDataType, level, exponent))
            .gradientExpression(makeWeightedAsymmetricPowerGradientExpression(predictionsTensor.getDataType(), level, exponent))
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
        CustomLoss rawAsymmetricPowerLoss = CustomLoss::Builder()
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

    finalizeLossReporting();
}

json AsymmetricPowerLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "asymmetric_power_loss";
    j["level"] = level;
    j["exponent"] = exponent;
    if (isRagged()) {
        j["loss_shape"] = lossShape;
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
    } else {
        j["loss_shape"] = lossShape;
    }
    return j;
}

void AsymmetricPowerLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in AsymmetricPowerLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "asymmetric_power_loss")
        throw runtime_error("Layer type mismatch in AsymmetricPowerLoss::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "AsymmetricPowerLoss");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "AsymmetricPowerLoss");
        AsymmetricPowerLoss::Builder builder;
        builder.network(*network).predictions(predictions).labels(labels).level(j.value("level", 0.5f)).exponent(j.value("exponent", 1.5f));
        if (j.contains("example_weights_tensor")) {
            const uint64_t weightsId = j.at("example_weights_tensor").at("id").get<uint64_t>();
            builder.exampleWeights(network->getApiTensorByOriginalId(weightsId));
        }
        builder.lossDataType(j.at("loss_data_type").get<DataType>());
        builder.lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
        switch (j.at("loss_shape").get<LossShape>()) {
            case LossShape::NONE: builder.reportsNoLoss(); break;
            case LossShape::BATCH: builder.reportsBatchLoss(); break;
            case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break;
            case LossShape::RAW: builder.reportsRawLoss(); break;
            case LossShape::PER_OUTPUT:
                throw runtime_error("Serialized ragged AsymmetricPowerLoss cannot use LossShape::PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    AsymmetricPowerLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.level = j.value("level", 0.5f);
    loss.exponent = j.value("exponent", 1.5f);
    validateLevel(loss.level);
    validateExponent(loss.exponent);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        loss.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("asymmetric_power_loss", &Thor::AsymmetricPowerLoss::deserialize);
    return true;
}();
}  // namespace
