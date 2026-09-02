#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"

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

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& predictionValueDimensions) {
    if (predictionValueDimensions.empty())
        throw invalid_argument("MeanPowerError ragged prediction values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(predictionValueDimensions.size(), 1);
    dimensions.front() = predictionValueDimensions.front();
    return dimensions;
}

ThorImplementation::DynamicExpression makeMeanPowerLossExpression(float exponent, DataType lossDataType) {
    validateExponent(exponent);
    ThorImplementation::Expression loss = meanPowerLossExpression(exponent).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMeanPowerLossExpression(
    float exponent, DataType lossDataType, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validateExponent(exponent);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
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

ThorImplementation::DynamicExpression makeWeightedMeanPowerGradientExpression(
    float exponent, DataType predictionsDataType, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validateExponent(exponent);
    validatePredictionsDType(predictionsDataType);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
    ThorImplementation::Expression gradient =
        (meanPowerGradientExpression(exponent) * exampleWeights).withOutputDType(predictionsDataType);
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

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("MeanPowerError LossShape::PER_OUTPUT is undefined for ragged sequences.");

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
                throw invalid_argument("MeanPowerError ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
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
            rawBuilder.lossExpression(makeWeightedMeanPowerLossExpression(exponent, lossDataType, predictionValueDimensions))
                .gradientExpression(makeWeightedMeanPowerGradientExpression(exponent, predictionsTensor.getDataType(), predictionValueDimensions))
                .exampleWeights(weightBroadcast.getRaggedFeatureOutput())
                .exampleWeightsName(kExampleWeightsName);
        } else {
            rawBuilder.lossExpression(makeMeanPowerLossExpression(exponent, lossDataType))
                .gradientExpression(makeMeanPowerGradientExpression(exponent, predictionsTensor.getDataType()));
        }

        RaggedCustomLoss rawMeanPowerError = rawBuilder.build();
        raggedRawLossTensor = rawMeanPowerError.getRaggedRawLoss();
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

    finalizeLossReporting();
}

json MeanPowerError::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "mean_power_error";
    j["exponent"] = exponent;
    if (isRagged()) {
        // Unlike the graph-visible raw support layer, the MeanPowerError facade remembers
        // the user-requested reporting shape when serialized directly.
        j["loss_shape"] = lossShape;
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
    }
    return j;
}

void MeanPowerError::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MeanPowerError::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mean_power_error")
        throw runtime_error("Layer type mismatch in MeanPowerError::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "MeanPowerError");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "MeanPowerError");
        MeanPowerError::Builder builder;
        builder.network(*network).predictions(predictions).labels(labels).exponent(j.value("exponent", 1.5f));
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
                throw runtime_error("Serialized ragged MeanPowerError cannot use LossShape::PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

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
