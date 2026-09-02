#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"

#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"

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

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& predictionValueDimensions) {
    if (predictionValueDimensions.empty())
        throw invalid_argument("QuantileLoss ragged prediction values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(predictionValueDimensions.size(), 1);
    dimensions.front() = predictionValueDimensions.front();
    return dimensions;
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

ThorImplementation::DynamicExpression makeWeightedQuantileLossExpression(
    DataType lossDataType, float quantile, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
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

ThorImplementation::DynamicExpression makeWeightedQuantileGradientExpression(
    DataType predictionsDataType, float quantile, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
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
    THOR_THROW_IF_FALSE(quantile > 0.0f && quantile < 1.0f);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("QuantileLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");

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
                throw invalid_argument("QuantileLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
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
            rawBuilder.lossExpression(makeWeightedQuantileLossExpression(lossDataType, quantile, predictionValueDimensions))
                .gradientExpression(makeWeightedQuantileGradientExpression(predictionsTensor.getDataType(), quantile, predictionValueDimensions))
                .exampleWeights(weightBroadcast.getRaggedFeatureOutput())
                .exampleWeightsName(kExampleWeightsName);
        } else {
            rawBuilder.lossExpression(makeQuantileLossExpression(lossDataType, quantile))
                .gradientExpression(makeQuantileGradientExpression(predictionsTensor.getDataType(), quantile));
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
        MultiInputCustomLoss rawQuantileLoss = MultiInputCustomLoss::Builder()
            .network(*network)
            .lossExpression(makeWeightedQuantileLossExpression(lossDataType, quantile))
            .gradientExpression(makeWeightedQuantileGradientExpression(predictionsTensor.getDataType(), quantile))
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

    finalizeLossReporting();
}

json QuantileLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "quantile_loss";
    j["quantile"] = quantile;
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

void QuantileLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in QuantileLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "quantile_loss")
        throw runtime_error("Layer type mismatch in QuantileLoss::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "QuantileLoss");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "QuantileLoss");
        QuantileLoss::Builder builder;
        builder.network(*network).predictions(predictions).labels(labels).quantile(j.value("quantile", 0.5f));
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
                throw runtime_error("Serialized ragged QuantileLoss cannot use LossShape::PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    QuantileLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.quantile = j.value("quantile", 0.5f);
    THOR_THROW_IF_FALSE(loss.quantile > 0.0f && loss.quantile < 1.0f);
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
    Thor::Loss::register_layer("quantile_loss", &Thor::QuantileLoss::deserialize);
    return true;
}();
}  // namespace
