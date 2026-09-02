#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/ExpectileLoss.h"

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

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& predictionValueDimensions) {
    if (predictionValueDimensions.empty())
        throw invalid_argument("ExpectileLoss ragged prediction values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(predictionValueDimensions.size(), 1);
    dimensions.front() = predictionValueDimensions.front();
    return dimensions;
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

ThorImplementation::DynamicExpression makeWeightedExpectileLossExpression(
    DataType lossDataType, float expectile, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
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

ThorImplementation::DynamicExpression makeWeightedExpectileGradientExpression(
    DataType predictionsDataType, float expectile, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions =
        ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels =
        ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
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
    THOR_THROW_IF_FALSE(expectile > 0.0f && expectile < 1.0f);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("ExpectileLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");

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
                throw invalid_argument("ExpectileLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
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
            rawBuilder.lossExpression(makeWeightedExpectileLossExpression(lossDataType, expectile, predictionValueDimensions))
                .gradientExpression(makeWeightedExpectileGradientExpression(predictionsTensor.getDataType(), expectile, predictionValueDimensions))
                .exampleWeights(weightBroadcast.getRaggedFeatureOutput())
                .exampleWeightsName(kExampleWeightsName);
        } else {
            rawBuilder.lossExpression(makeExpectileLossExpression(lossDataType, expectile))
                .gradientExpression(makeExpectileGradientExpression(predictionsTensor.getDataType(), expectile));
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
        MultiInputCustomLoss rawExpectileLoss = MultiInputCustomLoss::Builder()
            .network(*network)
            .lossExpression(makeWeightedExpectileLossExpression(lossDataType, expectile))
            .gradientExpression(makeWeightedExpectileGradientExpression(predictionsTensor.getDataType(), expectile))
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
            .gradientExpression(makeExpectileGradientExpression(predictionsTensor.getDataType(), expectile))
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

    finalizeLossReporting();
}

json ExpectileLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "expectile_loss";
    j["expectile"] = expectile;
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

void ExpectileLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in ExpectileLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "expectile_loss")
        throw runtime_error("Layer type mismatch in ExpectileLoss::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "ExpectileLoss");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "ExpectileLoss");
        ExpectileLoss::Builder builder;
        builder.network(*network).predictions(predictions).labels(labels).expectile(j.value("expectile", 0.5f));
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
                throw runtime_error("Serialized ragged ExpectileLoss cannot use LossShape::PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    ExpectileLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.expectile = j.value("expectile", 0.5f);
    THOR_THROW_IF_FALSE(loss.expectile > 0.0f && loss.expectile < 1.0f);
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
    Thor::Loss::register_layer("expectile_loss", &Thor::ExpectileLoss::deserialize);
    return true;
}();
}  // namespace
