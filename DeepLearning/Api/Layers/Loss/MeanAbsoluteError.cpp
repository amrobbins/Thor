#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"

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
    ThorImplementation::RegressionLossDType::validateLabelsDType("MAE", dtype);
}

void validatePredictionsDType(DataType dtype) {
    ThorImplementation::RegressionLossDType::validatePredictionsDType("MAE", dtype);
}

void validateExampleWeights(Tensor predictions, Tensor labels, std::optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == predictions || exampleWeights.value() == labels)
        throw runtime_error("MAE example_weights tensor must be distinct from predictions and labels.");
    const DataType dtype = exampleWeights.value().getDataType();
    ThorImplementation::RegressionLossDType::validateExampleWeightDType("MAE", dtype);
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        throw runtime_error("MAE example_weights dimensions must be [1] for per-example weights or match predictions dimensions.");
    }
}

ThorImplementation::DynamicExpression makeMAELossExpression(DataType lossDataType) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = (predictions - labels).abs().withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

vector<uint64_t> raggedPackedScalarWeightBroadcastDimensions(const vector<uint64_t>& predictionValueDimensions) {
    if (predictionValueDimensions.empty())
        throw invalid_argument("MAE ragged prediction values must have a packed-capacity dimension.");
    vector<uint64_t> dimensions(predictionValueDimensions.size(), 1);
    dimensions.front() = predictionValueDimensions.front();
    return dimensions;
}

ThorImplementation::DynamicExpression makeWeightedMAELossExpression(
    DataType lossDataType, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression exampleWeights =
        ThorImplementation::Expression::input(kExampleWeightsName, DataType::FP32, DataType::FP32);
    if (raggedPredictionValueDimensions.has_value()) {
        // SegmentedBroadcast materializes one scalar per packed token as [N,1].
        // Reshape that scalar carrier to [N,1,...,1] so ordinary expression
        // broadcasting scales every trailing loss element for that token. This
        // also handles scalar ragged values, whose packed shape is simply [N].
        exampleWeights = exampleWeights.reshape(
            raggedPackedScalarWeightBroadcastDimensions(raggedPredictionValueDimensions.value()));
    }
    ThorImplementation::Expression loss = ((predictions - labels).abs() * exampleWeights).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeMAEGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression positive(1.0);
    ThorImplementation::Expression negative(-1.0);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression sign =
        ThorImplementation::Expression::where(diff > zero, positive, ThorImplementation::Expression::where(diff < zero, negative, zero));
    ThorImplementation::Expression gradient =
        (sign * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor())).withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWeightedMAEGradientExpression(
    DataType predictionsDataType, optional<vector<uint64_t>> raggedPredictionValueDimensions = nullopt) {
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
    ThorImplementation::Expression positive(1.0);
    ThorImplementation::Expression negative(-1.0);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression sign =
        ThorImplementation::Expression::where(diff > zero, positive, ThorImplementation::Expression::where(diff < zero, negative, zero));
    ThorImplementation::Expression gradient =
        (sign * exampleWeights * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void MAE::buildSupportLayersAndAddToNetwork() {
    ThorImplementation::RegressionLossDType::validateLossDType("MAE", lossDataType);
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("MAE LossShape::PER_OUTPUT is undefined for ragged sequences.");

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
                throw invalid_argument("MAE ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
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
            rawBuilder.lossExpression(makeWeightedMAELossExpression(lossDataType, predictionValueDimensions))
                .gradientExpression(makeWeightedMAEGradientExpression(predictionsTensor.getDataType(), predictionValueDimensions))
                .exampleWeights(weightBroadcast.getRaggedFeatureOutput())
                .exampleWeightsName(kExampleWeightsName);
        } else {
            rawBuilder.lossExpression(makeMAELossExpression(lossDataType))
                .gradientExpression(makeMAEGradientExpression(predictionsTensor.getDataType()));
        }

        RaggedCustomLoss rawMAE = rawBuilder.build();
        raggedRawLossTensor = rawMAE.getRaggedRawLoss();
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
        MultiInputCustomLoss rawMAE = MultiInputCustomLoss::Builder()
                                          .network(*network)
                                          .lossExpression(makeWeightedMAELossExpression(lossDataType))
                                          .gradientExpression(makeWeightedMAEGradientExpression(predictionsTensor.getDataType()))
                                          .input(kPredictionsName, predictionsTensor, std::string(kGradientName))
                                          .auxiliaryInput(kLabelsName, labelsTensor)
                                          .auxiliaryInput(kExampleWeightsName, exampleWeightsTensor.value())
                                          .lossName(kLossName)
                                          .lossDataType(lossDataType)
                                          .lossWeight(lossWeight.value_or(1.0f))
                                          .reportsRawLoss()
                                          .build();
        lossShaperInput = rawMAE.getLoss();
    } else {
        CustomLoss rawMAE = CustomLoss::Builder()
                                .network(*network)
                                .lossExpression(makeMAELossExpression(lossDataType))
                                .gradientExpression(makeMAEGradientExpression(predictionsTensor.getDataType()))
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
        lossShaperInput = rawMAE.getLoss();
    }

    finalizeLossReporting();
}

json MAE::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "mae";
    if (isRagged()) {
        // Unlike the graph-visible raw support layer, the MAE facade remembers
        // the user-requested reporting shape when serialized directly.
        j["loss_shape"] = lossShape;
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
    }
    return j;
}

void MAE::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MAE::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mae")
        throw runtime_error("Layer type mismatch in MAE::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "MAE");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "MAE");
        MAE::Builder builder;
        builder.network(*network).predictions(predictions).labels(labels);
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
                throw runtime_error("Serialized ragged MAE cannot use LossShape::PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    MAE meanAbsoluteError;
    meanAbsoluteError.lossShape = j.at("loss_shape").get<LossShape>();
    meanAbsoluteError.lossDataType = j.at("loss_data_type").get<DataType>();

    meanAbsoluteError.lossWeight = ThorImplementation::lossWeightFromJson(j);
    meanAbsoluteError.predictionsTensor = predictions;
    meanAbsoluteError.labelsTensor = labels;
    if (j.contains("example_weights_tensor")) {
        originalTensorId = j["example_weights_tensor"].at("id").get<uint64_t>();
        meanAbsoluteError.exampleWeightsTensor = network->getApiTensorByOriginalId(originalTensorId);
    }
    meanAbsoluteError.network = network;
    meanAbsoluteError.initialized = true;
    meanAbsoluteError.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mae", &Thor::MAE::deserialize);
    return true;
}();
}  // namespace
