#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.h"

#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";
constexpr float kEpsilon = 0.0001f;
constexpr float kMaxMagnitude = 1000.0f;

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
            throw runtime_error("Unsupported MAPE label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw runtime_error("Unsupported MAPE predictions dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
}

ThorImplementation::Expression signOf(const ThorImplementation::Expression& value) {
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression one(1.0f);
    ThorImplementation::Expression negativeOne(-1.0f);
    return ThorImplementation::Expression::where(
        value > zero, one, ThorImplementation::Expression::where(value < zero, negativeOne, zero));
}

ThorImplementation::Expression stabilizedMapeLabels(const ThorImplementation::Expression& labels) {
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression positiveEpsilon(kEpsilon);
    ThorImplementation::Expression negativeEpsilon(-kEpsilon);
    return ThorImplementation::Expression::where(
        labels < zero,
        ThorImplementation::Expression::where(labels > negativeEpsilon, negativeEpsilon, labels),
        ThorImplementation::Expression::where(labels > positiveEpsilon, labels, positiveEpsilon));
}

ThorImplementation::DynamicExpression makeMapeLossExpression(DataType lossDataType) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression originalDiff = predictions - labels;
    ThorImplementation::Expression effectiveLabels = stabilizedMapeLabels(labels);
    ThorImplementation::Expression effectiveDiff = predictions - effectiveLabels;
    ThorImplementation::Expression percentage = (effectiveDiff / effectiveLabels).abs() * ThorImplementation::Expression(100.0f);
    ThorImplementation::Expression capped = percentage.min(ThorImplementation::Expression(kMaxMagnitude));
    ThorImplementation::Expression loss =
        ThorImplementation::Expression::where(originalDiff == zero, zero, capped).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeMapeGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0f);
    ThorImplementation::Expression originalDiff = predictions - labels;
    ThorImplementation::Expression effectiveLabels = stabilizedMapeLabels(labels);
    ThorImplementation::Expression effectiveDiff = predictions - effectiveLabels;
    ThorImplementation::Expression gradient = signOf(effectiveDiff) * ThorImplementation::Expression(100.0f) / effectiveLabels.abs();
    gradient = gradient.clamp(-kMaxMagnitude, kMaxMagnitude);
    gradient = ThorImplementation::Expression::where(originalDiff == zero, zero, gradient);
    gradient = (gradient * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
                   .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void MAPE::buildSupportLayersAndAddToNetwork() {
    if (isRagged()) {
        validatePredictionsDType(predictionsTensor.getDataType());
        validateLabelsDType(labelsTensor.getDataType());
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("MAPE LossShape::PER_OUTPUT is undefined for ragged sequences.");

        RaggedCustomLoss rawMape = RaggedCustomLoss::Builder()
                                        .network(*network)
                                        .lossExpression(makeMapeLossExpression(lossDataType))
                                        .gradientExpression(makeMapeGradientExpression(predictionsTensor.getDataType()))
                                        .predictions(raggedPredictionsTensor.value())
                                        .labels(raggedLabelsTensor.value())
                                        .predictionsName(kPredictionsName)
                                        .labelsName(kLabelsName)
                                        .lossName(kLossName)
                                        .gradientName(kGradientName)
                                        .lossDataType(lossDataType)
                                        .lossWeight(lossWeight.value_or(1.0f))
                                        .build();
        raggedRawLossTensor = rawMape.getRaggedRawLoss();
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

    MAPE meanAbsolutePercentageError = MAPE::Builder()
                                           .network(*network)
                                           .predictions(predictionsTensor)
                                           .labels(labelsTensor)
                                           .reportsRawLoss()
                                           .lossDataType(lossDataType)
                                           .lossWeight(lossWeight.value_or(1.0f))
                                           .build();

    lossShaperInput = meanAbsolutePercentageError.getLoss();
    finalizeLossReporting();
}

json MAPE::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "mape";
    if (isRagged()) {
        j["loss_shape"] = lossShape;
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
    }
    return j;
}

void MAPE::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in MAPE::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "mape")
        throw runtime_error("Layer type mismatch in MAPE::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "MAPE");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "MAPE");
        MAPE::Builder builder;
        builder.network(*network).predictions(predictions).labels(labels);
        builder.lossDataType(j.at("loss_data_type").get<DataType>());
        builder.lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
        switch (j.at("loss_shape").get<LossShape>()) {
            case LossShape::NONE: builder.reportsNoLoss(); break;
            case LossShape::BATCH: builder.reportsBatchLoss(); break;
            case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break;
            case LossShape::RAW: builder.reportsRawLoss(); break;
            case LossShape::PER_OUTPUT:
                throw runtime_error("Serialized ragged MAPE cannot use LossShape::PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    MAPE meanAbsolutePercentageError;
    meanAbsolutePercentageError.lossShape = j.at("loss_shape").get<LossShape>();
    meanAbsolutePercentageError.lossDataType = j.at("loss_data_type").get<DataType>();
    meanAbsolutePercentageError.lossWeight = ThorImplementation::lossWeightFromJson(j);

    uint64_t originalTensorId;
    originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    meanAbsolutePercentageError.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    meanAbsolutePercentageError.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);

    meanAbsolutePercentageError.lossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);
    meanAbsolutePercentageError.initialized = true;
    meanAbsolutePercentageError.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("mape", &Thor::MAPE::deserialize);
    return true;
}();
}  // namespace
