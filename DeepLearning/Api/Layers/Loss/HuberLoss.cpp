#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/HuberLoss.h"

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
            throw runtime_error("Unsupported HuberLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported HuberLoss predictions dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::Expression signOf(const ThorImplementation::Expression& diff) {
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression positive(1.0);
    ThorImplementation::Expression negative(-1.0);
    return ThorImplementation::Expression::where(diff > zero,
                                                 positive,
                                                 ThorImplementation::Expression::where(diff < zero, negative, zero));
}

ThorImplementation::DynamicExpression makeHuberLossExpression(DataType lossDataType, float delta) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression deltaExpr(delta);
    ThorImplementation::Expression half(0.5);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression absDiff = diff.abs();
    ThorImplementation::Expression quadratic = half * diff * diff;
    ThorImplementation::Expression linear = deltaExpr * (absDiff - (half * deltaExpr));
    ThorImplementation::Expression loss = ThorImplementation::Expression::where(absDiff <= deltaExpr, quadratic, linear).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeHuberGradientExpression(DataType predictionsDataType, float delta) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression deltaExpr(delta);
    ThorImplementation::Expression diff = predictions - labels;
    ThorImplementation::Expression absDiff = diff.abs();
    ThorImplementation::Expression quadraticGrad = diff;
    ThorImplementation::Expression linearGrad = deltaExpr * signOf(diff);
    ThorImplementation::Expression gradient =
        (ThorImplementation::Expression::where(absDiff <= deltaExpr, quadraticGrad, linearGrad) *
         ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void HuberLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
    THOR_THROW_IF_FALSE(delta > 0.0f);

    if (isRagged()) {
        if (lossShape == LossShape::PER_OUTPUT)
            throw invalid_argument("HuberLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");

        RaggedCustomLoss rawHuberLoss = RaggedCustomLoss::Builder()
                                            .network(*network)
                                            .lossExpression(makeHuberLossExpression(lossDataType, delta))
                                            .gradientExpression(makeHuberGradientExpression(predictionsTensor.getDataType(), delta))
                                            .predictions(raggedPredictionsTensor.value())
                                            .labels(raggedLabelsTensor.value())
                                            .predictionsName(kPredictionsName)
                                            .labelsName(kLabelsName)
                                            .lossName(kLossName)
                                            .gradientName(kGradientName)
                                            .lossDataType(lossDataType)
                                            .lossWeight(lossWeight.value_or(1.0f))
                                            .build();
        raggedRawLossTensor = rawHuberLoss.getRaggedRawLoss();
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

    CustomLoss rawHuberLoss = CustomLoss::Builder()
                                  .network(*network)
                                  .lossExpression(makeHuberLossExpression(lossDataType, delta))
                                  .gradientExpression(makeHuberGradientExpression(predictionsTensor.getDataType(), delta))
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

    lossShaperInput = rawHuberLoss.getLoss();
    finalizeLossReporting();
}

json HuberLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["delta"] = delta;
    if (isRagged()) {
        j["ragged_predictions"] = raggedPredictionsTensor->architectureJson();
        j["ragged_labels"] = raggedLabelsTensor->architectureJson();
        if (raggedRawLossTensor.has_value()) j["ragged_raw_loss"] = raggedRawLossTensor->architectureJson();
    }
    return j;
}

void HuberLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in HuberLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "huber_loss")
        throw runtime_error("Layer type mismatch in HuberLoss::deserialize: " + j.at("layer_type").get<std::string>());

    if (j.contains("ragged_predictions")) {
        RaggedTensor predictions = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_predictions"), network, "HuberLoss");
        RaggedTensor labels = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_labels"), network, "HuberLoss");
        HuberLoss::Builder builder;
        builder.network(*network).predictions(predictions).labels(labels).delta(j.value("delta", 1.0f));
        builder.lossDataType(j.at("loss_data_type").get<DataType>());
        builder.lossWeight(ThorImplementation::lossWeightFromJson(j).value_or(1.0f));
        switch (j.at("loss_shape").get<LossShape>()) {
            case LossShape::NONE: builder.reportsNoLoss(); break;
            case LossShape::BATCH: builder.reportsBatchLoss(); break;
            case LossShape::PER_EXAMPLE: builder.reportsPerExampleLoss(); break;
            case LossShape::RAW: builder.reportsRawLoss(); break;
            case LossShape::PER_OUTPUT:
                throw runtime_error("Serialized ragged HuberLoss cannot use LossShape::PER_OUTPUT.");
        }
        (void)builder.build();
        return;
    }

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    HuberLoss huberLoss;
    huberLoss.lossShape = j.at("loss_shape").get<LossShape>();
    huberLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    huberLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    huberLoss.delta = j.value("delta", 1.0f);
    huberLoss.predictionsTensor = predictions;
    huberLoss.labelsTensor = labels;
    huberLoss.network = network;
    huberLoss.initialized = true;
    huberLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("huber_loss", &Thor::HuberLoss::deserialize);
    return true;
}();
}  // namespace
