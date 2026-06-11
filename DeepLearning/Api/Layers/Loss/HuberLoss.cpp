#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/HuberLoss.h"

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
    THOR_THROW_IF_FALSE(delta > 0.0f);

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

json HuberLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["delta"] = delta;
    return j;
}

void HuberLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in HuberLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "huber_loss")
        throw runtime_error("Layer type mismatch in HuberLoss::deserialize: " + j.at("layer_type").get<std::string>());

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
