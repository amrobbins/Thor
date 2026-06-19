#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"

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
            throw runtime_error("Unsupported QuantileLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported QuantileLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
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

}  // namespace

void QuantileLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(quantile > 0.0f && quantile < 1.0f);

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

json QuantileLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["quantile"] = quantile;
    return j;
}

void QuantileLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in QuantileLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "quantile_loss")
        throw runtime_error("Layer type mismatch in QuantileLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    QuantileLoss quantileLoss;
    quantileLoss.lossShape = j.at("loss_shape").get<LossShape>();
    quantileLoss.lossDataType = j.at("loss_data_type").get<DataType>();

    quantileLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    quantileLoss.quantile = j.value("quantile", 0.5f);
    quantileLoss.predictionsTensor = predictions;
    quantileLoss.labelsTensor = labels;
    quantileLoss.network = network;
    quantileLoss.initialized = true;
    quantileLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("quantile_loss", &Thor::QuantileLoss::deserialize);
    return true;
}();
}  // namespace
