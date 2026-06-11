#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/ContrastiveLoss.h"

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
            throw runtime_error("Unsupported ContrastiveLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported ContrastiveLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::Expression positivePairMask(const ThorImplementation::Expression& labels) {
    return labels > ThorImplementation::Expression(0.5);
}

ThorImplementation::Expression hingeDistance(const ThorImplementation::Expression& distances, float margin) {
    ThorImplementation::Expression zero(0.0);
    return (ThorImplementation::Expression(margin) - distances).max(zero);
}

ThorImplementation::DynamicExpression makeContrastiveLossExpression(DataType lossDataType, float margin) {
    validatePredictionsDType(lossDataType);

    ThorImplementation::Expression distances = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);

    ThorImplementation::Expression positiveLoss = distances * distances;
    ThorImplementation::Expression hinge = hingeDistance(distances, margin);
    ThorImplementation::Expression negativeLoss = hinge * hinge;
    ThorImplementation::Expression loss = ThorImplementation::Expression::where(positivePairMask(labels), positiveLoss, negativeLoss)
                                          .withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeContrastiveGradientExpression(DataType predictionsDataType, float margin) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression distances = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression two(2.0);
    ThorImplementation::Expression marginExpr(margin);
    ThorImplementation::Expression hinge = (marginExpr - distances).max(zero);

    ThorImplementation::Expression positiveGradient = two * distances;
    ThorImplementation::Expression negativeGradient = ThorImplementation::Expression::where(hinge > zero, -two * hinge, zero);
    ThorImplementation::Expression gradient =
        (ThorImplementation::Expression::where(positivePairMask(labels), positiveGradient, negativeGradient) *
         ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void ContrastiveLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(margin > 0.0f);

    CustomLoss rawContrastiveLoss = CustomLoss::Builder()
                                        .network(*network)
                                        .lossExpression(makeContrastiveLossExpression(lossDataType, margin))
                                        .gradientExpression(makeContrastiveGradientExpression(predictionsTensor.getDataType(), margin))
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

    lossShaperInput = rawContrastiveLoss.getLoss();

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

json ContrastiveLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["margin"] = margin;
    return j;
}

void ContrastiveLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in ContrastiveLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "contrastive_loss")
        throw runtime_error("Layer type mismatch in ContrastiveLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    ContrastiveLoss contrastiveLoss;
    contrastiveLoss.lossShape = j.at("loss_shape").get<LossShape>();
    contrastiveLoss.lossDataType = j.at("loss_data_type").get<DataType>();

    contrastiveLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    contrastiveLoss.margin = j.value("margin", 1.0f);
    contrastiveLoss.predictionsTensor = predictions;
    contrastiveLoss.labelsTensor = labels;
    contrastiveLoss.network = network;
    contrastiveLoss.initialized = true;
    contrastiveLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("contrastive_loss", &Thor::ContrastiveLoss::deserialize);
    return true;
}();
}  // namespace
