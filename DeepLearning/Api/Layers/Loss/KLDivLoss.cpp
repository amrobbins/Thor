#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/KLDivLoss.h"

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
constexpr double kTargetLogEpsilon = 1.0e-36;

void validateLabelsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported KLDivLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported KLDivLoss predictions dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeKLDivLossExpression(DataType lossDataType) {
    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression logProbabilities = logits.logSoftmax();
    ThorImplementation::Expression safeLabels = labels.max(ThorImplementation::Expression(kTargetLogEpsilon));
    ThorImplementation::Expression loss = (labels * (safeLabels.ln() - logProbabilities)).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeKLDivGradientExpression(DataType predictionsDataType) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression gradient =
        ((logits.softmax() - labels) * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void KLDivLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());

    CustomLoss rawKLDivLoss = CustomLoss::Builder()
                                  .network(*network)
                                  .lossExpression(makeKLDivLossExpression(lossDataType))
                                  .gradientExpression(makeKLDivGradientExpression(predictionsTensor.getDataType()))
                                  .predictions(predictionsTensor)
                                  .labels(labelsTensor)
                                  .predictionsName(kPredictionsName)
                                  .labelsName(kLabelsName)
                                  .lossName(kLossName)
                                  .gradientName(kGradientName)
                                  .lossDataType(lossDataType)
                                  .reportsRawLoss()
                                  .build();

    lossShaperInput = rawKLDivLoss.getLoss();

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

json KLDivLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "kl_div_loss";
    j["loss_shape"] = lossShape;
    return j;
}

void KLDivLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in KLDivLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "kl_div_loss")
        throw runtime_error("Layer type mismatch in KLDivLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    KLDivLoss klDivLoss;
    klDivLoss.lossShape = j.at("loss_shape").get<LossShape>();
    klDivLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    klDivLoss.predictionsTensor = predictions;
    klDivLoss.labelsTensor = labels;
    klDivLoss.network = network;
    klDivLoss.initialized = true;
    klDivLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("kl_div_loss", &Thor::KLDivLoss::deserialize);
    return true;
}();
}  // namespace
