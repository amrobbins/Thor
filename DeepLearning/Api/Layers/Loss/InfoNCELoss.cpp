#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/InfoNCELoss.h"

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
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported InfoNCELoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported InfoNCELoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeInfoNCELossExpression(DataType lossDataType, float temperature) {
    validatePredictionsDType(lossDataType);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scaledLogits = logits / ThorImplementation::Expression(temperature);
    ThorImplementation::Expression logProbabilities = scaledLogits.logSoftmax();
    ThorImplementation::Expression loss = (-(labels * logProbabilities)).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeInfoNCEGradientExpression(DataType predictionsDataType, float temperature) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scaledLogits = logits / ThorImplementation::Expression(temperature);
    ThorImplementation::Expression probabilities = scaledLogits.softmax();
    ThorImplementation::Expression labelMass = labels.reduce_sum({1}, {});
    ThorImplementation::Expression gradient = (((probabilities * labelMass) - labels) /
                                               ThorImplementation::Expression(temperature) *
                                               ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
                                                  .withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void InfoNCELoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(temperature > 0.0f);

    CustomLoss rawInfoNCELoss = CustomLoss::Builder()
                                    .network(*network)
                                    .lossExpression(makeInfoNCELossExpression(lossDataType, temperature))
                                    .gradientExpression(makeInfoNCEGradientExpression(predictionsTensor.getDataType(), temperature))
                                    .predictions(predictionsTensor)
                                    .labels(labelsTensor)
                                    .predictionsName(kPredictionsName)
                                    .labelsName(kLabelsName)
                                    .lossName(kLossName)
                                    .gradientName(kGradientName)
                                    .lossDataType(lossDataType)
                                    .reportsRawLoss()
                                    .build();

    lossShaperInput = rawInfoNCELoss.getLoss();

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

json InfoNCELoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "info_nce_loss";
    j["loss_shape"] = lossShape;
    j["temperature"] = temperature;
    return j;
}

void InfoNCELoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in InfoNCELoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "info_nce_loss")
        throw runtime_error("Layer type mismatch in InfoNCELoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    InfoNCELoss infoNCELoss;
    infoNCELoss.lossShape = j.at("loss_shape").get<LossShape>();
    infoNCELoss.lossDataType = j.at("loss_data_type").get<DataType>();
    infoNCELoss.temperature = j.value("temperature", 1.0f);
    infoNCELoss.predictionsTensor = predictions;
    infoNCELoss.labelsTensor = labels;
    infoNCELoss.network = network;
    infoNCELoss.initialized = true;
    infoNCELoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("info_nce_loss", &Thor::InfoNCELoss::deserialize);
    return true;
}();
}  // namespace
