#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/ListNetLoss.h"

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
        throw runtime_error("Unsupported ListNetLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported ListNetLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeListNetLossExpression(DataType lossDataType,
                                                                float scoreTemperature,
                                                                float labelTemperature) {
    validatePredictionsDType(lossDataType);

    ThorImplementation::Expression scores = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(scoreTemperature);
    ThorImplementation::Expression scaledLabels = labels / ThorImplementation::Expression(labelTemperature);
    ThorImplementation::Expression targetProbabilities = scaledLabels.softmax();
    ThorImplementation::Expression logProbabilities = scaledScores.logSoftmax();
    ThorImplementation::Expression perDocumentLoss = -(targetProbabilities * logProbabilities);
    ThorImplementation::Expression perListLoss = perDocumentLoss.reduce_sum({1}, {}, DataType::FP32).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, perListLoss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeListNetGradientExpression(DataType predictionsDataType,
                                                                    float scoreTemperature,
                                                                    float labelTemperature) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression scores = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(scoreTemperature);
    ThorImplementation::Expression scaledLabels = labels / ThorImplementation::Expression(labelTemperature);
    ThorImplementation::Expression scoreProbabilities = scaledScores.softmax();
    ThorImplementation::Expression targetProbabilities = scaledLabels.softmax();
    ThorImplementation::Expression gradient = ((scoreProbabilities - targetProbabilities) /
                                               ThorImplementation::Expression(scoreTemperature) *
                                               ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
                                                  .withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void ListNetLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions()[0] > 1);
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == labelsTensor.getDimensions());
    THOR_THROW_IF_FALSE(scoreTemperature > 0.0f);
    THOR_THROW_IF_FALSE(labelTemperature > 0.0f);

    CustomLoss rawListNetLoss = CustomLoss::Builder()
                                    .network(*network)
                                    .lossExpression(makeListNetLossExpression(lossDataType, scoreTemperature, labelTemperature))
                                    .gradientExpression(makeListNetGradientExpression(predictionsTensor.getDataType(),
                                                                                     scoreTemperature,
                                                                                     labelTemperature))
                                    .predictions(predictionsTensor)
                                    .labels(labelsTensor)
                                    .predictionsName(kPredictionsName)
                                    .labelsName(kLabelsName)
                                    .lossName(kLossName)
                                    .gradientName(kGradientName)
                                    .lossDataType(lossDataType)
                                    .reportsRawLoss()
                                    .build();

    lossShaperInput = rawListNetLoss.getLoss();

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

json ListNetLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "list_net_loss";
    j["loss_shape"] = lossShape;
    j["score_temperature"] = scoreTemperature;
    j["label_temperature"] = labelTemperature;
    return j;
}

void ListNetLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in ListNetLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "list_net_loss")
        throw runtime_error("Layer type mismatch in ListNetLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    ListNetLoss listNetLoss;
    listNetLoss.lossShape = j.at("loss_shape").get<LossShape>();
    listNetLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    listNetLoss.scoreTemperature = j.value("score_temperature", 1.0f);
    listNetLoss.labelTemperature = j.value("label_temperature", 1.0f);
    listNetLoss.predictionsTensor = predictions;
    listNetLoss.labelsTensor = labels;
    listNetLoss.network = network;
    listNetLoss.initialized = true;
    listNetLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("list_net_loss", &Thor::ListNetLoss::deserialize);
    return true;
}();
}  // namespace
