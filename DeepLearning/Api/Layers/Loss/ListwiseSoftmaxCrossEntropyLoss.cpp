#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/ListwiseSoftmaxCrossEntropyLoss.h"

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
        throw runtime_error("Unsupported ListwiseSoftmaxCrossEntropyLoss label dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported ListwiseSoftmaxCrossEntropyLoss predictions dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeListwiseSoftmaxCrossEntropyLossExpression(DataType lossDataType, float temperature) {
    validatePredictionsDType(lossDataType);

    ThorImplementation::Expression scores = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(temperature);
    ThorImplementation::Expression logProbabilities = scaledScores.logSoftmax();
    ThorImplementation::Expression perDocumentLoss = -(labels * logProbabilities);
    ThorImplementation::Expression perListLoss = perDocumentLoss.reduce_sum({1}, {}, DataType::FP32).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, perListLoss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeListwiseSoftmaxCrossEntropyGradientExpression(DataType predictionsDataType, float temperature) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression scores = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(temperature);
    ThorImplementation::Expression probabilities = scaledScores.softmax();
    ThorImplementation::Expression labelMass = labels.reduce_sum({1}, {}, DataType::FP32);
    ThorImplementation::Expression gradient = (((probabilities * labelMass) - labels) /
                                               ThorImplementation::Expression(temperature) *
                                               ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
                                                  .withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void ListwiseSoftmaxCrossEntropyLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions()[0] > 1);
    THOR_THROW_IF_FALSE(predictionsTensor.getDimensions() == labelsTensor.getDimensions());
    THOR_THROW_IF_FALSE(temperature > 0.0f);

    CustomLoss rawListwiseSoftmaxCrossEntropyLoss = CustomLoss::Builder()
                                                       .network(*network)
                                                       .lossExpression(makeListwiseSoftmaxCrossEntropyLossExpression(lossDataType,
                                                                                                                     temperature))
                                                       .gradientExpression(makeListwiseSoftmaxCrossEntropyGradientExpression(
                                                           predictionsTensor.getDataType(),
                                                           temperature))
                                                       .predictions(predictionsTensor)
                                                       .labels(labelsTensor)
                                                       .predictionsName(kPredictionsName)
                                                       .labelsName(kLabelsName)
                                                       .lossName(kLossName)
                                                       .gradientName(kGradientName)
                                                       .lossDataType(lossDataType)
                                                       .reportsRawLoss()
                                                       .build();

    lossShaperInput = rawListwiseSoftmaxCrossEntropyLoss.getLoss();

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

json ListwiseSoftmaxCrossEntropyLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "listwise_softmax_cross_entropy_loss";
    j["loss_shape"] = lossShape;
    j["temperature"] = temperature;
    return j;
}

void ListwiseSoftmaxCrossEntropyLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in ListwiseSoftmaxCrossEntropyLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "listwise_softmax_cross_entropy_loss")
        throw runtime_error("Layer type mismatch in ListwiseSoftmaxCrossEntropyLoss::deserialize: " +
                            j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    ListwiseSoftmaxCrossEntropyLoss listwiseSoftmaxCrossEntropyLoss;
    listwiseSoftmaxCrossEntropyLoss.lossShape = j.at("loss_shape").get<LossShape>();
    listwiseSoftmaxCrossEntropyLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    listwiseSoftmaxCrossEntropyLoss.temperature = j.value("temperature", 1.0f);
    listwiseSoftmaxCrossEntropyLoss.predictionsTensor = predictions;
    listwiseSoftmaxCrossEntropyLoss.labelsTensor = labels;
    listwiseSoftmaxCrossEntropyLoss.network = network;
    listwiseSoftmaxCrossEntropyLoss.initialized = true;
    listwiseSoftmaxCrossEntropyLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("listwise_softmax_cross_entropy_loss", &Thor::ListwiseSoftmaxCrossEntropyLoss::deserialize);
    return true;
}();
}  // namespace
