#include "DeepLearning/Api/Layers/Loss/ListwiseSoftmaxCrossEntropyLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListwiseLossCommon.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

namespace Common = ListwiseLossCommon;

ThorImplementation::DynamicExpression makeListwiseSoftmaxCrossEntropyLossExpression(DataType lossDataType,
                                                                                    float temperature,
                                                                                    bool useMask) {
    Common::validateLossDataType("ListwiseSoftmaxCrossEntropyLoss", lossDataType);
    Common::validatePositiveTemperature("ListwiseSoftmaxCrossEntropyLoss", "temperature", temperature);

    ThorImplementation::Expression scores = Common::predictionsInput();
    ThorImplementation::Expression labels = Common::labelsInput();
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(temperature);
    ThorImplementation::Expression effectiveLabels = labels;
    ThorImplementation::Expression validMask(1.0);
    if (useMask) {
        validMask = Common::validDocumentMask();
        scaledScores = Common::maskPaddedDocuments(scaledScores, validMask);
        effectiveLabels = Common::zeroPaddedDocuments(labels, validMask);
    }

    ThorImplementation::Expression logProbabilities = scaledScores.logSoftmax();
    ThorImplementation::Expression perDocumentLoss = -(effectiveLabels * logProbabilities);
    if (useMask)
        perDocumentLoss = Common::zeroPaddedDocuments(perDocumentLoss, validMask);
    ThorImplementation::Expression perListLoss = perDocumentLoss.reduce_sum({1}, {}, DataType::FP32).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{Common::kLossName, perListLoss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeListwiseSoftmaxCrossEntropyGradientExpression(DataType predictionsDataType,
                                                                                        float temperature,
                                                                                        bool useMask) {
    Common::validatePredictionsDType("ListwiseSoftmaxCrossEntropyLoss", predictionsDataType);
    Common::validatePositiveTemperature("ListwiseSoftmaxCrossEntropyLoss", "temperature", temperature);

    ThorImplementation::Expression scores = Common::predictionsInput();
    ThorImplementation::Expression labels = Common::labelsInput();
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(temperature);
    ThorImplementation::Expression effectiveLabels = labels;
    ThorImplementation::Expression validMask(1.0);
    if (useMask) {
        validMask = Common::validDocumentMask();
        scaledScores = Common::maskPaddedDocuments(scaledScores, validMask);
        effectiveLabels = Common::zeroPaddedDocuments(labels, validMask);
    }

    ThorImplementation::Expression probabilities = scaledScores.softmax();
    ThorImplementation::Expression labelMass = effectiveLabels.reduce_sum({1}, {}, DataType::FP32);
    ThorImplementation::Expression gradient = (((probabilities * labelMass) - effectiveLabels) /
                                               ThorImplementation::Expression(temperature) *
                                               ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()));
    if (useMask)
        gradient = Common::zeroPaddedDocuments(gradient, validMask);
    gradient = gradient.withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{Common::kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void ListwiseSoftmaxCrossEntropyLoss::buildSupportLayersAndAddToNetwork() {
    Common::validateFixedSizeListwiseTensors("ListwiseSoftmaxCrossEntropyLoss", predictionsTensor, labelsTensor, maskTensor);
    Common::validateLossDataType("ListwiseSoftmaxCrossEntropyLoss", lossDataType);
    Common::validatePositiveTemperature("ListwiseSoftmaxCrossEntropyLoss", "temperature", temperature);

    lossShaperInput = Common::buildRawListwiseLoss(*network,
                                                   predictionsTensor,
                                                   labelsTensor,
                                                   maskTensor,
                                                   makeListwiseSoftmaxCrossEntropyLossExpression(lossDataType,
                                                                                                 temperature,
                                                                                                 maskTensor.has_value()),
                                                   makeListwiseSoftmaxCrossEntropyGradientExpression(predictionsTensor.getDataType(),
                                                                                                     temperature,
                                                                                                     maskTensor.has_value()),
                                                   lossDataType);
    lossTensor = Common::shapeRawListwiseLoss(*network, lossShaperInput, lossShape);
}

json ListwiseSoftmaxCrossEntropyLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "listwise_softmax_cross_entropy_loss";
    j["loss_shape"] = lossShape;
    j["temperature"] = temperature;
    j["has_mask"] = maskTensor.has_value();
    if (maskTensor.has_value())
        j["mask_tensor"] = maskTensor.value().architectureJson();
    return j;
}

void ListwiseSoftmaxCrossEntropyLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in ListwiseSoftmaxCrossEntropyLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "listwise_softmax_cross_entropy_loss")
        throw runtime_error("Layer type mismatch in ListwiseSoftmaxCrossEntropyLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    std::optional<Tensor> mask = std::nullopt;
    if (j.value("has_mask", false) || j.contains("mask_tensor")) {
        originalTensorId = j.at("mask_tensor").at("id").get<uint64_t>();
        mask = network->getApiTensorByOriginalId(originalTensorId);
    }

    ListwiseSoftmaxCrossEntropyLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.temperature = j.value("temperature", 1.0f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    loss.maskTensor = mask;
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("listwise_softmax_cross_entropy_loss", &Thor::ListwiseSoftmaxCrossEntropyLoss::deserialize);
    return true;
}();
}  // namespace
