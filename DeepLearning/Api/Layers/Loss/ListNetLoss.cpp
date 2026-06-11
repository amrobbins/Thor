#include "DeepLearning/Api/Layers/Loss/ListNetLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListwiseLossCommon.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

namespace Common = ListwiseLossCommon;

ThorImplementation::DynamicExpression makeListNetLossExpression(DataType lossDataType,
                                                                float scoreTemperature,
                                                                float labelTemperature,
                                                                bool useMask) {
    Common::validateLossDataType("ListNetLoss", lossDataType);
    Common::validatePositiveTemperature("ListNetLoss", "scoreTemperature", scoreTemperature);
    Common::validatePositiveTemperature("ListNetLoss", "labelTemperature", labelTemperature);

    ThorImplementation::Expression scores = Common::predictionsInput();
    ThorImplementation::Expression labels = Common::labelsInput();
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(scoreTemperature);
    ThorImplementation::Expression scaledLabels = labels / ThorImplementation::Expression(labelTemperature);
    ThorImplementation::Expression validMask(1.0);
    if (useMask) {
        validMask = Common::validDocumentMask();
        scaledScores = Common::maskPaddedDocuments(scaledScores, validMask);
        scaledLabels = Common::maskPaddedDocuments(scaledLabels, validMask);
    }

    ThorImplementation::Expression targetProbabilities = scaledLabels.softmax();
    ThorImplementation::Expression logProbabilities = scaledScores.logSoftmax();
    ThorImplementation::Expression perDocumentLoss = -(targetProbabilities * logProbabilities);
    if (useMask)
        perDocumentLoss = Common::zeroPaddedDocuments(perDocumentLoss, validMask);
    ThorImplementation::Expression perListLoss = perDocumentLoss.reduce_sum({1}, {}, DataType::FP32).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{Common::kLossName, perListLoss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeListNetGradientExpression(DataType predictionsDataType,
                                                                    float scoreTemperature,
                                                                    float labelTemperature,
                                                                    bool useMask) {
    Common::validatePredictionsDType("ListNetLoss", predictionsDataType);
    Common::validatePositiveTemperature("ListNetLoss", "scoreTemperature", scoreTemperature);
    Common::validatePositiveTemperature("ListNetLoss", "labelTemperature", labelTemperature);

    ThorImplementation::Expression scores = Common::predictionsInput();
    ThorImplementation::Expression labels = Common::labelsInput();
    ThorImplementation::Expression scaledScores = scores / ThorImplementation::Expression(scoreTemperature);
    ThorImplementation::Expression scaledLabels = labels / ThorImplementation::Expression(labelTemperature);
    ThorImplementation::Expression validMask(1.0);
    if (useMask) {
        validMask = Common::validDocumentMask();
        scaledScores = Common::maskPaddedDocuments(scaledScores, validMask);
        scaledLabels = Common::maskPaddedDocuments(scaledLabels, validMask);
    }

    ThorImplementation::Expression scoreProbabilities = scaledScores.softmax();
    ThorImplementation::Expression targetProbabilities = scaledLabels.softmax();
    ThorImplementation::Expression gradient = ((scoreProbabilities - targetProbabilities) /
                                               ThorImplementation::Expression(scoreTemperature) *
                                               ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()));
    if (useMask)
        gradient = Common::zeroPaddedDocuments(gradient, validMask);
    gradient = gradient.withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{Common::kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void ListNetLoss::buildSupportLayersAndAddToNetwork() {
    Common::validateFixedSizeListwiseTensors("ListNetLoss", predictionsTensor, labelsTensor, maskTensor);
    Common::validateLossDataType("ListNetLoss", lossDataType);
    Common::validatePositiveTemperature("ListNetLoss", "scoreTemperature", scoreTemperature);
    Common::validatePositiveTemperature("ListNetLoss", "labelTemperature", labelTemperature);

    lossShaperInput = Common::buildRawListwiseLoss(*network,
                                                   predictionsTensor,
                                                   labelsTensor,
                                                   maskTensor,
                                                   makeListNetLossExpression(lossDataType,
                                                                             scoreTemperature,
                                                                             labelTemperature,
                                                                             maskTensor.has_value()),
                                                   makeListNetGradientExpression(predictionsTensor.getDataType(),
                                                                                 scoreTemperature,
                                                                                 labelTemperature,
                                                                                 maskTensor.has_value()),
                                                   lossDataType,
                                                   lossWeight);
    lossTensor = Common::shapeRawListwiseLoss(*network, lossShaperInput, lossShape);
}

json ListNetLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "list_net_loss";
    j["loss_shape"] = lossShape;
    j["score_temperature"] = scoreTemperature;
    j["label_temperature"] = labelTemperature;
    j["has_mask"] = maskTensor.has_value();
    if (maskTensor.has_value())
        j["mask_tensor"] = maskTensor.value().architectureJson();
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

    std::optional<Tensor> mask = std::nullopt;
    if (j.value("has_mask", false) || j.contains("mask_tensor")) {
        originalTensorId = j.at("mask_tensor").at("id").get<uint64_t>();
        mask = network->getApiTensorByOriginalId(originalTensorId);
    }

    ListNetLoss listNetLoss;
    listNetLoss.lossShape = j.at("loss_shape").get<LossShape>();
    listNetLoss.lossDataType = j.at("loss_data_type").get<DataType>();

    listNetLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    listNetLoss.scoreTemperature = j.value("score_temperature", 1.0f);
    listNetLoss.labelTemperature = j.value("label_temperature", 1.0f);
    listNetLoss.predictionsTensor = predictions;
    listNetLoss.labelsTensor = labels;
    listNetLoss.maskTensor = mask;
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
