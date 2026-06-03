#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char *kPredictionsName = "predictions";
constexpr const char *kLabelsName = "labels";
constexpr const char *kLossName = "loss";
constexpr const char *kGradientName = "predictions_grad";

ThorImplementation::DynamicExpression makeBinaryCrossEntropyLossExpression(DataType lossDataType) {
    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);

    // Numerically stable BCE-with-logits:
    //   max(x, 0) - x * y + log1p(exp(-abs(x)))
    ThorImplementation::Expression loss = (logits.max(zero) - (logits * labels) + (-logits.abs()).exp().log1p()).withOutputDType(lossDataType);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeBinaryCrossEntropyGradientExpression(DataType predictionsDataType) {
    THOR_THROW_IF_FALSE(predictionsDataType == DataType::FP16 || predictionsDataType == DataType::FP32);

    ThorImplementation::Expression logits = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression gradient =
        ((logits.sigmoid() - labels) * ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void BinaryCrossEntropy::buildSupportLayersAndAddToNetwork() {
    THOR_THROW_IF_FALSE(!rawLossAddedToNetwork);

    CustomLoss rawBinaryCrossEntropy = CustomLoss::Builder()
                                           .network(*network)
                                           .lossExpression(makeBinaryCrossEntropyLossExpression(lossDataType))
                                           .gradientExpression(makeBinaryCrossEntropyGradientExpression(predictionsTensor.getDataType()))
                                           .predictions(predictionsTensor)
                                           .labels(labelsTensor)
                                           .predictionsName(kPredictionsName)
                                           .labelsName(kLabelsName)
                                           .lossName(kLossName)
                                           .gradientName(kGradientName)
                                           .reportsRawLoss()
                                           .lossDataType(lossDataType)
                                           .build();
    lossShaperInput = rawBinaryCrossEntropy.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed in this case.
        THOR_THROW_IF_FALSE(lossShape == LossShape::ELEMENTWISE);
        lossTensor = lossShaperInput;
    }
}

json BinaryCrossEntropy::architectureJson() const {
    // The thing that is deserialized must be just the base layer, any helper layers
    // are themselves deserialized. So loss_shape is set to RAW.

    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "binary_cross_entropy";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = LossShape::RAW;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.architectureJson();
    j["predictions_tensor"] = predictionsTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();

    return j;
}

void BinaryCrossEntropy::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in BinaryCrossEntropy::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "binary_cross_entropy")
        throw runtime_error("Layer type mismatch in BinaryCrossEntropy::deserialize: " + j.at("layer_type").get<std::string>());

    // Only connect the historical single raw-loss BCE layer and add it to the network.
    // New saves will contain a raw custom_loss support layer instead.
    BinaryCrossEntropy binaryCrossEntropy;
    THOR_THROW_IF_FALSE(j.at("loss_shape").get<LossShape>() == LossShape::RAW);
    binaryCrossEntropy.lossShape = LossShape::RAW;
    binaryCrossEntropy.lossDataType = j.at("loss_data_type").get<DataType>();

    uint64_t originalTensorId;
    originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    binaryCrossEntropy.predictionsTensor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    binaryCrossEntropy.labelsTensor = network->getApiTensorByOriginalId(originalTensorId);
    binaryCrossEntropy.rawLossAddedToNetwork = true;

    binaryCrossEntropy.lossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);
    binaryCrossEntropy.lossShaperInput = binaryCrossEntropy.lossTensor;

    binaryCrossEntropy.initialized = true;
    binaryCrossEntropy.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("binary_cross_entropy", &Thor::BinaryCrossEntropy::deserialize);
    return true;
}();
}  // namespace
