#include "DeepLearning/Api/Layers/Loss/CtcLoss.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void CtcLoss::buildSupportLayersAndAddToNetwork() {
    CtcLoss::Builder rawBuilder;
    rawBuilder.network(*network)
        .predictions(predictionsTensor)
        .labels(labelsTensor)
        .labelLengths(labelLengthsTensor)
        .inputLengths(inputLengthsTensor)
        .lossDataType(lossDataType)
        .rawLossAddedToNetwork()
        .reportsRawLoss();
    rawBuilder.lossWeight(lossWeight.value_or(1.0f));
    if (oobGradientMode == ThorImplementation::CtcLossOobGradientMode::ZERO)
        rawBuilder.zeroOutOfBoundsGradients();
    else
        rawBuilder.skipOutOfBoundsGradients();

    CtcLoss rawLoss;
    rawBuilder.populateAndAdd(rawLoss);
    lossShaperInput = rawLoss.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json CtcLoss::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "ctc_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = LossShape::RAW;
    j["loss_data_type"] = lossDataType;
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    j["predictions_tensor"] = predictionsTensor.architectureJson();
    j["labels_tensor"] = labelsTensor.architectureJson();
    j["label_lengths_tensor"] = labelLengthsTensor.architectureJson();
    j["input_lengths_tensor"] = inputLengthsTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    j["max_label_length"] = maxLabelLength;
    j["oob_gradient_mode"] = oobGradientMode == ThorImplementation::CtcLossOobGradientMode::ZERO ? "zero" : "skip";
    return j;
}

void CtcLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in CtcLoss::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "ctc_loss")
        throw runtime_error("Layer type mismatch in CtcLoss::deserialize: " + j.at("layer_type").get<string>());

    THOR_THROW_IF_FALSE(j.at("loss_shape").get<LossShape>() == LossShape::RAW);

    const uint64_t predictionsId = j.at("predictions_tensor").at("id").get<uint64_t>();
    const uint64_t labelsId = j.at("labels_tensor").at("id").get<uint64_t>();
    const uint64_t labelLengthsId = j.at("label_lengths_tensor").at("id").get<uint64_t>();
    const uint64_t inputLengthsId = j.at("input_lengths_tensor").at("id").get<uint64_t>();

    CtcLoss ctcLoss;
    ctcLoss.rawLossAddedToNetwork = true;
    ctcLoss.predictionsTensor = network->getApiTensorByOriginalId(predictionsId);
    ctcLoss.labelsTensor = network->getApiTensorByOriginalId(labelsId);
    ctcLoss.labelLengthsTensor = network->getApiTensorByOriginalId(labelLengthsId);
    ctcLoss.inputLengthsTensor = network->getApiTensorByOriginalId(inputLengthsId);
    ctcLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    ctcLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    ctcLoss.lossShape = LossShape::RAW;
    ctcLoss.lossTensor = Tensor::deserialize(j.at("loss_shaper_input_tensor"));
    ctcLoss.lossShaperInput = ctcLoss.lossTensor;
    ctcLoss.maxLabelLength = j.value("max_label_length", static_cast<uint32_t>(ctcLoss.labelsTensor.getDimensions().at(0)));
    const string oobMode = j.value("oob_gradient_mode", string("zero"));
    if (oobMode == "zero")
        ctcLoss.oobGradientMode = ThorImplementation::CtcLossOobGradientMode::ZERO;
    else if (oobMode == "skip")
        ctcLoss.oobGradientMode = ThorImplementation::CtcLossOobGradientMode::SKIP;
    else
        throw runtime_error("Unsupported CtcLoss oob_gradient_mode: " + oobMode);
    ctcLoss.initialized = true;
    ctcLoss.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Loss::register_layer("ctc_loss", &Thor::CtcLoss::deserialize);
    return true;
}();
}  // namespace
