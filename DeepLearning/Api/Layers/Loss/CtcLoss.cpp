#include "DeepLearning/Api/Layers/Loss/CtcLoss.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void CtcLoss::buildSupportLayersAndAddToNetwork() {
    CtcLoss::Builder rawBuilder;
    rawBuilder.network(*network)
        .predictions(predictionsTensor)
        .labels(labelsRaggedTensor)
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

    finalizeLossReporting();
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
    j["labels_ragged_tensor"] = labelsRaggedTensor.architectureJson();
    j["input_lengths_tensor"] = inputLengthsTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    j["oob_gradient_mode"] = oobGradientMode == ThorImplementation::CtcLossOobGradientMode::ZERO ? "zero" : "skip";
    return j;
}

void CtcLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<string>() != "2.0.0")
        throw runtime_error("Unsupported version in CtcLoss::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "ctc_loss")
        throw runtime_error("Layer type mismatch in CtcLoss::deserialize: " + j.at("layer_type").get<string>());

    THOR_THROW_IF_FALSE(j.at("loss_shape").get<LossShape>() == LossShape::RAW);

    const uint64_t predictionsId = j.at("predictions_tensor").at("id").get<uint64_t>();
    const json& labelsJson = j.at("labels_ragged_tensor");
    const string labelsVersion = labelsJson.at("version").get<string>();
    if (labelsVersion != "1.0.0" && labelsVersion != "1.1.0")
        throw runtime_error("Unsupported RaggedTensor version in CtcLoss::deserialize: " + labelsVersion);
    if (labelsJson.at("ragged_rank").get<uint32_t>() != 1)
        throw runtime_error("CtcLoss requires ragged_rank 1 labels.");
    const uint64_t labelValuesId = labelsJson.at("values").at("id").get<uint64_t>();
    const uint64_t labelOffsetsId = labelsJson.at("offsets").at("id").get<uint64_t>();
    const uint64_t inputLengthsId = j.at("input_lengths_tensor").at("id").get<uint64_t>();

    CtcLoss ctcLoss;
    ctcLoss.rawLossAddedToNetwork = true;
    ctcLoss.predictionsTensor = network->getApiTensorByOriginalId(predictionsId);
    Tensor labelValues = network->getApiTensorByOriginalId(labelValuesId);
    Tensor labelOffsets = network->getApiTensorByOriginalId(labelOffsetsId);
    ctcLoss.labelsRaggedTensor = labelsJson.contains("max_values_per_row")
        ? RaggedTensor(labelValues, labelOffsets, labelsJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(labelValues, labelOffsets);
    THOR_THROW_IF_FALSE(ctcLoss.labelsRaggedTensor.getBatchSize() == labelsJson.at("batch_size").get<uint64_t>());
    THOR_THROW_IF_FALSE(ctcLoss.labelsRaggedTensor.getMaxTotalValues() == labelsJson.at("max_total_values").get<uint64_t>());
    THOR_THROW_IF_FALSE(ctcLoss.predictionsTensor.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(ctcLoss.predictionsTensor.getDimensions().size() == 2);
    THOR_THROW_IF_FALSE(ctcLoss.labelsRaggedTensor.getValuesDataType() == DataType::INT32);
    THOR_THROW_IF_FALSE(ctcLoss.labelsRaggedTensor.getTrailingDimensions().empty());
    ctcLoss.labelsTensor = ctcLoss.labelsRaggedTensor.getValues();
    ctcLoss.inputLengthsTensor = network->getApiTensorByOriginalId(inputLengthsId);
    THOR_THROW_IF_FALSE(ThorImplementation::isCudnnCtcLengthDataType(ctcLoss.inputLengthsTensor.getDataType()));
    THOR_THROW_IF_FALSE(ctcLoss.inputLengthsTensor.getDimensions() == vector<uint64_t>{1});
    ctcLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    ctcLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    ctcLoss.lossShape = LossShape::RAW;
    ctcLoss.lossTensor = Tensor::deserialize(j.at("loss_shaper_input_tensor"));
    ctcLoss.lossShaperInput = ctcLoss.lossTensor;
    const string oobMode = j.at("oob_gradient_mode").get<string>();
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
