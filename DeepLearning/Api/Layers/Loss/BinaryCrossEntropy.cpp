#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

void BinaryCrossEntropy::buildSupportLayersAndAddToNetwork() {
    THOR_THROW_IF_FALSE(!rawLossAddedToNetwork);

    BinaryCrossEntropy rawBinaryCrossEntropy = BinaryCrossEntropy::Builder()
                                                .network(*network)
                                                .predictions(predictionsTensor)
                                                .labels(labelsTensor)
                                                .rawLossAddedToNetwork()
                                                .reportsElementwiseLoss()
                                                .lossDataType(lossDataType)
                                                .build();
    lossShaperInput = rawBinaryCrossEntropy.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getFeatureOutput().value();
    } else {
        // No loss shaper needed in this case
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

    // Only connect the single raw-loss layer and add it to the network, like when it was built.
    // Helper loss-shaper layers deserialize themselves.
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
