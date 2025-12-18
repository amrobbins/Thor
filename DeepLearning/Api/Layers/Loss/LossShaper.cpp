#include "DeepLearning/Api/Layers/Loss/LossShaper.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json LossShaper::serialize(const string &storageDir, Stream stream) const {
    // The thing that is deserialized must be just the base layers, any helper layers
    // are themselves deserialized. So loss_shape set to LossShape::ELEMENTWISE

    json j;
    j["factory"] = Layer::Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = outputLossType;
    j["loss_input"] = lossInput.serialize();
    j["loss_output"] = lossOutput.serialize();

    return j;
}

void LossShaper::deserialize(const json &j, Network *network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in LossShaper::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "loss_shaper")
        throw runtime_error("Layer type mismatch in LossShaper::deserialize: " + j.at("layer_type").get<std::string>());

    LossShaper lossShaper;
    lossShaper.outputLossType = j.at("loss_shape").get<ThorImplementation::LossShaper::OutputLossType>();

    uint64_t originalTensorId;
    originalTensorId = j["loss_input"].at("id").get<uint64_t>();
    lossShaper.lossInput = network->getApiTensorByOriginalId(originalTensorId);
    lossShaper.lossOutput = Tensor::deserialize(j["loss_output"]);
    lossShaper.initialized = true;
    lossShaper.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("loss_shaper", &Thor::LossShaper::deserialize);
    return true;
}();
}  // namespace
