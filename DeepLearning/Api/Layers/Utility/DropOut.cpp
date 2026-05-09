#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json DropOut::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();

    j["drop_proportion"] = dropProportion;

    return j;
}

void DropOut::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in DropOut::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "drop_out")
        throw runtime_error("Layer type mismatch in DropOut::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["feature_input"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    DropOut dropOut;
    dropOut.featureInput = featureInput;
    dropOut.featureOutput = featureOutput;
    dropOut.dropProportion = j.at("drop_proportion").get<float>();
    dropOut.initialized = true;
    dropOut.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("drop_out", &Thor::DropOut::deserialize);
    return true;
}();
}  // namespace
