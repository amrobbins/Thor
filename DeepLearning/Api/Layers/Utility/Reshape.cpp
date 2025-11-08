#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Reshape::Reshape() = default;
Reshape::~Reshape() = default;

json Reshape::serialize(const string &storageDir, Stream stream) const {
    assert(initialized);
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInput.get().serialize();
    j["feature_output"] = featureOutput.get().serialize();

    return j;
}

void Reshape::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Reshape::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "reshape")
        throw runtime_error("Layer type mismatch in Reshape::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["feature_input"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    Reshape reshape;
    reshape.featureInput = featureInput;
    reshape.featureOutput = featureOutput;
    reshape.initialized = true;
    reshape.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::registry["reshape"] = &Thor::Reshape::deserialize;
    return true;
}();
}
