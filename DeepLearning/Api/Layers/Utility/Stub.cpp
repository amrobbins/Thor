#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Stub::Stub() = default;
Stub::~Stub() = default;

json Stub::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(!featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["input_tensor"] = featureInput.value().architectureJson();

    return j;
}

void Stub::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Stub::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "stub")
        throw runtime_error("Layer type mismatch in Stub::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["input_tensor"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

    Stub stub;
    stub.featureInput = featureInput;
    stub.initialized = true;
    stub.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("stub", &Thor::Stub::deserialize);
    return true;
}();
}  // namespace
