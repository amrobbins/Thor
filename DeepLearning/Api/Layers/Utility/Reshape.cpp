#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Reshape::Reshape() = default;
Reshape::~Reshape() = default;

json Reshape::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();

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
    reshape.newDimensions = {0U};
    for (uint32_t i = 0; i < featureOutput.getDimensions().size(); ++i)
        reshape.newDimensions.push_back(featureOutput.getDimensions()[i]);
    if (reshape.featureInput.value().getTotalNumElements() != reshape.featureOutput.value().getTotalNumElements())
        throw runtime_error("In Reshape::deserialize, input num elements " + to_string(reshape.featureInput.value().getTotalNumElements()) +
                            ", output num elements " + to_string(reshape.featureOutput.value().getTotalNumElements()) +
                            ". These must match.");
    reshape.initialized = true;
    reshape.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("reshape", &Thor::Reshape::deserialize);
    return true;
}();
}  // namespace
