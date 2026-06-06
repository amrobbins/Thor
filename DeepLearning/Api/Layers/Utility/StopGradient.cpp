#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/StopGradient.h"
#include "DeepLearning/Api/Network/Network.h"

#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace Thor {

StopGradient::StopGradient() = default;
StopGradient::~StopGradient() = default;

json StopGradient::architectureJson() const {
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

void StopGradient::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in StopGradient::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "stop_gradient")
        throw runtime_error("Layer type mismatch in StopGradient::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["feature_input"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    StopGradient stopGradient;
    stopGradient.featureInput = featureInput;
    stopGradient.featureOutput = featureOutput;
    stopGradient.initialized = true;
    stopGradient.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("stop_gradient", &Thor::StopGradient::deserialize);
    return true;
}();
}  // namespace
