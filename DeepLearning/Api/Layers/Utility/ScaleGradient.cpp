#include "DeepLearning/Api/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cmath>
#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace Thor {

ScaleGradient::ScaleGradient() = default;
ScaleGradient::~ScaleGradient() = default;

json ScaleGradient::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(scale.has_value());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.value().architectureJson();
    j["feature_output"] = featureOutput.value().architectureJson();
    j["scale"] = scale.value();
    return j;
}

void ScaleGradient::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in ScaleGradient::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "scale_gradient")
        throw runtime_error("Layer type mismatch in ScaleGradient::deserialize: " + j.at("layer_type").get<string>());

    const float scale = j.at("scale").get<float>();
    if (!std::isfinite(scale))
        throw runtime_error("ScaleGradient::deserialize requires a finite scale.");

    nlohmann::json input = j.at("feature_input").get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
    Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

    ScaleGradient scaleGradient;
    scaleGradient.featureInput = featureInput;
    scaleGradient.featureOutput = featureOutput;
    scaleGradient.scale = scale;
    scaleGradient.initialized = true;
    scaleGradient.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("scale_gradient", &Thor::ScaleGradient::deserialize);
    return true;
}();
}  // namespace
