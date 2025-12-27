#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json GradientRivet::serialize(const string &storageDir, Stream stream) const {
    assert(initialized);
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    j["input_tensor"] = featureInput.get().serialize();
    j["output_tensor"] = featureOutput.get().serialize();

    return j;
}

void GradientRivet::deserialize(const json &j, Network *network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in GradientRivet::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "gradient_rivet")
        throw runtime_error("Layer type mismatch in GradientRivet::deserialize: " + j.at("layer_type").get<string>());

    nlohmann::json input = j["input_tensor"].get<nlohmann::json>();
    uint64_t originalTensorId = input.at("id").get<uint64_t>();
    Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);
    Tensor featureOutput = Tensor::deserialize(j.at("output_tensor").get<nlohmann::json>());

    GradientRivet gradientRivet;
    gradientRivet.featureInput = featureInput;
    gradientRivet.featureOutput = featureOutput;
    gradientRivet.initialized = true;
    gradientRivet.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("gradient_rivet", &Thor::GradientRivet::deserialize);
    return true;
}();
}  // namespace
