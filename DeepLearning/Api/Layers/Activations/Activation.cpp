#include "DeepLearning/Api/Layers/Activations/Activation.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

unordered_map<string, Activation::Deserializer>& Activation::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Activation::register_layer(string name, Deserializer fn) { get_registry().emplace(move(name), move(fn)); }

json Activation::serialize(const std::string& storageDir, Stream stream) const {
    assert(initialized);
    assert(featureInput.isPresent());
    assert(featureOutput.isPresent());

    json j;
    j["factory"] = Layer::Factory::Activation.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["feature_input"] = featureInput.get().serialize();
    j["feature_output"] = featureOutput.get().serialize();
    return j;
}

void Activation::deserialize(const json& j, Network* network) {
    assert(j.at("factory").get<std::string>() == Layer::Factory::Activation);
    std::string type = j.at("layer_type").get<std::string>();

    unordered_map<string, Activation::Deserializer>& registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw std::runtime_error("Unknown activation type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}

}  // namespace Thor
