#include "DeepLearning/Api/Layers/Loss/Loss.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json Loss::serialize(const string &storageDir, Stream stream) const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["labels_tensor"] = labelsTensor.serialize();
    j["predictions_tensor"] = predictionsTensor.serialize();
    j["loss_shaper_input_tensor"] = lossShaperInput.serialize();
    j["loss_tensor"] = lossTensor.serialize();

    return j;
}

unordered_map<string, Loss::Deserializer> &Loss::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Loss::register_layer(string name, Deserializer fn) { get_registry().emplace(move(name), move(fn)); }

void Loss::deserialize(const nlohmann::json &j, Network *network) {
    assert(j.at("factory").get<std::string>() == Layer::Factory::Loss.value());
    std::string type = j.at("layer_type").get<std::string>();

    unordered_map<string, Loss::Deserializer> &registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw std::runtime_error("Unknown activation type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}

}  // namespace Thor
