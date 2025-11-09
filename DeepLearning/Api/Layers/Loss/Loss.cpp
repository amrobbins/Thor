#include "DeepLearning/Api/Layers/Loss/Loss.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

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
