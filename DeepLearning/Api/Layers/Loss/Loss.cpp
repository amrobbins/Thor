#include "DeepLearning/Api/Layers/Loss/Loss.h"

using namespace Thor;
using namespace std;

unordered_map<string, function<void(const nlohmann::json&, Network*)>> Loss::registry;

void Loss::deserialize(const nlohmann::json &j, Network *network) {
    assert(j.at("factory").get<std::string>() == Layer::Factory::Loss.value());
    std::string type = j.at("layer_type").get<std::string>();

    auto it = registry.find(type);
    if (it == registry.end())
        throw std::runtime_error("Unknown activation type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}
