#include "DeepLearning/Api/Layers/Metrics/Metric.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

unordered_map<string, Metric::Deserializer> &Metric::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Metric::register_layer(string name, Deserializer fn) { get_registry().emplace(move(name), move(fn)); }

void Metric::deserialize(const nlohmann::json &j, Network *network) {
    assert(j.at("factory").get<std::string>() == Layer::Factory::Metric.value());
    std::string type = j.at("layer_type").get<std::string>();

    unordered_map<string, Layer::Deserializer> &registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw std::runtime_error("Unknown activation type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}

json Metric::serialize(thor_file::TarWriter &archiveWriter, Stream stream) const {
    json j;
    j["factory"] = Layer::Factory::Metric.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());

    // Input connections
    j["predictions"] = getPredictions().serialize();
    j["labels"] = labelsTensor.serialize();

    // Output connections
    j["metric"] = metricTensor.serialize();

    return j;
}

}  // namespace Thor