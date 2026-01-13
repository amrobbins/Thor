#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"

using namespace Thor;
using namespace std;

namespace Thor {

unordered_map<string, TrainableWeightsBiasesLayer::Deserializer> &TrainableWeightsBiasesLayer::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void TrainableWeightsBiasesLayer::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }

void TrainableWeightsBiasesLayer::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network) {
    assert(j.at("factory").get<string>() == Layer::Factory::Learning.value());
    string type = j.at("layer_type").get<string>();

    unordered_map<string, TrainableWeightsBiasesLayer::Deserializer> &registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw runtime_error("Unknown trainable layer type: " + type);

    auto deserializer = it->second;
    deserializer(archiveReader, j, network);
}

void TrainableWeightsBiasesLayer::removeOptimizer() { this->optimizer.reset(); }

}  // namespace Thor
