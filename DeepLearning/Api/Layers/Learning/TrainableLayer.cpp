#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"

using namespace Thor;
using namespace std;

namespace Thor {

unordered_map<string, TrainableLayer::Deserializer> &TrainableLayer::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void TrainableLayer::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }

void TrainableLayer::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network) {
    assert(j.at("factory").get<string>() == Layer::Factory::Learning.value());
    string type = j.at("layer_type").get<string>();

    unordered_map<string, TrainableLayer::Deserializer> &registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw runtime_error("Unknown trainable layer type: " + type);

    auto deserializer = it->second;
    deserializer(archiveReader, j, network);
}

void TrainableLayer::serializeParameters(thor_file::TarWriter &archiveWriter,
                                 Stream stream,
                                 bool saveOptimizerState,
                                 ThorImplementation::StampedNetwork &stampedNetwork) {
    // FIXME: Layers that inherit from Thor::TrainableLayer will call Thor::TrainableLayer::serializeParameters(...)
    //        to use common code to serialize all of the parameters, that takes care of the parameters themselves and their optimizers.
}


void TrainableLayer::deserializeParameters(thor_file::TarReader &archiveReader,
                                 Stream stream,
                                 bool loadOptimizerState,
                                 ThorImplementation::StampedNetwork &stampedNetwork) {
    // FIXME: Layers that inherit from Thor::TrainableLayer will call Thor::TrainableLayer::deserializeParameters(...)
    //        to use common code to deserialize all of the parameters, that takes care of the parameters themselves and their optimizers.
}

}  // namespace Thor
