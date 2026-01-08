#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Network/Network.h"

#include "DeepLearning/Implementation/Layers/Loss.h"

using namespace Thor;
using namespace std;

atomic<int64_t> Layer::nextId(2);

const Layer::Factory Layer::Factory::Activation{"activation"};
const Layer::Factory Layer::Factory::Layer{"layer"};
const Layer::Factory Layer::Factory::Learning{"learning"};
const Layer::Factory Layer::Factory::Loss{"loss"};
const Layer::Factory Layer::Factory::Metric{"metric"};

unordered_map<string, Layer::Deserializer>& Layer::get_registry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Layer::register_layer(string name, Deserializer fn) { get_registry().emplace(std::move(name), std::move(fn)); }

void Layer::addToNetwork(Network* network) {
    assert(isInitialized());
    network->addToNetwork(this);
}

// void Layer::addTemplateLayerToNetwork(Network* network) {
//     assert(isInitialized());
//     network->addTemplateLayerToNetwork(this, nextId.fetch_add(1));
// }

void Layer::connectTwoLayers(shared_ptr<ThorImplementation::Layer> drivingLayer,
                             shared_ptr<ThorImplementation::Layer> loadingLayer,
                             const shared_ptr<Thor::Layer> drivingApiLayer,
                             const shared_ptr<Thor::Layer> loadingApiLayer,
                             const Thor::Tensor connectingApiTensor) {
    int drivingLayerConnectionType = drivingApiLayer == nullptr ? 0 : drivingApiLayer->getConnectionType(connectingApiTensor);
    int loadingLayerConnectionType = loadingApiLayer == nullptr ? 0 : loadingApiLayer->getConnectionType(connectingApiTensor);

    drivingLayer->connectToNextLayer(loadingLayer.get(), drivingLayerConnectionType, loadingLayerConnectionType);
}

void Layer::deserialize(thor_file::TarReader& archiveReader, const nlohmann::json& j, Network* network) {
    string factory = j.at("factory").get<string>();
    if (factory == Factory::Activation) {
        Activation::deserialize(j, network);
        return;
    } else if (factory == Factory::Learning) {
        TrainableWeightsBiasesLayer::deserialize(archiveReader, j, network);
        return;
    } else if (factory == Factory::Loss) {
        Loss::deserialize(j, network);
        return;
    } else if (factory == Factory::Metric) {
        Metric::deserialize(j, network);
        return;
    }

    if (factory != Factory::Layer) {
        throw runtime_error("Unknown layer factory: " + factory);
    }

    string type = j.at("layer_type").get<string>();

    unordered_map<string, Layer::Deserializer>& registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
        throw runtime_error("Unknown layer type: " + type);

    auto deserializer = it->second;
    deserializer(j, network);
}
