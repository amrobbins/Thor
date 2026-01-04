#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Optimizer::Optimizer() : id(nextId.fetch_add(1)) { network = nullptr; }

atomic<int64_t> Optimizer::nextId(2);

void Optimizer::addToNetwork(Network *network) {
    assert(network != nullptr);
    this->network = network;
    network->addToNetwork(this);
}

unordered_map<string, float> Optimizer::getAllHyperParameters() {
    assert(network != nullptr);
    assert(network->getNumStamps() >= 1);
    assert(network->getNumTrainableLayers() >= 1);

    // All optimizer instances must have the same parameters.
    ThorImplementation::StampedNetwork &stampedNetwork = network->getStampedNetwork(0);

    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayer(0);
    if (trainableLayer->hasOptimizer()) {
        shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = trainableLayer->getOptimizer();
        return physicalOptimizer->getAllHyperParameters();
    } else {
        return {};
    }
}

// Update all optimizers in the network, each belonging to a trainable layer.
// The thought with this pattern is that the base optimizer keeps the knowledge about how to properly update all optimizers in the network.
void Optimizer::updateHyperParameters(Network *network, uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    uint32_t numStamps = network->getNumStamps();
    unordered_set<uint32_t> updated;
    for (uint32_t i = 0; i < numStamps; ++i) {
        ThorImplementation::StampedNetwork &stamp = network->getStampedNetwork(i);
        uint32_t gpuNum = stamp.getGpuNum();
        if (!updated.contains(gpuNum)) {
            updated.insert(gpuNum);
            uint64_t numTrainableLayers = stamp.getNumTrainableLayers();
            for (uint32_t j = 0; j < numTrainableLayers; ++j) {
                std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> &trainableLayer = stamp.getTrainableLayer(j);
                if (trainableLayer->hasOptimizer()) {
                    shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = trainableLayer->getOptimizer();
                    physicalOptimizer->updateHyperParameters(epoch, batch, batchesPerEpoch);
                }
            }
        }
    }
}

string Optimizer::getVersion() const { return "1.0.0"; }

unordered_map<string, Optimizer::Deserializer> &Optimizer::getRegistry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Optimizer::registerLayer(string name, Deserializer fn) { getRegistry().emplace(std::move(name), std::move(fn)); }

shared_ptr<Optimizer> Optimizer::deserialize(thor_file::TarReader &archiveReader, const json &j) {
    assert(j.contains("optimizer_type"));
    string optimizerType = j.at("optimizer_type").get<string>();

    unordered_map<string, Deserializer> &registry = getRegistry();
    auto it = registry.find(optimizerType);
    if (it == registry.end())
        throw runtime_error("Unknown optimizer type: " + optimizerType);

    Deserializer deserializer = it->second;
    return deserializer(archiveReader, j);
}

}  // namespace Thor
