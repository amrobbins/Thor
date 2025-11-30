#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

Optimizer::Optimizer() : id(nextId.fetch_add(1)) { network = nullptr; }

Optimizer::~Optimizer() = default;

atomic<int64_t> Optimizer::nextId(2);

void Optimizer::addToNetwork(Network *network) {
    assert(network != nullptr);
    this->network = network;
    network->addToNetwork(this);
}

//// For future multi-gpu support, optimizers for the same layer on different GPU's will need to accumulate into a single weights memory
//// and then broadcast the updated weights to the optimizers on the other gpus.
// void Optimizer::attachToNetworkLayers() {
//     assert(network != nullptr);
//
//     uint32_t numTrainableLayers = network->getNumTrainableLayers();
//     for (uint32_t i = 0; i < numTrainableLayers; ++i) {
//         shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = network->getTrainableLayer(i);
//         trainableLayer->attachOptimizer(this->clone());
//     }
// }

// void Optimizer::attachToPhysicalLayer(uint64_t apiLayerId) {
//     assert(network != nullptr);
//     assert(network->getNumStamps() >= 1);
//
//     // I need to stamp one optimizer per trainable layer per GPU.
//     // When weights are involved they are shared, and then the only factor when a single GPU
//     // is that the update is divided by the total batch size (i.e. size that go through all stamps)
//     // only other factor when multiple GPUs is that each GPU's weights need to be accumulated
//     // (directly summed because each gradient is partial since used total batch size among all stamps on all gpus)
//
//     uint32_t numStamps = network->getNumStamps();
//     unordered_map<uint32_t, shared_ptr<ThorImplementation::Optimizer>> optimizerPerGpu;
//     for (uint32_t i = 0; i < numStamps; ++i) {
//         ThorImplementation::StampedNetwork &stampedNetwork = network->getStampedNetwork(i);
//         shared_ptr<ThorImplementation::Layer> physicalLayerStamp = stampedNetwork.getPhysicalLayerFromApiLayer(apiLayerId);
//         shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer =
//             dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(physicalLayerStamp);
//         assert(trainableLayer != nullptr);
//         // I need to stamp once per GPU, but connect once per stamp.
//         // Initialization should be handled together with layer initialization, not here.
//         uint32_t stampGpu = stampedNetwork.getGpuNum();
//         if (optimizerPerGpu.count(stampGpu) == 0) {
//             shared_ptr<ThorImplementation::Optimizer> optimizer = stamp();
//             optimizerPerGpu[stampGpu] = optimizer;
//             assert(optimizer->getGradientUpdateStream().isInitialized());
//             // Multiple stamps on same GPU must share a single gradient update stream.
//             // This is automatic because the optimizer owns the gradient update stream and the layers share the optimizer.
//             // FIXME: That logic should be moved outside of Adam and into Optimizer.
//             // FIXME: Optimizer initialization moved to layer
//             // FIXME: create isInitializationMaster, (for i == 0), so just one stamp inits when all told to do so.
//             // FIXME: So now in terms of ownership and lifecycle, a layer will own its optimizer, so rather than the map I have here
//             //        So when I tell a layer to serialize, I will need to specify whether or not it should serialize its optimizer
//             //        And serialization only happens for a single stamp of course.
//
//         }
//         trainableLayer->setOptimizer(optimizerPerGpu[stampGpu]);
//     }
// }

// void Optimizer::disconnectFromNetwork() {
//     if (network == nullptr)
//         return;
//
//     vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
//     for (uint32_t i = 0; i < stamps.size(); ++i) {
//         ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
//         for (uint32_t j = 0; j < stampedNetwork.getTrainableLayers().size(); ++j) {
//             shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayers()[j];
//             trainableLayer->clearOptimizer();
//         }
//     }
//
//     for (auto gpuIt = optimizers.begin(); gpuIt != optimizers.end(); ++gpuIt) {
//         unordered_map<int64_t, ThorImplementation::Optimizer *> gpuOptimizers = gpuIt->second;
//         gpuOptimizers.clear();
//     }
//     for (auto gpuIt = optimizersShared.begin(); gpuIt != optimizersShared.end(); ++gpuIt) {
//         unordered_map<int64_t, shared_ptr<ThorImplementation::Optimizer>> gpuOptimizers = gpuIt->second;
//         gpuOptimizers.clear();
//     }
//
//     network = nullptr;
// }
//
// void Optimizer::destroy() {
//     disconnectFromNetwork();
//     optimizers.clear();
// }

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
        if (updated.count(gpuNum) == 0) {
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

unordered_map<string, Optimizer::Deserializer> &Optimizer::getRegistry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Optimizer::registerLayer(string name, Deserializer fn) { getRegistry().emplace(move(name), move(fn)); }

void Optimizer::deserialize(const json &j, Network *network, uint64_t layerId) {
    // If there is no saved optimizer here, return.
    if (!j.contains("optimizer") || !j["optimizer"].is_object())
        return;

    json optimizerJ = j["optimizer"];
    string optimizerType = optimizerJ.at("type").get<string>();

    unordered_map<string, Deserializer> &registry = getRegistry();
    auto it = registry.find(optimizerType);
    if (it == registry.end())
        throw runtime_error("Unknown optimizer type: " + optimizerType);

    Deserializer deserializer = it->second;
    deserializer(j, network, layerId);
}