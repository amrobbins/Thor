#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace Thor;
using namespace std;

Optimizer::Optimizer() : id(nextId.fetch_add(1)) { network = nullptr; }

Optimizer::~Optimizer() { destroy(); }

atomic<int64_t> Optimizer::nextId(2);

void Optimizer::addToNetwork(Network *network) {
    assert(network != nullptr);
    this->network = network;
    network->addToNetwork(this);
}

// For future multi-gpu support, optimizers for the same layer on different GPU's will need to accumulate into a single weights memory
// and then broadcast the updated weights to the optimizers on the other gpus.
void Optimizer::attachToNetwork() {
    assert(network != nullptr);

    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        uint32_t stampGpu = stampedNetwork.getGpuNum();
        for (uint32_t j = 0; j < stampedNetwork.getTrainableLayers().size(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayers()[j];
            int64_t layerStampedId = trainableLayer->getStampedId();
            if (optimizersShared[stampGpu].count(layerStampedId) == 0) {
                shared_ptr<ThorImplementation::Optimizer> optimizer = stamp(trainableLayer);
                assert(optimizer->getGradientUpdateStream().isInitialized());
                optimizer->initialize();
                optimizersShared[stampGpu][layerStampedId] = optimizer;
                optimizers[stampGpu][layerStampedId] = optimizer.get();
            }

            assert(optimizers[stampGpu].count(layerStampedId) == 1);
            assert(optimizers[stampGpu][layerStampedId]->getGradientUpdateStream().isInitialized());
            trainableLayer->setOptimizer(optimizersShared[stampGpu][layerStampedId]);
        }
    }
}

void Optimizer::disconnectFromNetwork() {
    if (network == nullptr)
        return;

    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        for (uint32_t j = 0; j < stampedNetwork.getTrainableLayers().size(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayers()[j];
            trainableLayer->clearOptimizer();
        }
    }

    for (auto gpuIt = optimizers.begin(); gpuIt != optimizers.end(); ++gpuIt) {
        unordered_map<int64_t, ThorImplementation::Optimizer *> gpuOptimizers = gpuIt->second;
        gpuOptimizers.clear();
    }
    for (auto gpuIt = optimizersShared.begin(); gpuIt != optimizersShared.end(); ++gpuIt) {
        unordered_map<int64_t, shared_ptr<ThorImplementation::Optimizer>> gpuOptimizers = gpuIt->second;
        gpuOptimizers.clear();
    }

    network = nullptr;
}

void Optimizer::destroy() {
    disconnectFromNetwork();
    optimizers.clear();
}

unordered_map<string, float> Optimizer::getAllHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    assert(network != nullptr);

    // All optimizer instances must have the same parameters.
    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    assert(!stamps.empty());
    assert(!stamps[0].getTrainableLayers().empty());

    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stamps[0].getTrainableLayers()[0];
    Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
    assert(maybeOptimizer.isPresent());
    shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
    return optimizer->getAllHyperParameters(epoch, batch, batchesPerEpoch);
}

unordered_map<string, float> Optimizer::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    assert(network != nullptr);

    // Each trainable layer has it's own optimizer which is shared across all stamps on a gpu
    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    assert(!stamps.empty());

    unordered_map<string, float> updatedHyperParameters;
    for (auto gpuIt = optimizers.begin(); gpuIt != optimizers.end(); ++gpuIt) {
        unordered_map<int64_t, ThorImplementation::Optimizer *> gpuOptimizers = gpuIt->second;
        for (auto it = gpuOptimizers.begin(); it != gpuOptimizers.end(); ++it) {
            ThorImplementation::Optimizer *optimizer = it->second;
            updatedHyperParameters = optimizer->updateHyperParameters(epoch, batch, batchesPerEpoch);
        }
    }

    return updatedHyperParameters;
}