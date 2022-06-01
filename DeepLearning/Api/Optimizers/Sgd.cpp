#include "Sgd.h"

using namespace std;

Sgd::Sgd() {}

Sgd::~Sgd() {}

void Sgd::setNetwork(Thor::Network *network) { this->network = network; }

unordered_map<string, float> Sgd::updateParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    unordered_map<string, float> updatedParameters;

    assert(network != nullptr);

    if (!parametersInitialized || (epoch != currentEpoch && decay > 0.0)) {
        parametersInitialized = true;
        currentEpoch = epoch;

        vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
        if (!stamps.empty())
            updatedParameters = initializeStampedNetworkParameters(stamps[0], epoch, batch, batchesPerEpoch);
        for (uint32_t i = 1; i < stamps.size(); ++i) {
            initializeStampedNetworkParameters(stamps[i], epoch, batch, batchesPerEpoch);
        }
    }

    return updatedParameters;
}

unordered_map<string, float> Sgd::initializeStampedNetworkParameters(ThorImplementation::StampedNetwork &stampedNetwork,
                                                                     uint64_t epoch,
                                                                     uint64_t batch,
                                                                     uint64_t batchesPerEpoch) {
    float currentLearningRate = initialLearningRate * pow(1.0 - (double)decay, (double)epoch);
    for (uint32_t i = 0; i < stampedNetwork.trainableLayers.size(); ++i) {
        ThorImplementation::TrainableWeightsBiasesLayer *trainableLayer = stampedNetwork.trainableLayers[i];
        trainableLayer->setLearningRate(currentLearningRate);
        // FIXME: support for momentum
    }

    unordered_map<string, float> updatedParameters;
    updatedParameters["currentLearningRate"] = currentLearningRate;
    return updatedParameters;
}

std::unordered_map<std::string, float> Sgd::getAllParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    float currentLearningRate = initialLearningRate * pow((double)decay, (double)epoch);
    unordered_map<string, float> parameters;
    parameters["currentLearningRate"] = currentLearningRate;
    parameters["initialLearningRate"] = initialLearningRate;
    parameters["decay"] = decay;
    parameters["momentum"] = momentum;
    parameters["useNesterov"] = useNesterov ? 1.0f : 0.0f;
    return parameters;
}
