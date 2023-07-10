#include "DeepLearning/Api/Optimizers/Sgd.h"

using namespace std;
using namespace Thor;

Sgd::~Sgd() {}

shared_ptr<ThorImplementation::Optimizer> Sgd::stamp(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) {
    Optional<ThorImplementation::Tensor> errorInput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(trainableLayer->getErrorInputs());
    Optional<ThorImplementation::Tensor> errorOutput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(trainableLayer->getErrorOutputs());
    return make_shared<ThorImplementation::Sgd>(
        trainableLayer, initialLearningRate, decay, momentum, useNesterovMomentum, errorInput, errorOutput);
}

void Sgd::setConstantLearningRate(float newCurrentLearningRate) {
    setDecay(0.0f);
    setInitialLearningRate(newCurrentLearningRate);
}

shared_ptr<Optimizer> Sgd::clone() const { return std::make_shared<Sgd>(*this); }

/*
 * decay will still apply from epoch 0.
 */
void Sgd::setDecay(float newDecay) {
    unordered_map<string, float> updatedParameters;

    assert(network != nullptr);

    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        for (uint32_t j = 0; j < stampedNetwork.trainableLayers.size(); ++j) {
            // Ensure that an sgd optimizer is infact attached to each trainable layer, and then set each of their learning rate.
            ThorImplementation::TrainableWeightsBiasesLayer *trainableLayer = stampedNetwork.trainableLayers[j];
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            sgd->setDecay(newDecay);
        }
    }
}

/*
 * decay computation will always apply from epoch 0, not from when setLearningRate() is called.
 */
void Sgd::setInitialLearningRate(float newInitialLearningRate) {
    unordered_map<string, float> updatedParameters;

    assert(network != nullptr);

    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        for (uint32_t j = 0; j < stampedNetwork.trainableLayers.size(); ++j) {
            // Ensure that an sgd optimizer is infact attached to each trainable layer, and then set each of their learning rate.
            ThorImplementation::TrainableWeightsBiasesLayer *trainableLayer = stampedNetwork.trainableLayers[j];
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            sgd->setInitialLearningRate(newInitialLearningRate);
        }
    }
}

void Sgd::setMomentum(float newMomentum) {
    unordered_map<string, float> updatedParameters;

    assert(network != nullptr);

    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        for (uint32_t j = 0; j < stampedNetwork.trainableLayers.size(); ++j) {
            // Ensure that an sgd optimizer is infact attached to each trainable layer, and then set each of their learning rate.
            ThorImplementation::TrainableWeightsBiasesLayer *trainableLayer = stampedNetwork.trainableLayers[j];
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            sgd->setMomentum(newMomentum);
        }
    }
}

void Sgd::setUseNesterovMomentum(bool newUseNesterovMomentum) {
    unordered_map<string, float> updatedParameters;

    assert(network != nullptr);

    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        for (uint32_t j = 0; j < stampedNetwork.trainableLayers.size(); ++j) {
            // Ensure that an sgd optimizer is infact attached to each trainable layer, and then set each of their learning rate.
            ThorImplementation::TrainableWeightsBiasesLayer *trainableLayer = stampedNetwork.trainableLayers[j];
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            sgd->setUseNesterovMomentum(newUseNesterovMomentum);
        }
    }
}