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
    assert(decay < 1.0);

    decay = newDecay;
    updateParameters();
}

/*
 * decay computation will always apply from epoch 0, not from when setLearningRate() is called.
 */
void Sgd::setInitialLearningRate(float newInitialLearningRate) {
    assert(newInitialLearningRate >= 0.0f);

    initialLearningRate = newInitialLearningRate;
    updateParameters();
}

void Sgd::setMomentum(float newMomentum) {
    assert(momentum >= 0.0f);

    momentum = newMomentum;
    updateParameters();
}

void Sgd::setUseNesterovMomentum(bool newUseNesterovMomentum) {
    useNesterovMomentum = newUseNesterovMomentum;
    updateParameters();
}

void Sgd::updateParameters() {
    assert(network != nullptr);
    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        for (uint32_t j = 0; j < stampedNetwork.getTrainableLayers().size(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayers()[j];
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            if (sgd == nullptr || sgd->getId() != getId())
                continue;
            sgd->setInitialLearningRate(initialLearningRate);
            sgd->setDecay(decay);
            sgd->setMomentum(momentum);
            sgd->setUseNesterovMomentum(useNesterovMomentum);
        }
    }
}

float Sgd::getInitialLearningRate() { return initialLearningRate; }

float Sgd::getDecay() { return decay; }

float Sgd::getMomentum() { return momentum; }

bool Sgd::getUseNesterovMomentum() { return useNesterovMomentum; }