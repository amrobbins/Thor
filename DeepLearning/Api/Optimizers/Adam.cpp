#include "DeepLearning/Api/Optimizers/Adam.h"

using namespace std;
using namespace Thor;

Adam::~Adam() {}

shared_ptr<ThorImplementation::Optimizer> Adam::stamp(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) {
    Optional<ThorImplementation::Tensor> errorInput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(trainableLayer->getErrorInputs());
    Optional<ThorImplementation::Tensor> errorOutput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(trainableLayer->getErrorOutputs());
    return make_shared<ThorImplementation::Adam>(trainableLayer, alpha, beta1, beta2, epsilon, errorInput, errorOutput);
}

void Adam::setAlpha(float newAlpha) {
    alpha = newAlpha;
    updateParameters();
}

void Adam::setBeta1(float newBeta1) {
    beta1 = newBeta1;
    updateParameters();
}

void Adam::setBeta2(float newBeta2) {
    beta2 = newBeta2;
    updateParameters();
}

void Adam::setEpsilon(float newEpsilon) {
    epsilon = newEpsilon;
    updateParameters();
}

float Adam::getAlpha() { return alpha; }

float Adam::getBeta1() { return beta1; }

float Adam::getBeta2() { return beta2; }

float Adam::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> Adam::clone() const { return std::make_shared<Adam>(*this); }

void Adam::updateParameters() {
    assert(network != nullptr);
    vector<ThorImplementation::StampedNetwork> stamps = network->getStampedNetworks();
    for (uint32_t i = 0; i < stamps.size(); ++i) {
        ThorImplementation::StampedNetwork stampedNetwork = stamps[i];
        for (uint32_t j = 0; j < stampedNetwork.getTrainableLayers().size(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayers()[j];
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Adam> adam = dynamic_pointer_cast<ThorImplementation::Adam>(optimizer);
            if (adam == nullptr || adam->getId() != getId())
                continue;
            adam->setAlpha(alpha);
            adam->setBeta1(beta1);
            adam->setBeta2(beta2);
            adam->setEpsilon(epsilon);
        }
    }
}