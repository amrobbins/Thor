#include "DeepLearning/Api/Optimizers/Sgd.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

// Sgd::~Sgd() {}

shared_ptr<ThorImplementation::Optimizer> Sgd::stamp(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) {
    return make_shared<ThorImplementation::Sgd>(
        trainableLayer, initialLearningRate, decay, momentum, useNesterovMomentum, startResumeEpoch);
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
    uint32_t numStamps = network->getNumStamps();
    for (uint32_t i = 0; i < numStamps; ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network->getStampedNetwork(i);
        uint32_t numTrainableLayers = stampedNetwork.getNumTrainableLayers();
        for (uint32_t j = 0; j < numTrainableLayers; ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> &trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybePhysicalOptimizer = trainableLayer->getOptimizer();
            assert(maybePhysicalOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybePhysicalOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> physicalSgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            if (physicalSgd == nullptr || physicalSgd->getId() != getId())
                continue;
            physicalSgd->setInitialLearningRate(initialLearningRate);
            physicalSgd->setDecay(decay);
            physicalSgd->setMomentum(momentum);
            physicalSgd->setUseNesterovMomentum(useNesterovMomentum);
        }
    }
}

float Sgd::getInitialLearningRate() { return initialLearningRate; }

float Sgd::getDecay() { return decay; }

float Sgd::getMomentum() { return momentum; }

bool Sgd::getUseNesterovMomentum() { return useNesterovMomentum; }

json Sgd::serialize(const string &storageDir,
                    Stream stream,
                    TrainableWeightsBiasesLayer const *owningLayer,
                    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalOwningLayer) const {
    json j;
    j["optimizer_type"] = string("sgd");
    j["version"] = getVersion();

    // Get the params from the physical layer and record them.
    shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalOwningLayer->getOptimizer();
    shared_ptr<ThorImplementation::Sgd> physicalSgd = dynamic_pointer_cast<ThorImplementation::Sgd>(physicalOptimizer);
    assert(physicalSgd != nullptr);

    j["initial_learning_rate"] = physicalSgd->getInitialLearningRate();
    j["decay"] = physicalSgd->getDecay();
    j["momentum"] = physicalSgd->getMomentum();
    j["use_nesterov"] = physicalSgd->getUseNesterovMomentum();
    j["epoch"] = physicalSgd->getEpoch();

    return j;
}

shared_ptr<Optimizer> Sgd::deserialize(const json &j) {
    if (j.at("optimizer_type").get<string>() != "sgd")
        throw runtime_error("Layer type mismatch in Sgd::deserialize: " + j.at("type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Sgd::deserialize: " + j["version"].get<string>());

    float initialLearningRate = j.at("initial_learning_rate").get<float>();
    float decay = j.at("decay").get<float>();
    float momentum = j.at("momentum").get<float>();
    float useNesterov = j.at("use_nesterov").get<float>();
    float epoch = j.at("epoch").get<float>();

    Sgd sgd;
    sgd.initialLearningRate = initialLearningRate;
    sgd.decay = decay;
    sgd.momentum = momentum;
    sgd.useNesterovMomentum = useNesterov;
    sgd.startResumeEpoch = epoch;
    return sgd.clone();
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("sgd", &Thor::Sgd::deserialize);
    return true;
}();
}  // namespace
