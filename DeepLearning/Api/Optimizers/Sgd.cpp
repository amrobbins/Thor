#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

// Sgd::~Sgd() {}

shared_ptr<ThorImplementation::Optimizer> Sgd::stamp(shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) {
    return make_shared<ThorImplementation::Sgd>(
        getId(), trainableLayer, initialLearningRate, decay, momentum, useNesterovMomentum, startResumeEpoch);
}

void Sgd::setConstantLearningRate(float newCurrentLearningRate, PlacedNetwork *placedNetwork) {
    setDecay(0.0f, placedNetwork);
    setInitialLearningRate(newCurrentLearningRate, placedNetwork);
}

shared_ptr<Optimizer> Sgd::clone() const { return make_shared<Sgd>(*this); }

/*
 * decay will still apply from epoch 0.
 */
void Sgd::setDecay(float newDecay, PlacedNetwork *placedNetwork) {
    assert(decay < 1.0);

    decay = newDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

/*
 * decay computation will always apply from epoch 0, not from when setLearningRate() is called.
 */
void Sgd::setInitialLearningRate(float newInitialLearningRate, PlacedNetwork *placedNetwork) {
    assert(newInitialLearningRate >= 0.0f);

    initialLearningRate = newInitialLearningRate;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Sgd::setMomentum(float newMomentum, PlacedNetwork *placedNetwork) {
    assert(momentum >= 0.0f);

    momentum = newMomentum;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Sgd::setUseNesterovMomentum(bool newUseNesterovMomentum, PlacedNetwork *placedNetwork) {
    useNesterovMomentum = newUseNesterovMomentum;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Sgd::updateParameters(PlacedNetwork *placedNetwork) {
    assert(placedNetwork != nullptr);
    uint32_t numStamps = placedNetwork->getNumStamps();
    for (uint32_t i = 0; i < numStamps; ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = placedNetwork->getStampedNetwork(i);
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

uint64_t Sgd::getEpoch() { return startResumeEpoch; }

json Sgd::architectureJson() const {
    // FIXME: I need to save id, then I will need originalId like layer.

    json j;
    j["optimizer_type"] = string("sgd");
    j["version"] = getVersion();

    j["initial_learning_rate"] = initialLearningRate;
    j["decay"] = decay;
    j["momentum"] = momentum;
    j["use_nesterov"] = useNesterovMomentum;
    j["epoch"] = 0;

    return j;
}

json Sgd::serialize(thor_file::TarWriter &archiveWriter,
                    Stream stream,
                    TrainableWeightsBiasesLayer const *owningLayer,
                    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalOwningLayer,
                    bool saveOptimizerState) const {
    json j = architectureJson();

    if (physicalOwningLayer != nullptr && saveOptimizerState) {
        shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalOwningLayer->getOptimizer();
        shared_ptr<ThorImplementation::Sgd> physicalSgd = dynamic_pointer_cast<ThorImplementation::Sgd>(physicalOptimizer);
        assert(physicalSgd != nullptr);
        j["epoch"] = physicalSgd->getEpoch();
    }

    return j;
}

shared_ptr<Optimizer> Sgd::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const json &j, Network *network) {
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
