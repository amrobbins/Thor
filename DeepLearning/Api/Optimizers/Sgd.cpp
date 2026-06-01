#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Sgd::Sgd() : Optimizer() {}

Sgd::Sgd(uint64_t originalId) : Optimizer(originalId) {}

// Sgd::~Sgd() {}

shared_ptr<ThorImplementation::Optimizer> Sgd::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::Sgd>(getId(), initialLearningRate, decay, momentum, useNesterovMomentum, startResumeEpoch);
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
    THOR_THROW_IF_FALSE(newDecay <= 1.0f);
    THOR_THROW_IF_FALSE(newDecay >= 0.0f);

    decay = newDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

/*
 * decay computation will always apply from epoch 0, not from when setLearningRate() is called.
 */
void Sgd::setInitialLearningRate(float newInitialLearningRate, PlacedNetwork *placedNetwork) {
    THOR_THROW_IF_FALSE(newInitialLearningRate >= 0.0f);

    initialLearningRate = newInitialLearningRate;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Sgd::setMomentum(float newMomentum, PlacedNetwork *placedNetwork) {
    THOR_THROW_IF_FALSE(newMomentum >= 0.0f);

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
    // FIXME: re-implement after the current per-parameter optimizer pattern is centralized.
    (void)placedNetwork;
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

    j["id"] = getId();

    j["initial_learning_rate"] = initialLearningRate;
    j["decay"] = decay;
    j["momentum"] = momentum;
    j["use_nesterov"] = useNesterovMomentum;
    j["epoch"] = startResumeEpoch;

    return j;
}

json Sgd::serialize(thor_file::TarWriter &archiveWriter,
                    Stream stream,
                    shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                    std::string filenamePrefix,
                    bool saveOptimizerState) const {
    (void)archiveWriter;
    (void)stream;
    (void)filenamePrefix;

    json j = architectureJson();
    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        shared_ptr<ThorImplementation::Sgd> physicalSgd = dynamic_pointer_cast<ThorImplementation::Sgd>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalSgd != nullptr);
        j["epoch"] = physicalSgd->getEpoch();
    }
    return j;
}

shared_ptr<Optimizer> Sgd::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const json &j, Network *network) {
    (void)archiveReader;
    (void)network;
    if (j.at("optimizer_type").get<string>() != "sgd")
        throw runtime_error("Layer type mismatch in Sgd::deserialize: " + j.at("type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Sgd::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();

    float initialLearningRate = j.at("initial_learning_rate").get<float>();
    float decay = j.at("decay").get<float>();
    float momentum = j.at("momentum").get<float>();
    bool useNesterov = j.at("use_nesterov").get<bool>();
    uint64_t epoch = j.at("epoch").get<uint64_t>();

    Sgd sgd(originalId);
    sgd.originalId = originalId;
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
