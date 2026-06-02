#include "DeepLearning/Api/Optimizers/Lars.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Lars::Lars() : Optimizer() {}

Lars::Lars(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> Lars::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::Lars>(
        getId(), alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum);
}

void Lars::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lars::setMomentum(float newMomentum, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newMomentum >= 0.0f);
    momentum = newMomentum;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lars::setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newWeightDecay >= 0.0f);
    weightDecay = newWeightDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lars::setTrustCoefficient(float newTrustCoefficient, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newTrustCoefficient > 0.0f);
    trustCoefficient = newTrustCoefficient;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lars::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lars::setUseNesterovMomentum(bool newUseNesterovMomentum, PlacedNetwork* placedNetwork) {
    useNesterovMomentum = newUseNesterovMomentum;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Lars::getAlpha() { return alpha; }

float Lars::getMomentum() { return momentum; }

float Lars::getWeightDecay() { return weightDecay; }

float Lars::getTrustCoefficient() { return trustCoefficient; }

float Lars::getEpsilon() { return epsilon; }

bool Lars::getUseNesterovMomentum() { return useNesterovMomentum; }

shared_ptr<Optimizer> Lars::clone() const { return make_shared<Lars>(*this); }

void Lars::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement Lars::updateParameters()");
}

json Lars::architectureJson() const {
    json j;
    j["optimizer_type"] = string("lars");
    j["version"] = getVersion();
    j["id"] = getId();
    j["alpha"] = alpha;
    j["momentum"] = momentum;
    j["weight_decay"] = weightDecay;
    j["trust_coefficient"] = trustCoefficient;
    j["epsilon"] = epsilon;
    j["use_nesterov"] = useNesterovMomentum;
    return j;
}

json Lars::serialize(thor_file::TarWriter& archiveWriter,
                     Stream stream,
                     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                     std::string filenamePrefix,
                     bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_lars";
        string velocityFile = optimizerName + "_velocity.gds";
        j["velocity_tensor"] = velocityFile;

        shared_ptr<ThorImplementation::Lars> physicalLars = dynamic_pointer_cast<ThorImplementation::Lars>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalLars != nullptr);

        optional<ThorImplementation::Tensor> velocity = physicalLars->getParameter("velocity")->getStorage();
        if (velocity.has_value())
            archiveWriter.addArchiveFile(velocityFile, velocity.value());
    }

    return j;
}

shared_ptr<Optimizer> Lars::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "lars")
        throw runtime_error("Optimizer type mismatch in Lars::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Lars::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float alpha = j.at("alpha").get<float>();
    float momentum = j.at("momentum").get<float>();
    float weightDecay = j.at("weight_decay").get<float>();
    float trustCoefficient = j.at("trust_coefficient").get<float>();
    float epsilon = j.at("epsilon").get<float>();
    bool useNesterovMomentum = j.at("use_nesterov").get<bool>();

    optional<string> velocityFile;
    if (j.contains("velocity_tensor"))
        velocityFile = j.at("velocity_tensor").get<string>();

    Lars lars(originalId);
    lars.alpha = alpha;
    lars.momentum = momentum;
    lars.weightDecay = weightDecay;
    lars.trustCoefficient = trustCoefficient;
    lars.epsilon = epsilon;
    lars.useNesterovMomentum = useNesterovMomentum;
    lars.archiveReader = archiveReader;
    lars.velocityFile = velocityFile;
    return lars.clone();
}

vector<Event> Lars::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                               bool isFirstStamp,
                               shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                               optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Lars> physicalLars = dynamic_pointer_cast<ThorImplementation::Lars>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalLars != nullptr);

    ThorImplementation::Tensor velocity = physicalLars->getParameter("velocity")->getStorage().value();
    Stream stream = physicalLars->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::Lars> sisterPhysicalLars = dynamic_pointer_cast<ThorImplementation::Lars>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalLars != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalLars->getParameter("velocity")->getStorage().has_value());
        velocity.copyFromAsync(sisterPhysicalLars->getParameter("velocity")->getStorage().value(), stream);
    } else if (velocityFile.has_value()) {
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(velocityFile.value(), velocity);

        archiveReader = nullptr;
        velocityFile.reset();
    } else {
        velocity.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("lars", &Thor::Lars::deserialize);
    return true;
}();
}  // namespace
