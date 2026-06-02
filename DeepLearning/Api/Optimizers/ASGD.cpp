#include "DeepLearning/Api/Optimizers/ASGD.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

ASGD::ASGD() : Optimizer() {}

ASGD::ASGD(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> ASGD::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    auto physicalASGD = make_shared<ThorImplementation::ASGD>(getId(), alpha, lambd, power, t0, weightDecay);
    physicalASGD->setT(t);
    return physicalASGD;
}

void ASGD::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void ASGD::setLambd(float newLambd, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newLambd >= 0.0f);
    lambd = newLambd;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void ASGD::setPower(float newPower, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newPower >= 0.0f);
    power = newPower;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void ASGD::setT0(float newT0, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newT0 >= 1.0f);
    t0 = newT0;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void ASGD::setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newWeightDecay >= 0.0f);
    weightDecay = newWeightDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float ASGD::getAlpha() { return alpha; }

float ASGD::getLambd() { return lambd; }

float ASGD::getPower() { return power; }

float ASGD::getT0() { return t0; }

float ASGD::getWeightDecay() { return weightDecay; }

float ASGD::getT() { return t; }

shared_ptr<Optimizer> ASGD::clone() const { return make_shared<ASGD>(*this); }

void ASGD::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement ASGD::updateParameters()");
}

json ASGD::architectureJson() const {
    json j;
    j["optimizer_type"] = string("asgd");
    j["version"] = getVersion();
    j["id"] = getId();
    j["t"] = t;
    j["alpha"] = alpha;
    j["lambd"] = lambd;
    j["power"] = power;
    j["t0"] = t0;
    j["weight_decay"] = weightDecay;
    return j;
}

json ASGD::serialize(thor_file::TarWriter& archiveWriter,
                     Stream stream,
                     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                     std::string filenamePrefix,
                     bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_asgd";
        string averagedWeightsFile = optimizerName + "_averaged_weights.gds";
        j["averaged_weights_tensor"] = averagedWeightsFile;

        shared_ptr<ThorImplementation::ASGD> physicalASGD = dynamic_pointer_cast<ThorImplementation::ASGD>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalASGD != nullptr);
        optional<ThorImplementation::Tensor> averagedWeights = physicalASGD->getParameter("averaged_weights")->getStorage();
        if (averagedWeights.has_value())
            archiveWriter.addArchiveFile(averagedWeightsFile, averagedWeights.value());

        j["t"] = physicalASGD->getT();
    }

    return j;
}

shared_ptr<Optimizer> ASGD::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "asgd")
        throw runtime_error("Optimizer type mismatch in ASGD::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in ASGD::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float t = j.at("t").get<float>();
    float alpha = j.at("alpha").get<float>();
    float lambd = j.at("lambd").get<float>();
    float power = j.at("power").get<float>();
    float t0 = j.at("t0").get<float>();
    float weightDecay = j.at("weight_decay").get<float>();

    optional<string> averagedWeightsFile;
    if (j.contains("averaged_weights_tensor"))
        averagedWeightsFile = j.at("averaged_weights_tensor").get<string>();

    ASGD asgd(originalId);
    asgd.t = t;
    asgd.alpha = alpha;
    asgd.lambd = lambd;
    asgd.power = power;
    asgd.t0 = t0;
    asgd.weightDecay = weightDecay;
    asgd.archiveReader = archiveReader;
    asgd.averagedWeightsFile = averagedWeightsFile;
    return asgd.clone();
}

vector<Event> ASGD::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                               bool isFirstStamp,
                               shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                               optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::ASGD> physicalASGD = dynamic_pointer_cast<ThorImplementation::ASGD>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalASGD != nullptr);

    ThorImplementation::Tensor averagedWeights = physicalASGD->getParameter("averaged_weights")->getStorage().value();
    Stream stream = physicalASGD->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::ASGD> sisterPhysicalASGD = dynamic_pointer_cast<ThorImplementation::ASGD>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalASGD != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalASGD->getParameter("averaged_weights")->getStorage().has_value());
        averagedWeights.copyFromAsync(sisterPhysicalASGD->getParameter("averaged_weights")->getStorage().value(), stream);
    } else if (averagedWeightsFile.has_value()) {
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(averagedWeightsFile.value(), averagedWeights);

        archiveReader = nullptr;
        averagedWeightsFile.reset();
    } else {
        averagedWeights.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("asgd", &Thor::ASGD::deserialize);
    return true;
}();
}  // namespace
