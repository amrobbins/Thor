#include "DeepLearning/Api/Optimizers/Adagrad.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Adagrad::Adagrad() : Optimizer() {}

Adagrad::Adagrad(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> Adagrad::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::Adagrad>(getId(), alpha, epsilon);
}

void Adagrad::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adagrad::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Adagrad::getAlpha() { return alpha; }

float Adagrad::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> Adagrad::clone() const { return make_shared<Adagrad>(*this); }

void Adagrad::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement Adagrad::updateParameters()");
}

json Adagrad::architectureJson() const {
    json j;
    j["optimizer_type"] = string("adagrad");
    j["version"] = getVersion();
    j["id"] = getId();
    j["alpha"] = alpha;
    j["epsilon"] = epsilon;
    return j;
}

json Adagrad::serialize(thor_file::TarWriter& archiveWriter,
                        Stream stream,
                        shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                        std::string filenamePrefix,
                        bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_adagrad";
        string accumulatorFile = optimizerName + "_accumulator.gds";
        j["accumulator_tensor"] = accumulatorFile;

        shared_ptr<ThorImplementation::Adagrad> physicalAdagrad = dynamic_pointer_cast<ThorImplementation::Adagrad>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalAdagrad != nullptr);
        optional<ThorImplementation::Tensor> accumulator = physicalAdagrad->getParameter("accumulator")->getStorage();
        if (accumulator.has_value())
            archiveWriter.addArchiveFile(accumulatorFile, accumulator.value());
    }

    return j;
}

shared_ptr<Optimizer> Adagrad::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "adagrad")
        throw runtime_error("Layer type mismatch in Adagrad::deserialize: " + j.at("type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Adagrad::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float alpha = j.at("alpha").get<float>();
    float epsilon = j.at("epsilon").get<float>();

    optional<string> accumulatorFile;
    if (j.contains("accumulator_tensor"))
        accumulatorFile = j.at("accumulator_tensor").get<string>();

    Adagrad adagrad(originalId);
    adagrad.alpha = alpha;
    adagrad.epsilon = epsilon;
    adagrad.archiveReader = archiveReader;
    adagrad.accumulatorFile = accumulatorFile;
    return adagrad.clone();
}

vector<Event> Adagrad::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                  optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Adagrad> physicalAdagrad = dynamic_pointer_cast<ThorImplementation::Adagrad>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalAdagrad != nullptr);

    ThorImplementation::Tensor accumulator = physicalAdagrad->getParameter("accumulator")->getStorage().value();
    Stream stream = physicalAdagrad->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::Adagrad> sisterPhysicalAdagrad = dynamic_pointer_cast<ThorImplementation::Adagrad>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalAdagrad != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalAdagrad->getParameter("accumulator")->getStorage().has_value());
        accumulator.copyFromAsync(sisterPhysicalAdagrad->getParameter("accumulator")->getStorage().value(), stream);
    } else if (accumulatorFile.has_value()) {
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(accumulatorFile.value(), accumulator);

        archiveReader = nullptr;
        accumulatorFile.reset();
    } else {
        accumulator.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("adagrad", &Thor::Adagrad::deserialize);
    return true;
}();
}  // namespace
