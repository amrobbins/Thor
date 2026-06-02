#include "DeepLearning/Api/Optimizers/RAdam.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

RAdam::RAdam() : Optimizer() {}

RAdam::RAdam(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> RAdam::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    auto physicalRAdam = make_shared<ThorImplementation::RAdam>(getId(), alpha, beta1, beta2, epsilon);
    physicalRAdam->setT(t);
    return physicalRAdam;
}

void RAdam::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void RAdam::setBeta1(float newBeta1, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta1 >= 0.0f && newBeta1 < 1.0f);
    beta1 = newBeta1;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void RAdam::setBeta2(float newBeta2, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta2 >= 0.0f && newBeta2 < 1.0f);
    beta2 = newBeta2;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void RAdam::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float RAdam::getAlpha() { return alpha; }

float RAdam::getBeta1() { return beta1; }

float RAdam::getBeta2() { return beta2; }

float RAdam::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> RAdam::clone() const { return make_shared<RAdam>(*this); }

void RAdam::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement RAdam::updateParameters()");
}

json RAdam::architectureJson() const {
    json j;
    j["optimizer_type"] = string("radam");
    j["version"] = getVersion();
    j["id"] = getId();
    j["t"] = t;
    j["alpha"] = alpha;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    j["epsilon"] = epsilon;
    return j;
}

json RAdam::serialize(thor_file::TarWriter& archiveWriter,
                      Stream stream,
                      shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                      std::string filenamePrefix,
                      bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_radam";
        string mFile = optimizerName + "_m.gds";
        string vFile = optimizerName + "_v.gds";
        j["m_tensor"] = mFile;
        j["v_tensor"] = vFile;

        shared_ptr<ThorImplementation::RAdam> physicalRAdam = dynamic_pointer_cast<ThorImplementation::RAdam>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalRAdam != nullptr);

        optional<ThorImplementation::Tensor> m = physicalRAdam->getParameter("m")->getStorage();
        if (m.has_value())
            archiveWriter.addArchiveFile(mFile, m.value());

        optional<ThorImplementation::Tensor> v = physicalRAdam->getParameter("v")->getStorage();
        if (v.has_value())
            archiveWriter.addArchiveFile(vFile, v.value());

        j["t"] = physicalRAdam->getT();
    }

    return j;
}

shared_ptr<Optimizer> RAdam::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "radam")
        throw runtime_error("Optimizer type mismatch in RAdam::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in RAdam::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float t = j.at("t").get<float>();
    float alpha = j.at("alpha").get<float>();
    float beta1 = j.at("beta1").get<float>();
    float beta2 = j.at("beta2").get<float>();
    float epsilon = j.at("epsilon").get<float>();

    optional<string> mFile;
    optional<string> vFile;
    if (j.contains("m_tensor")) {
        THOR_THROW_IF_FALSE(j.contains("v_tensor"));
        mFile = j.at("m_tensor").get<string>();
        vFile = j.at("v_tensor").get<string>();
    }

    RAdam radam(originalId);
    radam.t = t;
    radam.alpha = alpha;
    radam.beta1 = beta1;
    radam.beta2 = beta2;
    radam.epsilon = epsilon;
    radam.archiveReader = archiveReader;
    radam.mFile = mFile;
    radam.vFile = vFile;
    return radam.clone();
}

vector<Event> RAdam::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                bool isFirstStamp,
                                shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::RAdam> physicalRAdam = dynamic_pointer_cast<ThorImplementation::RAdam>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalRAdam != nullptr);

    ThorImplementation::Tensor m = physicalRAdam->getParameter("m")->getStorage().value();
    ThorImplementation::Tensor v = physicalRAdam->getParameter("v")->getStorage().value();
    Stream stream = physicalRAdam->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::RAdam> sisterPhysicalRAdam = dynamic_pointer_cast<ThorImplementation::RAdam>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalRAdam != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalRAdam->getParameter("m")->getStorage().has_value());
        THOR_THROW_IF_FALSE(sisterPhysicalRAdam->getParameter("v")->getStorage().has_value());
        m.copyFromAsync(sisterPhysicalRAdam->getParameter("m")->getStorage().value(), stream);
        v.copyFromAsync(sisterPhysicalRAdam->getParameter("v")->getStorage().value(), stream);
    } else if (mFile.has_value()) {
        THOR_THROW_IF_FALSE(vFile.has_value());
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(mFile.value(), m);
        archiveReader->registerReadRequest(vFile.value(), v);

        archiveReader = nullptr;
        mFile.reset();
        vFile.reset();
    } else {
        m.memsetAsync(stream, 0);
        v.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("radam", &Thor::RAdam::deserialize);
    return true;
}();
}  // namespace
