#include "DeepLearning/Api/Optimizers/NAdam.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

NAdam::NAdam() : Optimizer() {}

NAdam::NAdam(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> NAdam::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    auto physicalNAdam = make_shared<ThorImplementation::NAdam>(getId(), alpha, beta1, beta2, epsilon);
    physicalNAdam->setT(t);
    return physicalNAdam;
}

void NAdam::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void NAdam::setBeta1(float newBeta1, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta1 >= 0.0f && newBeta1 < 1.0f);
    beta1 = newBeta1;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void NAdam::setBeta2(float newBeta2, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta2 >= 0.0f && newBeta2 < 1.0f);
    beta2 = newBeta2;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void NAdam::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float NAdam::getAlpha() { return alpha; }

float NAdam::getBeta1() { return beta1; }

float NAdam::getBeta2() { return beta2; }

float NAdam::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> NAdam::clone() const { return make_shared<NAdam>(*this); }

void NAdam::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement NAdam::updateParameters()");
}

json NAdam::architectureJson() const {
    json j;
    j["optimizer_type"] = string("nadam");
    j["version"] = getVersion();
    j["id"] = getId();
    j["t"] = t;
    j["alpha"] = alpha;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    j["epsilon"] = epsilon;
    return j;
}

json NAdam::serialize(thor_file::TarWriter& archiveWriter,
                      Stream stream,
                      shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                      std::string filenamePrefix,
                      bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_nadam";
        string mFile = optimizerName + "_m.gds";
        string vFile = optimizerName + "_v.gds";
        j["m_tensor"] = mFile;
        j["v_tensor"] = vFile;

        shared_ptr<ThorImplementation::NAdam> physicalNAdam = dynamic_pointer_cast<ThorImplementation::NAdam>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalNAdam != nullptr);

        optional<ThorImplementation::Tensor> m = physicalNAdam->getParameter("m")->getStorage();
        if (m.has_value())
            archiveWriter.addArchiveFile(mFile, m.value());

        optional<ThorImplementation::Tensor> v = physicalNAdam->getParameter("v")->getStorage();
        if (v.has_value())
            archiveWriter.addArchiveFile(vFile, v.value());

        j["t"] = physicalNAdam->getT();
    }

    return j;
}

shared_ptr<Optimizer> NAdam::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "nadam")
        throw runtime_error("Optimizer type mismatch in NAdam::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in NAdam::deserialize: " + j["version"].get<string>());

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

    NAdam nadam(originalId);
    nadam.t = t;
    nadam.alpha = alpha;
    nadam.beta1 = beta1;
    nadam.beta2 = beta2;
    nadam.epsilon = epsilon;
    nadam.archiveReader = archiveReader;
    nadam.mFile = mFile;
    nadam.vFile = vFile;
    return nadam.clone();
}

vector<Event> NAdam::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                bool isFirstStamp,
                                shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::NAdam> physicalNAdam = dynamic_pointer_cast<ThorImplementation::NAdam>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalNAdam != nullptr);

    ThorImplementation::Tensor m = physicalNAdam->getParameter("m")->getStorage().value();
    ThorImplementation::Tensor v = physicalNAdam->getParameter("v")->getStorage().value();
    Stream stream = physicalNAdam->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::NAdam> sisterPhysicalNAdam = dynamic_pointer_cast<ThorImplementation::NAdam>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalNAdam != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalNAdam->getParameter("m")->getStorage().has_value());
        THOR_THROW_IF_FALSE(sisterPhysicalNAdam->getParameter("v")->getStorage().has_value());
        m.copyFromAsync(sisterPhysicalNAdam->getParameter("m")->getStorage().value(), stream);
        v.copyFromAsync(sisterPhysicalNAdam->getParameter("v")->getStorage().value(), stream);
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
    Thor::Optimizer::registerLayer("nadam", &Thor::NAdam::deserialize);
    return true;
}();
}  // namespace
