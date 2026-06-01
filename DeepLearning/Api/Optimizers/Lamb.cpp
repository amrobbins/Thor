#include "DeepLearning/Api/Optimizers/Lamb.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Lamb::Lamb() : Optimizer() {}

Lamb::Lamb(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> Lamb::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    auto physicalLamb = make_shared<ThorImplementation::Lamb>(getId(), alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon);
    physicalLamb->setT(t);
    return physicalLamb;
}

void Lamb::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lamb::setBeta1(float newBeta1, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta1 >= 0.0f && newBeta1 < 1.0f);
    beta1 = newBeta1;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lamb::setBeta2(float newBeta2, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta2 >= 0.0f && newBeta2 < 1.0f);
    beta2 = newBeta2;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lamb::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lamb::setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newWeightDecay >= 0.0f);
    weightDecay = newWeightDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Lamb::setTrustRatioEpsilon(float newTrustRatioEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newTrustRatioEpsilon > 0.0f);
    trustRatioEpsilon = newTrustRatioEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Lamb::getAlpha() { return alpha; }

float Lamb::getBeta1() { return beta1; }

float Lamb::getBeta2() { return beta2; }

float Lamb::getEpsilon() { return epsilon; }

float Lamb::getWeightDecay() { return weightDecay; }

float Lamb::getTrustRatioEpsilon() { return trustRatioEpsilon; }

shared_ptr<Optimizer> Lamb::clone() const { return make_shared<Lamb>(*this); }

void Lamb::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement Lamb::updateParameters()");
}

json Lamb::architectureJson() const {
    json j;
    j["optimizer_type"] = string("lamb");
    j["version"] = getVersion();
    j["id"] = getId();
    j["t"] = t;
    j["alpha"] = alpha;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    j["epsilon"] = epsilon;
    j["weight_decay"] = weightDecay;
    j["trust_ratio_epsilon"] = trustRatioEpsilon;
    return j;
}

json Lamb::serialize(thor_file::TarWriter& archiveWriter,
                     Stream stream,
                     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                     std::string filenamePrefix,
                     bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_lamb";
        string mFile = optimizerName + "_m.gds";
        string vFile = optimizerName + "_v.gds";
        j["m_tensor"] = mFile;
        j["v_tensor"] = vFile;

        shared_ptr<ThorImplementation::Lamb> physicalLamb = dynamic_pointer_cast<ThorImplementation::Lamb>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalLamb != nullptr);

        optional<ThorImplementation::Tensor> m = physicalLamb->getParameter("m")->getStorage();
        if (m.has_value())
            archiveWriter.addArchiveFile(mFile, m.value());

        optional<ThorImplementation::Tensor> v = physicalLamb->getParameter("v")->getStorage();
        if (v.has_value())
            archiveWriter.addArchiveFile(vFile, v.value());

        j["t"] = physicalLamb->getT();
    }

    return j;
}

shared_ptr<Optimizer> Lamb::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "lamb")
        throw runtime_error("Optimizer type mismatch in Lamb::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Lamb::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float t = j.at("t").get<float>();
    float alpha = j.at("alpha").get<float>();
    float beta1 = j.at("beta1").get<float>();
    float beta2 = j.at("beta2").get<float>();
    float epsilon = j.at("epsilon").get<float>();
    float weightDecay = j.at("weight_decay").get<float>();
    float trustRatioEpsilon = j.value("trust_ratio_epsilon", 1e-6f);

    optional<string> mFile;
    optional<string> vFile;
    if (j.contains("m_tensor")) {
        THOR_THROW_IF_FALSE(j.contains("v_tensor"));
        mFile = j.at("m_tensor").get<string>();
        vFile = j.at("v_tensor").get<string>();
    }

    Lamb lamb(originalId);
    lamb.t = t;
    lamb.alpha = alpha;
    lamb.beta1 = beta1;
    lamb.beta2 = beta2;
    lamb.epsilon = epsilon;
    lamb.weightDecay = weightDecay;
    lamb.trustRatioEpsilon = trustRatioEpsilon;
    lamb.archiveReader = archiveReader;
    lamb.mFile = mFile;
    lamb.vFile = vFile;
    return lamb.clone();
}

vector<Event> Lamb::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                               bool isFirstStamp,
                               shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                               optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Lamb> physicalLamb = dynamic_pointer_cast<ThorImplementation::Lamb>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalLamb != nullptr);

    ThorImplementation::Tensor m = physicalLamb->getParameter("m")->getStorage().value();
    ThorImplementation::Tensor v = physicalLamb->getParameter("v")->getStorage().value();
    Stream stream = physicalLamb->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::Lamb> sisterPhysicalLamb = dynamic_pointer_cast<ThorImplementation::Lamb>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalLamb != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalLamb->getParameter("m")->getStorage().has_value());
        THOR_THROW_IF_FALSE(sisterPhysicalLamb->getParameter("v")->getStorage().has_value());
        m.copyFromAsync(sisterPhysicalLamb->getParameter("m")->getStorage().value(), stream);
        v.copyFromAsync(sisterPhysicalLamb->getParameter("v")->getStorage().value(), stream);
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
    Thor::Optimizer::registerLayer("lamb", &Thor::Lamb::deserialize);
    return true;
}();
}  // namespace
