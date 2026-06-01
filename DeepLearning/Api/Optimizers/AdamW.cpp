#include "DeepLearning/Api/Optimizers/AdamW.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <optional>

using namespace std;
using json = nlohmann::json;

namespace Thor {

AdamW::AdamW() : Optimizer() {}

AdamW::AdamW(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> AdamW::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::AdamW>(getId(), alpha, beta1, beta2, epsilon, weightDecay);
}

void AdamW::setAlpha(float newAlpha, PlacedNetwork *placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void AdamW::setBeta1(float newBeta1, PlacedNetwork *placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta1 >= 0.0f && newBeta1 < 1.0f);
    beta1 = newBeta1;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void AdamW::setBeta2(float newBeta2, PlacedNetwork *placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta2 >= 0.0f && newBeta2 < 1.0f);
    beta2 = newBeta2;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void AdamW::setEpsilon(float newEpsilon, PlacedNetwork *placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void AdamW::setWeightDecay(float newWeightDecay, PlacedNetwork *placedNetwork) {
    THOR_THROW_IF_FALSE(newWeightDecay >= 0.0f);
    weightDecay = newWeightDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float AdamW::getAlpha() { return alpha; }

float AdamW::getBeta1() { return beta1; }

float AdamW::getBeta2() { return beta2; }

float AdamW::getEpsilon() { return epsilon; }

float AdamW::getWeightDecay() { return weightDecay; }

shared_ptr<Optimizer> AdamW::clone() const { return make_shared<AdamW>(*this); }

void AdamW::updateParameters(PlacedNetwork *placedNetwork) {
    // FIXME: re-implement after the current per-parameter optimizer pattern is centralized.
    // This mirrors Adam/Sgd's current placed-network update behavior.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement AdamW::updateParameters()");
}

json AdamW::architectureJson() const {
    json j;
    j["optimizer_type"] = string("adamw");
    j["version"] = getVersion();

    j["id"] = getId();

    j["t"] = t;
    j["alpha"] = alpha;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    j["epsilon"] = epsilon;
    j["weight_decay"] = weightDecay;

    return j;
}

json AdamW::serialize(thor_file::TarWriter &archiveWriter,
                      Stream stream,
                      shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                      std::string filenamePrefix,
                      bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_adamw";
        string mFile = optimizerName + "_m.gds";
        string vFile = optimizerName + "_v.gds";
        j["m_tensor"] = mFile;
        j["v_tensor"] = vFile;

        shared_ptr<ThorImplementation::AdamW> physicalAdamW = dynamic_pointer_cast<ThorImplementation::AdamW>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalAdamW != nullptr);
        std::optional<ThorImplementation::Tensor> m = physicalAdamW->getParameter("m")->getStorage();

        if (m.has_value())
            archiveWriter.addArchiveFile(mFile, m.value());

        std::optional<ThorImplementation::Tensor> v = physicalAdamW->getParameter("v")->getStorage();
        if (v.has_value())
            archiveWriter.addArchiveFile(vFile, v.value());

        j["t"] = physicalAdamW->getT();
    }

    return j;
}

shared_ptr<Optimizer> AdamW::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const json &j, Network *network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "adamw")
        throw runtime_error("Optimizer type mismatch in AdamW::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in AdamW::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float t = j.at("t").get<float>();
    float alpha = j.at("alpha").get<float>();
    float beta1 = j.at("beta1").get<float>();
    float beta2 = j.at("beta2").get<float>();
    float epsilon = j.at("epsilon").get<float>();
    float weightDecay = j.at("weight_decay").get<float>();

    std::optional<string> mFile;
    std::optional<string> vFile;
    if (j.contains("m_tensor")) {
        THOR_THROW_IF_FALSE(j.contains("v_tensor"));
        mFile = j.at("m_tensor").get<string>();
        vFile = j.at("v_tensor").get<string>();
    }

    AdamW adamw(originalId);
    adamw.t = t;
    adamw.alpha = alpha;
    adamw.beta1 = beta1;
    adamw.beta2 = beta2;
    adamw.epsilon = epsilon;
    adamw.weightDecay = weightDecay;
    adamw.archiveReader = archiveReader;
    adamw.mFile = mFile;
    adamw.vFile = vFile;
    return adamw.clone();
}

vector<Event> AdamW::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                bool isFirstStamp,
                                shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                std::optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::AdamW> physicalAdamW = dynamic_pointer_cast<ThorImplementation::AdamW>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalAdamW != nullptr);

    ThorImplementation::Tensor m = physicalAdamW->getParameter("m")->getStorage().value();
    ThorImplementation::Tensor v = physicalAdamW->getParameter("v")->getStorage().value();
    Stream stream = physicalAdamW->getGradientUpdateStream();

    // Parameter values are initialized right now, based on 1 of 3 methods:
    // 1. Copy from another optimizer whose parameters have already been set - when stamping more than one stamp
    //      * So this is once per GPU since multiple stamps on the same GPU share the weights
    // 2. Copy from a file - when loading a saved network with a saved optimizer
    // 3. Run an initializer to set the weights - on an untrained network or when the optimizer has not been saved
    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::AdamW> sisterPhysicalAdamW = dynamic_pointer_cast<ThorImplementation::AdamW>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalAdamW != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalAdamW->getParameter("m")->getStorage().has_value());
        THOR_THROW_IF_FALSE(sisterPhysicalAdamW->getParameter("v")->getStorage().has_value());
        m.copyFromAsync(sisterPhysicalAdamW->getParameter("m")->getStorage().value(), stream);
        v.copyFromAsync(sisterPhysicalAdamW->getParameter("v")->getStorage().value(), stream);
    } else if (mFile.has_value()) {
        THOR_THROW_IF_FALSE(vFile.has_value());
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(mFile.value(), m);
        archiveReader->registerReadRequest(vFile.value(), v);

        // Can't use the files later, they may not still be there
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
    Thor::Optimizer::registerLayer("adamw", &Thor::AdamW::deserialize);
    return true;
}();
}  // namespace
