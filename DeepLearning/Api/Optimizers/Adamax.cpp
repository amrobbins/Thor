#include "DeepLearning/Api/Optimizers/Adamax.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Adamax::Adamax() : Optimizer() {}

Adamax::Adamax(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> Adamax::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    auto physicalAdamax = make_shared<ThorImplementation::Adamax>(getId(), alpha, beta1, beta2, epsilon);
    physicalAdamax->setT(t);
    return physicalAdamax;
}

void Adamax::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adamax::setBeta1(float newBeta1, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta1 >= 0.0f && newBeta1 < 1.0f);
    beta1 = newBeta1;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adamax::setBeta2(float newBeta2, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta2 >= 0.0f && newBeta2 < 1.0f);
    beta2 = newBeta2;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adamax::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Adamax::getAlpha() { return alpha; }

float Adamax::getBeta1() { return beta1; }

float Adamax::getBeta2() { return beta2; }

float Adamax::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> Adamax::clone() const { return make_shared<Adamax>(*this); }

void Adamax::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement Adamax::updateParameters()");
}

json Adamax::architectureJson() const {
    json j;
    j["optimizer_type"] = string("adamax");
    j["version"] = getVersion();
    j["id"] = getId();
    j["t"] = t;
    j["alpha"] = alpha;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    j["epsilon"] = epsilon;
    return j;
}

json Adamax::serialize(thor_file::TarWriter& archiveWriter,
                       Stream stream,
                       shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                       std::string filenamePrefix,
                       bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_adamax";
        string mFile = optimizerName + "_m.gds";
        string uFile = optimizerName + "_u.gds";
        j["m_tensor"] = mFile;
        j["u_tensor"] = uFile;

        shared_ptr<ThorImplementation::Adamax> physicalAdamax = dynamic_pointer_cast<ThorImplementation::Adamax>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalAdamax != nullptr);

        optional<ThorImplementation::Tensor> m = physicalAdamax->getParameter("m")->getStorage();
        if (m.has_value())
            archiveWriter.addArchiveFile(mFile, m.value());

        optional<ThorImplementation::Tensor> u = physicalAdamax->getParameter("u")->getStorage();
        if (u.has_value())
            archiveWriter.addArchiveFile(uFile, u.value());

        j["t"] = physicalAdamax->getT();
    }

    return j;
}

shared_ptr<Optimizer> Adamax::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "adamax")
        throw runtime_error("Optimizer type mismatch in Adamax::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Adamax::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float t = j.at("t").get<float>();
    float alpha = j.at("alpha").get<float>();
    float beta1 = j.at("beta1").get<float>();
    float beta2 = j.at("beta2").get<float>();
    float epsilon = j.at("epsilon").get<float>();

    optional<string> mFile;
    optional<string> uFile;
    if (j.contains("m_tensor")) {
        THOR_THROW_IF_FALSE(j.contains("u_tensor"));
        mFile = j.at("m_tensor").get<string>();
        uFile = j.at("u_tensor").get<string>();
    }

    Adamax adamax(originalId);
    adamax.t = t;
    adamax.alpha = alpha;
    adamax.beta1 = beta1;
    adamax.beta2 = beta2;
    adamax.epsilon = epsilon;
    adamax.archiveReader = archiveReader;
    adamax.mFile = mFile;
    adamax.uFile = uFile;
    return adamax.clone();
}

vector<Event> Adamax::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                 bool isFirstStamp,
                                 shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                 optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Adamax> physicalAdamax = dynamic_pointer_cast<ThorImplementation::Adamax>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalAdamax != nullptr);

    ThorImplementation::Tensor m = physicalAdamax->getParameter("m")->getStorage().value();
    ThorImplementation::Tensor u = physicalAdamax->getParameter("u")->getStorage().value();
    Stream stream = physicalAdamax->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::Adamax> sisterPhysicalAdamax = dynamic_pointer_cast<ThorImplementation::Adamax>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalAdamax != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalAdamax->getParameter("m")->getStorage().has_value());
        THOR_THROW_IF_FALSE(sisterPhysicalAdamax->getParameter("u")->getStorage().has_value());
        m.copyFromAsync(sisterPhysicalAdamax->getParameter("m")->getStorage().value(), stream);
        u.copyFromAsync(sisterPhysicalAdamax->getParameter("u")->getStorage().value(), stream);
    } else if (mFile.has_value()) {
        THOR_THROW_IF_FALSE(uFile.has_value());
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(mFile.value(), m);
        archiveReader->registerReadRequest(uFile.value(), u);

        archiveReader = nullptr;
        mFile.reset();
        uFile.reset();
    } else {
        m.memsetAsync(stream, 0);
        u.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("adamax", &Thor::Adamax::deserialize);
    return true;
}();
}  // namespace
