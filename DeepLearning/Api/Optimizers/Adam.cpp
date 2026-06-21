#include "DeepLearning/Api/Optimizers/Adam.h"
#include <optional>
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Adam::Adam() : Optimizer() {}

Adam::Adam(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> Adam::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    auto physicalAdam = make_shared<ThorImplementation::Adam>(getId(), alpha, beta1, beta2, epsilon, amsgrad);
    physicalAdam->setT(t);
    return physicalAdam;
}

void Adam::setAlpha(float newAlpha, PlacedNetwork *placedNetwork) {
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adam::setBeta1(float newBeta1, PlacedNetwork *placedNetwork) {
    beta1 = newBeta1;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adam::setBeta2(float newBeta2, PlacedNetwork *placedNetwork) {
    beta2 = newBeta2;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adam::setEpsilon(float newEpsilon, PlacedNetwork *placedNetwork) {
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Adam::getAlpha() { return alpha; }

float Adam::getBeta1() { return beta1; }

float Adam::getBeta2() { return beta2; }

float Adam::getEpsilon() { return epsilon; }

bool Adam::getAmsgrad() { return amsgrad; }

shared_ptr<Optimizer> Adam::clone() const { return make_shared<Adam>(*this); }

void Adam::updateParameters(PlacedNetwork *placedNetwork) {
    // FIXME: re-implement
    // THOR_THROW_IF_FALSE(placedNetwork != nullptr);
    // uint32_t numStamps = placedNetwork->getNumStamps();
    // for (uint32_t i = 0; i < numStamps; ++i) {
    //     ThorImplementation::StampedNetwork &stampedNetwork = placedNetwork->getStampedNetwork(i);
    //     uint32_t numTrainableLayers = stampedNetwork.getNumTrainableLayers();
    //     for (uint32_t j = 0; j < numTrainableLayers; ++j) {
    //         shared_ptr<ThorImplementation::TrainableLayer> &trainableLayer = stampedNetwork.getTrainableLayer(j);
    //         std::optional<shared_ptr<ThorImplementation::Optimizer>> maybePhysicalOptimizer = trainableLayer->getOptimizer();
    //         THOR_THROW_IF_FALSE(maybePhysicalOptimizer.has_value());
    //         shared_ptr<ThorImplementation::Optimizer> optimizer = maybePhysicalOptimizer.value();
    //         shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(optimizer);
    //         if (physicalAdam == nullptr || physicalAdam->getId() != getId())
    //             continue;
    //         physicalAdam->setAlpha(alpha);
    //         physicalAdam->setBeta1(beta1);
    //         physicalAdam->setBeta2(beta2);
    //         physicalAdam->setEpsilon(epsilon);
    //     }
    // }
    throw runtime_error("FIXME: Implement Adam::updateParameters()");
}

json Adam::architectureJson() const {
    json j;
    j["optimizer_type"] = string("adam");
    j["version"] = getVersion();

    j["id"] = getId();

    // Get the params and weights from the physical layer.
    // Record the params and dump the weights to files.
    j["t"] = t;
    j["alpha"] = alpha;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    j["epsilon"] = epsilon;
    j["amsgrad"] = amsgrad;

    return j;
}

// In progress update: There will be one optimizer per parameter. Previously this had weightsM and biasesM, not now.
// FIXME: It would be better if m and v where parameters, so Adam inherits from Parameterizable, and they have zerosInitializer
//        Should make that change before furthering the pattern. Optimizer should inherit from parameterizable,
//        so optimizer serialization is standardized.
json Adam::serialize(thor_file::TarWriter &archiveWriter,
                     Stream stream,
                     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                     std::string filenamePrefix,
                     bool saveOptimizerState) const {
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalAdam != nullptr);

        string optimizerName = filenamePrefix + "_adam";
        string mFile = optimizerName + "_m.gds";
        string vFile = optimizerName + "_v.gds";
        j["m_tensor"] = mFile;
        j["v_tensor"] = vFile;

        std::optional<ThorImplementation::Tensor> m = physicalAdam->getParameter("m")->getStorage();
        if (m.has_value())
            archiveWriter.addArchiveFile(mFile, m.value());

        std::optional<ThorImplementation::Tensor> v = physicalAdam->getParameter("v")->getStorage();
        if (v.has_value())
            archiveWriter.addArchiveFile(vFile, v.value());

        if (physicalAdam->getAmsgrad()) {
            string vhatFile = optimizerName + "_vhat.gds";
            j["vhat_tensor"] = vhatFile;
            std::optional<ThorImplementation::Tensor> vhat = physicalAdam->getParameter("vhat")->getStorage();
            if (vhat.has_value())
                archiveWriter.addArchiveFile(vhatFile, vhat.value());
        }

        j["t"] = physicalAdam->getT();
        j["amsgrad"] = physicalAdam->getAmsgrad();
    }

    return j;
}

shared_ptr<Optimizer> Adam::deserialize(shared_ptr<thor_file::TarReader> &archiveReader, const json &j, Network *network) {
    if (j.at("optimizer_type").get<string>() != "adam")
        throw runtime_error("Layer type mismatch in Adam::deserialize: " + j.at("type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Adam::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float t = j.at("t").get<float>();
    float alpha = j.at("alpha").get<float>();
    float beta1 = j.at("beta1").get<float>();
    float beta2 = j.at("beta2").get<float>();
    float epsilon = j.at("epsilon").get<float>();
    bool amsgrad = j.value("amsgrad", false);

    std::optional<string> mFile;
    std::optional<string> vFile;
    std::optional<string> vhatFile;
    std::optional<string> mBiasFile;
    std::optional<string> vBiasFile;
    if (j.contains("m_tensor")) {
        THOR_THROW_IF_FALSE(j.contains("v_tensor"));
        mFile = j.at("m_tensor").get<string>();
        vFile = j.at("v_tensor").get<string>();
        if (amsgrad) {
            THOR_THROW_IF_FALSE(j.contains("vhat_tensor"));
            vhatFile = j.at("vhat_tensor").get<string>();
        }
    }
    if (j.contains("m_bias_tensor")) {
        THOR_THROW_IF_FALSE(j.contains("v_bias_tensor"));
        mBiasFile = j.at("m_bias_tensor").get<string>();
        vBiasFile = j.at("v_bias_tensor").get<string>();
    }

    Adam adam(originalId);
    adam.t = t;
    adam.alpha = alpha;
    adam.beta1 = beta1;
    adam.beta2 = beta2;
    adam.epsilon = epsilon;
    adam.amsgrad = amsgrad;
    adam.archiveReader = archiveReader;
    adam.mFile = mFile;
    adam.vFile = vFile;
    adam.vhatFile = vhatFile;
    adam.mBiasFile = mBiasFile;
    adam.vBiasFile = vBiasFile;
    return adam.clone();
}

vector<Event> Adam::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                               bool isFirstStamp,
                               shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                               std::optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalAdam != nullptr);

    ThorImplementation::Tensor m = physicalAdam->getParameter("m")->getStorage().value();
    ThorImplementation::Tensor v = physicalAdam->getParameter("v")->getStorage().value();
    std::optional<ThorImplementation::Tensor> vhat;
    if (physicalAdam->getAmsgrad()) {
        vhat = physicalAdam->getParameter("vhat")->getStorage().value();
    }
    Stream stream = physicalAdam->getGradientUpdateStream();

    // Parameter values are initialized right now, based on 1 of 3 methods:
    // 1. Copy from another optimizer whose parameters have already been set - when stamping more than one stamp
    //      * So this is once per GPU since multiple stamps on the same GPU share the weights
    // 2. Copy from a file - when loading a saved network with a saved optimizer
    // 3. Run an initializer to set the weights - on an untrained network or when the optimizer has not been saved
    if (!isFirstStamp) {
        // 1. Copy from another layer whose weights have already been set - when stamping more than one stamp
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::Adam> sisterPhysicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalAdam != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalAdam->getParameter("m")->getStorage().has_value());
        THOR_THROW_IF_FALSE(sisterPhysicalAdam->getParameter("v")->getStorage().has_value());
        m.copyFromAsync(sisterPhysicalAdam->getParameter("m")->getStorage().value(), stream);
        v.copyFromAsync(sisterPhysicalAdam->getParameter("v")->getStorage().value(), stream);
        if (physicalAdam->getAmsgrad()) {
            THOR_THROW_IF_FALSE(sisterPhysicalAdam->getAmsgrad());
            THOR_THROW_IF_FALSE(vhat.has_value());
            THOR_THROW_IF_FALSE(sisterPhysicalAdam->getParameter("vhat")->getStorage().has_value());
            vhat->copyFromAsync(sisterPhysicalAdam->getParameter("vhat")->getStorage().value(), stream);
        }
    } else if (mFile.has_value()) {
        THOR_THROW_IF_FALSE(vFile.has_value());
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(mFile.value(), m);
        archiveReader->registerReadRequest(vFile.value(), v);
        if (physicalAdam->getAmsgrad()) {
            THOR_THROW_IF_FALSE(vhat.has_value());
            THOR_THROW_IF_FALSE(vhatFile.has_value());
            archiveReader->registerReadRequest(vhatFile.value(), vhat.value());
        }

        // Can't use the files later, they may not still be there
        archiveReader = nullptr;
        mFile.reset();
        vFile.reset();
        vhatFile.reset();
        mBiasFile.reset();
        vBiasFile.reset();
    } else {
        m.memsetAsync(stream, 0);
        v.memsetAsync(stream, 0);
        if (vhat.has_value())
            vhat->memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("adam", &Thor::Adam::deserialize);
    return true;
}();
}  // namespace
