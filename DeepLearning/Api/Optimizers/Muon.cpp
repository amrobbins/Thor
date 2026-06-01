#include "DeepLearning/Api/Optimizers/Muon.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Muon::Muon() : Optimizer() {}

Muon::Muon(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<Optimizer> Muon::makeDefaultFallbackOptimizer() { return AdamW::Builder().build(); }

shared_ptr<ThorImplementation::Optimizer> Muon::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    THOR_THROW_IF_FALSE(fallbackOptimizer != nullptr);
    return make_shared<ThorImplementation::Muon>(getId(),
                                                 alpha,
                                                 beta,
                                                 epsilon,
                                                 weightDecay,
                                                 nesterov,
                                                 orthogonalizationOptions,
                                                 fallbackOptimizer->stamp(nullptr));
}

void Muon::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Muon::setBeta(float newBeta, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta >= 0.0f && newBeta < 1.0f);
    beta = newBeta;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Muon::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    orthogonalizationOptions.epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Muon::setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newWeightDecay >= 0.0f);
    weightDecay = newWeightDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Muon::setNesterov(bool newNesterov, PlacedNetwork* placedNetwork) {
    nesterov = newNesterov;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Muon::getAlpha() const { return alpha; }
float Muon::getBeta() const { return beta; }
float Muon::getEpsilon() const { return epsilon; }
float Muon::getWeightDecay() const { return weightDecay; }
bool Muon::getNesterov() const { return nesterov; }
uint32_t Muon::getNumIterations() const { return orthogonalizationOptions.numIterations; }
float Muon::getCoefficientA() const { return static_cast<float>(orthogonalizationOptions.coefficientA); }
float Muon::getCoefficientB() const { return static_cast<float>(orthogonalizationOptions.coefficientB); }
float Muon::getCoefficientC() const { return static_cast<float>(orthogonalizationOptions.coefficientC); }
bool Muon::getTransposeTallMatrices() const { return orthogonalizationOptions.transposeTallMatrices; }
shared_ptr<Optimizer> Muon::getFallbackOptimizer() const { return fallbackOptimizer; }

shared_ptr<Optimizer> Muon::clone() const {
    auto cloned = make_shared<Muon>(*this);
    if (fallbackOptimizer != nullptr)
        cloned->fallbackOptimizer = fallbackOptimizer->clone();
    return cloned;
}

void Muon::updateParameters(PlacedNetwork* placedNetwork) {
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement Muon::updateParameters()");
}

json Muon::architectureJson() const {
    json j;
    j["optimizer_type"] = string("muon");
    j["version"] = getVersion();
    j["id"] = getId();
    j["alpha"] = alpha;
    j["beta"] = beta;
    j["epsilon"] = epsilon;
    j["weight_decay"] = weightDecay;
    j["nesterov"] = nesterov;
    j["num_iterations"] = orthogonalizationOptions.numIterations;
    j["coefficient_a"] = orthogonalizationOptions.coefficientA;
    j["coefficient_b"] = orthogonalizationOptions.coefficientB;
    j["coefficient_c"] = orthogonalizationOptions.coefficientC;
    j["transpose_tall_matrices"] = orthogonalizationOptions.transposeTallMatrices;
    THOR_THROW_IF_FALSE(fallbackOptimizer != nullptr);
    j["fallback_optimizer"] = fallbackOptimizer->architectureJson();
    return j;
}

json Muon::serialize(thor_file::TarWriter& archiveWriter,
                     Stream stream,
                     shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                     string filenamePrefix,
                     bool saveOptimizerState) const {
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        shared_ptr<ThorImplementation::Muon> physicalMuon = dynamic_pointer_cast<ThorImplementation::Muon>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalMuon != nullptr);
        shared_ptr<ThorImplementation::Optimizer> selected = physicalMuon->getSelectedOptimizer();
        THOR_THROW_IF_FALSE(selected != nullptr);

        if (physicalMuon->isUsingMuonMatrixPath()) {
            string optimizerName = filenamePrefix + "_muon";
            string momentumTensorFile = optimizerName + "_momentum.gds";
            j["selected_optimizer"] = string("muon");
            j["momentum_tensor"] = momentumTensorFile;
            optional<ThorImplementation::Tensor> momentum = selected->getParameter("momentum")->getStorage();
            if (momentum.has_value())
                archiveWriter.addArchiveFile(momentumTensorFile, momentum.value());
        } else {
            j["selected_optimizer"] = string("fallback");
            j["fallback_optimizer_state"] = fallbackOptimizer->serialize(
                archiveWriter, stream, selected, filenamePrefix + "_muon_fallback", saveOptimizerState);
        }
    }

    return j;
}

shared_ptr<Optimizer> Muon::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "muon")
        throw runtime_error("Optimizer type mismatch in Muon::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Muon::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    Muon muon(originalId);
    muon.alpha = j.at("alpha").get<float>();
    muon.beta = j.at("beta").get<float>();
    muon.epsilon = j.at("epsilon").get<float>();
    muon.weightDecay = j.at("weight_decay").get<float>();
    muon.nesterov = j.value("nesterov", true);
    muon.orthogonalizationOptions.numIterations = j.at("num_iterations").get<uint32_t>();
    muon.orthogonalizationOptions.coefficientA = j.at("coefficient_a").get<double>();
    muon.orthogonalizationOptions.coefficientB = j.at("coefficient_b").get<double>();
    muon.orthogonalizationOptions.coefficientC = j.at("coefficient_c").get<double>();
    muon.orthogonalizationOptions.epsilon = muon.epsilon;
    muon.orthogonalizationOptions.transposeTallMatrices = j.value("transpose_tall_matrices", true);
    muon.orthogonalizationOptions.computeDType = ThorImplementation::DataType::FP32;
    muon.orthogonalizationOptions.outputDType = ThorImplementation::DataType::FP32;

    if (j.contains("fallback_optimizer_state")) {
        muon.fallbackOptimizer = Optimizer::deserialize(archiveReader, j.at("fallback_optimizer_state"), network);
    } else if (j.contains("fallback_optimizer")) {
        muon.fallbackOptimizer = Optimizer::deserialize(archiveReader, j.at("fallback_optimizer"), network);
    } else {
        muon.fallbackOptimizer = makeDefaultFallbackOptimizer();
    }

    if (j.contains("momentum_tensor")) {
        muon.archiveReader = archiveReader;
        muon.momentumFile = j.at("momentum_tensor").get<string>();
    }

    return muon.clone();
}

vector<Event> Muon::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                               bool isFirstStamp,
                               shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                               optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Muon> physicalMuon = dynamic_pointer_cast<ThorImplementation::Muon>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalMuon != nullptr);
    shared_ptr<ThorImplementation::Optimizer> selected = physicalMuon->getSelectedOptimizer();
    THOR_THROW_IF_FALSE(selected != nullptr);

    shared_ptr<ThorImplementation::Muon> sisterMuon = dynamic_pointer_cast<ThorImplementation::Muon>(sisterPhysicalOptimizer);
    shared_ptr<ThorImplementation::Optimizer> sisterSelected = sisterMuon != nullptr ? sisterMuon->getSelectedOptimizer() : nullptr;

    if (physicalMuon->isUsingFallbackPath()) {
        THOR_THROW_IF_FALSE(fallbackOptimizer != nullptr);
        return fallbackOptimizer->initialize(selected, isFirstStamp, sisterSelected, sisterOptimizerLoadedEvent);
    }

    ThorImplementation::Tensor momentum = selected->getParameter("momentum")->getStorage().value();
    Stream stream = selected->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterSelected != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        THOR_THROW_IF_FALSE(sisterSelected->getParameter("momentum")->getStorage().has_value());
        momentum.copyFromAsync(sisterSelected->getParameter("momentum")->getStorage().value(), stream);
    } else if (momentumFile.has_value()) {
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(momentumFile.value(), momentum);
        archiveReader = nullptr;
        momentumFile.reset();
    } else {
        momentum.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("muon", &Thor::Muon::deserialize);
    return true;
}();
}  // namespace
