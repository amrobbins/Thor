#include "DeepLearning/Api/Optimizers/RMSprop.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

RMSprop::RMSprop() : Optimizer() {}

RMSprop::RMSprop(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> RMSprop::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::RMSprop>(getId(), alpha, rho, epsilon);
}

void RMSprop::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void RMSprop::setRho(float newRho, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newRho >= 0.0f && newRho < 1.0f);
    rho = newRho;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void RMSprop::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float RMSprop::getAlpha() { return alpha; }

float RMSprop::getRho() { return rho; }

float RMSprop::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> RMSprop::clone() const { return make_shared<RMSprop>(*this); }

void RMSprop::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement RMSprop::updateParameters()");
}

json RMSprop::architectureJson() const {
    json j;
    j["optimizer_type"] = string("rmsprop");
    j["version"] = getVersion();
    j["id"] = getId();
    j["alpha"] = alpha;
    j["rho"] = rho;
    j["epsilon"] = epsilon;
    return j;
}

json RMSprop::serialize(thor_file::TarWriter& archiveWriter,
                        Stream stream,
                        shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                        std::string filenamePrefix,
                        bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_rmsprop";
        string squareAverageFile = optimizerName + "_square_average.gds";
        j["square_average_tensor"] = squareAverageFile;

        shared_ptr<ThorImplementation::RMSprop> physicalRMSprop = dynamic_pointer_cast<ThorImplementation::RMSprop>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalRMSprop != nullptr);
        optional<ThorImplementation::Tensor> squareAverage = physicalRMSprop->getParameter("square_average")->getStorage();
        if (squareAverage.has_value())
            archiveWriter.addArchiveFile(squareAverageFile, squareAverage.value());
    }

    return j;
}

shared_ptr<Optimizer> RMSprop::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "rmsprop")
        throw runtime_error("Layer type mismatch in RMSprop::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in RMSprop::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float alpha = j.at("alpha").get<float>();
    float rho = j.at("rho").get<float>();
    float epsilon = j.at("epsilon").get<float>();

    optional<string> squareAverageFile;
    if (j.contains("square_average_tensor"))
        squareAverageFile = j.at("square_average_tensor").get<string>();

    RMSprop rmsprop(originalId);
    rmsprop.alpha = alpha;
    rmsprop.rho = rho;
    rmsprop.epsilon = epsilon;
    rmsprop.archiveReader = archiveReader;
    rmsprop.squareAverageFile = squareAverageFile;
    return rmsprop.clone();
}

vector<Event> RMSprop::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                  optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::RMSprop> physicalRMSprop = dynamic_pointer_cast<ThorImplementation::RMSprop>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalRMSprop != nullptr);

    ThorImplementation::Tensor squareAverage = physicalRMSprop->getParameter("square_average")->getStorage().value();
    Stream stream = physicalRMSprop->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::RMSprop> sisterPhysicalRMSprop = dynamic_pointer_cast<ThorImplementation::RMSprop>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalRMSprop != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalRMSprop->getParameter("square_average")->getStorage().has_value());
        squareAverage.copyFromAsync(sisterPhysicalRMSprop->getParameter("square_average")->getStorage().value(), stream);
    } else if (squareAverageFile.has_value()) {
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        archiveReader->registerReadRequest(squareAverageFile.value(), squareAverage);

        archiveReader = nullptr;
        squareAverageFile.reset();
    } else {
        squareAverage.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("rmsprop", &Thor::RMSprop::deserialize);
    return true;
}();
}  // namespace
