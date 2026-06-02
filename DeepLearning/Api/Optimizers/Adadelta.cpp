#include "DeepLearning/Api/Optimizers/Adadelta.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Adadelta::Adadelta() : Optimizer() {}

Adadelta::Adadelta(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> Adadelta::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::Adadelta>(getId(), alpha, rho, epsilon);
}

void Adadelta::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adadelta::setRho(float newRho, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newRho >= 0.0f && newRho < 1.0f);
    rho = newRho;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adadelta::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Adadelta::getAlpha() { return alpha; }

float Adadelta::getRho() { return rho; }

float Adadelta::getEpsilon() { return epsilon; }

shared_ptr<Optimizer> Adadelta::clone() const { return make_shared<Adadelta>(*this); }

void Adadelta::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement Adadelta::updateParameters()");
}

json Adadelta::architectureJson() const {
    json j;
    j["optimizer_type"] = string("adadelta");
    j["version"] = getVersion();
    j["id"] = getId();
    j["alpha"] = alpha;
    j["rho"] = rho;
    j["epsilon"] = epsilon;
    return j;
}

json Adadelta::serialize(thor_file::TarWriter& archiveWriter,
                         Stream stream,
                         shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                         std::string filenamePrefix,
                         bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        string optimizerName = filenamePrefix + "_adadelta";
        string gradientSquareAverageFile = optimizerName + "_gradient_square_average.gds";
        string updateSquareAverageFile = optimizerName + "_update_square_average.gds";
        j["gradient_square_average_tensor"] = gradientSquareAverageFile;
        j["update_square_average_tensor"] = updateSquareAverageFile;

        shared_ptr<ThorImplementation::Adadelta> physicalAdadelta = dynamic_pointer_cast<ThorImplementation::Adadelta>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalAdadelta != nullptr);
        optional<ThorImplementation::Tensor> gradientSquareAverage =
            physicalAdadelta->getParameter("gradient_square_average")->getStorage();
        if (gradientSquareAverage.has_value())
            archiveWriter.addArchiveFile(gradientSquareAverageFile, gradientSquareAverage.value());

        optional<ThorImplementation::Tensor> updateSquareAverage =
            physicalAdadelta->getParameter("update_square_average")->getStorage();
        if (updateSquareAverage.has_value())
            archiveWriter.addArchiveFile(updateSquareAverageFile, updateSquareAverage.value());
    }

    return j;
}

shared_ptr<Optimizer> Adadelta::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "adadelta")
        throw runtime_error("Optimizer type mismatch in Adadelta::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Adadelta::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();
    float alpha = j.at("alpha").get<float>();
    float rho = j.at("rho").get<float>();
    float epsilon = j.at("epsilon").get<float>();

    optional<string> gradientSquareAverageFile;
    if (j.contains("gradient_square_average_tensor"))
        gradientSquareAverageFile = j.at("gradient_square_average_tensor").get<string>();

    optional<string> updateSquareAverageFile;
    if (j.contains("update_square_average_tensor"))
        updateSquareAverageFile = j.at("update_square_average_tensor").get<string>();

    Adadelta adadelta(originalId);
    adadelta.alpha = alpha;
    adadelta.rho = rho;
    adadelta.epsilon = epsilon;
    adadelta.archiveReader = archiveReader;
    adadelta.gradientSquareAverageFile = gradientSquareAverageFile;
    adadelta.updateSquareAverageFile = updateSquareAverageFile;
    return adadelta.clone();
}

vector<Event> Adadelta::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                   bool isFirstStamp,
                                   shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                   optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Adadelta> physicalAdadelta = dynamic_pointer_cast<ThorImplementation::Adadelta>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalAdadelta != nullptr);

    ThorImplementation::Tensor gradientSquareAverage =
        physicalAdadelta->getParameter("gradient_square_average")->getStorage().value();
    ThorImplementation::Tensor updateSquareAverage = physicalAdadelta->getParameter("update_square_average")->getStorage().value();
    Stream stream = physicalAdadelta->getGradientUpdateStream();

    if (!isFirstStamp) {
        THOR_THROW_IF_FALSE(sisterPhysicalOptimizer != nullptr);
        if (sisterOptimizerLoadedEvent.has_value())
            stream.waitEvent(sisterOptimizerLoadedEvent.value());
        shared_ptr<ThorImplementation::Adadelta> sisterPhysicalAdadelta =
            dynamic_pointer_cast<ThorImplementation::Adadelta>(sisterPhysicalOptimizer);
        THOR_THROW_IF_FALSE(sisterPhysicalAdadelta != nullptr);
        THOR_THROW_IF_FALSE(sisterPhysicalAdadelta->getParameter("gradient_square_average")->getStorage().has_value());
        THOR_THROW_IF_FALSE(sisterPhysicalAdadelta->getParameter("update_square_average")->getStorage().has_value());
        gradientSquareAverage.copyFromAsync(
            sisterPhysicalAdadelta->getParameter("gradient_square_average")->getStorage().value(), stream);
        updateSquareAverage.copyFromAsync(sisterPhysicalAdadelta->getParameter("update_square_average")->getStorage().value(), stream);
    } else if (gradientSquareAverageFile.has_value() || updateSquareAverageFile.has_value()) {
        THOR_THROW_IF_FALSE(archiveReader != nullptr);
        THOR_THROW_IF_FALSE(gradientSquareAverageFile.has_value());
        THOR_THROW_IF_FALSE(updateSquareAverageFile.has_value());
        archiveReader->registerReadRequest(gradientSquareAverageFile.value(), gradientSquareAverage);
        archiveReader->registerReadRequest(updateSquareAverageFile.value(), updateSquareAverage);

        archiveReader = nullptr;
        gradientSquareAverageFile.reset();
        updateSquareAverageFile.reset();
    } else {
        gradientSquareAverage.memsetAsync(stream, 0);
        updateSquareAverage.memsetAsync(stream, 0);
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("adadelta", &Thor::Adadelta::deserialize);
    return true;
}();
}  // namespace
