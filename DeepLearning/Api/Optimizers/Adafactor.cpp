#include "DeepLearning/Api/Optimizers/Adafactor.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

using namespace std;
using json = nlohmann::json;

namespace Thor {

Adafactor::Adafactor() : Optimizer() {}

Adafactor::Adafactor(uint64_t originalId) : Optimizer(originalId) {}

shared_ptr<ThorImplementation::Optimizer> Adafactor::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::Adafactor>(getId(), alpha, beta2, epsilon, weightDecay, factorSecondMoment);
}

void Adafactor::setAlpha(float newAlpha, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newAlpha > 0.0f);
    alpha = newAlpha;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adafactor::setBeta2(float newBeta2, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newBeta2 >= 0.0f && newBeta2 < 1.0f);
    beta2 = newBeta2;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adafactor::setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newEpsilon > 0.0f);
    epsilon = newEpsilon;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adafactor::setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork) {
    THOR_THROW_IF_FALSE(newWeightDecay >= 0.0f);
    weightDecay = newWeightDecay;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

void Adafactor::setFactorSecondMoment(bool newFactorSecondMoment, PlacedNetwork* placedNetwork) {
    factorSecondMoment = newFactorSecondMoment;
    if (placedNetwork != nullptr)
        updateParameters(placedNetwork);
}

float Adafactor::getAlpha() { return alpha; }

float Adafactor::getBeta2() { return beta2; }

float Adafactor::getEpsilon() { return epsilon; }

float Adafactor::getWeightDecay() { return weightDecay; }

bool Adafactor::getFactorSecondMoment() { return factorSecondMoment; }

shared_ptr<Optimizer> Adafactor::clone() const { return make_shared<Adafactor>(*this); }

void Adafactor::updateParameters(PlacedNetwork* placedNetwork) {
    // FIXME: re-implement once API-side post-placement optimizer mutation is restored for all optimizers.
    (void)placedNetwork;
    throw runtime_error("FIXME: Implement Adafactor::updateParameters()");
}

json Adafactor::architectureJson() const {
    json j;
    j["optimizer_type"] = string("adafactor");
    j["version"] = getVersion();
    j["id"] = getId();
    j["alpha"] = alpha;
    j["beta2"] = beta2;
    j["epsilon"] = epsilon;
    j["weight_decay"] = weightDecay;
    j["factor_second_moment"] = factorSecondMoment;
    return j;
}

json Adafactor::serialize(thor_file::TarWriter& archiveWriter,
                          Stream stream,
                          shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                          string filenamePrefix,
                          bool saveOptimizerState) const {
    (void)stream;
    json j = architectureJson();

    if (saveOptimizerState) {
        THOR_THROW_IF_FALSE(physicalOptimizer != nullptr);
        THOR_THROW_IF_FALSE(!filenamePrefix.empty());

        shared_ptr<ThorImplementation::Adafactor> physicalAdafactor = dynamic_pointer_cast<ThorImplementation::Adafactor>(physicalOptimizer);
        THOR_THROW_IF_FALSE(physicalAdafactor != nullptr);
        shared_ptr<ThorImplementation::Optimizer> selected = physicalAdafactor->getSelectedOptimizer();
        THOR_THROW_IF_FALSE(selected != nullptr);

        string optimizerName = filenamePrefix + "_adafactor";
        if (physicalAdafactor->isUsingFactoredPath()) {
            string rowFile = optimizerName + "_row_second_moment.gds";
            string columnFile = optimizerName + "_column_second_moment.gds";
            j["selected_optimizer"] = string("factored");
            j["row_second_moment_tensor"] = rowFile;
            j["column_second_moment_tensor"] = columnFile;

            optional<ThorImplementation::Tensor> rowSecondMoment = selected->getParameter("row_second_moment")->getStorage();
            if (rowSecondMoment.has_value())
                archiveWriter.addArchiveFile(rowFile, rowSecondMoment.value());

            optional<ThorImplementation::Tensor> columnSecondMoment = selected->getParameter("column_second_moment")->getStorage();
            if (columnSecondMoment.has_value())
                archiveWriter.addArchiveFile(columnFile, columnSecondMoment.value());
        } else {
            string secondMomentFile = optimizerName + "_second_moment.gds";
            j["selected_optimizer"] = string("unfactored");
            j["second_moment_tensor"] = secondMomentFile;

            optional<ThorImplementation::Tensor> secondMoment = selected->getParameter("second_moment")->getStorage();
            if (secondMoment.has_value())
                archiveWriter.addArchiveFile(secondMomentFile, secondMoment.value());
        }
    }

    return j;
}

shared_ptr<Optimizer> Adafactor::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    (void)network;
    if (j.at("optimizer_type").get<string>() != "adafactor")
        throw runtime_error("Optimizer type mismatch in Adafactor::deserialize: " + j.at("optimizer_type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Adafactor::deserialize: " + j["version"].get<string>());

    uint64_t originalId = j.at("id").get<uint64_t>();

    Adafactor adafactor(originalId);
    adafactor.alpha = j.at("alpha").get<float>();
    adafactor.beta2 = j.at("beta2").get<float>();
    adafactor.epsilon = j.at("epsilon").get<float>();
    adafactor.weightDecay = j.value("weight_decay", 0.0f);
    adafactor.factorSecondMoment = j.value("factor_second_moment", true);

    if (j.contains("selected_optimizer"))
        adafactor.selectedOptimizerStateKind = j.at("selected_optimizer").get<string>();

    if (j.contains("second_moment_tensor"))
        adafactor.secondMomentFile = j.at("second_moment_tensor").get<string>();
    if (j.contains("row_second_moment_tensor"))
        adafactor.rowSecondMomentFile = j.at("row_second_moment_tensor").get<string>();
    if (j.contains("column_second_moment_tensor"))
        adafactor.columnSecondMomentFile = j.at("column_second_moment_tensor").get<string>();

    if (adafactor.secondMomentFile.has_value() || adafactor.rowSecondMomentFile.has_value() || adafactor.columnSecondMomentFile.has_value())
        adafactor.archiveReader = archiveReader;

    return adafactor.clone();
}

vector<Event> Adafactor::initialize(shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                    bool isFirstStamp,
                                    shared_ptr<ThorImplementation::Optimizer> sisterPhysicalOptimizer,
                                    optional<Event> sisterOptimizerLoadedEvent) {
    shared_ptr<ThorImplementation::Adafactor> physicalAdafactor = dynamic_pointer_cast<ThorImplementation::Adafactor>(physicalOptimizer);
    THOR_THROW_IF_FALSE(physicalAdafactor != nullptr);
    shared_ptr<ThorImplementation::Optimizer> selected = physicalAdafactor->getSelectedOptimizer();
    THOR_THROW_IF_FALSE(selected != nullptr);

    shared_ptr<ThorImplementation::Adafactor> sisterAdafactor = dynamic_pointer_cast<ThorImplementation::Adafactor>(sisterPhysicalOptimizer);
    shared_ptr<ThorImplementation::Optimizer> sisterSelected = sisterAdafactor != nullptr ? sisterAdafactor->getSelectedOptimizer() : nullptr;

    Stream stream = selected->getGradientUpdateStream();

    if (physicalAdafactor->isUsingFactoredPath()) {
        if (selectedOptimizerStateKind.has_value())
            THOR_THROW_IF_FALSE(selectedOptimizerStateKind.value() == "factored");
        THOR_THROW_IF_FALSE(!secondMomentFile.has_value());

        ThorImplementation::Tensor rowSecondMoment = selected->getParameter("row_second_moment")->getStorage().value();
        ThorImplementation::Tensor columnSecondMoment = selected->getParameter("column_second_moment")->getStorage().value();

        if (!isFirstStamp) {
            THOR_THROW_IF_FALSE(sisterSelected != nullptr);
            if (sisterOptimizerLoadedEvent.has_value())
                stream.waitEvent(sisterOptimizerLoadedEvent.value());
            THOR_THROW_IF_FALSE(sisterSelected->getParameter("row_second_moment")->getStorage().has_value());
            THOR_THROW_IF_FALSE(sisterSelected->getParameter("column_second_moment")->getStorage().has_value());
            rowSecondMoment.copyFromAsync(sisterSelected->getParameter("row_second_moment")->getStorage().value(), stream);
            columnSecondMoment.copyFromAsync(sisterSelected->getParameter("column_second_moment")->getStorage().value(), stream);
        } else if (rowSecondMomentFile.has_value() || columnSecondMomentFile.has_value()) {
            THOR_THROW_IF_FALSE(archiveReader != nullptr);
            THOR_THROW_IF_FALSE(rowSecondMomentFile.has_value());
            THOR_THROW_IF_FALSE(columnSecondMomentFile.has_value());
            archiveReader->registerReadRequest(rowSecondMomentFile.value(), rowSecondMoment);
            archiveReader->registerReadRequest(columnSecondMomentFile.value(), columnSecondMoment);

            archiveReader = nullptr;
            rowSecondMomentFile.reset();
            columnSecondMomentFile.reset();
            selectedOptimizerStateKind.reset();
        } else {
            rowSecondMoment.memsetAsync(stream, 0);
            columnSecondMoment.memsetAsync(stream, 0);
        }
    } else {
        if (selectedOptimizerStateKind.has_value())
            THOR_THROW_IF_FALSE(selectedOptimizerStateKind.value() == "unfactored");
        THOR_THROW_IF_FALSE(!rowSecondMomentFile.has_value());
        THOR_THROW_IF_FALSE(!columnSecondMomentFile.has_value());

        ThorImplementation::Tensor secondMoment = selected->getParameter("second_moment")->getStorage().value();

        if (!isFirstStamp) {
            THOR_THROW_IF_FALSE(sisterSelected != nullptr);
            if (sisterOptimizerLoadedEvent.has_value())
                stream.waitEvent(sisterOptimizerLoadedEvent.value());
            THOR_THROW_IF_FALSE(sisterSelected->getParameter("second_moment")->getStorage().has_value());
            secondMoment.copyFromAsync(sisterSelected->getParameter("second_moment")->getStorage().value(), stream);
        } else if (secondMomentFile.has_value()) {
            THOR_THROW_IF_FALSE(archiveReader != nullptr);
            archiveReader->registerReadRequest(secondMomentFile.value(), secondMoment);

            archiveReader = nullptr;
            secondMomentFile.reset();
            selectedOptimizerStateKind.reset();
        } else {
            secondMoment.memsetAsync(stream, 0);
        }
    }

    return {stream.putEvent(false, true)};
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Optimizer::registerLayer("adafactor", &Thor::Adafactor::deserialize);
    return true;
}();
}  // namespace
