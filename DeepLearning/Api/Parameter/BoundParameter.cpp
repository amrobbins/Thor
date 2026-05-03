#include "DeepLearning/Api/Parameter/BoundParameter.h"

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

#include "DeepLearning/Api/Network/StampedNetwork.h"

#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {
std::shared_ptr<ThorImplementation::PhysicalParameter> getImplementationParameter(ThorImplementation::StampedNetwork& stampedNetwork,
                                                                                  uint64_t apiLayerId,
                                                                                  const std::string& parameterName) {
    shared_ptr<ThorImplementation::Layer> physicalLayer = stampedNetwork.getPhysicalLayerFromApiLayer(apiLayerId);
    if (physicalLayer == nullptr) {
        throw runtime_error("BoundParameter could not find the physical layer for api layer id " + to_string(apiLayerId) + ".");
    }

    shared_ptr<ThorImplementation::Parameterizable> physicalParameterizable =
        dynamic_pointer_cast<ThorImplementation::Parameterizable>(physicalLayer);
    if (physicalParameterizable == nullptr) {
        throw runtime_error("BoundParameter api layer id " + to_string(apiLayerId) + " is not parameterizable in the placed network.");
    }

    return physicalParameterizable->getParameter(parameterName);
}

std::shared_ptr<ThorImplementation::PhysicalParameter> getImplementationParameter(PlacedNetwork* placedNetwork,
                                                                                  uint64_t apiLayerId,
                                                                                  const std::string& parameterName,
                                                                                  uint64_t stampIndex) {
    if (placedNetwork == nullptr)
        throw runtime_error("BoundParameter is not associated with a placed network.");

    if (stampIndex >= placedNetwork->getNumStamps())
        throw runtime_error("BoundParameter stamp index out of range.");
    ThorImplementation::StampedNetwork stampedNetwork = placedNetwork->getStampedNetwork(stampIndex);
    return getImplementationParameter(stampedNetwork, apiLayerId, parameterName);
}

}  // namespace

BoundParameter::BoundParameter(std::shared_ptr<ParameterSpecification> parameter, PlacedNetwork* placedNetwork, uint64_t apiLayerId)
    : parameter(std::move(parameter)), placedNetwork(placedNetwork), apiLayerId(apiLayerId) {
    if (this->parameter == nullptr)
        throw runtime_error("Cannot create a BoundParameter from a null Parameter.");
    if (this->placedNetwork == nullptr)
        throw runtime_error("Cannot create a BoundParameter without a placed network.");
}

const std::string& BoundParameter::getName() const { return parameter->getName(); }

bool BoundParameter::isTrainable() const { return parameter->isTrainable(); }

bool BoundParameter::isTrainingEnabled() const {
    if (!isTrainable()) {
        return false;
    }
    if (placedNetwork->getNumStamps() == 0) {
        return parameter->isTrainingInitiallyEnabled();
    }
    bool enabled = getImplementationParameter(placedNetwork, apiLayerId, parameter->getName(), 0)->isTrainingEnabled();
    for (uint64_t stampIndex = 1; stampIndex < placedNetwork->getNumStamps(); ++stampIndex) {
        bool otherEnabled = getImplementationParameter(placedNetwork, apiLayerId, parameter->getName(), stampIndex)->isTrainingEnabled();
        if (otherEnabled != enabled) {
            throw runtime_error("BoundParameter found inconsistent training-enabled states across stamps for parameter '" +
                                parameter->getName() + "'.");
        }
    }
    return enabled;
}

void BoundParameter::setTrainingEnabled(bool enabled) {
    if (!isTrainable()) {
        throw runtime_error("Only trainable parameters may toggle training enabled. Parameter '" + parameter->getName() +
                            "' is not trainable.");
    }
    // Parameters are never serialized - their training enabled state follows the BoundParameter once bound.
    // parameter->trainingInitiallyEnabled = enabled;
    for (uint64_t stampIndex = 0; stampIndex < placedNetwork->getNumStamps(); ++stampIndex) {
        const shared_ptr<ThorImplementation::PhysicalParameter>& boundParameter =
            getImplementationParameter(placedNetwork, apiLayerId, parameter->getName(), stampIndex);
        boundParameter->setTrainingEnabled(enabled);
    }
}

bool BoundParameter::hasOptimizer() const { return parameter->hasOptimizer(); }

json BoundParameter::serialize(json parameterJson,
                               std::shared_ptr<ParameterSpecification> parameterSpecification,
                               thor_file::TarWriter& archiveWriter,
                               Stream stream,
                               bool saveOptimizerState,
                               ThorImplementation::StampedNetwork& stampedNetwork,
                               const string& filenamePrefix,
                               const uint64_t apiLayerId) {
    // parameterJson is the contents of json[...][layer][parameters][thisParameter]
    // It is mutated here to include the files that are written to the archive.
    shared_ptr<ThorImplementation::PhysicalParameter> physicalParameter =
        getImplementationParameter(stampedNetwork, apiLayerId, parameterSpecification->getName());

    // Serialize the initializer
    shared_ptr<Initializer> apiInitializer = parameterSpecification->getInitializer();
    assert(apiInitializer != nullptr);
    shared_ptr<ThorImplementation::Initializer> physicalInitializer = physicalParameter->getInitializer();
    assert(physicalInitializer != nullptr);
    json initializerJson = apiInitializer->serialize(archiveWriter, stream, physicalInitializer, filenamePrefix);
    parameterJson["initializer"] = initializerJson;

    // Serialize the optimizer if present
    if (parameterSpecification->hasOptimizer()) {
        std::shared_ptr<Optimizer> optimizer = parameterSpecification->getOptimizer();
        assert(optimizer != nullptr);
        json optimizerJson =
            optimizer->serialize(archiveWriter, stream, physicalParameter->getOptimizer(), filenamePrefix, saveOptimizerState);
        parameterJson["optimizer"] = optimizerJson;
    }

    // Serialize the parameter values
    Optional<ThorImplementation::Tensor> physicalStorage = physicalParameter->getStorage();
    assert(physicalStorage.isPresent());
    string parameterStorageFile = (filenamePrefix + "_parameter_" + parameterSpecification->getName());
    parameterJson["parameter_storage"] = parameterStorageFile;
    archiveWriter.addArchiveFile(parameterStorageFile, physicalStorage.get());

    return parameterJson;
}

}  // namespace Thor
