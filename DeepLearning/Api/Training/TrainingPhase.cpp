#include "DeepLearning/Api/Training/TrainingPhase.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {
namespace {

std::shared_ptr<Network> deserializePhaseNetwork(const json& networkJson, std::shared_ptr<thor_file::TarReader> archiveReader) {
    const std::string networkName = networkJson.value("name", std::string("training_phase_network"));
    auto network = std::make_shared<Network>(networkName);

    if (networkJson.contains("default_optimizer")) {
        network->setDefaultOptimizer(Optimizer::deserialize(archiveReader, networkJson.at("default_optimizer"), network.get()));
    }

    const json& layers = networkJson.at("layers");
    if (!layers.is_array()) {
        throw std::runtime_error("TrainingPhase network 'layers' is not a JSON array.");
    }

    for (const json& layerJson : layers) {
        Layer::deserialize(archiveReader, layerJson, network.get());
    }

    if (networkJson.contains("ragged_network_inputs")) {
        const json& raggedInputs = networkJson.at("ragged_network_inputs");
        if (!raggedInputs.is_array()) {
            throw std::runtime_error("TrainingPhase network 'ragged_network_inputs' is not a JSON array.");
        }
        for (const json& raggedInputJson : raggedInputs) {
            if (raggedInputJson.at("version").get<std::string>() != "1.0.0") {
                throw std::runtime_error("Unsupported TrainingPhase network ragged_network_inputs version: " +
                                         raggedInputJson.at("version").get<std::string>());
            }
            const std::string name = raggedInputJson.at("name").get<std::string>();
            const std::string valuesInputName = raggedInputJson.at("values_input_name").get<std::string>();
            const std::string offsetsInputName = raggedInputJson.at("offsets_input_name").get<std::string>();
            RaggedTensor raggedTensor = RaggedTensor::deserialize(raggedInputJson.at("ragged_tensor"), archiveReader.get());
            network->registerRaggedNetworkInput(name, raggedTensor, valuesInputName, offsetsInputName);
        }
    }

    return network;
}

}  // namespace

TrainingPhase::TrainingPhase(std::string name, std::shared_ptr<Network> network, bool enabled)
    : name(std::move(name)), network(std::move(network)), enabled(enabled), initialized(true) {
    validate();
}

TrainingPhase::TrainingPhase(std::string name,
                             std::vector<Tensor> lossRoots,
                             std::map<std::string, Tensor> outputs,
                             std::vector<std::string> dependsOn,
                             bool enabled)
    : name(std::move(name)),
      legacyLossRoots(std::move(lossRoots)),
      legacyOutputs(std::move(outputs)),
      legacyDependsOn(std::move(dependsOn)),
      enabled(enabled),
      initialized(true) {
    validate();
}

void TrainingPhase::enable() {
    validate();
    enabled = true;
}

void TrainingPhase::disable() {
    validate();
    enabled = false;
}

void TrainingPhase::setEnabled(bool enabled) {
    validate();
    this->enabled = enabled;
}

const std::vector<Tensor>& TrainingPhase::getLossRoots() const {
    validate();
    if (network == nullptr) {
        return legacyLossRoots;
    }
    refreshNetworkDerivedCaches();
    return cachedNetworkLossRoots;
}

const std::map<std::string, Tensor>& TrainingPhase::getOutputs() const {
    validate();
    if (network == nullptr) {
        return legacyOutputs;
    }
    refreshNetworkDerivedCaches();
    return cachedNetworkOutputs;
}

void TrainingPhase::refreshNetworkDerivedCaches() const {
    if (network == nullptr || networkDerivedCachesValid) {
        return;
    }

    cachedNetworkLossRoots = network->getLossRootTensors();
    cachedNetworkOutputs.clear();

    const uint32_t numLayers = network->getNumLayers();
    for (uint32_t layerIndex = 0; layerIndex < numLayers; ++layerIndex) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(network->getLayer(layerIndex));
        if (output == nullptr) {
            continue;
        }
        const std::string outputName = output->getName();
        if (outputName.empty()) {
            throw std::runtime_error("TrainingPhase network-backed outputs must have non-empty NetworkOutput names.");
        }
        if (!output->getFeatureInput().has_value()) {
            throw std::runtime_error("TrainingPhase network-backed output '" + outputName + "' has no input tensor.");
        }
        if (!cachedNetworkOutputs.emplace(outputName, output->getFeatureInput().value()).second) {
            throw std::runtime_error("TrainingPhase network contains duplicate NetworkOutput name '" + outputName + "'.");
        }
    }

    networkDerivedCachesValid = true;
}

void TrainingPhase::validate() const {
    if (!initialized) {
        return;
    }
    if (name.empty()) {
        throw std::runtime_error("TrainingPhase requires a non-empty name.");
    }

    if (network != nullptr) {
        if (!legacyLossRoots.empty() || !legacyOutputs.empty() || !legacyDependsOn.empty()) {
            throw std::runtime_error("TrainingPhase cannot combine a phase network with legacy loss roots, outputs, or dependencies.");
        }
        if (network->getNetworkName().empty()) {
            throw std::runtime_error("TrainingPhase network must have a non-empty Network name.");
        }
        return;
    }

    for (const Tensor& lossRoot : legacyLossRoots) {
        if (!lossRoot.isInitialized()) {
            throw std::runtime_error("TrainingPhase loss roots must all be initialized tensors.");
        }
    }

    std::set<std::string> seenOutputNames;
    for (const auto& [outputName, outputTensor] : legacyOutputs) {
        if (outputName.empty()) {
            throw std::runtime_error("TrainingPhase output names must be non-empty.");
        }
        if (!outputTensor.isInitialized()) {
            throw std::runtime_error("TrainingPhase outputs must all be initialized tensors.");
        }
        if (!seenOutputNames.insert(outputName).second) {
            throw std::runtime_error("TrainingPhase output names must be unique.");
        }
    }

    std::set<std::string> seenDependencies;
    for (const std::string& dependency : legacyDependsOn) {
        if (dependency.empty()) {
            throw std::runtime_error("TrainingPhase dependency names must be non-empty.");
        }
        if (dependency == name) {
            throw std::runtime_error("TrainingPhase '" + name + "' cannot depend on itself.");
        }
        if (!seenDependencies.insert(dependency).second) {
            throw std::runtime_error("TrainingPhase '" + name + "' contains duplicate dependency '" + dependency + "'.");
        }
    }
}

json TrainingPhase::architectureJson() const {
    validate();

    json j;
    j["version"] = network == nullptr ? "1.0.0" : getVersion();
    j["name"] = name;
    j["enabled"] = enabled;

    if (network != nullptr) {
        json networkJson = network->architectureJson();
        networkJson["name"] = network->getNetworkName();
        j["network"] = std::move(networkJson);
        return j;
    }

    // Legacy serialized shape retained for compatibility with existing saved TrainingPrograms.
    j["loss_roots"] = json::array();
    for (const Tensor& lossRoot : legacyLossRoots) {
        j["loss_roots"].push_back(lossRoot.architectureJson());
    }

    j["outputs"] = json::object();
    for (const auto& [outputName, outputTensor] : legacyOutputs) {
        j["outputs"][outputName] = outputTensor.architectureJson();
    }

    if (!legacyDependsOn.empty()) {
        j["depends_on"] = json::array();
        for (const std::string& dependency : legacyDependsOn) {
            j["depends_on"].push_back(dependency);
        }
    }

    return j;
}

std::string TrainingPhase::architectureJsonString() const { return architectureJson().dump(); }

TrainingPhase TrainingPhase::deserialize(const json& j, std::shared_ptr<thor_file::TarReader> archiveReader) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported TrainingPhase version: " + version);
    }

    const bool enabled = j.contains("enabled") ? j.at("enabled").get<bool>() : true;

    if (j.contains("network")) {
        return TrainingPhase(j.at("name").get<std::string>(), deserializePhaseNetwork(j.at("network"), archiveReader), enabled);
    }

    std::vector<Tensor> lossRoots;
    if (j.contains("loss_roots")) {
        for (const json& lossRootJson : j.at("loss_roots")) {
            lossRoots.push_back(Tensor::deserialize(lossRootJson, archiveReader.get()));
        }
    }

    std::map<std::string, Tensor> outputs;
    if (j.contains("outputs")) {
        for (auto it = j.at("outputs").begin(); it != j.at("outputs").end(); ++it) {
            outputs.emplace(it.key(), Tensor::deserialize(it.value(), archiveReader.get()));
        }
    }

    std::vector<std::string> dependsOn;
    if (j.contains("depends_on")) {
        for (const json& dependencyJson : j.at("depends_on")) {
            dependsOn.push_back(dependencyJson.get<std::string>());
        }
    }

    return TrainingPhase(j.at("name").get<std::string>(), std::move(lossRoots), std::move(outputs), std::move(dependsOn), enabled);
}

}  // namespace Thor
