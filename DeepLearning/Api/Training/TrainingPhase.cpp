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
            const std::string raggedInputVersion = raggedInputJson.at("version").get<std::string>();
            if (raggedInputVersion != "1.0.0" && raggedInputVersion != "1.1.0") {
                throw std::runtime_error("Unsupported TrainingPhase network ragged_network_inputs version: " +
                                         raggedInputVersion);
            }
            const std::string name = raggedInputJson.at("name").get<std::string>();
            const std::string valuesInputName = raggedInputJson.at("values_input_name").get<std::string>();
            const std::string offsetsInputName = raggedInputJson.at("offsets_input_name").get<std::string>();
            const uint64_t valuesTensorId = raggedInputJson.at("values_tensor_id").get<uint64_t>();
            const uint64_t offsetsTensorId = raggedInputJson.at("offsets_tensor_id").get<uint64_t>();
            Tensor values = network->getApiTensorByOriginalId(valuesTensorId);
            Tensor offsets = network->getApiTensorByOriginalId(offsetsTensorId);
            RaggedTensor raggedTensor = raggedInputJson.contains("max_values_per_row")
                ? RaggedTensor(values, offsets, raggedInputJson.at("max_values_per_row").get<uint64_t>())
                : RaggedTensor(values, offsets);
            network->registerRaggedNetworkInput(name, raggedTensor, valuesInputName, offsetsInputName);
        }
    }

    if (networkJson.contains("ragged_network_outputs")) {
        const json& raggedOutputs = networkJson.at("ragged_network_outputs");
        if (!raggedOutputs.is_array()) {
            throw std::runtime_error("TrainingPhase network 'ragged_network_outputs' is not a JSON array.");
        }
        for (const json& raggedOutputJson : raggedOutputs) {
            const std::string raggedOutputVersion = raggedOutputJson.at("version").get<std::string>();
            if (raggedOutputVersion != "1.0.0" && raggedOutputVersion != "1.1.0") {
                throw std::runtime_error("Unsupported TrainingPhase network ragged_network_outputs version: " +
                                         raggedOutputVersion);
            }
            const std::string name = raggedOutputJson.at("name").get<std::string>();
            const std::string valuesOutputName = raggedOutputJson.at("values_output_name").get<std::string>();
            const std::string offsetsOutputName = raggedOutputJson.at("offsets_output_name").get<std::string>();
            const uint64_t valuesTensorId = raggedOutputJson.at("values_tensor_id").get<uint64_t>();
            const uint64_t offsetsTensorId = raggedOutputJson.at("offsets_tensor_id").get<uint64_t>();
            Tensor values = network->getApiTensorByOriginalId(valuesTensorId);
            Tensor offsets = network->getApiTensorByOriginalId(offsetsTensorId);
            RaggedTensor raggedTensor = raggedOutputJson.contains("max_values_per_row")
                ? RaggedTensor(values, offsets, raggedOutputJson.at("max_values_per_row").get<uint64_t>())
                : RaggedTensor(values, offsets);
            network->registerRaggedNetworkOutput(name, raggedTensor, valuesOutputName, offsetsOutputName);
        }
    }

    return network;
}

}  // namespace

TrainingPhase::TrainingPhase(std::string name, std::shared_ptr<Network> network, bool enabled)
    : name(std::move(name)), network(std::move(network)), enabled(enabled), initialized(true) {
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

const std::map<std::string, Tensor>& TrainingPhase::getOutputs() const {
    validate();
    refreshNetworkDerivedCaches();
    return cachedNetworkOutputs;
}

void TrainingPhase::refreshNetworkDerivedCaches() const {
    if (network == nullptr || networkDerivedCachesValid) {
        return;
    }

    cachedNetworkOutputs.clear();

    const uint32_t numLayers = network->getNumLayers();
    for (uint32_t layerIndex = 0; layerIndex < numLayers; ++layerIndex) {
        std::shared_ptr<NetworkOutput> output = std::dynamic_pointer_cast<NetworkOutput>(network->getLayer(layerIndex));
        if (output == nullptr) {
            continue;
        }
        const std::string outputName = output->getName();
        if (outputName.empty()) {
            throw std::runtime_error("TrainingPhase network outputs must have non-empty NetworkOutput names.");
        }
        if (!output->getFeatureInput().has_value()) {
            throw std::runtime_error("TrainingPhase network output '" + outputName + "' has no input tensor.");
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
    if (network == nullptr) {
        throw std::runtime_error("TrainingPhase requires a phase Network.");
    }
    if (network->getNetworkName().empty()) {
        throw std::runtime_error("TrainingPhase network must have a non-empty Network name.");
    }
}

json TrainingPhase::architectureJson() const {
    validate();

    json j;
    j["version"] = getVersion();
    j["name"] = name;
    j["enabled"] = enabled;

    json networkJson = network->architectureJson();
    networkJson["name"] = network->getNetworkName();
    j["network"] = std::move(networkJson);
    return j;
}

std::string TrainingPhase::architectureJsonString() const { return architectureJson().dump(); }

TrainingPhase TrainingPhase::deserialize(const json& j, std::shared_ptr<thor_file::TarReader> archiveReader) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.1.0") {
        throw std::runtime_error("Unsupported TrainingPhase version: " + version);
    }
    if (!j.contains("network")) {
        throw std::runtime_error("TrainingPhase serialized data requires a phase Network.");
    }

    const bool enabled = j.contains("enabled") ? j.at("enabled").get<bool>() : true;
    return TrainingPhase(j.at("name").get<std::string>(), deserializePhaseNetwork(j.at("network"), archiveReader), enabled);
}

}  // namespace Thor
