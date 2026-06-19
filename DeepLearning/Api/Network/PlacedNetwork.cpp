#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include "Utilities/Common/Event.h"

#include <utility>
#include <stdexcept>
#include <set>

using namespace std;
using json = nlohmann::json;

namespace Thor {

PlacedNetwork::~PlacedNetwork() {
    for (uint32_t i = 0; i < stampedNetworks.size(); ++i) {
        // Calls parentCleanup then cleanUp then clears all the shared pointers:
        stampedNetworks[i].clear();
    }
    stampedNetworks.clear();
}

void PlacedNetwork::save(const std::string &directory, bool overwrite, bool saveOptimizerState) {
    network.save(stampedNetworks, directory, overwrite, saveOptimizerState);
}

std::map<std::string, ThorImplementation::Tensor> PlacedNetwork::infer(std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                                                       uint64_t stampIndex) {
    std::map<std::string, ThorImplementation::Tensor> batchOutputs;
    std::map<std::string, Event> outputReadyEvents;
    Event done = submitBatch(stampIndex, std::move(batchInputs), batchOutputs, outputReadyEvents, true);
    done.synchronize();
    return batchOutputs;
}

std::map<std::string, ThorImplementation::Tensor> PlacedNetwork::infer(const Batch& batchInputs, uint64_t stampIndex) {
    std::map<std::string, ThorImplementation::Tensor> batchOutputs;
    std::map<std::string, Event> outputReadyEvents;
    Event done = submitBatch(stampIndex, batchInputs, batchOutputs, outputReadyEvents, true);
    done.synchronize();
    return batchOutputs;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    if (!isInferenceOnly) {
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors()));
    }
    return stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                 batchOutputs,
                                                 outputReadyEvents,
                                                 isInferenceOnly,
                                                 reusableProcessingFinishedEvent,
                                                 waitForOutputsOnProcessingStream);
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 const std::vector<Tensor>& activeTrainingLossRoots,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    if (!isInferenceOnly) {
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(activeTrainingLossRoots);
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
    }
    return stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                 batchOutputs,
                                                 outputReadyEvents,
                                                 isInferenceOnly,
                                                 reusableProcessingFinishedEvent,
                                                 waitForOutputsOnProcessingStream);
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 const Batch& batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    if (!isInferenceOnly) {
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors()));
    }
    return stampedNetworks[stampIndex].sendBatch(batchInputs,
                                                 batchOutputs,
                                                 outputReadyEvents,
                                                 isInferenceOnly,
                                                 reusableProcessingFinishedEvent,
                                                 waitForOutputsOnProcessingStream);
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 const Batch& batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 const std::vector<Tensor>& activeTrainingLossRoots,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    if (!isInferenceOnly) {
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(activeTrainingLossRoots);
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
    }
    return stampedNetworks[stampIndex].sendBatch(batchInputs,
                                                 batchOutputs,
                                                 outputReadyEvents,
                                                 isInferenceOnly,
                                                 reusableProcessingFinishedEvent,
                                                 waitForOutputsOnProcessingStream);
}

std::vector<uint64_t> PlacedNetwork::getActiveTrainingRawLossOriginalIdsForDebug(uint64_t stampIndex) const {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    return stampedNetworks[stampIndex].getActiveTrainingRawLossOriginalIdsForDebug();
}

void PlacedNetwork::extendOutputWritableEvents(uint64_t stampIndex, Event event) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    stampedNetworks[stampIndex].extendOutputWritableEvents(event);
}

std::vector<ParameterReference> PlacedNetwork::getTrainableParameterReferences(bool trainingEnabledOnly) {
    return network.getTrainableParameterReferences(trainingEnabledOnly);
}

BoundParameter PlacedNetwork::resolveParameterReference(const ParameterReference& parameterReference) {
    return network.resolveParameterReference(this, parameterReference);
}

std::vector<BoundParameter> PlacedNetwork::resolveParameterReferences(const std::vector<ParameterReference>& parameterReferences) {
    return network.resolveParameterReferences(this, parameterReferences);
}

bool PlacedNetwork::hasApiTensor(const Tensor& tensor) {
    return tensor.isInitialized() && network.hasApiTensorByOriginalId(tensor.getOriginalId());
}

Tensor PlacedNetwork::resolveApiTensor(const Tensor& tensor) {
    if (!tensor.isInitialized()) {
        throw std::runtime_error("Cannot resolve an uninitialized Tensor against a placed network.");
    }
    return network.resolveApiTensorByOriginalId(tensor.getOriginalId());
}

std::vector<Tensor> PlacedNetwork::resolveApiTensors(const std::vector<Tensor>& tensors) {
    std::vector<Tensor> resolved;
    resolved.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        resolved.push_back(resolveApiTensor(tensor));
    }
    return resolved;
}

bool PlacedNetwork::hasNetworkInput(const std::string& name) {
    if (stampedNetworks.empty()) {
        return false;
    }
    if (stampedNetworks[0].raggedInputNamedShared.count(name) != 0) {
        return true;
    }
    for (const auto& [raggedName, binding] : stampedNetworks[0].raggedInputNamedShared) {
        (void)raggedName;
        if (name == binding.valuesInputName || name == binding.offsetsInputName) {
            return false;
        }
    }
    return stampedNetworks[0].inputNamedShared.count(name) != 0;
}

std::vector<std::string> PlacedNetwork::getNetworkInputNames(uint64_t stampIndex) {
    if (stampIndex >= stampedNetworks.size()) {
        throw std::runtime_error("PlacedNetwork stamp index out of range while listing network inputs.");
    }
    std::set<std::string> raggedPhysicalNames;
    for (const auto& [name, binding] : stampedNetworks[stampIndex].raggedInputNamedShared) {
        (void)name;
        raggedPhysicalNames.insert(binding.valuesInputName);
        raggedPhysicalNames.insert(binding.offsetsInputName);
    }

    std::vector<std::string> names;
    names.reserve(stampedNetworks[stampIndex].inputNamedShared.size() + stampedNetworks[stampIndex].raggedInputNamedShared.size());
    for (const auto& [name, input] : stampedNetworks[stampIndex].inputNamedShared) {
        (void)input;
        if (raggedPhysicalNames.count(name) == 0) {
            names.push_back(name);
        }
    }
    for (const auto& [name, binding] : stampedNetworks[stampIndex].raggedInputNamedShared) {
        (void)binding;
        names.push_back(name);
    }
    return names;
}

}  // namespace Thor
