#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "Utilities/Common/Event.h"

#include <utility>
#include <stdexcept>
#include <set>
#include <chrono>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

uint64_t elapsedMicros(std::chrono::high_resolution_clock::time_point start,
                       std::chrono::high_resolution_clock::time_point finish) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
}

std::shared_ptr<ThorImplementation::PhysicalParameter> getPhysicalParameter(ThorImplementation::StampedNetwork& stampedNetwork,
                                                                            const ParameterReference& parameterReference) {
    std::shared_ptr<ThorImplementation::Layer> physicalLayer =
        stampedNetwork.getPhysicalLayerFromApiLayer(parameterReference.getParameterizableId());
    if (physicalLayer == nullptr) {
        throw std::runtime_error("Could not find physical layer for api layer id " +
                                 std::to_string(parameterReference.getParameterizableId()) +
                                 " while copying placed-network training state.");
    }

    std::shared_ptr<ThorImplementation::Parameterizable> physicalParameterizable =
        std::dynamic_pointer_cast<ThorImplementation::Parameterizable>(physicalLayer);
    if (physicalParameterizable == nullptr) {
        throw std::runtime_error("Physical layer for api layer id " +
                                 std::to_string(parameterReference.getParameterizableId()) +
                                 " is not parameterizable while copying placed-network training state.");
    }

    return physicalParameterizable->getParameter(parameterReference.getParameterName());
}

void copyTensorState(ThorImplementation::Tensor destination, ThorImplementation::Tensor source, Stream& stream, const std::string& description) {
    if (destination.getDescriptor() != source.getDescriptor()) {
        throw std::runtime_error("Cannot copy placed-network training state for " + description +
                                 ": source and destination tensor descriptors differ.");
    }
    destination.copyFromAsync(source, stream);
}

void copyOptimizerTensorState(ThorImplementation::Optimizer& destination, ThorImplementation::Optimizer& source, Stream& stream, const std::string& description) {
    if (!destination.isCompiled() || !source.isCompiled()) {
        return;
    }

    const std::vector<std::string> parameterNames = destination.getOptimizerParameterNames();
    for (const std::string& parameterName : parameterNames) {
        ThorImplementation::Tensor destinationTensor = destination.getOptimizerParameterTensor(parameterName);
        ThorImplementation::Tensor sourceTensor = source.getOptimizerParameterTensor(parameterName);
        copyTensorState(destinationTensor, sourceTensor, stream, description + " optimizer parameter '" + parameterName + "'");
    }
}

}  // namespace

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
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = std::chrono::high_resolution_clock::now();
    if (!isInferenceOnly) {
        const auto activeRootsStart = std::chrono::high_resolution_clock::now();
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors());
        const auto activeRootsFinish = std::chrono::high_resolution_clock::now();
        const auto setActiveRootsStart = std::chrono::high_resolution_clock::now();
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = std::chrono::high_resolution_clock::now();
        if (submitTiming != nullptr) {
            submitTiming->activeLossRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveLossRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeLossRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = std::chrono::high_resolution_clock::now();
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = std::chrono::high_resolution_clock::now();
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                 const std::map<std::string, Event>& inputReadyEvents,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = std::chrono::high_resolution_clock::now();
    if (!isInferenceOnly) {
        const auto activeRootsStart = std::chrono::high_resolution_clock::now();
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors());
        const auto activeRootsFinish = std::chrono::high_resolution_clock::now();
        const auto setActiveRootsStart = std::chrono::high_resolution_clock::now();
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = std::chrono::high_resolution_clock::now();
        if (submitTiming != nullptr) {
            submitTiming->activeLossRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveLossRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeLossRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = std::chrono::high_resolution_clock::now();
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                                           inputReadyEvents,
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = std::chrono::high_resolution_clock::now();
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 const std::vector<Tensor>& activeTrainingLossRoots,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = std::chrono::high_resolution_clock::now();
    if (!isInferenceOnly) {
        const auto activeRootsStart = std::chrono::high_resolution_clock::now();
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(activeTrainingLossRoots);
        const auto activeRootsFinish = std::chrono::high_resolution_clock::now();
        const auto setActiveRootsStart = std::chrono::high_resolution_clock::now();
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = std::chrono::high_resolution_clock::now();
        if (submitTiming != nullptr) {
            submitTiming->activeLossRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveLossRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeLossRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = std::chrono::high_resolution_clock::now();
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(std::move(batchInputs),
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = std::chrono::high_resolution_clock::now();
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 const Batch& batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = std::chrono::high_resolution_clock::now();
    if (!isInferenceOnly) {
        const auto activeRootsStart = std::chrono::high_resolution_clock::now();
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(network.getLossRootTensors());
        const auto activeRootsFinish = std::chrono::high_resolution_clock::now();
        const auto setActiveRootsStart = std::chrono::high_resolution_clock::now();
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = std::chrono::high_resolution_clock::now();
        if (submitTiming != nullptr) {
            submitTiming->activeLossRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveLossRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeLossRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = std::chrono::high_resolution_clock::now();
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(batchInputs,
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = std::chrono::high_resolution_clock::now();
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

Event PlacedNetwork::submitBatch(uint64_t stampIndex,
                                 const Batch& batchInputs,
                                 std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                                 std::map<std::string, Event>& outputReadyEvents,
                                 bool isInferenceOnly,
                                 const std::vector<Tensor>& activeTrainingLossRoots,
                                 Event* reusableProcessingFinishedEvent,
                                 bool waitForOutputsOnProcessingStream,
                                 ThorImplementation::BatchSubmissionTiming* submitTiming,
                                 std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    const auto totalStart = std::chrono::high_resolution_clock::now();
    if (!isInferenceOnly) {
        const auto activeRootsStart = std::chrono::high_resolution_clock::now();
        std::vector<Tensor> activeRawLossRoots = network.getRawLossTensorsForTrainingRoots(activeTrainingLossRoots);
        const auto activeRootsFinish = std::chrono::high_resolution_clock::now();
        const auto setActiveRootsStart = std::chrono::high_resolution_clock::now();
        stampedNetworks[stampIndex].setActiveTrainingLossRoots(activeRawLossRoots);
        const auto setActiveRootsFinish = std::chrono::high_resolution_clock::now();
        if (submitTiming != nullptr) {
            submitTiming->activeLossRootsMicros += elapsedMicros(activeRootsStart, activeRootsFinish);
            submitTiming->setActiveLossRootsMicros += elapsedMicros(setActiveRootsStart, setActiveRootsFinish);
            submitTiming->activeLossRootCount += activeRawLossRoots.size();
        }
    }
    const auto sendBatchStart = std::chrono::high_resolution_clock::now();
    Event processingFinishedEvent = stampedNetworks[stampIndex].sendBatch(batchInputs,
                                                                           batchOutputs,
                                                                           outputReadyEvents,
                                                                           isInferenceOnly,
                                                                           reusableProcessingFinishedEvent,
                                                                           waitForOutputsOnProcessingStream,
                                                                           submitTiming,
                                                                           outputSlotIndex);
    const auto sendBatchFinish = std::chrono::high_resolution_clock::now();
    if (submitTiming != nullptr) {
        submitTiming->sendBatchMicros += elapsedMicros(sendBatchStart, sendBatchFinish);
        submitTiming->totalMicros += elapsedMicros(totalStart, sendBatchFinish);
    }
    return processingFinishedEvent;
}

std::vector<uint64_t> PlacedNetwork::getActiveTrainingRawLossOriginalIdsForDebug(uint64_t stampIndex) const {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    return stampedNetworks[stampIndex].getActiveTrainingRawLossOriginalIdsForDebug();
}

void PlacedNetwork::preallocateInputSlots(uint32_t numSlots) {
    THOR_THROW_IF_FALSE(numSlots >= 1);
    for (ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        stampedNetwork.preallocateInputSlots(numSlots);
    }
}

void PlacedNetwork::preallocateOutputSlots(uint32_t numSlots) {
    THOR_THROW_IF_FALSE(numSlots >= 1);
    for (ThorImplementation::StampedNetwork& stampedNetwork : stampedNetworks) {
        stampedNetwork.preallocateOutputSlots(numSlots);
    }
}

void PlacedNetwork::extendOutputWritableEvents(uint64_t stampIndex, Event event, std::optional<uint32_t> outputSlotIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());
    stampedNetworks[stampIndex].extendOutputWritableEvents(event, outputSlotIndex);
}

void PlacedNetwork::copyTrainingStateFrom(PlacedNetwork& source) {
    if (source.getNumStamps() != getNumStamps()) {
        throw std::runtime_error("Cannot copy placed-network training state between networks with different stamp counts.");
    }

    const std::vector<ParameterReference> parameterReferences = getTrainableParameterReferences(/*trainingEnabledOnly=*/false);
    if (parameterReferences.empty()) {
        return;
    }

    std::vector<Stream> copyStreams;
    for (uint64_t stampIndex = 0; stampIndex < stampedNetworks.size(); ++stampIndex) {
        ThorImplementation::StampedNetwork& destinationStamp = stampedNetworks[stampIndex];
        ThorImplementation::StampedNetwork& sourceStamp = source.getStampedNetwork(stampIndex);

        for (const ParameterReference& parameterReference : parameterReferences) {
            std::shared_ptr<ThorImplementation::PhysicalParameter> destinationParameter =
                getPhysicalParameter(destinationStamp, parameterReference);
            std::shared_ptr<ThorImplementation::PhysicalParameter> sourceParameter =
                getPhysicalParameter(sourceStamp, parameterReference);

            if (destinationParameter == nullptr || sourceParameter == nullptr) {
                throw std::runtime_error("Cannot copy placed-network training state for parameter '" +
                                         parameterReference.getParameterName() + "': missing physical parameter.");
            }
            if (!destinationParameter->getStorage().has_value() || !sourceParameter->getStorage().has_value()) {
                throw std::runtime_error("Cannot copy placed-network training state for parameter '" +
                                         parameterReference.getParameterName() + "': parameter storage is not initialized.");
            }

            const std::string description = "api layer " + std::to_string(parameterReference.getParameterizableId()) +
                                            " parameter '" + parameterReference.getParameterName() + "'";
            ThorImplementation::Tensor destinationStorage = destinationParameter->getStorage().value();
            Stream copyStream = Stream::getNextUploadStream(destinationStorage.getPlacement().getDeviceNum());
            copyTensorState(destinationStorage, sourceParameter->getStorage().value(), copyStream, description);

            if (destinationParameter->hasOptimizer() && sourceParameter->hasOptimizer()) {
                copyOptimizerTensorState(*destinationParameter->getOptimizer(),
                                         *sourceParameter->getOptimizer(),
                                         copyStream,
                                         description);
            }
            copyStreams.push_back(copyStream);
        }
    }

    for (Stream& copyStream : copyStreams) {
        copyStream.synchronize();
    }
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
