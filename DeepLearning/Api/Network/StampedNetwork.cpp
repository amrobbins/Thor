#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Diagnostics/TrainingDiagnostics.h"

#include <limits>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <set>
#if THOR_ENABLE_BATCH_SUBMISSION_TIMING
#include <chrono>
#endif

namespace ThorImplementation {

namespace {

#if THOR_ENABLE_BATCH_SUBMISSION_TIMING
using BatchTimingClock = std::chrono::high_resolution_clock;
using BatchTimingTimePoint = BatchTimingClock::time_point;

BatchTimingTimePoint timingNow(const BatchSubmissionTiming* submitTiming) {
    return submitTiming == nullptr ? BatchTimingTimePoint{} : BatchTimingClock::now();
}

uint64_t elapsedMicros(BatchTimingTimePoint start, BatchTimingTimePoint finish) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
}
#else
struct BatchTimingTimePoint {};

constexpr BatchTimingTimePoint timingNow(const BatchSubmissionTiming*) {
    return {};
}

constexpr uint64_t elapsedMicros(BatchTimingTimePoint, BatchTimingTimePoint) {
    return 0;
}
#endif

}  // namespace

void StampedNetwork::setActiveTrainingLossRoots(const std::vector<Thor::Tensor>& activeRawLossRoots) {
    for (const Thor::Tensor& rawLossRoot : activeRawLossRoots) {
        THOR_THROW_IF_FALSE(rawLossRoot.isInitialized());
    }

    std::set<ThorImplementation::Loss*> activePhysicalLosses;
    for (const Thor::Tensor& rawLossRoot : activeRawLossRoots) {
        auto drivingLayerIt = apiTensorToPhysicalDrivingLayerShared.find(rawLossRoot);
        if (drivingLayerIt == apiTensorToPhysicalDrivingLayerShared.end()) {
            throw std::runtime_error("Active raw loss tensor with original id " + std::to_string(rawLossRoot.getOriginalId()) +
                                     " is not present in the stamped network.");
        }
        std::shared_ptr<ThorImplementation::Loss> physicalLoss =
            std::dynamic_pointer_cast<ThorImplementation::Loss>(drivingLayerIt->second);
        if (physicalLoss == nullptr) {
            throw std::runtime_error("Active raw loss tensor with original id " + std::to_string(rawLossRoot.getOriginalId()) +
                                     " is not driven by a physical loss layer.");
        }
        activePhysicalLosses.insert(physicalLoss.get());
    }

    for (const auto& [apiLayerId, physicalLayer] : apiLayerToPhysicalLayerShared) {
        (void)apiLayerId;
        std::shared_ptr<ThorImplementation::Loss> physicalLoss = std::dynamic_pointer_cast<ThorImplementation::Loss>(physicalLayer);
        if (physicalLoss == nullptr) {
            continue;
        }

        const bool active = activePhysicalLosses.count(physicalLoss.get()) != 0;
        physicalLoss->setTrainingActive(active);
        if (!active) {
            physicalLoss->pruneTrainingBackpropPathIfInactive();
        }
    }
}

std::vector<uint64_t> StampedNetwork::getActiveTrainingRawLossOriginalIdsForDebug() const {
    std::vector<uint64_t> result;
    for (const auto& [apiTensor, physicalLayer] : apiTensorToPhysicalDrivingLayerShared) {
        std::shared_ptr<ThorImplementation::Loss> physicalLoss = std::dynamic_pointer_cast<ThorImplementation::Loss>(physicalLayer);
        if (physicalLoss != nullptr && physicalLoss->isTrainingActive()) {
            result.push_back(apiTensor.getOriginalId());
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<Event> StampedNetwork::getSynchronizeEvents() const {
    std::vector<Event> events = initializationDoneEvents;
    std::set<const Layer*> visitedLayers;

    auto appendLayerEvents = [&](const auto& layers) {
        for (const auto& layer : layers) {
            if (layer == nullptr || !visitedLayers.insert(layer.get()).second)
                continue;
            std::vector<Event> layerEvents = layer->getSynchronizeEvents();
            events.insert(events.end(),
                          std::make_move_iterator(layerEvents.begin()),
                          std::make_move_iterator(layerEvents.end()));
        }
    };

    appendLayerEvents(inputsShared);
    appendLayerEvents(outputsShared);
    appendLayerEvents(trainableLayersShared);
    appendLayerEvents(otherLayersShared);
    return events;
}


void StampedNetwork::initialize(bool initializeWeights, bool copyWeightsFromOtherStamp, StampedNetwork *otherStamp) {
    // First, ensure the shared pointers and raw pointers match
    for (auto it = inputsShared.begin(); it != inputsShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(inputs, it->get()) == 1);
    for (auto it = outputsShared.begin(); it != outputsShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(outputs, it->get()) == 1);
    for (auto it = trainableLayersShared.begin(); it != trainableLayersShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(trainableLayers, it->get()) == 1);
    for (auto it = otherLayersShared.begin(); it != otherLayersShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(otherLayers, it->get()) == 1);
    for (auto it = apiTensorToPhysicalDrivingLayerShared.begin(); it != apiTensorToPhysicalDrivingLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(apiTensorToPhysicalDrivingLayer.count(it->first) == 1);
        THOR_THROW_IF_FALSE(apiTensorToPhysicalDrivingLayer[it->first] == it->second.get());
    }
    for (auto it = apiLayerToPhysicalLayerShared.begin(); it != apiLayerToPhysicalLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(apiLayerToPhysicalLayer.count(it->first) == 1);
        THOR_THROW_IF_FALSE(apiLayerToPhysicalLayer[it->first] == it->second.get());
    }
    for (auto it = physicalLayerToApiLayerShared.begin(); it != physicalLayerToApiLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(physicalLayerToApiLayer.count(it->first.get()) == 1);
        THOR_THROW_IF_FALSE(physicalLayerToApiLayer[it->first.get()] == it->second);
    }
    for (auto it = apiTensorToApiDrivingLayerShared.begin(); it != apiTensorToApiDrivingLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(it->first) == 1);
        THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer[it->first] == it->second.get());
    }
    for (auto it = inputNamedShared.begin(); it != inputNamedShared.end(); ++it) {
        THOR_THROW_IF_FALSE(inputNamed.count(it->first) == 1);
        THOR_THROW_IF_FALSE(inputNamed[it->first] == it->second.get());
    }
    for (auto it = raggedInputNamedShared.begin(); it != raggedInputNamedShared.end(); ++it) {
        THOR_THROW_IF_FALSE(raggedInputNamed.count(it->first) == 1);
        THOR_THROW_IF_FALSE(raggedInputNamed[it->first].valuesInputName == it->second.valuesInputName);
        THOR_THROW_IF_FALSE(raggedInputNamed[it->first].offsetsInputName == it->second.offsetsInputName);
        THOR_THROW_IF_FALSE(raggedInputNamed[it->first].descriptor == it->second.descriptor);
    }
    for (auto it = outputNamedShared.begin(); it != outputNamedShared.end(); ++it) {
        THOR_THROW_IF_FALSE(outputNamed.count(it->first) == 1);
        THOR_THROW_IF_FALSE(outputNamed[it->first] == it->second.get());
    }

    // // FIXME: This overlaps + fights with newer deserialization/initialization logic
    // // Now that checks have been run, initialize the stamp
    // THOR_THROW_IF_FALSE(!(initializeWeights && copyWeightsFromOtherStamp));
    // if (initializeWeights) {
    //     // Weights are shared by all stamps so weights are only initialized once
    //     for (uint32_t i = 0; i < initializers.size(); ++i)
    //         initializers[i]->initialize();
    // } else if (copyWeightsFromOtherStamp) {
    //     // Every GPU needs its a copy of the weights, if they have already been initialized in a weights memory, then copy that memory
    //     // to the target GPU.
    //     THOR_THROW_IF_FALSE(otherStamp != nullptr);
    //     // FIXME use trainable layer stamped ids to copy weights and when present biases from other stamp to this stamp
    //     std::unordered_map<uint64_t, ThorImplementation::TrainableLayer *> trainableLayerMap;
    //     for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
    //         trainableLayerMap[trainableLayers[i]->getStampedId()] = trainableLayers[i];
    //     }
    //     std::vector<Stream> streams;
    //     Stream stream;
    //     for (uint32_t i = 0; i < otherStamp->trainableLayers.size(); ++i) {
    //         uint32_t stampedId = otherStamp->trainableLayers[i]->getStampedId();
    //         if (i == 0) {
    //             streams.push_back(trainableLayerMap[stampedId]->getStreams()[0]);
    //         }
    //         Tensor uninitializedWeights = trainableLayerMap[stampedId]->getWeights();
    //         std::optional<Tensor> uninitializedBiases = trainableLayerMap[stampedId]->getBiases();
    //         ThorImplementation::TrainableLayer *initializedLayer = otherStamp->trainableLayers[i];
    //         Tensor initializedWeights = initializedLayer->getWeights();
    //         std::optional<Tensor> initializedBiases = initializedLayer->getBiases();
    //         uninitializedWeights.copyFromAsync(initializedWeights, streams.back());
    //         if (initializedBiases.has_value()) {
    //             THOR_THROW_IF_FALSE(uninitializedBiases.has_value());
    //             uninitializedBiases.value().copyFromAsync(initializedBiases.value(), stream);
    //         }
    //     }
    //     for (uint32_t i = 0; i < streams.size(); ++i) {
    //         streams[i].synchronize();
    //     }
    // }

    // // FIXME: get rid of implementation layer initialize, that is owned by API layer. Implementation layer has compile.
    // // so implementationLayer.compile then apiLayer.initialize()
    // for (uint32_t i = 0; i < inputs.size(); ++i) {
    //     inputs[i]->parentInitialize();
    //     inputs[i]->initialize();
    // }
    // for (uint32_t i = 0; i < outputs.size(); ++i) {
    //     outputs[i]->parentInitialize();
    //     outputs[i]->initialize();
    // }
    // for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
    //     trainableLayers[i]->parentInitialize();
    //     trainableLayers[i]->initialize();
    // }
    // for (uint32_t i = 0; i < otherLayers.size(); ++i) {
    //     otherLayers[i]->parentInitialize();
    //     otherLayers[i]->initialize();
    // }
}

// Note that all processing is finished at the end of any input stream of the stamp.
// Note *input* stream - this is not the case for the loader streams
Event StampedNetwork::sendBatch(std::map<std::string, Tensor> batchInputs,
                                std::map<std::string, Tensor> &batchOutputs,
                                std::map<std::string, Event> &outputReadyEvents,
                                bool isInferenceOnly,
                                Event* reusableProcessingFinishedEvent,
                                bool waitForOutputsOnProcessingStream,
                                BatchSubmissionTiming* submitTiming,
                                std::optional<uint32_t> outputSlotIndex) {
    static const std::map<std::string, Event> noInputReadyEvents;
    return sendBatch(std::move(batchInputs),
                     noInputReadyEvents,
                     batchOutputs,
                     outputReadyEvents,
                     isInferenceOnly,
                     reusableProcessingFinishedEvent,
                     waitForOutputsOnProcessingStream,
                     submitTiming,
                     outputSlotIndex);
}

Event StampedNetwork::sendBatch(std::map<std::string, Tensor> batchInputs,
                                const std::map<std::string, Event>& inputReadyEvents,
                                std::map<std::string, Tensor> &batchOutputs,
                                std::map<std::string, Event> &outputReadyEvents,
                                bool isInferenceOnly,
                                Event* reusableProcessingFinishedEvent,
                                bool waitForOutputsOnProcessingStream,
                                BatchSubmissionTiming* submitTiming,
                                std::optional<uint32_t> outputSlotIndex) {
    std::optional<uint32_t> batchSize;
    const auto unwrapStart = timingNow(submitTiming);
    for (const auto &[inputName, inputTensor] : batchInputs) {
        (void)inputName;
        const std::vector<uint64_t> dimensions = inputTensor.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty());
        THOR_THROW_IF_FALSE(dimensions[0] <= std::numeric_limits<uint32_t>::max());
        if (!batchSize.has_value()) {
            batchSize = static_cast<uint32_t>(dimensions[0]);
        } else {
            THOR_THROW_IF_FALSE(batchSize.value() == dimensions[0]);
        }
    }
    THOR_THROW_IF_FALSE(batchSize.has_value());
    for (const auto& [inputName, _] : inputReadyEvents) {
        (void)_;
        THOR_THROW_IF_FALSE(batchInputs.count(inputName) == 1);
    }
    const auto unwrapFinish = timingNow(submitTiming);
    BatchSubmissionTiming localTiming;
    Event processingFinishedEvent = sendPhysicalBatch(std::move(batchInputs),
                                                       inputReadyEvents,
                                                       batchOutputs,
                                                       outputReadyEvents,
                                                       isInferenceOnly,
                                                       batchSize.value(),
                                                       reusableProcessingFinishedEvent,
                                                       waitForOutputsOnProcessingStream,
                                                       submitTiming == nullptr ? nullptr : &localTiming,
                                                       outputSlotIndex);
    if (submitTiming != nullptr) {
        localTiming.batchUnwrapMicros += elapsedMicros(unwrapStart, unwrapFinish);
        accumulateBatchSubmissionTiming(*submitTiming, localTiming);
    }
    return processingFinishedEvent;
}

Event StampedNetwork::sendBatch(const Batch& batchInputs,
                                std::map<std::string, Tensor> &batchOutputs,
                                std::map<std::string, Event> &outputReadyEvents,
                                bool isInferenceOnly,
                                Event* reusableProcessingFinishedEvent,
                                bool waitForOutputsOnProcessingStream,
                                BatchSubmissionTiming* submitTiming,
                                std::optional<uint32_t> outputSlotIndex) {
    std::map<std::string, Tensor> physicalBatchInputs;
    std::optional<uint32_t> batchSize;
    const auto unwrapStart = timingNow(submitTiming);

    auto requireConsistentBatchSize = [&batchSize](uint64_t candidate) {
        THOR_THROW_IF_FALSE(candidate <= std::numeric_limits<uint32_t>::max());
        if (!batchSize.has_value()) {
            batchSize = static_cast<uint32_t>(candidate);
        } else {
            THOR_THROW_IF_FALSE(batchSize.value() == candidate);
        }
    };

    for (const auto& [name, value] : batchInputs.values()) {
        if (std::holds_alternative<Tensor>(value)) {
            Tensor inputTensor = std::get<Tensor>(value);
            const std::vector<uint64_t> dimensions = inputTensor.getDescriptor().getDimensions();
            THOR_THROW_IF_FALSE(!dimensions.empty());
            requireConsistentBatchSize(dimensions[0]);
            THOR_THROW_IF_FALSE(physicalBatchInputs.emplace(name, inputTensor).second);
        } else if (std::holds_alternative<RaggedTensor>(value)) {
            auto raggedIt = raggedInputNamed.find(name);
            THOR_THROW_IF_FALSE(raggedIt != raggedInputNamed.end());
            const RaggedInputBinding& binding = raggedIt->second;
            RaggedTensor raggedTensor = std::get<RaggedTensor>(value);
            THOR_THROW_IF_FALSE(raggedTensor.getDescriptor() == binding.descriptor);
            requireConsistentBatchSize(raggedTensor.getBatchSize());
            THOR_THROW_IF_FALSE(physicalBatchInputs.emplace(binding.valuesInputName, raggedTensor.getValues()).second);
            THOR_THROW_IF_FALSE(physicalBatchInputs.emplace(binding.offsetsInputName, raggedTensor.getOffsets()).second);
        } else {
            THOR_UNREACHABLE();
        }
    }

    THOR_THROW_IF_FALSE(batchSize.has_value());
    const auto unwrapFinish = timingNow(submitTiming);
    BatchSubmissionTiming localTiming;
    static const std::map<std::string, Event> noInputReadyEvents;
    Event processingFinishedEvent = sendPhysicalBatch(std::move(physicalBatchInputs),
                                                       noInputReadyEvents,
                                                       batchOutputs,
                                                       outputReadyEvents,
                                                       isInferenceOnly,
                                                       batchSize.value(),
                                                       reusableProcessingFinishedEvent,
                                                       waitForOutputsOnProcessingStream,
                                                       submitTiming == nullptr ? nullptr : &localTiming,
                                                       outputSlotIndex);
    if (submitTiming != nullptr) {
        localTiming.batchUnwrapMicros += elapsedMicros(unwrapStart, unwrapFinish);
        accumulateBatchSubmissionTiming(*submitTiming, localTiming);
    }
    return processingFinishedEvent;
}

Event StampedNetwork::sendPhysicalBatch(std::map<std::string, Tensor> batchInputs,
                                        const std::map<std::string, Event>& inputReadyEvents,
                                        std::map<std::string, Tensor> &batchOutputs,
                                        std::map<std::string, Event> &outputReadyEvents,
                                        bool isInferenceOnly,
                                        uint32_t batchSize,
                                        Event* reusableProcessingFinishedEvent,
                                        bool waitForOutputsOnProcessingStream,
                                        BatchSubmissionTiming* submitTiming,
                                        std::optional<uint32_t> outputSlotIndex) {
    const auto physicalStart = timingNow(submitTiming);
    THOR_THROW_IF_FALSE(batchInputs.size() == inputs.size());

    const uint32_t queueSlot = outputSlotIndex.value_or(0);
    const uint32_t outputSlot = queueSlot;
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->setActiveInputSlot(queueSlot);
    }
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        outputs[i]->setActiveOutputSlot(outputSlot);
    }

    const auto inputForwardStart = timingNow(submitTiming);
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        auto it = batchInputs.find(inputs[i]->getName());
        THOR_THROW_IF_FALSE(it != batchInputs.end());
        Tensor inputTensor = it->second;
        const auto readyIt = inputReadyEvents.find(inputs[i]->getName());
        if (readyIt != inputReadyEvents.end()) {
            inputs[i]->forward(inputTensor, isInferenceOnly, readyIt->second, batchSize);
        } else {
            inputs[i]->forward(inputTensor, isInferenceOnly, batchSize);
        }
    }
    const auto inputForwardFinish = timingNow(submitTiming);

    // Capture each NetworkOutput-owned ready event.  NetworkOutput may offload its
    // value through a dedicated download stream when the requested output placement
    // differs from the producing layer placement (for example GPU loss -> CPU stats
    // tensor).  In that case getStream() is the producing/compute stream, not the
    // stream that owns the final D2H copy.  Consumers that need materialized outputs
    // must wait on the NetworkOutput ready event, not on the producer stream.
    const auto outputCollectStart = timingNow(submitTiming);
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        batchOutputs[outputs[i]->getName()] = outputs[i]->getFeatureOutputForSlot(outputSlot).value();
        Event outputReadyEvent = outputs[i]->getOutputReadyEventForSlot(outputSlot);
        outputReadyEvents[outputs[i]->getName()] = outputReadyEvent;
    }
    const auto outputCollectFinish = timingNow(submitTiming);

    const auto outputWaitStart = timingNow(submitTiming);
    if (waitForOutputsOnProcessingStream) {
        for (const auto& [outputName, outputReadyEvent] : outputReadyEvents) {
            (void)outputName;
            inputs[0]->getStream().waitEvent(outputReadyEvent);
        }
    }
    const auto outputWaitFinish = timingNow(submitTiming);

    // Processing is finished when the stream from input 0 is ready.  The native queued
    // trainer deliberately does not fold CPU output/stat readiness back into this stream;
    // it waits for outputReadyEvents on a side stream so the single stamp can queue
    // future input work while host stat extraction catches up.
    // The native queued trainer passes a per-in-flight reusable event here so the
    // hot training path does not allocate/destroy a CUDA event for every batch.
    Event processingFinishedEvent;
    const auto processingEventStart = timingNow(submitTiming);
    if (reusableProcessingFinishedEvent != nullptr) {
        inputs[0]->getStream().putEvent(*reusableProcessingFinishedEvent, true, true);
        processingFinishedEvent = *reusableProcessingFinishedEvent;
    } else {
        processingFinishedEvent = inputs[0]->getStream().putEvent(true, true);
    }
    const auto processingEventFinish = timingNow(submitTiming);

    // The streams from all other inputs wait for the stream from input 0 to be ready
    const auto inputFanoutStart = timingNow(submitTiming);
    for (uint i = 1; i < inputs.size(); ++i) {
        inputs[i]->getStream().waitEvent(processingFinishedEvent);
    }
    const auto inputFanoutFinish = timingNow(submitTiming);

    if (submitTiming != nullptr) {
        submitTiming->physicalTotalMicros += elapsedMicros(physicalStart, inputFanoutFinish);
        submitTiming->inputForwardMicros += elapsedMicros(inputForwardStart, inputForwardFinish);
        submitTiming->outputCollectMicros += elapsedMicros(outputCollectStart, outputCollectFinish);
        submitTiming->outputWaitOnProcessingMicros += elapsedMicros(outputWaitStart, outputWaitFinish);
        submitTiming->processingEventMicros += elapsedMicros(processingEventStart, processingEventFinish);
        submitTiming->inputFanoutMicros += elapsedMicros(inputFanoutStart, inputFanoutFinish);
        submitTiming->numInputs += inputs.size();
        submitTiming->numOutputs += outputs.size();
    }

    return processingFinishedEvent;
}

void StampedNetwork::clear() {
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->cleanup();
    }
    inputs.clear();

    for (uint32_t i = 0; i < outputs.size(); ++i) {
        outputs[i]->cleanup();
    }
    outputs.clear();

    for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
        trainableLayers[i]->cleanup();
    }
    trainableLayers.clear();

    for (uint32_t i = 0; i < otherLayers.size(); ++i) {
        otherLayers[i]->cleanup();
    }
    otherLayers.clear();

    apiTensorToPhysicalDrivingLayer.clear();
    apiLayerToPhysicalLayer.clear();
    physicalLayerToApiLayer.clear();
    apiTensorToApiDrivingLayer.clear();
    inputNamed.clear();
    raggedInputNamed.clear();
    outputNamed.clear();

    inputsShared.clear();
    outputsShared.clear();
    trainableLayersShared.clear();
    otherLayersShared.clear();
    initializationDoneEvents.clear();
    apiTensorToPhysicalDrivingLayerShared.clear();
    apiLayerToPhysicalLayerShared.clear();
    physicalLayerToApiLayerShared.clear();
    apiTensorToApiDrivingLayerShared.clear();
    inputNamedShared.clear();
    raggedInputNamedShared.clear();
    outputNamedShared.clear();
}

void StampedNetwork::preallocateInputSlots(uint32_t numSlots) {
    THOR_THROW_IF_FALSE(numSlots >= 1);
    for (NetworkInput* input : inputs) {
        input->preallocateInputSlots(numSlots);
    }
}

void StampedNetwork::preallocateOutputSlots(uint32_t numSlots) {
    THOR_THROW_IF_FALSE(numSlots >= 1);
    for (NetworkOutput* output : outputs) {
        output->preallocateOutputSlots(numSlots);
    }
}

void StampedNetwork::extendOutputWritableEvents(Event event, std::optional<uint32_t> outputSlotIndex) {
    if (outputSlotIndex.has_value()) {
        const uint32_t outputSlot = outputSlotIndex.value();
        for (NetworkOutput* output : outputs) {
            output->extendOutputWritableEventForSlot(outputSlot, event);
        }
    } else {
        for (NetworkOutput* output : outputs) {
            output->extendOutputWritableEvent(event);
        }
    }
}

}  // namespace ThorImplementation
