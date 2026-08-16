#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"

#include <tuple>
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"
#include "DeepLearning/Implementation/Diagnostics/TrainingDiagnostics.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <exception>
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


void StampedNetwork::initializeProcessingDataStreamJoin() {
    THOR_THROW_IF_FALSE(!inputs.empty());

    processingDataStreams.clear();
    processingDataStreamEvents.clear();

    const Stream processingStream = inputs[0]->getStream();
    std::set<uint64_t> streamIds;
    streamIds.insert(processingStream.getId());

    auto appendStream = [&](const Stream& stream) {
        // getProcessingStreams() may omit unused handles entirely, but tolerate
        // an uninitialized handle defensively. Distinct layers commonly share
        // fanout/data/gradient-pool streams, so deduplicate by CUDA stream id.
        if (!stream.isInitialized())
            return;
        THOR_THROW_IF_FALSE(stream.getGpuNum() == processingStream.getGpuNum());
        if (!streamIds.insert(stream.getId()).second)
            return;
        processingDataStreams.push_back(stream);
        processingDataStreamEvents.emplace_back(stream.getGpuNum(), false, false);
    };

    auto appendLayerStreams = [&](Layer* layer) {
        THOR_THROW_IF_FALSE(layer != nullptr);
        for (const Stream& stream : layer->getProcessingStreams())
            appendStream(stream);
    };

    for (NetworkInput* input : inputs)
        appendLayerStreams(input);
    for (NetworkOutput* output : outputs)
        appendLayerStreams(output);
    for (TrainableLayer* trainableLayer : trainableLayers)
        appendLayerStreams(trainableLayer);
    for (Layer* layer : otherLayers)
        appendLayerStreams(layer);
}

void StampedNetwork::joinProcessingDataStreams(const Stream& processingStream) {
    THOR_THROW_IF_FALSE(processingDataStreams.size() == processingDataStreamEvents.size());
    for (size_t i = 0; i < processingDataStreams.size(); ++i) {
        processingDataStreams[i].putEvent(processingDataStreamEvents[i], false, false);
        processingStream.waitEvent(processingDataStreamEvents[i]);
    }
}

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

void StampedNetwork::setTrainingDropoutEnabled(bool enabled) {
    for (const auto& [apiLayerId, physicalLayer] : apiLayerToPhysicalLayerShared) {
        (void)apiLayerId;
        std::shared_ptr<TrainingDropoutControllable> controllable =
            std::dynamic_pointer_cast<TrainingDropoutControllable>(physicalLayer);
        if (controllable != nullptr) {
            controllable->setTrainingDropoutEnabled(enabled);
        }
    }
}

bool StampedNetwork::isTrainingDropoutEnabled() const {
    for (const auto& [apiLayerId, physicalLayer] : apiLayerToPhysicalLayerShared) {
        (void)apiLayerId;
        std::shared_ptr<TrainingDropoutControllable> controllable =
            std::dynamic_pointer_cast<TrainingDropoutControllable>(physicalLayer);
        if (controllable != nullptr && !controllable->isTrainingDropoutEnabled()) {
            return false;
        }
    }
    return true;
}

std::vector<PartialBatchIncompatibility>
StampedNetwork::getPartialBatchIncompatibilities() const {
    std::vector<PartialBatchIncompatibility> incompatibilities;
    std::set<Layer*> seen;
    auto inspect = [&](Layer* layer) {
        THOR_THROW_IF_FALSE(layer != nullptr);
        if (!seen.insert(layer).second || layer->supportsPartialBatches()) {
            return;
        }
        uint64_t reportedLayerId = layer->getId();
        const auto apiLayerIt = physicalLayerToApiLayer.find(layer);
        if (apiLayerIt != physicalLayerToApiLayer.end()) {
            reportedLayerId = apiLayerIt->second;
        }
        incompatibilities.push_back(PartialBatchIncompatibility{
            reportedLayerId, layer->getName(), layer->getType()});
    };
    for (TrainableLayer* layer : trainableLayers) {
        inspect(layer);
    }
    for (Layer* layer : otherLayers) {
        inspect(layer);
    }
    std::sort(
        incompatibilities.begin(),
        incompatibilities.end(),
        [](const PartialBatchIncompatibility& lhs,
           const PartialBatchIncompatibility& rhs) {
            return std::tie(lhs.layerId, lhs.layerType, lhs.layerName) <
                   std::tie(rhs.layerId, rhs.layerType, rhs.layerName);
        });
    return incompatibilities;
}

uint32_t StampedNetwork::getNumTrainingDropoutControllableLayers() const {
    uint32_t count = 0;
    for (const auto& [apiLayerId, physicalLayer] : apiLayerToPhysicalLayerShared) {
        (void)apiLayerId;
        if (std::dynamic_pointer_cast<TrainingDropoutControllable>(physicalLayer) != nullptr) {
            ++count;
        }
    }
    return count;
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
// Note *input* stream - this is not the case for the batch-source streams
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
    if (!raggedInputNamed.empty()) {
        throw std::logic_error(
            "StampedNetwork::sendBatch(map<string, Tensor>) cannot represent logical RaggedNetworkInput values. "
            "Submit a Thor::Batch containing RaggedTensor entries instead.");
    }

    std::optional<uint32_t> physicalBatchCapacity;
    const auto unwrapStart = timingNow(submitTiming);
    for (const auto &[inputName, inputTensor] : batchInputs) {
        (void)inputName;
        const std::vector<uint64_t> dimensions = inputTensor.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty());
        THOR_THROW_IF_FALSE(dimensions[0] <= std::numeric_limits<uint32_t>::max());
        if (!physicalBatchCapacity.has_value()) {
            physicalBatchCapacity = static_cast<uint32_t>(dimensions[0]);
        } else {
            THOR_THROW_IF_FALSE(physicalBatchCapacity.value() == dimensions[0]);
        }
    }
    THOR_THROW_IF_FALSE(physicalBatchCapacity.has_value());
    for (const auto& [inputName, _] : inputReadyEvents) {
        (void)_;
        THOR_THROW_IF_FALSE(batchInputs.count(inputName) == 1);
    }
    const auto unwrapFinish = timingNow(submitTiming);
    std::map<std::string, PhysicalBatchInput> physicalBatchInputs;
    for (auto& [name, tensor] : batchInputs) {
        THOR_THROW_IF_FALSE(physicalBatchInputs.emplace(name, PhysicalBatchInput{std::move(tensor), std::nullopt}).second);
    }
    BatchSubmissionTiming localTiming;
    Event processingFinishedEvent = sendPhysicalBatch(std::move(physicalBatchInputs),
                                                       inputReadyEvents,
                                                       batchOutputs,
                                                       outputReadyEvents,
                                                       isInferenceOnly,
                                                       physicalBatchCapacity.value(),
                                                       physicalBatchCapacity.value(),
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
    std::map<std::string, PhysicalBatchInput> physicalBatchInputs;
    std::optional<uint32_t> physicalBatchCapacity;
    const auto unwrapStart = timingNow(submitTiming);

    auto requireConsistentBatchCapacity = [&physicalBatchCapacity](uint64_t candidate) {
        THOR_THROW_IF_FALSE(candidate <= std::numeric_limits<uint32_t>::max());
        if (!physicalBatchCapacity.has_value()) {
            physicalBatchCapacity = static_cast<uint32_t>(candidate);
        } else {
            THOR_THROW_IF_FALSE(physicalBatchCapacity.value() == candidate);
        }
    };

    for (const auto& [name, value] : batchInputs.values()) {
        const std::optional<Thor::BatchSourceReference> sourceReference =
            batchInputs.getSourceReference(name);
        if (std::holds_alternative<Tensor>(value)) {
            Tensor inputTensor = std::get<Tensor>(value);
            const std::vector<uint64_t> dimensions = inputTensor.getDescriptor().getDimensions();
            THOR_THROW_IF_FALSE(!dimensions.empty());
            requireConsistentBatchCapacity(dimensions[0]);
            THOR_THROW_IF_FALSE(
                physicalBatchInputs.emplace(
                    name,
                    PhysicalBatchInput{inputTensor, sourceReference}).second);
        } else if (std::holds_alternative<RaggedTensor>(value)) {
            auto raggedIt = raggedInputNamed.find(name);
            THOR_THROW_IF_FALSE(raggedIt != raggedInputNamed.end());
            const RaggedInputBinding& binding = raggedIt->second;
            RaggedTensor raggedTensor = std::get<RaggedTensor>(value);
            THOR_THROW_IF_FALSE(raggedTensor.getDescriptor() == binding.descriptor);
            requireConsistentBatchCapacity(raggedTensor.getBatchSize());
            const std::optional<uint64_t> activeValueCount =
                raggedTensor.getHostActiveValueCountIfAvailable();
            THOR_THROW_IF_FALSE(
                physicalBatchInputs.emplace(
                    binding.valuesInputName,
                    PhysicalBatchInput{raggedTensor.getValues(), sourceReference}).second);
            THOR_THROW_IF_FALSE(
                physicalBatchInputs.emplace(
                    binding.offsetsInputName,
                    PhysicalBatchInput{raggedTensor.getOffsets(),
                                       sourceReference,
                                       raggedTensor.getRowPartitionRuntime().getDescriptor(),
                                       activeValueCount}).second);
        } else if (std::holds_alternative<Thor::DeviceBatchReference>(value)) {
            Thor::DeviceBatchReference reference = std::get<Thor::DeviceBatchReference>(value);
            requireConsistentBatchCapacity(reference.getBatchCapacity());
            THOR_THROW_IF_FALSE(
                physicalBatchInputs.emplace(
                    name,
                    PhysicalBatchInput{std::move(reference), sourceReference}).second);
        } else {
            THOR_UNREACHABLE();
        }
    }

    THOR_THROW_IF_FALSE(physicalBatchCapacity.has_value());
    const uint32_t validExampleCount =
        batchInputs.getValidExampleCount().value_or(physicalBatchCapacity.value());
    THOR_THROW_IF_FALSE(validExampleCount <= physicalBatchCapacity.value());
    const auto unwrapFinish = timingNow(submitTiming);
    BatchSubmissionTiming localTiming;
    static const std::map<std::string, Event> noInputReadyEvents;
    Event processingFinishedEvent = sendPhysicalBatch(std::move(physicalBatchInputs),
                                                       noInputReadyEvents,
                                                       batchOutputs,
                                                       outputReadyEvents,
                                                       isInferenceOnly,
                                                       physicalBatchCapacity.value(),
                                                       validExampleCount,
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

Event StampedNetwork::sendPhysicalBatch(std::map<std::string, PhysicalBatchInput> batchInputs,
                                        const std::map<std::string, Event>& inputReadyEvents,
                                        std::map<std::string, Tensor> &batchOutputs,
                                        std::map<std::string, Event> &outputReadyEvents,
                                        bool isInferenceOnly,
                                        uint32_t physicalBatchCapacity,
                                        uint32_t validExampleCount,
                                        Event* reusableProcessingFinishedEvent,
                                        bool waitForOutputsOnProcessingStream,
                                        BatchSubmissionTiming* submitTiming,
                                        std::optional<uint32_t> outputSlotIndex) {
    const auto physicalStart = timingNow(submitTiming);
    THOR_THROW_IF_FALSE(batchInputs.size() == inputs.size());
    THOR_THROW_IF_FALSE(physicalBatchCapacity >= 1);
    THOR_THROW_IF_FALSE(validExampleCount >= 1);
    THOR_THROW_IF_FALSE(validExampleCount <= physicalBatchCapacity);

    if (validExampleCount < physicalBatchCapacity) {
        auto requirePartialBatchSupport = [](ThorImplementation::Layer* layer) {
            THOR_THROW_IF_FALSE(layer != nullptr);
            if (!layer->supportsPartialBatches()) {
                const std::string layerName = layer->getName().empty() ? std::string("<unnamed>") : layer->getName();
                throw std::logic_error("Layer '" + layerName + "' of type " + layer->getType() +
                                       " does not define exact partial-batch semantics.");
            }
        };
        for (ThorImplementation::TrainableLayer* layer : trainableLayers)
            requirePartialBatchSupport(layer);
        for (ThorImplementation::Layer* layer : otherLayers)
            requirePartialBatchSupport(layer);
    }

    const uint32_t queueSlot = outputSlotIndex.value_or(0);
    const uint32_t outputSlot = queueSlot;
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->setActiveInputSlot(queueSlot);
    }
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        outputs[i]->setActiveOutputSlot(outputSlot);
    }
    {
        std::set<Metric*> configuredMetrics;
        for (const auto& [outputName, metric] : metricStatisticsByOutputNameShared) {
            (void)outputName;
            if (metric != nullptr && configuredMetrics.insert(metric.get()).second)
                metric->setActiveMetricStatisticSlot(outputSlot);
        }
    }

    const auto inputForwardStart = timingNow(submitTiming);

    // A logical RaggedNetworkInput materializes values and offsets through two
    // physical NetworkInput ports. Tail canonicalization is a values-copy concern,
    // but its extent comes exclusively from the matching offsets-owned runtime.
    // Keep this association transient: no ragged state is stored on values tensors.
    std::map<std::string, uint64_t> raggedValuesActiveValueCounts;
    for (const auto& [logicalName, binding] : raggedInputNamed) {
        (void)logicalName;
        auto valuesBatchIt = batchInputs.find(binding.valuesInputName);
        auto offsetsBatchIt = batchInputs.find(binding.offsetsInputName);
        if (valuesBatchIt == batchInputs.end() || offsetsBatchIt == batchInputs.end()) continue;

        THOR_THROW_IF_FALSE(offsetsBatchIt->second.rowPartitionDescriptor.has_value());
        THOR_THROW_IF_FALSE(offsetsBatchIt->second.rowPartitionHostActiveValueCount.has_value());
        const uint64_t activeValueCount = offsetsBatchIt->second.rowPartitionHostActiveValueCount.value();
        THOR_THROW_IF_FALSE(
            raggedValuesActiveValueCounts.emplace(binding.valuesInputName, activeValueCount).second);
    }

    for (uint32_t i = 0; i < inputs.size(); ++i) {
        auto it = batchInputs.find(inputs[i]->getName());
        THOR_THROW_IF_FALSE(it != batchInputs.end());
        const auto readyIt = inputReadyEvents.find(inputs[i]->getName());
        const auto raggedValuesIt = raggedValuesActiveValueCounts.find(inputs[i]->getName());
        if (std::holds_alternative<Tensor>(it->second.value)) {
            Tensor inputTensor = std::get<Tensor>(it->second.value);
            if (it->second.rowPartitionDescriptor.has_value()) {
                THOR_THROW_IF_FALSE(raggedValuesIt == raggedValuesActiveValueCounts.end());
                if (readyIt != inputReadyEvents.end()) {
                    inputs[i]->forwardRowPartitionOffsets(
                        inputTensor,
                        isInferenceOnly,
                        readyIt->second,
                        it->second.rowPartitionDescriptor.value(),
                        it->second.rowPartitionHostActiveValueCount,
                        validExampleCount,
                        it->second.sourceReference);
                } else {
                    inputs[i]->forwardRowPartitionOffsets(
                        inputTensor,
                        isInferenceOnly,
                        it->second.rowPartitionDescriptor.value(),
                        it->second.rowPartitionHostActiveValueCount,
                        validExampleCount,
                        it->second.sourceReference);
                }
            } else if (raggedValuesIt != raggedValuesActiveValueCounts.end()) {
                const uint64_t activeValueCount = raggedValuesIt->second;
                if (readyIt != inputReadyEvents.end()) {
                    inputs[i]->forwardRaggedValues(
                        inputTensor,
                        isInferenceOnly,
                        readyIt->second,
                        activeValueCount,
                        validExampleCount,
                        it->second.sourceReference);
                } else {
                    inputs[i]->forwardRaggedValues(
                        inputTensor,
                        isInferenceOnly,
                        activeValueCount,
                        validExampleCount,
                        it->second.sourceReference);
                }
            } else if (readyIt != inputReadyEvents.end()) {
                inputs[i]->forward(
                    inputTensor,
                    isInferenceOnly,
                    readyIt->second,
                    validExampleCount,
                    it->second.sourceReference);
            } else {
                inputs[i]->forward(
                    inputTensor,
                    isInferenceOnly,
                    validExampleCount,
                    it->second.sourceReference);
            }
        } else if (std::holds_alternative<Thor::DeviceBatchReference>(it->second.value)) {
            THOR_THROW_IF_FALSE(readyIt == inputReadyEvents.end());
            THOR_THROW_IF_FALSE(raggedValuesIt == raggedValuesActiveValueCounts.end());
            inputs[i]->forward(
                std::get<Thor::DeviceBatchReference>(it->second.value),
                isInferenceOnly,
                validExampleCount,
                it->second.sourceReference);
        } else {
            THOR_UNREACHABLE();
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

    // A stamp uses statically connected activation tensors. Before advertising that
    // the batch's GPU processing is complete, join every stream declared by the
    // layers' getProcessingStreams() contract back onto input 0's stream. This
    // includes secondary fanout consumers and trainable gradient/update streams,
    // both of which may still read current-batch graph tensors after the primary
    // data stream has advanced. This is especially important for fanout branches whose
    // consumer cannot enqueue work until another NetworkInput arrives (for example a
    // weighted metric waiting for its weights): the fanout's producer stream alone
    // does not cover that deferred branch. Auxiliary output/download streams remain
    // outside this barrier and are waited independently through outputReadyEvents.
    Event processingFinishedEvent;
    const auto processingEventStart = timingNow(submitTiming);
    joinProcessingDataStreams(inputs[0]->getStream());
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

void StampedNetwork::clearImpl(bool propagateCleanupFailure) {
    processingDataStreamEvents.clear();
    processingDataStreams.clear();

    std::exception_ptr firstCleanupFailure;
    auto cleanupLayers = [&](auto& layers) {
        for (auto* layer : layers) {
            if (layer == nullptr) {
                continue;
            }
            try {
                layer->cleanup();
            } catch (...) {
                if (firstCleanupFailure == nullptr) {
                    firstCleanupFailure = std::current_exception();
                }
            }
        }
        layers.clear();
    };

    // Continue through every layer even when CUDA is already reporting an
    // error. A partial model stamp may own independent raw CUDA allocations in
    // many cleanup() implementations; stopping at the first exception leaks the
    // rest of the physical graph.
    cleanupLayers(inputs);
    cleanupLayers(outputs);
    cleanupLayers(trainableLayers);
    cleanupLayers(otherLayers);

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
    gradientUpdateStreamPool.reset();
    otherLayersShared.clear();
    initializationDoneEvents.clear();
    apiTensorToPhysicalDrivingLayerShared.clear();
    apiLayerToPhysicalLayerShared.clear();
    physicalLayerToApiLayerShared.clear();
    apiTensorToApiDrivingLayerShared.clear();
    inputNamedShared.clear();
    raggedInputNamedShared.clear();
    outputNamedShared.clear();
    metricStatisticsByOutputNameShared.clear();

    if (propagateCleanupFailure && firstCleanupFailure != nullptr) {
        std::rethrow_exception(firstCleanupFailure);
    }
}

void StampedNetwork::clear() { clearImpl(/*propagateCleanupFailure=*/true); }

void StampedNetwork::clearNoThrow() noexcept {
    try {
        clearImpl(/*propagateCleanupFailure=*/false);
    } catch (...) {
        // clearImpl(false) is designed not to throw, but destruction and failed
        // startup cleanup must never terminate if a future container/member
        // cleanup path becomes exceptional.
    }
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
    std::set<Metric*> configuredMetrics;
    for (const auto& [outputName, metric] : metricStatisticsByOutputNameShared) {
        (void)outputName;
        if (metric != nullptr && configuredMetrics.insert(metric.get()).second)
            metric->preallocateMetricStatisticSlots(numSlots);
    }
}

std::map<std::string, MetricBatchStatisticTensors> StampedNetwork::getMetricBatchStatisticTensorsForSlot(
    uint32_t slotIndex) const {
    std::map<std::string, MetricBatchStatisticTensors> statistics;
    for (const auto& [outputName, metric] : metricStatisticsByOutputNameShared) {
        THOR_THROW_IF_FALSE(metric != nullptr);
        std::optional<MetricBatchStatisticTensors> metricStatistics =
            metric->getMetricBatchStatisticTensorsForSlot(slotIndex);
        if (metricStatistics.has_value())
            statistics.emplace(outputName, std::move(metricStatistics.value()));
    }
    return statistics;
}

void StampedNetwork::extendMetricStatisticWritableEvents(Event event, std::optional<uint32_t> outputSlotIndex) {
    const uint32_t slotIndex = outputSlotIndex.value_or(0);
    std::set<Metric*> extendedMetrics;
    for (const auto& [outputName, metric] : metricStatisticsByOutputNameShared) {
        (void)outputName;
        if (metric != nullptr && extendedMetrics.insert(metric.get()).second)
            metric->extendMetricStatisticWritableEventForSlot(slotIndex, event);
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
