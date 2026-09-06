#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"

#include <tuple>
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"
#include "DeepLearning/Implementation/Diagnostics/TrainingDiagnostics.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <exception>
#include <functional>
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

uint64_t checkedFlopAdd(uint64_t lhs, uint64_t rhs, const char* where) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(where) + " FLOP count overflow.");
    }
    return lhs + rhs;
}

uint64_t checkedFlopMul(uint64_t lhs, uint64_t rhs, const char* where) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(where) + " FLOP count overflow.");
    }
    return lhs * rhs;
}

}  // namespace

std::shared_ptr<Layer> StampedNetwork::selectRowPartitionDrivingLayer(
    RowPartitionId rowPartitionId,
    RaggedPartitionRequirement requirement) const {
    const RowPartitionPhysicalization* physicalization = nullptr;
    auto externalIt = externalRowPartitions.find(rowPartitionId);
    if (externalIt != externalRowPartitions.end()) physicalization = &externalIt->second;
    auto internalIt = internalRowPartitions.find(rowPartitionId);
    if (internalIt != internalRowPartitions.end()) {
        THOR_THROW_IF_FALSE(physicalization == nullptr);
        physicalization = &internalIt->second;
    }
    if (physicalization == nullptr) {
        throw std::logic_error("Row-partition physicalization is missing for the requested logical partition.");
    }
    if (!consumesAnyRaggedPartitionInformation(requirement)) {
        throw std::logic_error(
            "Logical row-partition token reached a physical consumer that declares no partition requirement.");
    }
    if (hasRaggedPartitionRequirement(requirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT) &&
        hasRaggedPartitionRequirement(requirement, RaggedPartitionRequirement::DEVICE_OFFSETS)) {
        throw std::logic_error(
            "One logical row-partition input port cannot simultaneously consume DEVICE_ACTIVE_COUNT and DEVICE_OFFSETS; "
            "the physical operator must expose distinct structural inputs if it needs both representations.");
    }
    auto requireAggregated = [&](RaggedPartitionRequirement requested, const char* representation) {
        if (!hasRaggedPartitionRequirement(physicalization->requirements, requested)) {
            throw std::logic_error(
                std::string("Row-partition consumer requested ") + representation +
                " after placement aggregation omitted that representation.");
        }
    };

    if (hasRaggedPartitionRequirement(requirement, RaggedPartitionRequirement::DEVICE_OFFSETS)) {
        requireAggregated(RaggedPartitionRequirement::DEVICE_OFFSETS, "DEVICE_OFFSETS");
        if (physicalization->offsetsDrivingLayer == nullptr) {
            throw std::logic_error("Row-partition DEVICE_OFFSETS representation was not materialized.");
        }
        return physicalization->offsetsDrivingLayer;
    }
    if (hasRaggedPartitionRequirement(requirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT)) {
        requireAggregated(RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT, "DEVICE_ACTIVE_COUNT");
        if (physicalization->activeCountDrivingLayer == nullptr) {
            throw std::logic_error("Row-partition DEVICE_ACTIVE_COUNT representation was not materialized.");
        }
        return physicalization->activeCountDrivingLayer;
    }
    if (hasRaggedPartitionRequirement(requirement, RaggedPartitionRequirement::HOST_EXTENT)) {
        requireAggregated(RaggedPartitionRequirement::HOST_EXTENT, "HOST_EXTENT");
        if (physicalization->hostCarrierDrivingLayer == nullptr) {
            throw std::logic_error("Row-partition HOST_EXTENT carrier was not materialized.");
        }
        return physicalization->hostCarrierDrivingLayer;
    }

    throw std::logic_error("Row-partition consumer declared an unsupported partition requirement.");
}

void StampedNetwork::auditRowPartitionPhysicalizations() const {
    std::set<const NetworkInput*> expectedManagedInputs;

    auto auditManagedRepresentations = [&](const RowPartitionPhysicalization& physicalization) {
        const bool needsHostExtent = hasRaggedPartitionRequirement(
            physicalization.requirements, RaggedPartitionRequirement::HOST_EXTENT);
        const bool needsActiveCount = hasRaggedPartitionRequirement(
            physicalization.requirements, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
        const bool needsOffsets = hasRaggedPartitionRequirement(
            physicalization.requirements, RaggedPartitionRequirement::DEVICE_OFFSETS);

        THOR_THROW_IF_FALSE((physicalization.hostCarrierDrivingLayer != nullptr) == needsHostExtent);
        THOR_THROW_IF_FALSE((physicalization.activeCountInput != nullptr) == needsActiveCount);
        THOR_THROW_IF_FALSE((physicalization.activeCountDrivingLayer != nullptr) == needsActiveCount);
        THOR_THROW_IF_FALSE((physicalization.offsetsInput != nullptr) == needsOffsets);
        THOR_THROW_IF_FALSE((physicalization.offsetsDrivingLayer != nullptr) == needsOffsets);

        auto auditManagedInput = [&](const std::shared_ptr<NetworkInput>& input,
                                     const std::vector<uint64_t>& expectedDimensions) {
            THOR_THROW_IF_FALSE(input != nullptr);
            THOR_THROW_IF_FALSE(expectedManagedInputs.insert(input.get()).second);
            THOR_THROW_IF_FALSE(input->getFeatureOutput().has_value());
            THOR_THROW_IF_FALSE(input->getFeatureOutput()->getDimensions() == expectedDimensions);
            THOR_THROW_IF_FALSE(input->getFeatureOutput()->getDataType() == physicalization.descriptor.getOffsetsDataType());
            THOR_THROW_IF_FALSE(std::find(inputsShared.begin(), inputsShared.end(), input) == inputsShared.end());
            for (const auto& [name, publicInput] : inputNamedShared) {
                (void)name;
                THOR_THROW_IF_FALSE(publicInput != input);
            }
        };

        if (needsActiveCount) auditManagedInput(physicalization.activeCountInput, {1});
        if (needsOffsets) {
            auditManagedInput(physicalization.offsetsInput,
                              physicalization.descriptor.getOffsetsDescriptor().getDimensions());
        }
    };

    for (const auto& [rowPartitionId, physicalization] : externalRowPartitions) {
        THOR_THROW_IF_FALSE(rowPartitionId != 0);
        THOR_THROW_IF_FALSE(physicalization.rowPartitionId == rowPartitionId);
        THOR_THROW_IF_FALSE(physicalization.logicalPartitionToken.isInitialized());
        THOR_THROW_IF_FALSE(!physicalization.valuesInputName.empty());
        THOR_THROW_IF_FALSE(physicalization.ownerValuesInput != nullptr);
        auto logicalIt = externalRowPartitionByLogicalPartitionToken.find(physicalization.logicalPartitionToken);
        THOR_THROW_IF_FALSE(logicalIt != externalRowPartitionByLogicalPartitionToken.end());
        THOR_THROW_IF_FALSE(logicalIt->second == rowPartitionId);
        auto valuesIt = externalRowPartitionByValuesInputName.find(physicalization.valuesInputName);
        THOR_THROW_IF_FALSE(valuesIt != externalRowPartitionByValuesInputName.end());
        THOR_THROW_IF_FALSE(valuesIt->second == rowPartitionId);
        auditManagedRepresentations(physicalization);
    }

    for (const auto& [rowPartitionId, physicalization] : internalRowPartitions) {
        THOR_THROW_IF_FALSE(rowPartitionId != 0);
        THOR_THROW_IF_FALSE(physicalization.rowPartitionId == rowPartitionId);
        THOR_THROW_IF_FALSE(physicalization.logicalPartitionToken.isInitialized());
        THOR_THROW_IF_FALSE(physicalization.logicalValuesTensor.isInitialized());
        THOR_THROW_IF_FALSE(physicalization.producerPhysicalLayer != nullptr);
        THOR_THROW_IF_FALSE(!physicalization.sourceRowPartitionIds.empty());
        auto logicalIt = internalRowPartitionByLogicalPartitionToken.find(physicalization.logicalPartitionToken);
        THOR_THROW_IF_FALSE(logicalIt != internalRowPartitionByLogicalPartitionToken.end());
        THOR_THROW_IF_FALSE(logicalIt->second == rowPartitionId);
        for (RowPartitionId sourceId : physicalization.sourceRowPartitionIds) THOR_THROW_IF_FALSE(sourceId != 0);
        if (physicalization.derivationKind == InternalRowPartitionPhysicalization::DerivationKind::SEQUENCE_SLICE) {
            THOR_THROW_IF_FALSE(physicalization.sourceRowPartitionIds.size() == 1);
            THOR_THROW_IF_FALSE(physicalization.sliceLength > 0);
            THOR_THROW_IF_FALSE(hasRaggedPartitionRequirement(
                physicalization.requirements, RaggedPartitionRequirement::DEVICE_OFFSETS));
        }
        auditManagedRepresentations(physicalization);
    }

    for (const auto& [valuesOutputName, binding] : raggedOutputByValuesOutputName) {
        THOR_THROW_IF_FALSE(!valuesOutputName.empty());
        THOR_THROW_IF_FALSE(binding.rowPartitionId != 0);
        const RowPartitionPhysicalization* physicalization = nullptr;
        auto externalIt = externalRowPartitions.find(binding.rowPartitionId);
        if (externalIt != externalRowPartitions.end()) physicalization = &externalIt->second;
        auto internalIt = internalRowPartitions.find(binding.rowPartitionId);
        if (internalIt != internalRowPartitions.end()) {
            THOR_THROW_IF_FALSE(physicalization == nullptr);
            physicalization = &internalIt->second;
        }
        THOR_THROW_IF_FALSE(physicalization != nullptr);
        THOR_THROW_IF_FALSE(physicalization->descriptor == binding.descriptor);
    }

    THOR_THROW_IF_FALSE(expectedManagedInputs.size() == managedPartitionInputsShared.size());
    THOR_THROW_IF_FALSE(managedPartitionInputsShared.size() == managedPartitionInputs.size());
    for (size_t i = 0; i < managedPartitionInputsShared.size(); ++i) {
        THOR_THROW_IF_FALSE(managedPartitionInputsShared[i] != nullptr);
        THOR_THROW_IF_FALSE(managedPartitionInputs[i] == managedPartitionInputsShared[i].get());
        THOR_THROW_IF_FALSE(expectedManagedInputs.count(managedPartitionInputs[i]) == 1);
    }
}

uint64_t StampedNetwork::getFloatingPointOperationsCurrentBatchForward() {
    uint64_t total = 0;
    for (ThorImplementation::TrainableLayer* layer : trainableLayers) {
        THOR_THROW_IF_FALSE(layer != nullptr);
        total = checkedFlopAdd(total, layer->flopCountForward(), "StampedNetwork forward");
    }
    for (ThorImplementation::Layer* layer : otherLayers) {
        THOR_THROW_IF_FALSE(layer != nullptr);
        total = checkedFlopAdd(
            total,
            checkedFlopMul(layer->floatingPointOperationsPerExampleForward(), batchSize, "StampedNetwork forward"),
            "StampedNetwork forward");
    }
    return total;
}

uint64_t StampedNetwork::getFloatingPointOperationsCurrentBatchBackward() {
    uint64_t total = 0;
    for (ThorImplementation::TrainableLayer* layer : trainableLayers) {
        THOR_THROW_IF_FALSE(layer != nullptr);
        total = checkedFlopAdd(total, layer->flopCountBackward(), "StampedNetwork backward");
    }
    for (ThorImplementation::Layer* layer : otherLayers) {
        THOR_THROW_IF_FALSE(layer != nullptr);
        total = checkedFlopAdd(
            total,
            checkedFlopMul(layer->floatingPointOperationsPerExampleBackward(), batchSize, "StampedNetwork backward"),
            "StampedNetwork backward");
    }
    return total;
}

uint64_t StampedNetwork::getFloatingPointOperationsCurrentBatchTraining() {
    return checkedFlopAdd(getFloatingPointOperationsCurrentBatchForward(),
                          getFloatingPointOperationsCurrentBatchBackward(),
                          "StampedNetwork training");
}


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
    for (NetworkInput* input : managedPartitionInputs)
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
    appendLayerEvents(managedPartitionInputsShared);
    appendLayerEvents(outputsShared);
    appendLayerEvents(trainableLayersShared);
    appendLayerEvents(otherLayersShared);
    return events;
}


void StampedNetwork::initialize(bool initializeWeights, bool copyWeightsFromOtherStamp, StampedNetwork *otherStamp) {
    // First, ensure the shared pointers and raw pointers match
    for (auto it = inputsShared.begin(); it != inputsShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(inputs, it->get()) == 1);
    for (auto it = managedPartitionInputsShared.begin(); it != managedPartitionInputsShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(managedPartitionInputs, it->get()) == 1);
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
        THOR_THROW_IF_FALSE(raggedInputNamed[it->first].partitionInputName == it->second.partitionInputName);
        THOR_THROW_IF_FALSE(raggedInputNamed[it->first].descriptor == it->second.descriptor);
        THOR_THROW_IF_FALSE(raggedInputNamed[it->first].rowPartitionId == it->second.rowPartitionId);
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

    std::map<RowPartitionId, std::vector<uint64_t>> submittedHostOffsetsByPartition;
    for (const auto& [name, value] : batchInputs.values()) {
        const std::optional<Thor::BatchSourceReference> sourceReference =
            batchInputs.getSourceReference(name);
        if (std::holds_alternative<Tensor>(value)) {
            Tensor inputTensor = std::get<Tensor>(value);
            auto raggedIt = raggedInputNamed.find(name);
            if (raggedIt != raggedInputNamed.end()) {
                const RaggedInputBinding& binding = raggedIt->second;
                if (binding.ownsPartition()) {
                    throw std::runtime_error(
                        "StampedNetwork::sendBatch partition-owning ragged input '" + name +
                        "' requires a PhysicalRaggedTensor; only shared-partition ragged inputs may be supplied as packed PhysicalTensor values.");
                }
                if (inputTensor.getDescriptor() != binding.descriptor.getValuesDescriptor()) {
                    throw std::runtime_error(
                        "StampedNetwork::sendBatch shared-partition ragged input '" + name +
                        "' values descriptor mismatch: received=" + inputTensor.getDescriptor().toString() +
                        " expected=" + binding.descriptor.getValuesDescriptor().toString() + ".");
                }
                requireConsistentBatchCapacity(binding.descriptor.getBatchSize());
                THOR_THROW_IF_FALSE(
                    physicalBatchInputs.emplace(
                        binding.valuesInputName,
                        PhysicalBatchInput{inputTensor, sourceReference}).second);
            } else {
                const std::vector<uint64_t> dimensions = inputTensor.getDescriptor().getDimensions();
                THOR_THROW_IF_FALSE(!dimensions.empty());
                requireConsistentBatchCapacity(dimensions[0]);
                THOR_THROW_IF_FALSE(
                    physicalBatchInputs.emplace(
                        name,
                        PhysicalBatchInput{inputTensor, sourceReference}).second);
            }
        } else if (std::holds_alternative<RaggedTensor>(value)) {
            auto raggedIt = raggedInputNamed.find(name);
            THOR_THROW_IF_FALSE(raggedIt != raggedInputNamed.end());
            const RaggedInputBinding& binding = raggedIt->second;
            RaggedTensor raggedTensor = std::get<RaggedTensor>(value);
            if (raggedTensor.getDescriptor() != binding.descriptor) {
                throw std::runtime_error(
                    "StampedNetwork::sendBatch ragged input '" + name +
                    "' descriptor mismatch: received=" + raggedTensor.getDescriptor().toString() +
                    " expected=" + binding.descriptor.toString() + ".");
            }
            requireConsistentBatchCapacity(raggedTensor.getBatchSize());
            std::optional<std::vector<uint64_t>> maybeHostOffsets =
                raggedTensor.getHostOffsetsIfAvailable();
            Tensor submittedOffsets = raggedTensor.getOffsets();
            if (submittedOffsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
                std::vector<uint64_t> cpuOffsets(raggedTensor.getBatchSize() + 1, 0);
                if (submittedOffsets.getDataType() == DataType::UINT32) {
                    const uint32_t* raw = submittedOffsets.getMemPtr<uint32_t>();
                    for (uint64_t i = 0; i <= raggedTensor.getBatchSize(); ++i) cpuOffsets[i] = raw[i];
                } else {
                    THOR_THROW_IF_FALSE(submittedOffsets.getDataType() == DataType::UINT64);
                    const uint64_t* raw = submittedOffsets.getMemPtr<uint64_t>();
                    for (uint64_t i = 0; i <= raggedTensor.getBatchSize(); ++i) cpuOffsets[i] = raw[i];
                }
                if (maybeHostOffsets.has_value() && maybeHostOffsets.value() != cpuOffsets) {
                    throw std::runtime_error(
                        "StampedNetwork::sendBatch ragged input '" + name +
                        "' has CPU offsets that disagree with its authoritative host row partition.");
                }
                if (!maybeHostOffsets.has_value()) {
                    raggedTensor.getRowPartitionRuntime().setHostOffsets(cpuOffsets);
                    maybeHostOffsets = cpuOffsets;
                }
            }
            if (!maybeHostOffsets.has_value()) {
                throw std::runtime_error(
                    "StampedNetwork::sendBatch ragged input '" + name +
                    "' has no authoritative host row partition. GPU-only offsets cannot establish ragged semantics without an implicit device-to-host synchronization.");
            }
            const std::vector<uint64_t>& hostOffsets = maybeHostOffsets.value();
            THOR_THROW_IF_FALSE(binding.rowPartitionId != 0);
            auto [it, inserted] = submittedHostOffsetsByPartition.emplace(binding.rowPartitionId, hostOffsets);
            if (!inserted && it->second != hostOffsets) {
                throw std::runtime_error(
                    "StampedNetwork::sendBatch ragged inputs sharing logical partition " +
                    std::to_string(binding.rowPartitionId) +
                    " were submitted with different authoritative host offsets.");
            }
            PhysicalBatchInput valuesInput{raggedTensor.getValues(), sourceReference};
            if (binding.ownsPartition()) {
                // The owning packed-values input is also the host-state carrier.
                // Publishing metadata here costs no device partition allocation and
                // lets HOST_EXTENT consumers route to an ordinary values tensor.
                valuesInput.rowPartitionDescriptor =
                    raggedTensor.getRowPartitionRuntime().getDescriptor();
                valuesInput.rowPartitionHostOffsets = hostOffsets;
                valuesInput.logicalRowPartitionId = binding.rowPartitionId;
            }
            THOR_THROW_IF_FALSE(
                physicalBatchInputs.emplace(binding.valuesInputName, std::move(valuesInput)).second);
            if (binding.ownsPartition()) {
                // The physical offsets input is hidden from the user-visible input
                // map. Thor populates it from the authoritative host partition
                // using the same managed NetworkInput ring as RP6A.
                auto physicalizationIt = externalRowPartitions.find(binding.rowPartitionId);
                THOR_THROW_IF_FALSE(physicalizationIt != externalRowPartitions.end());
                if (physicalizationIt->second.activeCountInput != nullptr) {
                    const std::string& physicalInputName =
                        physicalizationIt->second.activeCountInput->getName();
                    THOR_THROW_IF_FALSE(
                        physicalBatchInputs.emplace(
                            physicalInputName,
                            PhysicalBatchInput{ManagedRowPartitionOffsetsInput{
                                raggedTensor.getRowPartitionRuntime().getDescriptor(),
                                hostOffsets,
                                binding.rowPartitionId,
                                /*activeCountOnly=*/true}}).second);
                }
                if (physicalizationIt->second.offsetsInput != nullptr) {
                    const std::string& physicalInputName = physicalizationIt->second.offsetsInput->getName();
                    THOR_THROW_IF_FALSE(
                        physicalBatchInputs.emplace(
                            physicalInputName,
                            PhysicalBatchInput{ManagedRowPartitionOffsetsInput{
                                raggedTensor.getRowPartitionRuntime().getDescriptor(),
                                hostOffsets,
                                binding.rowPartitionId,
                                /*activeCountOnly=*/false}}).second);
                }
            }
        } else if (std::holds_alternative<Thor::DeviceBatchReference>(value)) {
            Thor::DeviceBatchReference reference = std::get<Thor::DeviceBatchReference>(value);
            auto raggedIt = raggedInputNamed.find(name);
            if (raggedIt != raggedInputNamed.end()) {
                const RaggedInputBinding& binding = raggedIt->second;
                if (binding.ownsPartition()) {
                    throw std::runtime_error(
                        "StampedNetwork::sendBatch partition-owning ragged input '" + name +
                        "' requires a PhysicalRaggedTensor; only shared-partition ragged inputs may be supplied as packed DeviceBatchReference values.");
                }
                if (reference.getOutputDescriptor() != binding.descriptor.getValuesDescriptor()) {
                    throw std::runtime_error(
                        "StampedNetwork::sendBatch shared-partition ragged input '" + name +
                        "' device-reference descriptor mismatch: received=" + reference.getOutputDescriptor().toString() +
                        " expected=" + binding.descriptor.getValuesDescriptor().toString() + ".");
                }
                requireConsistentBatchCapacity(binding.descriptor.getBatchSize());
                THOR_THROW_IF_FALSE(
                    physicalBatchInputs.emplace(
                        binding.valuesInputName,
                        PhysicalBatchInput{std::move(reference), sourceReference}).second);
            } else {
                requireConsistentBatchCapacity(reference.getBatchCapacity());
                THOR_THROW_IF_FALSE(
                    physicalBatchInputs.emplace(
                        name,
                        PhysicalBatchInput{std::move(reference), sourceReference}).second);
            }
        } else {
            THOR_UNREACHABLE();
        }
    }

    // Derive every internally-created partition from authoritative source host
    // state before any physical NetworkInput is forwarded. This makes the new
    // RowPartitionId semantic state independent of GPU-produced metadata and
    // supports recursive creator chains without a device-to-host readback.
    std::set<RowPartitionId> resolvingInternalPartitions;
    std::function<const std::vector<uint64_t>&(RowPartitionId)> resolveHostOffsets =
        [&](RowPartitionId rowPartitionId) -> const std::vector<uint64_t>& {
        auto published = submittedHostOffsetsByPartition.find(rowPartitionId);
        if (published != submittedHostOffsetsByPartition.end()) return published->second;

        auto internalIt = internalRowPartitions.find(rowPartitionId);
        if (internalIt == internalRowPartitions.end()) {
            throw std::runtime_error(
                "StampedNetwork::sendBatch cannot resolve authoritative host offsets for logical partition " +
                std::to_string(rowPartitionId) + ".");
        }
        if (!resolvingInternalPartitions.insert(rowPartitionId).second) {
            throw std::logic_error("Internal ragged row-partition derivations contain a cycle.");
        }

        const InternalRowPartitionPhysicalization& physicalization = internalIt->second;
        std::vector<uint64_t> derived(physicalization.descriptor.getBatchSize() + 1, 0);
        if (physicalization.derivationKind == InternalRowPartitionPhysicalization::DerivationKind::SEQUENCE_SLICE) {
            THOR_THROW_IF_FALSE(physicalization.sourceRowPartitionIds.size() == 1);
            const std::vector<uint64_t>& source = resolveHostOffsets(physicalization.sourceRowPartitionIds.front());
            THOR_THROW_IF_FALSE(source.size() == derived.size());
            for (uint64_t row = 0; row < physicalization.descriptor.getBatchSize(); ++row) {
                THOR_THROW_IF_FALSE(source[row] <= source[row + 1]);
                const uint64_t rowLength = source[row + 1] - source[row];
                const uint64_t slicedLength = rowLength <= physicalization.sliceStart
                    ? 0
                    : std::min<uint64_t>(physicalization.sliceLength, rowLength - physicalization.sliceStart);
                THOR_THROW_IF_FALSE(derived[row] <= std::numeric_limits<uint64_t>::max() - slicedLength);
                derived[row + 1] = derived[row] + slicedLength;
            }
        } else {
            THOR_THROW_IF_FALSE(
                physicalization.derivationKind == InternalRowPartitionPhysicalization::DerivationKind::SEQUENCE_CONCATENATE);
            for (RowPartitionId sourceId : physicalization.sourceRowPartitionIds) {
                const std::vector<uint64_t>& source = resolveHostOffsets(sourceId);
                THOR_THROW_IF_FALSE(source.size() == derived.size());
                for (uint64_t boundary = 0; boundary < derived.size(); ++boundary) {
                    THOR_THROW_IF_FALSE(derived[boundary] <= std::numeric_limits<uint64_t>::max() - source[boundary]);
                    derived[boundary] += source[boundary];
                }
            }
        }
        THOR_THROW_IF_FALSE(!derived.empty() && derived.front() == 0);
        THOR_THROW_IF_FALSE(derived.back() <= physicalization.descriptor.getMaxTotalValues());
        resolvingInternalPartitions.erase(rowPartitionId);
        return submittedHostOffsetsByPartition.emplace(rowPartitionId, std::move(derived)).first->second;
    };

    for (auto& [rowPartitionId, physicalization] : internalRowPartitions) {
        const std::vector<uint64_t>& hostOffsets = resolveHostOffsets(rowPartitionId);
        THOR_THROW_IF_FALSE(physicalization.producerPhysicalLayer != nullptr);
        const std::optional<Tensor> producerValues = physicalization.producerPhysicalLayer->getFeatureOutput();
        THOR_THROW_IF_FALSE(producerValues.has_value());
        RowPartitionRuntime::publishHostState(
            producerValues.value(), physicalization.descriptor, rowPartitionId, hostOffsets);

        auto addManagedInput = [&](const std::shared_ptr<NetworkInput>& input, bool activeCountOnly) {
            if (input == nullptr) return;
            THOR_THROW_IF_FALSE(physicalBatchInputs.emplace(
                input->getName(),
                PhysicalBatchInput{ManagedRowPartitionOffsetsInput{
                    physicalization.descriptor,
                    hostOffsets,
                    rowPartitionId,
                    activeCountOnly}}).second);
        };
        addManagedInput(physicalization.activeCountInput, /*activeCountOnly=*/true);
        addManagedInput(physicalization.offsetsInput, /*activeCountOnly=*/false);
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

    // A logical ragged output does not consume a physical partition representation.
    // The authoritative host partition is already known for this submission, so
    // attach it directly to the output slot's values tensor. This keeps output
    // semantics batch-local without inserting HOST_EXTENT fanouts into otherwise
    // values-only producer graphs. Metadata publication is host-only and does not
    // wait for, inspect, or copy device structural bytes.
    for (const auto& [valuesOutputName, binding] : raggedOutputByValuesOutputName) {
        auto outputIt = batchOutputs.find(valuesOutputName);
        if (outputIt == batchOutputs.end()) {
            // Inference placement may prune training-only outputs.
            continue;
        }
        const std::vector<uint64_t>& hostOffsets = resolveHostOffsets(binding.rowPartitionId);
        RowPartitionRuntime::publishHostState(
            outputIt->second, binding.descriptor, binding.rowPartitionId, hostOffsets);
    }

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
    THOR_THROW_IF_FALSE(batchInputs.size() == inputs.size() + managedPartitionInputs.size());
    THOR_THROW_IF_FALSE(physicalBatchCapacity >= 1);
    THOR_THROW_IF_FALSE(validExampleCount >= 1);
    THOR_THROW_IF_FALSE(validExampleCount <= physicalBatchCapacity);

    for (const auto& [inputName, input] : batchInputs) {
        if (std::holds_alternative<ManagedRowPartitionOffsetsInput>(input.value)) {
            THOR_THROW_IF_FALSE(!input.sourceReference.has_value());
            THOR_THROW_IF_FALSE(!input.rowPartitionDescriptor.has_value());
            THOR_THROW_IF_FALSE(!input.rowPartitionHostOffsets.has_value());
            const auto& managed = std::get<ManagedRowPartitionOffsetsInput>(input.value);
            THOR_THROW_IF_FALSE(managed.descriptor.getBatchSize() == physicalBatchCapacity);
            THOR_THROW_IF_FALSE(managed.hostOffsets.size() == managed.descriptor.getBatchSize() + 1);
            if (managed.logicalRowPartitionId.has_value())
                THOR_THROW_IF_FALSE(managed.logicalRowPartitionId.value() != 0);
            continue;
        }
        const bool carriesRowPartitionHostState = input.rowPartitionDescriptor.has_value();
        if (carriesRowPartitionHostState != input.rowPartitionHostOffsets.has_value() ||
            input.logicalRowPartitionId.has_value() != carriesRowPartitionHostState) {
            throw std::runtime_error(
                "StampedNetwork::sendPhysicalBatch physical input '" + inputName +
                "' must provide descriptor, logical partition id, and authoritative host offsets together.");
        }
    }

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
    for (NetworkInput* input : inputs) {
        input->setActiveInputSlot(queueSlot);
    }
    for (NetworkInput* input : managedPartitionInputs) {
        input->setActiveInputSlot(queueSlot);
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

    // A partition-owning logical RaggedNetworkInput always materializes values.
    // The values input carries authoritative host partition metadata, while hidden
    // active-count/full-offset inputs exist only when placement requested them.
    // Packed storage beyond the active prefix remains undefined.

    auto forwardPhysicalInput = [&](NetworkInput* input, bool managedPartitionInput) {
        THOR_THROW_IF_FALSE(input != nullptr);
        auto it = batchInputs.find(input->getName());
        THOR_THROW_IF_FALSE(it != batchInputs.end());
        const auto readyIt = inputReadyEvents.find(input->getName());
        if (managedPartitionInput) {
            THOR_THROW_IF_FALSE(readyIt == inputReadyEvents.end());
        }
        if (std::holds_alternative<Tensor>(it->second.value)) {
            Tensor inputTensor = std::get<Tensor>(it->second.value);
            if (it->second.rowPartitionDescriptor.has_value()) {
                THOR_THROW_IF_FALSE(it->second.logicalRowPartitionId.has_value());
                // Logical ragged Batch submission has no separate per-field ready
                // event. Publish authoritative host state on the ordinary values
                // NetworkInput before downstream notification.
                THOR_THROW_IF_FALSE(readyIt == inputReadyEvents.end());
                input->forwardWithRowPartitionHostState(
                    inputTensor,
                    isInferenceOnly,
                    validExampleCount,
                    it->second.rowPartitionDescriptor.value(),
                    it->second.logicalRowPartitionId.value(),
                    it->second.rowPartitionHostOffsets.value(),
                    it->second.sourceReference);
            } else if (readyIt != inputReadyEvents.end()) {
                input->forward(
                    inputTensor,
                    isInferenceOnly,
                    readyIt->second,
                    validExampleCount,
                    it->second.sourceReference);
            } else {
                input->forward(
                    inputTensor,
                    isInferenceOnly,
                    validExampleCount,
                    it->second.sourceReference);
            }
        } else if (std::holds_alternative<Thor::DeviceBatchReference>(it->second.value)) {
            THOR_THROW_IF_FALSE(readyIt == inputReadyEvents.end());
            input->forward(
                std::get<Thor::DeviceBatchReference>(it->second.value),
                isInferenceOnly,
                validExampleCount,
                it->second.sourceReference);
        } else if (std::holds_alternative<ManagedRowPartitionOffsetsInput>(it->second.value)) {
            THOR_THROW_IF_FALSE(readyIt == inputReadyEvents.end());
            THOR_THROW_IF_FALSE(!it->second.sourceReference.has_value());
            ManagedRowPartitionOffsetsInput managed =
                std::get<ManagedRowPartitionOffsetsInput>(std::move(it->second.value));
            if (managed.activeCountOnly) {
                THOR_THROW_IF_FALSE(managed.logicalRowPartitionId.has_value());
                input->forwardManagedRowPartitionActiveCount(
                    isInferenceOnly,
                    managed.descriptor,
                    validExampleCount,
                    std::move(managed.hostOffsets),
                    managed.logicalRowPartitionId.value());
            } else {
                input->forwardManagedRowPartitionOffsets(
                    isInferenceOnly,
                    managed.descriptor,
                    validExampleCount,
                    std::move(managed.hostOffsets),
                    managed.logicalRowPartitionId);
            }
        } else {
            THOR_UNREACHABLE();
        }
    };

    for (NetworkInput* input : inputs)
        forwardPhysicalInput(input, /*managedPartitionInput=*/false);
    for (NetworkInput* input : managedPartitionInputs)
        forwardPhysicalInput(input, /*managedPartitionInput=*/true);
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
    for (NetworkInput* input : managedPartitionInputs) {
        input->getStream().waitEvent(processingFinishedEvent);
    }
    const auto inputFanoutFinish = timingNow(submitTiming);

    if (submitTiming != nullptr) {
        submitTiming->physicalTotalMicros += elapsedMicros(physicalStart, inputFanoutFinish);
        submitTiming->inputForwardMicros += elapsedMicros(inputForwardStart, inputForwardFinish);
        submitTiming->outputCollectMicros += elapsedMicros(outputCollectStart, outputCollectFinish);
        submitTiming->outputWaitOnProcessingMicros += elapsedMicros(outputWaitStart, outputWaitFinish);
        submitTiming->processingEventMicros += elapsedMicros(processingEventStart, processingEventFinish);
        submitTiming->inputFanoutMicros += elapsedMicros(inputFanoutStart, inputFanoutFinish);
        submitTiming->numInputs += inputs.size() + managedPartitionInputs.size();
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
    cleanupLayers(managedPartitionInputs);
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
    managedPartitionInputsShared.clear();
    externalRowPartitions.clear();
    internalRowPartitions.clear();
    externalRowPartitionByLogicalPartitionToken.clear();
    internalRowPartitionByLogicalPartitionToken.clear();
    externalRowPartitionByValuesInputName.clear();
    raggedOutputByValuesOutputName.clear();
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
    for (NetworkInput* input : managedPartitionInputs) {
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
