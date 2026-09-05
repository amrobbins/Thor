#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Data/BatchSourceResource.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Layers/Metric.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include <vector>
#include <optional>
#include <cstdint>
#include <map>
#include <string>
#include <variant>

#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

namespace Thor {
class Network;
class PlacedNetwork;
}  // namespace Thor

namespace ThorImplementation {

struct ManagedRowPartitionOffsetsInput {
    RowPartitionDescriptor descriptor;
    std::vector<uint64_t> hostOffsets;
    std::optional<RowPartitionId> logicalRowPartitionId = std::nullopt;
    bool activeCountOnly = false;
};

struct PhysicalBatchInput {
    // ManagedRowPartitionOffsetsInput is intentionally a normal physical NetworkInput
    // submission. The difference from Tensor is ownership: Thor generates and owns
    // the host source bytes and the NetworkInput slot keeps them alive through the
    // asynchronous H2D copy.
    std::variant<Tensor, Thor::DeviceBatchReference, ManagedRowPartitionOffsetsInput> value;
    std::optional<Thor::BatchSourceReference> sourceReference;

    // Transitional compatibility for callers that still submit a materialized
    // physical offsets Tensor. Logical RaggedNetworkInput execution uses the managed
    // variant above instead.
    std::optional<RowPartitionDescriptor> rowPartitionDescriptor;
    std::optional<std::vector<uint64_t>> rowPartitionHostOffsets;
    std::optional<RowPartitionId> logicalRowPartitionId;
};

struct PartialBatchIncompatibility {
    uint64_t layerId = 0;
    std::string layerName;
    std::string layerType;

    bool operator==(const PartialBatchIncompatibility&) const = default;
};

struct BatchSubmissionTiming {
    uint64_t activeObjectiveRootsMicros = 0;
    uint64_t setActiveObjectiveRootsMicros = 0;
    uint64_t sendBatchMicros = 0;
    uint64_t batchUnwrapMicros = 0;
    uint64_t physicalTotalMicros = 0;
    uint64_t inputForwardMicros = 0;
    uint64_t outputCollectMicros = 0;
    uint64_t outputWaitOnProcessingMicros = 0;
    uint64_t processingEventMicros = 0;
    uint64_t inputFanoutMicros = 0;
    uint64_t totalMicros = 0;
    uint64_t numInputs = 0;
    uint64_t numOutputs = 0;
    uint64_t activeObjectiveRootCount = 0;
};

inline void accumulateBatchSubmissionTiming(BatchSubmissionTiming& dst, const BatchSubmissionTiming& src) {
    dst.activeObjectiveRootsMicros += src.activeObjectiveRootsMicros;
    dst.setActiveObjectiveRootsMicros += src.setActiveObjectiveRootsMicros;
    dst.sendBatchMicros += src.sendBatchMicros;
    dst.batchUnwrapMicros += src.batchUnwrapMicros;
    dst.physicalTotalMicros += src.physicalTotalMicros;
    dst.inputForwardMicros += src.inputForwardMicros;
    dst.outputCollectMicros += src.outputCollectMicros;
    dst.outputWaitOnProcessingMicros += src.outputWaitOnProcessingMicros;
    dst.processingEventMicros += src.processingEventMicros;
    dst.inputFanoutMicros += src.inputFanoutMicros;
    dst.totalMicros += src.totalMicros;
    dst.numInputs += src.numInputs;
    dst.numOutputs += src.numOutputs;
    dst.activeObjectiveRootCount += src.activeObjectiveRootCount;
}

/**
 * One physical network stamp. Host submission into a stamp is serialized: callers
 * must not submit two batches to the same StampedNetwork concurrently from different
 * host threads. A submission may enqueue asynchronous work on many CUDA streams; the
 * single-host-thread contract does not imply single-stream GPU execution.
 */
class StampedNetwork {
   private:
    struct LayerComparatorShared {
        bool operator()(const std::shared_ptr<Layer> &lhs, const std::shared_ptr<Layer> &rhs) const { return lhs->getId() < rhs->getId(); }
    };
    struct LayerComparator {
        bool operator()(const Layer *lhs, const Layer *rhs) const { return lhs->getId() < rhs->getId(); }
    };
    template <typename T>
    int count(std::vector<T> v, T item) {
        uint32_t c = 0;
        for (uint32_t i = 0; i < v.size(); ++i) {
            if (v[i] == item)
                c += 1;
        }
        return c;
    }

   public:
    uint32_t getGpuNum() const { return gpuNum; }
    uint64_t getFloatingPointOperationsPerExampleForward() const { return floatingPointOperationsPerExampleForward; }
    uint64_t getFloatingPointOperationsPerExampleBackward() const { return floatingPointOperationsPerExampleBackward; }
    uint64_t getFloatingPointOperationsPerExampleTraining() const {
        return floatingPointOperationsPerExampleForward + floatingPointOperationsPerExampleBackward;
    }
    // Batch-local logical FLOP totals. Trainable layers expose their stamped
    // execution-plan counts directly, which lets ragged Attention replace its
    // capacity estimate with the row lengths published for the current batch.
    // Non-trainable layers retain their existing fixed per-example accounting.
    [[nodiscard]] uint64_t getFloatingPointOperationsCurrentBatchForward();
    [[nodiscard]] uint64_t getFloatingPointOperationsCurrentBatchBackward();
    [[nodiscard]] uint64_t getFloatingPointOperationsCurrentBatchTraining();
    struct RaggedInputBinding {
        std::string valuesInputName;
        std::string offsetsInputName;
        std::optional<std::string> partitionInputName;
        ThorImplementation::RaggedTensorDescriptor descriptor;
        RowPartitionId rowPartitionId = 0;

        [[nodiscard]] bool ownsPartition() const { return !partitionInputName.has_value(); }
    };

    // One external logical partition may have requirement-driven Thor-managed
    // physical representations. HOST_EXTENT reuses the owning values carrier and
    // therefore allocates no partition tensor; DEVICE_ACTIVE_COUNT and
    // DEVICE_OFFSETS materialize hidden [1] and [B+1] NetworkInputs respectively.
    // None of these inputs are part of the public batch-input surface.
    struct ExternalRowPartitionPhysicalization {
        RowPartitionId rowPartitionId = 0;
        RowPartitionDescriptor descriptor;
        Thor::Tensor logicalOffsetsTensor;
        std::string valuesInputName;
        ThorImplementation::RaggedPartitionRequirement requirements =
            ThorImplementation::RaggedPartitionRequirement::NONE;
        std::shared_ptr<ThorImplementation::NetworkInput> ownerValuesInput;
        std::shared_ptr<ThorImplementation::Layer> hostCarrierDrivingLayer;
        std::shared_ptr<ThorImplementation::NetworkInput> activeCountInput;
        std::shared_ptr<ThorImplementation::Layer> activeCountDrivingLayer;
        std::shared_ptr<ThorImplementation::NetworkInput> offsetsInput;
        std::shared_ptr<ThorImplementation::Layer> offsetsDrivingLayer;
    };

    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> getInputs() { return inputsShared; }
    std::vector<std::shared_ptr<ThorImplementation::NetworkOutput>> getOutputs() { return outputsShared; }

    /**
     * Records synchronization events for every physical layer in this stamp.
     * Synchronizing every returned event guarantees that all work enqueued for
     * the stamp before this call, including parameter/optimizer initialization,
     * has completed. Callers must prevent concurrent batch submission while
     * taking this synchronization snapshot.
     */
    std::vector<Event> getSynchronizeEvents() const;

    std::vector<std::string> getNamedInputNames() const {
        std::vector<std::string> names;
        names.reserve(inputNamedShared.size());
        for (const auto& [name, _] : inputNamedShared) {
            names.push_back(name);
        }
        return names;
    }
    std::vector<std::string> getNamedOutputNames() const {
        std::vector<std::string> names;
        names.reserve(outputNamedShared.size());
        for (const auto& [name, _] : outputNamedShared) {
            names.push_back(name);
        }
        return names;
    }
    std::shared_ptr<ThorImplementation::NetworkInput> getNamedInput(const std::string& name) const {
        auto it = inputNamedShared.find(name);
        if (it == inputNamedShared.end()) {
            return nullptr;
        }
        return it->second;
    }
    std::shared_ptr<ThorImplementation::NetworkOutput> getNamedOutput(const std::string& name) const {
        auto it = outputNamedShared.find(name);
        if (it == outputNamedShared.end()) {
            return nullptr;
        }
        return it->second;
    }
    void preallocateInputSlots(uint32_t numSlots);
    void preallocateOutputSlots(uint32_t numSlots);
    [[nodiscard]] std::map<std::string, MetricBatchStatisticTensors> getMetricBatchStatisticTensorsForSlot(
        uint32_t slotIndex) const;
    void extendMetricStatisticWritableEvents(Event event, std::optional<uint32_t> outputSlotIndex = std::nullopt);

    uint64_t getNumTrainableLayers() { return trainableLayersShared.size(); }
    std::shared_ptr<ThorImplementation::TrainableLayer> &getTrainableLayer(uint64_t i) { return trainableLayersShared[i]; }
    std::vector<std::shared_ptr<ThorImplementation::Layer>> getOtherLayers() { return otherLayersShared; }
    void setActiveTrainingLossRoots(const std::vector<Thor::Tensor>& activeRawLossRoots);
    std::vector<uint64_t> getActiveTrainingRawLossOriginalIdsForDebug() const;
    void setTrainingDropoutEnabled(bool enabled);
    [[nodiscard]] bool isTrainingDropoutEnabled() const;
    [[nodiscard]] uint32_t getNumTrainingDropoutControllableLayers() const;
    [[nodiscard]] std::vector<PartialBatchIncompatibility>
    getPartialBatchIncompatibilities() const;

    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(uint64_t apiLayerId) {
        return apiLayerToPhysicalLayerShared[apiLayerId];
    }
    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(std::shared_ptr<Thor::Layer> apiLayer) {
        return apiLayerToPhysicalLayerShared[apiLayer->getId()];
    }
    // void recordIfParameterizable(std::shared_ptr<Thor::Layer> layer, std::shared_ptr<ThorImplementation::Layer> implementationLayer) {
    //     std::shared_ptr<Thor::Parameterizable> parameterizable = dynamic_pointer_cast<Thor::Parameterizable>(layer);
    //     if (parameterizable != nullptr) {
    //         auto implementationParameterizable = std::dynamic_pointer_cast<ThorImplementation::Parameterizable>(implementationLayer);
    //         THOR_THROW_IF_FALSE(implementationParameterizable != nullptr);
    //         apiParameterizableToPhysicalParameterizable[parameterizable->getId()] = implementationParameterizable;
    //     }
    // }
    // std::shared_ptr<ThorImplementation::Parameterizable> getPhysicalParameterizableFromApiParameterizable(uint64_t apiParameterizableId)
    // {
    //     auto it = apiParameterizableToPhysicalParameterizable.find(apiParameterizableId);
    //     THOR_THROW_IF_FALSE(it != apiParameterizableToPhysicalParameterizable.end());
    //     return it->second;
    // }
    // std::shared_ptr<ThorImplementation::Parameterizable> getPhysicalParameterizableFromApiParameterizable(
    //     std::shared_ptr<Thor::Layer> apiParameterizable) {
    //     THOR_THROW_IF_FALSE(apiParameterizable != nullptr);
    //     uint64_t apiParameterizableId = apiParameterizable->getId();
    //     return getPhysicalParameterizableFromApiParameterizable(apiParameterizableId);
    // }

#if defined(THOR_GTEST) || defined(__JETBRAINS_IDE__)
    std::map<uint64_t, std::shared_ptr<ThorImplementation::Layer>> getApiLayerToPhysicalLayer() { return apiLayerToPhysicalLayerShared; }
    std::shared_ptr<ThorImplementation::NetworkInput> getManagedPartitionOffsetsInputForTest(
        RowPartitionId rowPartitionId) const {
        auto it = externalRowPartitions.find(rowPartitionId);
        if (it == externalRowPartitions.end())
            return nullptr;
        return it->second.offsetsInput;
    }
    std::shared_ptr<ThorImplementation::NetworkInput> getManagedPartitionActiveCountInputForTest(
        RowPartitionId rowPartitionId) const {
        auto it = externalRowPartitions.find(rowPartitionId);
        if (it == externalRowPartitions.end())
            return nullptr;
        return it->second.activeCountInput;
    }
    std::optional<RaggedPartitionRequirement> getExternalRowPartitionRequirementsForTest(
        RowPartitionId rowPartitionId) const {
        auto it = externalRowPartitions.find(rowPartitionId);
        if (it == externalRowPartitions.end())
            return std::nullopt;
        return it->second.requirements;
    }
#endif

   protected:
    void initialize(bool initializeWeights, bool copyWeightsFromOtherStamp, StampedNetwork *otherStamp = nullptr);

    // Note that all processing is finished at the end of any input stream of the stamp.
    // Note *input* stream - this is not the case for the batch-source streams
    Event sendBatch(std::map<std::string, Tensor> batchInputs,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly,
                    Event* reusableProcessingFinishedEvent = nullptr,
                    bool waitForOutputsOnProcessingStream = true,
                    BatchSubmissionTiming* submitTiming = nullptr,
                    std::optional<uint32_t> outputSlotIndex = std::nullopt);
    Event sendBatch(std::map<std::string, Tensor> batchInputs,
                    const std::map<std::string, Event>& inputReadyEvents,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly,
                    Event* reusableProcessingFinishedEvent = nullptr,
                    bool waitForOutputsOnProcessingStream = true,
                    BatchSubmissionTiming* submitTiming = nullptr,
                    std::optional<uint32_t> outputSlotIndex = std::nullopt);

    Event sendBatch(const Batch& batchInputs,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly,
                    Event* reusableProcessingFinishedEvent = nullptr,
                    bool waitForOutputsOnProcessingStream = true,
                    BatchSubmissionTiming* submitTiming = nullptr,
                    std::optional<uint32_t> outputSlotIndex = std::nullopt);

    Event sendPhysicalBatch(std::map<std::string, PhysicalBatchInput> batchInputs,
                            const std::map<std::string, Event>& inputReadyEvents,
                            std::map<std::string, Tensor> &batchOutputs,
                            std::map<std::string, Event> &outputReadyEvents,
                            bool isInferenceOnly,
                            uint32_t physicalBatchCapacity,
                            uint32_t validExampleCount,
                            Event* reusableProcessingFinishedEvent = nullptr,
                            bool waitForOutputsOnProcessingStream = true,
                            BatchSubmissionTiming* submitTiming = nullptr,
                            std::optional<uint32_t> outputSlotIndex = std::nullopt);

    void extendOutputWritableEvents(Event event, std::optional<uint32_t> outputSlotIndex = std::nullopt);

    void clear();

    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputsShared;
    std::vector<std::shared_ptr<ThorImplementation::NetworkOutput>> outputsShared;
    std::vector<std::shared_ptr<ThorImplementation::TrainableLayer>> trainableLayersShared;
    std::shared_ptr<GradientUpdateStreamPool> gradientUpdateStreamPool;
    std::vector<std::shared_ptr<ThorImplementation::Layer>> otherLayersShared;
    std::vector<Event> initializationDoneEvents;
    std::map<Thor::Tensor, std::shared_ptr<ThorImplementation::Layer>> apiTensorToPhysicalDrivingLayerShared;
    std::map<uint64_t, std::shared_ptr<ThorImplementation::Layer>> apiLayerToPhysicalLayerShared;
    std::map<std::shared_ptr<ThorImplementation::Layer>, uint64_t, StampedNetwork::LayerComparatorShared> physicalLayerToApiLayerShared;
    std::map<Thor::Tensor, std::shared_ptr<Thor::Layer>> apiTensorToApiDrivingLayerShared;
    std::map<std::string, std::shared_ptr<ThorImplementation::NetworkInput>> inputNamedShared;
    std::map<std::string, RaggedInputBinding> raggedInputNamedShared;
    std::map<std::string, std::shared_ptr<ThorImplementation::NetworkOutput>> outputNamedShared;
    // Internal sufficient-statistic source for each public metric output. Hidden
    // ratio numerator/denominator tensors never enter outputNamedShared.
    std::map<std::string, std::shared_ptr<ThorImplementation::Metric>> metricStatisticsByOutputNameShared;

    // Thor-managed physical row-partition inputs are ordinary NetworkInputs for
    // execution/lifetime purposes, but are intentionally absent from inputsShared,
    // inputNamedShared, and the user-visible physical batch contract.
    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> managedPartitionInputsShared;
    std::vector<ThorImplementation::NetworkInput*> managedPartitionInputs;
    std::map<RowPartitionId, ExternalRowPartitionPhysicalization> externalRowPartitions;
    std::map<Thor::Tensor, RowPartitionId> externalRowPartitionByLogicalOffsetsTensor;
    std::map<std::string, RowPartitionId> externalRowPartitionByValuesInputName;

    // std::map<uint64_t, std::shared_ptr<ThorImplementation::Parameterizable>> apiParameterizableToPhysicalParameterizable;
    // FIXME: get rid of raw pointers
    // For performance, store and use the raw pointers
    std::vector<ThorImplementation::NetworkInput *> inputs;
    std::vector<ThorImplementation::NetworkOutput *> outputs;
    std::vector<ThorImplementation::TrainableLayer *> trainableLayers;
    std::vector<ThorImplementation::Layer *> otherLayers;
    std::map<Thor::Tensor, ThorImplementation::Layer *> apiTensorToPhysicalDrivingLayer;
    std::map<uint64_t, ThorImplementation::Layer *> apiLayerToPhysicalLayer;
    std::map<ThorImplementation::Layer *, uint64_t, StampedNetwork::LayerComparator> physicalLayerToApiLayer;
    std::map<Thor::Tensor, Thor::Layer *> apiTensorToApiDrivingLayer;
    std::map<std::string, ThorImplementation::NetworkInput *> inputNamed;
    std::map<std::string, RaggedInputBinding> raggedInputNamed;
    std::map<std::string, ThorImplementation::NetworkOutput *> outputNamed;

    uint32_t gpuNum;

    uint64_t bytesRequired;
    uint64_t batchSize;

    uint64_t floatingPointOperationsPerExampleForward;
    uint64_t floatingPointOperationsPerExampleBackward;

   private:
    void initializeProcessingDataStreamJoin();
    void joinProcessingDataStreams(const Stream& processingStream);
    [[nodiscard]] std::shared_ptr<ThorImplementation::Layer> selectExternalRowPartitionDrivingLayer(
        RowPartitionId rowPartitionId,
        RaggedPartitionRequirement requirement) const;
    void auditExternalRowPartitionPhysicalizations() const;
    void clearImpl(bool propagateCleanupFailure);
    void clearNoThrow() noexcept;

    // A placed stamp owns one statically connected activation tensor per graph
    // edge. Native queued execution may stage several batches concurrently, but
    // a later batch must not reuse those tensors until every stream reported by
    // Layer::getProcessingStreams() has finished consuming the current batch.
    // These cached streams/events form
    // the per-batch GPU processing barrier without pulling auxiliary D2H/output
    // streams into the critical path.
    std::vector<Stream> processingDataStreams;
    std::vector<Event> processingDataStreamEvents;

    friend class Thor::Network;
    friend class Thor::PlacedNetwork;
};

}  // namespace ThorImplementation
