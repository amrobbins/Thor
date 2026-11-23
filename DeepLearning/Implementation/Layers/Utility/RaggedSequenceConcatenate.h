#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/DistinctProducerStreamJoin.h"
#include "DeepLearning/Implementation/Layers/DistinctTargetStreamFanout.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Physical R9A sequence-axis concatenate. API input ports are laid out as
// [value_0 .. value_N-1, unique_partition_carrier_0 .. unique_partition_carrier_M-1].
// Distinct sequence values may share one canonical row partition, so only one
// host partition carrier is connected for each unique logical partition. The
// carrier payload is never interpreted by this layer. StampedNetwork derives the
// compact copy-span plan from authoritative host partitions every batch, in a
// table of its batch metadata ring (Utilities/Common/BatchMetadataRing.h), and
// stages its device address here.
//
// The physical layer produces packed values only. A newly created logical
// partition is physicalized by StampedNetwork: authoritative host offsets are
// published on this values output, while any requested [1] active-count or
// [B+1] offsets representation is a separate Thor-managed NetworkInput.
class RaggedSequenceConcatenate : public MultiConnectionLayer {
   public:
    RaggedSequenceConcatenate(uint32_t valueInputCount,
                              uint32_t uniquePartitionInputCount,
                              RaggedTensorDescriptor outputDescriptor)
        : valueInputCount(valueInputCount),
          uniquePartitionInputCount(uniquePartitionInputCount),
          outputDescriptor(std::move(outputDescriptor)) {
        if (valueInputCount < 2) throw std::invalid_argument("RaggedSequenceConcatenate requires at least two value inputs.");
        if (uniquePartitionInputCount == 0) {
            throw std::invalid_argument("RaggedSequenceConcatenate requires at least one host partition carrier input.");
        }

        inputPortCount = valueInputCount + uniquePartitionInputCount;
        previousLayers.resize(inputPortCount);
        featureInputs.resize(inputPortCount);
        errorOutputs.resize(inputPortCount);
        streams.resize(inputPortCount);
        forwardInputReadyEvents.resize(inputPortCount);

        featureOutputs.resize(1);
        errorInputs.resize(1);
        nextLayers.resize(1);
    }

    ~RaggedSequenceConcatenate() override = default;

    std::string getType() override { return "RaggedSequenceConcatenate"; }

    std::optional<Tensor> createFeatureOutputTensor() override { THOR_UNREACHABLE(); }

    // The copy plan's span format and capacity (valid once compiled). StampedNetwork declares a table of
    // getCopyPlanCapacityBytes() in its batch metadata ring for this layer and builds the plan there every batch.
    [[nodiscard]] RaggedSequenceCopySpanFormat getCopySpanFormat() const { return copySpanFormat; }
    [[nodiscard]] std::size_t getCopySpanBytes() const { return raggedSequenceCopySpanBytes(copySpanFormat); }
    [[nodiscard]] uint64_t getCopyPlanCapacitySpans() const { return maxCopySpanCount; }
    [[nodiscard]] uint64_t getCopyPlanCapacityBytes() const { return maxCopySpanBytes; }

    // This batch's copy plan: spanCount spans at device address copySpans, there once `uploaded` is reached. The
    // address holds until the batch's processing is done, so the backward reads the same plan.
    void stageBatchCopyPlan(const void *copySpans,
                            uint64_t spanCount,
                            Event uploaded,
                            uint32_t validExampleCount,
                            uint64_t activeOutputValues) {
        THOR_THROW_IF_FALSE(validExampleCount >= 1 && validExampleCount <= outputDescriptor.getBatchSize());
        THOR_THROW_IF_FALSE(spanCount <= maxCopySpanCount);
        if (activeOutputValues == 0) THOR_THROW_IF_FALSE(spanCount == 0);
        else THOR_THROW_IF_FALSE(spanCount != 0 && copySpans != nullptr);
        stagedCopySpans = copySpans;
        stagedCopyPlanUploaded = std::move(uploaded);
        stagedCopySpanCount = spanCount;
        stagedActiveOutputValues = activeOutputValues;
        stagedValidExampleCount = validExampleCount;
        copyPlanStaged = true;
    }

#if defined(THOR_GTEST) || defined(__JETBRAINS_IDE__)
    [[nodiscard]] uint64_t getStagedCopySpanCountForTest() const { return stagedCopySpanCount; }
#endif

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == inputPortCount);
        THOR_THROW_IF_FALSE(!featureInputs.empty() && featureInputs[0].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        validateConnectedInputs();
        ensureOutputAllocated();

        // Forward value-input and backward gradient destinations are immutable
        // while this compiled path remains executable. Post-compile loss-root
        // pruning can only remove the entire backward path; it cannot replace one
        // live gradient destination with another. The pointers are fixed here and
        // go to every launch as its parameters; an input without a gradient has a
        // null entry, whose spans the backward skips.
        valueInputPointers.assign(valueInputCount, nullptr);
        valueGradientPointers.assign(valueInputCount, nullptr);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            valueInputPointers[i] = featureInputs[i]->getMemPtr();
            if (errorOutputs[i].has_value()) valueGradientPointers[i] = errorOutputs[i]->getMemPtr();
        }

        const uint64_t batchSize = outputDescriptor.getBatchSize();
        if (batchSize > std::numeric_limits<uint64_t>::max() / 2 / valueInputCount) {
            throw std::invalid_argument("RaggedSequenceConcatenate maximum copy-span count overflow.");
        }
        // Every row of every input, with the plan's allowance for the long ones it cuts.
        maxCopySpanCount = raggedSequenceCopySpanCapacity(batchSize * valueInputCount);
        // Packed 8-byte spans unless the output or an input holds more values than they can address.
        uint64_t largestInputCapacity = 0;
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            largestInputCapacity = std::max<uint64_t>(largestInputCapacity, featureInputs[i]->getDescriptor().getDimensions()[0]);
        }
        copySpanFormat = raggedSequenceCopySpanFormatFor(
            valueInputCount, outputDescriptor.getValuesDescriptor().getDimensions()[0], largestInputCapacity);
        const std::size_t spanBytes = getCopySpanBytes();
        if (maxCopySpanCount > std::numeric_limits<uint64_t>::max() / spanBytes) {
            throw std::invalid_argument("RaggedSequenceConcatenate copy-span storage size overflow.");
        }
        maxCopySpanBytes = maxCopySpanCount * spanBytes;
        THOR_THROW_IF_FALSE(maxCopySpanBytes > 0);

        allFeatureInputTensorIds.clear();
        for (const std::optional<Tensor> &input : featureInputs) {
            THOR_THROW_IF_FALSE(input.has_value());
            allFeatureInputTensorIds.insert(input->getTensorId());
        }
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void cleanup() override {
        THOR_THROW_IF_FALSE(!featureInputs.empty() && featureInputs[0].has_value());
        valueInputPointers.clear();
        valueGradientPointers.clear();
        maxCopySpanCount = 0;
        maxCopySpanBytes = 0;
        stagedCopySpans = nullptr;
        stagedCopyPlanUploaded = Event();
        stagedCopySpanCount = 0;
        stagedActiveOutputValues = 0;
        stagedValidExampleCount = 0;
        copyPlanStaged = false;
        for (Event &event : forwardInputReadyEvents) event = Event();
        backwardOutputsReadyEvent = Event();
        MultiConnectionLayer::cleanup();
    }

    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        THOR_THROW_IF_FALSE(featureInput.has_value());
        const uint64_t batchSize = outputDescriptor.getBatchSize();
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1 && resolvedValidExampleCount <= physicalBatchCapacity);
        if (batchCardinalitySet) {
            THOR_THROW_IF_FALSE(currentValidExampleCount == resolvedValidExampleCount);
        } else {
            currentValidExampleCount = resolvedValidExampleCount;
            batchCardinalitySet = true;
        }

        auto waiting = stillWaitingForFeatureInputTensors.find(featureInput->getTensorId());
        THOR_THROW_IF_FALSE(waiting != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(waiting);
        if (!stillWaitingForFeatureInputTensors.empty()) return;
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;

        detail::waitForDistinctProducerStreams(streams[0], streams, forwardInputReadyEvents, 1);

        const TensorDescriptor valuesDescriptor = outputDescriptor.getValuesDescriptor();
        uint64_t elementsPerValue = 1;
        const std::vector<uint64_t> &dimensions = valuesDescriptor.getDimensions();
        for (uint32_t d = 1; d < dimensions.size(); ++d) elementsPerValue *= dimensions[d];

        // StampedNetwork derives the output partition and this batch's non-empty
        // copy spans together from authoritative host input partitions before any
        // physical input runs. CUDA consumes only that plan; device offsets never
        // participate in placement resolution.
        THOR_THROW_IF_FALSE(copyPlanStaged);
        THOR_THROW_IF_FALSE(stagedValidExampleCount == resolvedValidExampleCount);
        RowPartitionRuntime outputPartition = RowPartitionRuntime::fromHostStateCarrier(
            featureOutputs[0].value(), outputDescriptor.getRowPartition());
        THOR_THROW_IF_FALSE(
            outputPartition.requireHostOffset(resolvedValidExampleCount) == stagedActiveOutputValues);
        // The plan is in this stamp's batch metadata ring; this stream reads it once its upload is done. The backward
        // runs on this stream after the forward, so it needs no wait of its own.
        if (stagedCopySpanCount != 0) streams[0].waitEvent(stagedCopyPlanUploaded);
        launchRaggedSequenceConcatenate(featureOutputs[0]->getMemPtr(),
                                        valueInputPointers,
                                        stagedCopySpans,
                                        stagedCopySpanCount,
                                        TensorDescriptor::getElementSizeInBytes(valuesDescriptor.getDataType()),
                                        elementsPerValue,
                                        copySpanFormat,
                                        stagedActiveOutputValues,
                                        streams[0]);

        if (nextLayers[0].has_value())
            nextLayers[0].value()->forward(featureOutputs[0], validationPass, currentValidExampleCount);
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!errorInput.has_value()) return;
        if (!errorInputs[0].has_value() || errorInput.value() != errorInputs[0].value()) {
            throw std::logic_error("RaggedSequenceConcatenate received an unknown output gradient.");
        }

        const uint64_t batchSize = outputDescriptor.getBatchSize();
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1 && resolvedValidExampleCount <= physicalBatchCapacity);

        const TensorDescriptor valuesDescriptor = outputDescriptor.getValuesDescriptor();
        uint64_t elementsPerValue = 1;
        const std::vector<uint64_t> &dimensions = valuesDescriptor.getDimensions();
        for (uint32_t d = 1; d < dimensions.size(); ++d) elementsPerValue *= dimensions[d];

        // Reuse the exact forward copy plan and the compile-time gradient
        // destinations.
        THOR_THROW_IF_FALSE(copyPlanStaged);
        THOR_THROW_IF_FALSE(stagedValidExampleCount == resolvedValidExampleCount);
        RowPartitionRuntime outputPartition = RowPartitionRuntime::fromHostStateCarrier(
            featureOutputs[0].value(), outputDescriptor.getRowPartition());
        THOR_THROW_IF_FALSE(
            outputPartition.requireHostOffset(resolvedValidExampleCount) == stagedActiveOutputValues);
        launchRaggedSequenceConcatenateBackward(valueGradientPointers,
                                                errorInput->getMemPtr(),
                                                stagedCopySpans,
                                                stagedCopySpanCount,
                                                TensorDescriptor::getElementSizeInBytes(valuesDescriptor.getDataType()),
                                                elementsPerValue,
                                                copySpanFormat,
                                                stagedActiveOutputValues,
                                                streams[0]);

        ThorImplementation::detail::recordCompletionAndWaitOnDistinctTargetStreams(
            streams[0],
            backwardOutputsReadyEvent,
            valueInputCount,
            [this](std::size_t i) { return errorOutputs[i].has_value() && previousLayers[i].has_value(); },
            [this](std::size_t i) -> const Stream& { return streams[i]; });
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (!errorOutputs[i].has_value() || !previousLayers[i].has_value()) continue;
            previousLayers[i].value()->backward(errorOutputs[i], resolvedValidExampleCount);
        }
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        if (driverConnectionType != 0) {
            throw std::logic_error("RaggedSequenceConcatenate has only one physical output: packed values.");
        }
        if (nextLayers[0].has_value()) {
            throw std::logic_error("RaggedSequenceConcatenate values output was connected more than once without a fanout.");
        }
        ensureOutputAllocated();
        nextLayers[0] = nextLayer;

        const bool backPropagate = shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[0] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[0], streams[0], backPropagate, loaderConnectionType);
        if (!errorInputs[0].has_value()) pruneUpstreamValueGradients();
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer *previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || static_cast<uint32_t>(connectionType) >= inputPortCount) {
            throw std::logic_error("RaggedSequenceConcatenate input connection type is outside its declared input range.");
        }
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("RaggedSequenceConcatenate input port was connected more than once.");
        }

        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        streams[inputIndex] = stream;
        // Downstream ragged stamp may need this descriptor before all source ports connect.
        ensureOutputAllocated();
        if (inputIndex < valueInputCount && backPropagateError && !isInferenceOnly()) {
            errorOutputs[inputIndex] = featureInput->clone();
        } else {
            errorOutputs[inputIndex] = std::nullopt;
        }
        ensureNoDeviceCrossing();
        return errorOutputs[inputIndex];
    }

    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {
        THOR_THROW_IF_FALSE(oldErrorInput.has_value());
        if (!errorInputs[0].has_value() || errorInputs[0].value() != oldErrorInput.value()) {
            throw std::logic_error("RaggedSequenceConcatenate can replace only its values-output gradient.");
        }
        errorInputs[0] = newErrorInput;
        if (!newErrorInput.has_value()) pruneUpstreamValueGradients();
    }

    // Sequence concatenate preserves every active input value exactly once. The
    // output partition therefore has the sum of the active input extents. Backward
    // counts only slices whose input-gradient routes are actually authored. Every
    // extent query is O(1) in batch size because values tensors carry authoritative
    // host partition state; this loop is only over the fixed layer input arity.
    uint64_t logicalByteCountForward(uint64_t validExampleCount) override {
        try {
            if (featureOutputs.empty() || !featureOutputs[0].has_value() || featureInputs.size() < valueInputCount) return 0;
            const std::optional<uint64_t> outputActive = logicalActiveValues(
                featureOutputs[0].value(), outputDescriptor.getBatchSize(), outputDescriptor.getMaxTotalValues(), validExampleCount);
            if (!outputActive.has_value()) return 0;

            uint64_t bytes = 0;
            uint64_t summedInputActive = 0;
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (!featureInputs[i].has_value()) return 0;
                const uint64_t inputCapacity = featureInputs[i]->getDimensions().empty() ? 0 : featureInputs[i]->getDimensions()[0];
                const std::optional<uint64_t> inputActive = logicalActiveValues(
                    featureInputs[i].value(), outputDescriptor.getBatchSize(), inputCapacity, validExampleCount);
                if (!inputActive.has_value()) return 0;
                if (summedInputActive > std::numeric_limits<uint64_t>::max() - *inputActive) return 0;
                summedInputActive += *inputActive;
                bytes = checkedLogicalByteAdd(
                    bytes, logicalValueBytes(featureInputs[i].value(), inputCapacity, *inputActive,
                                             "RaggedSequenceConcatenate forward input"),
                    "RaggedSequenceConcatenate forward inputs");
            }
            if (summedInputActive != *outputActive) return 0;
            return checkedLogicalByteAdd(
                bytes, logicalValueBytes(featureOutputs[0].value(), outputDescriptor.getMaxTotalValues(),
                                         *outputActive, "RaggedSequenceConcatenate forward output"),
                "RaggedSequenceConcatenate forward");
        } catch (...) {
            return 0;
        }
    }

    uint64_t logicalByteCountBackward(uint64_t validExampleCount) override {
        try {
            if (errorInputs.empty() || !errorInputs[0].has_value() || featureInputs.size() < valueInputCount ||
                errorOutputs.size() < valueInputCount) return 0;
            uint64_t bytes = 0;
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (!errorOutputs[i].has_value()) continue;
                if (!featureInputs[i].has_value()) return 0;
                const uint64_t inputCapacity = featureInputs[i]->getDimensions().empty() ? 0 : featureInputs[i]->getDimensions()[0];
                const std::optional<uint64_t> inputActive = logicalActiveValues(
                    featureInputs[i].value(), outputDescriptor.getBatchSize(), inputCapacity, validExampleCount);
                if (!inputActive.has_value()) return 0;
                bytes = checkedLogicalByteAdd(
                    bytes, logicalValueBytes(errorInputs[0].value(), outputDescriptor.getMaxTotalValues(),
                                             *inputActive, "RaggedSequenceConcatenate backward upstream slice"),
                    "RaggedSequenceConcatenate backward upstream");
                bytes = checkedLogicalByteAdd(
                    bytes, logicalValueBytes(errorOutputs[i].value(), inputCapacity, *inputActive,
                                             "RaggedSequenceConcatenate backward input gradient"),
                    "RaggedSequenceConcatenate backward input gradient");
            }
            return bytes;
        } catch (...) {
            return 0;
        }
    }

   private:
    [[nodiscard]] static std::optional<uint64_t> logicalActiveValues(
        const Tensor& carrier, uint64_t batchSize, uint64_t valueCapacity, uint64_t validExampleCount) noexcept {
        try {
            if (valueCapacity == 0) return std::nullopt;
            if (validExampleCount == 0) {
                const auto active = RowPartitionRuntime::getPublishedHostActiveValueCountIfAvailable(carrier);
                if (!active.has_value() || *active > valueCapacity) return std::nullopt;
                return active;
            }
            if (validExampleCount > batchSize) return std::nullopt;
            const auto active = RowPartitionRuntime::getPublishedHostOffsetIfAvailable(carrier, validExampleCount);
            if (!active.has_value() || *active > valueCapacity) return std::nullopt;
            return active;
        } catch (...) {
            return std::nullopt;
        }
    }

    [[nodiscard]] static uint64_t logicalValueBytes(const Tensor& tensor,
                                                    uint64_t valueCapacity,
                                                    uint64_t activeValues,
                                                    const char* where) {
        if (valueCapacity == 0 || activeValues > valueCapacity) return 0;
        const uint64_t capacityBytes = tensor.getArraySizeInBytes();
        if (capacityBytes % valueCapacity != 0) return 0;
        const uint64_t bytesPerValue = capacityBytes / valueCapacity;
        if (bytesPerValue != 0 && activeValues > std::numeric_limits<uint64_t>::max() / bytesPerValue) {
            throw std::overflow_error(std::string(where) + " logical byte count overflow.");
        }
        return activeValues * bytesPerValue;
    }

    void ensureOutputAllocated() {
        if (featureOutputs[0].has_value()) return;
        std::optional<Tensor> firstInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(firstInput.has_value());
        featureOutputs[0] = Tensor(firstInput->getPlacement(), outputDescriptor.getValuesDescriptor());
    }

    void validateConnectedInputs() const {
        const TensorDescriptor referenceValues = outputDescriptor.getValuesDescriptor();
        const std::vector<uint64_t> referenceDimensions = referenceValues.getDimensions();
        const std::vector<uint64_t> referenceTrailing(referenceDimensions.begin() + 1, referenceDimensions.end());
        const TensorPlacement referencePlacement = featureInputs[0]->getPlacement();
        uint64_t summedCapacity = 0;
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            if (featureInputs[i]->getPlacement() != referencePlacement) {
                throw std::invalid_argument("RaggedSequenceConcatenate inputs must reside on one device.");
            }
            const TensorDescriptor &descriptor = featureInputs[i]->getDescriptor();
            const std::vector<uint64_t> dimensions = descriptor.getDimensions();
            if (descriptor.getDataType() != referenceValues.getDataType() || dimensions.size() != referenceDimensions.size() ||
                dimensions.empty()) {
                throw std::invalid_argument("RaggedSequenceConcatenate value inputs must share dtype and trailing rank.");
            }
            const std::vector<uint64_t> trailing(dimensions.begin() + 1, dimensions.end());
            if (trailing != referenceTrailing) {
                throw std::invalid_argument("RaggedSequenceConcatenate value inputs must share identical trailing dimensions.");
            }
            if (summedCapacity > std::numeric_limits<uint64_t>::max() - dimensions[0]) {
                throw std::invalid_argument("RaggedSequenceConcatenate input packed capacities overflow uint64.");
            }
            summedCapacity += dimensions[0];
        }
        if (referenceDimensions.empty() || summedCapacity != referenceDimensions[0]) {
            throw std::invalid_argument(
                "RaggedSequenceConcatenate output packed capacity must equal the sum of input packed capacities.");
        }
        for (uint32_t partitionPort = 0; partitionPort < uniquePartitionInputCount; ++partitionPort) {
            const uint32_t inputIndex = valueInputCount + partitionPort;
            THOR_THROW_IF_FALSE(featureInputs[inputIndex].has_value());
            if (featureInputs[inputIndex]->getPlacement() != referencePlacement) {
                throw std::invalid_argument("RaggedSequenceConcatenate host partition carriers must reside on the values device.");
            }
        }
    }

    void pruneUpstreamValueGradients() {
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (!errorOutputs[i].has_value()) continue;
            if (previousLayers[i].has_value()) previousLayers[i].value()->replaceErrorInput(errorOutputs[i], std::nullopt);
            errorOutputs[i] = std::nullopt;
        }
    }

    uint32_t valueInputCount = 0;
    uint32_t uniquePartitionInputCount = 0;
    uint32_t inputPortCount = 0;
    RaggedTensorDescriptor outputDescriptor;

    std::vector<void *> valueInputPointers;
    std::vector<void *> valueGradientPointers;
    RaggedSequenceCopySpanFormat copySpanFormat = RaggedSequenceCopySpanFormat::WIDE;
    uint64_t maxCopySpanCount = 0;
    uint64_t maxCopySpanBytes = 0;
    const void *stagedCopySpans = nullptr;
    Event stagedCopyPlanUploaded;
    uint64_t stagedCopySpanCount = 0;
    uint64_t stagedActiveOutputValues = 0;
    uint32_t stagedValidExampleCount = 0;
    bool copyPlanStaged = false;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    Event backwardOutputsReadyEvent;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
