#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/HostFunctionArgs.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Physical R9A sequence-axis concatenate. API input ports are laid out as
// [value_0 .. value_N-1, unique_partition_carrier_0 .. unique_partition_carrier_M-1].
// Distinct sequence values may share one canonical row partition, so only one
// host partition carrier is connected for each unique logical partition. The
// carrier payload is never interpreted by this layer. StampedNetwork derives and
// stages the compact non-empty copy-span plan directly from authoritative host
// partitions.
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

    RaggedSequenceCopySpan32* beginBatchCopyPlan32(uint64_t requiredSpanCapacity) {
        if (outputDescriptor.getOffsetsDataType() != DataType::UINT32) {
            throw std::logic_error("RaggedSequenceConcatenate requested UINT32 copy-plan storage for a non-UINT32 partition.");
        }
        THOR_THROW_IF_FALSE(requiredSpanCapacity <= maxCopySpanCount);
        prepareBatchCopyPlanHostStorageForWrite();
        return static_cast<RaggedSequenceCopySpan32*>(copySpans_h);
    }

    RaggedSequenceCopySpan64* beginBatchCopyPlan64(uint64_t requiredSpanCapacity) {
        if (outputDescriptor.getOffsetsDataType() != DataType::UINT64) {
            throw std::logic_error("RaggedSequenceConcatenate requested UINT64 copy-plan storage for a non-UINT64 partition.");
        }
        THOR_THROW_IF_FALSE(requiredSpanCapacity <= maxCopySpanCount);
        prepareBatchCopyPlanHostStorageForWrite();
        return static_cast<RaggedSequenceCopySpan64*>(copySpans_h);
    }

    void commitBatchCopyPlan(uint64_t spanCount,
                             uint32_t validExampleCount,
                             uint64_t activeOutputValues) {
        THOR_THROW_IF_FALSE(copyPlanHostStoragePrepared);
        THOR_THROW_IF_FALSE(validExampleCount >= 1 && validExampleCount <= outputDescriptor.getBatchSize());
        THOR_THROW_IF_FALSE(spanCount <= maxCopySpanCount);
        if (activeOutputValues == 0) THOR_THROW_IF_FALSE(spanCount == 0);
        else THOR_THROW_IF_FALSE(spanCount != 0);

        const std::size_t spanBytes = outputDescriptor.getOffsetsDataType() == DataType::UINT32
            ? sizeof(RaggedSequenceCopySpan32)
            : sizeof(RaggedSequenceCopySpan64);
        if (spanCount > std::numeric_limits<std::size_t>::max() / spanBytes) {
            throw std::invalid_argument("RaggedSequenceConcatenate staged copy-plan byte count overflow.");
        }
        const std::size_t bytes = static_cast<std::size_t>(spanCount) * spanBytes;
        THOR_THROW_IF_FALSE(bytes <= maxCopySpanBytes);

        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        if (bytes != 0) {
            CUDA_CHECK(cudaMemcpyAsync(copySpans_d,
                                       copySpans_h,
                                       bytes,
                                       cudaMemcpyHostToDevice,
                                       streams[0].getStream()));
            copyPlanUploadCompleteEvent.record(streams[0]);
            copyPlanUploadInFlight = true;
        } else {
            copyPlanUploadInFlight = false;
        }

        stagedCopySpanCount = spanCount;
        stagedActiveOutputValues = activeOutputValues;
        stagedValidExampleCount = validExampleCount;
        copyPlanStaged = true;
        copyPlanHostStoragePrepared = false;
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

        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&valueInputPointers_d), valueInputCount * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&valueGradientPointers_d), valueInputCount * sizeof(void *)));

        const uint64_t batchSize = outputDescriptor.getBatchSize();
        if (batchSize > std::numeric_limits<uint64_t>::max() / valueInputCount) {
            throw std::invalid_argument("RaggedSequenceConcatenate maximum copy-span count overflow.");
        }
        maxCopySpanCount = batchSize * valueInputCount;
        const std::size_t spanBytes = outputDescriptor.getOffsetsDataType() == DataType::UINT32
            ? sizeof(RaggedSequenceCopySpan32)
            : sizeof(RaggedSequenceCopySpan64);
        if (maxCopySpanCount > std::numeric_limits<std::size_t>::max() / spanBytes) {
            throw std::invalid_argument("RaggedSequenceConcatenate copy-span storage size overflow.");
        }
        maxCopySpanBytes = static_cast<std::size_t>(maxCopySpanCount) * spanBytes;
        THOR_THROW_IF_FALSE(maxCopySpanBytes > 0);
        CUDA_CHECK(cudaMalloc(&copySpans_d, maxCopySpanBytes));
        CUDA_CHECK(cudaHostAlloc(&copySpans_h, maxCopySpanBytes, cudaHostAllocPortable));
        copyPlanUploadCompleteEvent = Event(featureInputs[0]->getPlacement().getDeviceNum(), false, true);

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
        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        if (copyPlanUploadInFlight) copyPlanUploadCompleteEvent.synchronize();
        if (valueInputPointers_d != nullptr) CUDA_CHECK(cudaFree(valueInputPointers_d));
        if (valueGradientPointers_d != nullptr) CUDA_CHECK(cudaFree(valueGradientPointers_d));
        if (copySpans_d != nullptr) CUDA_CHECK(cudaFree(copySpans_d));
        if (copySpans_h != nullptr) CUDA_CHECK(cudaFreeHost(copySpans_h));
        valueInputPointers_d = nullptr;
        valueGradientPointers_d = nullptr;
        copySpans_d = nullptr;
        copySpans_h = nullptr;
        maxCopySpanCount = 0;
        maxCopySpanBytes = 0;
        stagedCopySpanCount = 0;
        stagedActiveOutputValues = 0;
        stagedValidExampleCount = 0;
        copyPlanStaged = false;
        copyPlanHostStoragePrepared = false;
        copyPlanUploadInFlight = false;
        copyPlanUploadCompleteEvent = Event();
        for (Event &event : forwardInputReadyEvents) event = Event();
        outputsReadyEvent = Event();
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

        for (uint32_t i = 1; i < inputPortCount; ++i) streams[0].waitFor(streams[i], forwardInputReadyEvents[i]);
        refreshPointerTables(streams[0]);

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
        launchRaggedSequenceConcatenate(featureOutputs[0]->getMemPtr(),
                                        valueInputPointers_d,
                                        copySpans_d,
                                        stagedCopySpanCount,
                                        TensorDescriptor::getElementSizeInBytes(valuesDescriptor.getDataType()),
                                        elementsPerValue,
                                        TensorDescriptor::getElementSizeInBytes(outputDescriptor.getOffsetsDataType()),
                                        stagedActiveOutputValues,
                                        streams[0]);

        streams[0].putEvent(outputsReadyEvent);
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

        // Reuse the exact forward copy plan. Refresh only the gradient pointer
        // table so late graph pruning/fusion cannot leave stale destinations.
        refreshGradientPointerTable(streams[0]);
        THOR_THROW_IF_FALSE(copyPlanStaged);
        THOR_THROW_IF_FALSE(stagedValidExampleCount == resolvedValidExampleCount);
        RowPartitionRuntime outputPartition = RowPartitionRuntime::fromHostStateCarrier(
            featureOutputs[0].value(), outputDescriptor.getRowPartition());
        THOR_THROW_IF_FALSE(
            outputPartition.requireHostOffset(resolvedValidExampleCount) == stagedActiveOutputValues);
        launchRaggedSequenceConcatenateBackward(valueGradientPointers_d,
                                                errorInput->getMemPtr(),
                                                copySpans_d,
                                                stagedCopySpanCount,
                                                TensorDescriptor::getElementSizeInBytes(valuesDescriptor.getDataType()),
                                                elementsPerValue,
                                                TensorDescriptor::getElementSizeInBytes(outputDescriptor.getOffsetsDataType()),
                                                stagedActiveOutputValues,
                                                streams[0]);

        streams[0].putEvent(backwardOutputsReadyEvent);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (!errorOutputs[i].has_value() || !previousLayers[i].has_value()) continue;
            if (i != 0) streams[i].waitEvent(backwardOutputsReadyEvent);
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

   private:
    struct PointerRefreshArgs : public HostFunctionArgsBase {
        std::vector<void *> valuePointers;
        std::vector<void *> gradientPointers;
    };
    static void releasePointerRefresh(void *) {}

    void prepareBatchCopyPlanHostStorageForWrite() {
        THOR_THROW_IF_FALSE(copySpans_d != nullptr && copySpans_h != nullptr);
        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        // The pinned staging storage can be reused once the preceding H2D copy
        // has completed; the prior value kernel does not read host staging.
        if (copyPlanUploadInFlight) {
            copyPlanUploadCompleteEvent.synchronize();
            copyPlanUploadInFlight = false;
        }
        copyPlanStaged = false;
        copyPlanHostStoragePrepared = true;
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

    void refreshPointerTables(Stream stream) {
        auto args = std::make_unique<PointerRefreshArgs>();
        args->valuePointers.resize(valueInputCount);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            args->valuePointers[i] = featureInputs[i]->getMemPtr();
        }
        CUDA_CHECK(cudaMemcpyAsync(valueInputPointers_d,
                                   args->valuePointers.data(),
                                   valueInputCount * sizeof(void *),
                                   cudaMemcpyHostToDevice,
                                   stream.getStream()));
        stream.enqueueHostFunction(&releasePointerRefresh, std::move(args));
    }

    void refreshGradientPointerTable(Stream stream) {
        auto args = std::make_unique<PointerRefreshArgs>();
        args->gradientPointers.resize(valueInputCount, nullptr);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (errorOutputs[i].has_value()) args->gradientPointers[i] = errorOutputs[i]->getMemPtr();
        }
        CUDA_CHECK(cudaMemcpyAsync(valueGradientPointers_d,
                                   args->gradientPointers.data(),
                                   valueInputCount * sizeof(void *),
                                   cudaMemcpyHostToDevice,
                                   stream.getStream()));
        stream.enqueueHostFunction(&releasePointerRefresh, std::move(args));
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

    void **valueInputPointers_d = nullptr;
    void **valueGradientPointers_d = nullptr;
    void *copySpans_d = nullptr;
    void *copySpans_h = nullptr;
    uint64_t maxCopySpanCount = 0;
    std::size_t maxCopySpanBytes = 0;
    uint64_t stagedCopySpanCount = 0;
    uint64_t stagedActiveOutputValues = 0;
    uint32_t stagedValidExampleCount = 0;
    bool copyPlanStaged = false;
    bool copyPlanHostStoragePrepared = false;
    bool copyPlanUploadInFlight = false;
    Event copyPlanUploadCompleteEvent;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    Event outputsReadyEvent;
    Event backwardOutputsReadyEvent;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
