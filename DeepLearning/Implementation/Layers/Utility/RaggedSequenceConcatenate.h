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
// [value_0 .. value_N-1, unique_offsets_0 .. unique_offsets_M-1].
// offsetPortForInput maps each logical sequence input to one of the unique
// structural offset ports, allowing distinct sequence values to share an exact
// canonical row partition without requiring duplicate graph edges.
//
// Output connection type 0 is packed values and 1 is the newly-produced offsets
// tensor. Only output 0 participates in autodiff.
class RaggedSequenceConcatenate : public MultiConnectionLayer {
   public:
    RaggedSequenceConcatenate(uint32_t valueInputCount,
                              uint32_t uniqueOffsetsInputCount,
                              std::vector<uint32_t> offsetPortForInput,
                              RaggedTensorDescriptor outputDescriptor)
        : valueInputCount(valueInputCount),
          uniqueOffsetsInputCount(uniqueOffsetsInputCount),
          offsetPortForInput(std::move(offsetPortForInput)),
          outputDescriptor(std::move(outputDescriptor)) {
        if (valueInputCount < 2) throw std::invalid_argument("RaggedSequenceConcatenate requires at least two value inputs.");
        if (uniqueOffsetsInputCount == 0) throw std::invalid_argument("RaggedSequenceConcatenate requires at least one offsets input.");
        if (this->offsetPortForInput.size() != valueInputCount) {
            throw std::invalid_argument("RaggedSequenceConcatenate offset-port mapping size must equal value input count.");
        }
        for (uint32_t port : this->offsetPortForInput) {
            if (port >= uniqueOffsetsInputCount) {
                throw std::invalid_argument("RaggedSequenceConcatenate offset-port mapping is out of range.");
            }
        }

        inputPortCount = valueInputCount + uniqueOffsetsInputCount;
        previousLayers.resize(inputPortCount);
        featureInputs.resize(inputPortCount);
        errorOutputs.resize(inputPortCount);
        streams.resize(inputPortCount);
        forwardInputReadyEvents.resize(inputPortCount);

        featureOutputs.resize(2);
        errorInputs.resize(2);
        nextLayers.resize(2);
    }

    ~RaggedSequenceConcatenate() override = default;

    std::string getType() override { return "RaggedSequenceConcatenate"; }

    std::optional<Tensor> createFeatureOutputTensor() override { THOR_UNREACHABLE(); }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == inputPortCount);
        THOR_THROW_IF_FALSE(!featureInputs.empty() && featureInputs[0].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        validateConnectedInputs();
        ensureOutputAllocated(/*outputIndex=*/0);
        ensureOutputAllocated(/*outputIndex=*/1);

        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&valueInputPointers_d), valueInputCount * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sequenceOffsetPointers_d), valueInputCount * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&valueGradientPointers_d), valueInputCount * sizeof(void *)));

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
        if (valueInputPointers_d != nullptr) CUDA_CHECK(cudaFree(valueInputPointers_d));
        if (sequenceOffsetPointers_d != nullptr) CUDA_CHECK(cudaFree(sequenceOffsetPointers_d));
        if (valueGradientPointers_d != nullptr) CUDA_CHECK(cudaFree(valueGradientPointers_d));
        valueInputPointers_d = nullptr;
        sequenceOffsetPointers_d = nullptr;
        valueGradientPointers_d = nullptr;
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

        RowPartitionRuntime outputPartition(featureOutputs[1].value(), outputDescriptor.getRowPartition());
        outputPartition.clearHostOffsets();
        outputPartition.clearHostActiveValueCount();
        outputPartition.clearHostMaxActiveRowLength();

        launchRaggedSequenceConcatenate(featureOutputs[0]->getMemPtr(),
                                        featureOutputs[1]->getMemPtr(),
                                        valueInputPointers_d,
                                        sequenceOffsetPointers_d,
                                        valueInputCount,
                                        TensorDescriptor::getElementSizeInBytes(valuesDescriptor.getDataType()),
                                        elementsPerValue,
                                        TensorDescriptor::getElementSizeInBytes(outputDescriptor.getOffsetsDataType()),
                                        batchSize,
                                        streams[0]);
        publishOutputHostPartition(outputPartition);

        streams[0].putEvent(outputsReadyEvent);
        for (uint32_t outputIndex = 0; outputIndex < 2; ++outputIndex) {
            if (!nextLayers[outputIndex].has_value()) continue;
            nextLayers[outputIndex].value()->forward(featureOutputs[outputIndex], validationPass, currentValidExampleCount);
        }
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!errorInput.has_value()) return;
        if (!errorInputs[0].has_value() || errorInput.value() != errorInputs[0].value()) {
            throw std::logic_error("RaggedSequenceConcatenate received a gradient for its structural offsets output.");
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

        // Offsets were refreshed on the corresponding forward pass. Refresh the
        // gradient destination table here as well so late graph pruning/fusion can
        // never leave a stale pointer in a compiled executable.
        refreshGradientPointerTable(streams[0]);
        launchRaggedSequenceConcatenateBackward(valueGradientPointers_d,
                                                errorInput->getMemPtr(),
                                                sequenceOffsetPointers_d,
                                                valueInputCount,
                                                TensorDescriptor::getElementSizeInBytes(valuesDescriptor.getDataType()),
                                                elementsPerValue,
                                                TensorDescriptor::getElementSizeInBytes(outputDescriptor.getOffsetsDataType()),
                                                batchSize,
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
        if (driverConnectionType < 0 || driverConnectionType > 1) {
            throw std::logic_error("RaggedSequenceConcatenate output connection type must be 0 (values) or 1 (offsets).");
        }
        const uint32_t outputIndex = static_cast<uint32_t>(driverConnectionType);
        if (nextLayers[outputIndex].has_value()) {
            throw std::logic_error("RaggedSequenceConcatenate output port was connected more than once without a fanout.");
        }
        ensureOutputAllocated(outputIndex);
        nextLayers[outputIndex] = nextLayer;

        const bool backPropagate = outputIndex == 0 && shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[outputIndex] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[outputIndex], streams[0], backPropagate, loaderConnectionType);
        if (outputIndex == 1 && errorInputs[outputIndex].has_value()) {
            throw std::logic_error("RaggedSequenceConcatenate structural offsets output cannot participate in autodiff.");
        }
        if (outputIndex == 0 && !errorInputs[0].has_value()) pruneUpstreamValueGradients();
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
        std::vector<void *> offsetPointers;
        std::vector<void *> gradientPointers;
    };
    static void releasePointerRefresh(void *) {}

    void ensureOutputAllocated(uint32_t outputIndex) {
        THOR_THROW_IF_FALSE(outputIndex < 2);
        if (featureOutputs[outputIndex].has_value()) return;
        std::optional<Tensor> firstInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(firstInput.has_value());
        const TensorPlacement placement = firstInput->getPlacement();
        const TensorDescriptor descriptor = outputIndex == 0 ? outputDescriptor.getValuesDescriptor() : outputDescriptor.getOffsetsDescriptor();
        featureOutputs[outputIndex] = Tensor(placement, descriptor);
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
        for (uint32_t offsetPort = 0; offsetPort < uniqueOffsetsInputCount; ++offsetPort) {
            const uint32_t inputIndex = valueInputCount + offsetPort;
            THOR_THROW_IF_FALSE(featureInputs[inputIndex].has_value());
            if (featureInputs[inputIndex]->getPlacement() != referencePlacement) {
                throw std::invalid_argument("RaggedSequenceConcatenate offsets must reside on the values device.");
            }
            if (featureInputs[inputIndex]->getDescriptor() != outputDescriptor.getOffsetsDescriptor()) {
                throw std::invalid_argument("RaggedSequenceConcatenate offsets inputs must share output batch size and offsets dtype.");
            }
        }
    }

    void refreshPointerTables(Stream stream) {
        auto args = std::make_unique<PointerRefreshArgs>();
        args->valuePointers.resize(valueInputCount);
        args->offsetPointers.resize(valueInputCount);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            args->valuePointers[i] = featureInputs[i]->getMemPtr();
            const uint32_t uniqueOffsetPort = offsetPortForInput[i];
            args->offsetPointers[i] = featureInputs[valueInputCount + uniqueOffsetPort]->getMemPtr();
        }
        CUDA_CHECK(cudaMemcpyAsync(valueInputPointers_d,
                                   args->valuePointers.data(),
                                   valueInputCount * sizeof(void *),
                                   cudaMemcpyHostToDevice,
                                   stream.getStream()));
        CUDA_CHECK(cudaMemcpyAsync(sequenceOffsetPointers_d,
                                   args->offsetPointers.data(),
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

    void publishOutputHostPartition(RowPartitionRuntime& outputPartition) {
        const uint64_t batchSize = outputDescriptor.getBatchSize();
        std::vector<std::vector<uint64_t>> inputHostOffsets;
        inputHostOffsets.reserve(valueInputCount);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            const std::vector<uint64_t> dimensions = featureInputs[i]->getDimensions();
            THOR_THROW_IF_FALSE(!dimensions.empty());
            RowPartitionRuntime inputPartition(
                featureInputs[valueInputCount + offsetPortForInput[i]].value(),
                RowPartitionDescriptor(batchSize, dimensions[0], outputDescriptor.getOffsetsDataType()));
            const std::optional<std::vector<uint64_t>> hostOffsets = inputPartition.getHostOffsetsIfAvailable();
            if (!hostOffsets.has_value()) return;
            inputHostOffsets.push_back(hostOffsets.value());
        }

        std::vector<uint64_t> outputHostOffsets(batchSize + 1, 0);
        for (uint64_t boundary = 0; boundary <= batchSize; ++boundary) {
            uint64_t sum = 0;
            for (const std::vector<uint64_t>& offsets : inputHostOffsets) {
                THOR_THROW_IF_FALSE(boundary < offsets.size());
                THOR_THROW_IF_FALSE(sum <= std::numeric_limits<uint64_t>::max() - offsets[boundary]);
                sum += offsets[boundary];
            }
            outputHostOffsets[boundary] = sum;
        }
        THOR_THROW_IF_FALSE(outputHostOffsets.back() <= outputDescriptor.getMaxTotalValues());
        outputPartition.setHostOffsets(std::move(outputHostOffsets));
    }

    void pruneUpstreamValueGradients() {
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (!errorOutputs[i].has_value()) continue;
            if (previousLayers[i].has_value()) previousLayers[i].value()->replaceErrorInput(errorOutputs[i], std::nullopt);
            errorOutputs[i] = std::nullopt;
        }
    }

    uint32_t valueInputCount = 0;
    uint32_t uniqueOffsetsInputCount = 0;
    uint32_t inputPortCount = 0;
    std::vector<uint32_t> offsetPortForInput;
    RaggedTensorDescriptor outputDescriptor;

    void **valueInputPointers_d = nullptr;
    void **sequenceOffsetPointers_d = nullptr;
    void **valueGradientPointers_d = nullptr;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    Event outputsReadyEvent;
    Event backwardOutputsReadyEvent;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
