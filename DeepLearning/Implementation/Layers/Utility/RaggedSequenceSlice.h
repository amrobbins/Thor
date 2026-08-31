#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceSlice.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Physical R9B sequence-axis slice. Input connection type 0 is packed values
// and 1 is canonical offsets. Output connection type 0 is compacted values and
// 1 is a newly produced canonical offsets tensor. Only the values ports
// participate in autodiff.
class RaggedSequenceSlice : public MultiConnectionLayer {
   public:
    RaggedSequenceSlice(uint64_t start,
                        uint64_t length,
                        RaggedTensorDescriptor inputDescriptor,
                        RaggedTensorDescriptor outputDescriptor)
        : start(start),
          length(length),
          inputDescriptor(std::move(inputDescriptor)),
          outputDescriptor(std::move(outputDescriptor)) {
        if (length == 0) throw std::invalid_argument("RaggedSequenceSlice length must be greater than zero.");
        if (this->inputDescriptor.getBatchSize() == 0) {
            throw std::invalid_argument("RaggedSequenceSlice requires a non-empty logical batch descriptor.");
        }
        if (this->inputDescriptor.getBatchSize() != this->outputDescriptor.getBatchSize() ||
            this->inputDescriptor.getValuesDataType() != this->outputDescriptor.getValuesDataType() ||
            this->inputDescriptor.getOffsetsDataType() != this->outputDescriptor.getOffsetsDataType() ||
            this->inputDescriptor.getTrailingDimensions() != this->outputDescriptor.getTrailingDimensions()) {
            throw std::invalid_argument("RaggedSequenceSlice input/output descriptors are incompatible.");
        }

        previousLayers.resize(2);
        featureInputs.resize(2);
        errorOutputs.resize(2);
        streams.resize(2);
        forwardInputReadyEvents.resize(2);

        featureOutputs.resize(2);
        errorInputs.resize(2);
        nextLayers.resize(2);
    }

    ~RaggedSequenceSlice() override = default;

    std::string getType() override { return "RaggedSequenceSlice"; }

    std::optional<Tensor> createFeatureOutputTensor() override { THOR_UNREACHABLE(); }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == 2);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        validateConnectedInputs();
        ensureOutputAllocated(/*outputIndex=*/0);
        ensureOutputAllocated(/*outputIndex=*/1);

        const TensorPlacement placement = featureInputs[0]->getPlacement();
        const uint64_t batchSize = inputDescriptor.getBatchSize();
        rowLengths = Tensor(placement, TensorDescriptor(inputDescriptor.getOffsetsDataType(), {batchSize}));
        lengthsToOffsetsPlan = prepareRowPartitionLengthsToOffsets(rowLengths, featureOutputs[1].value(), batchSize);
        scanTempStorage = Tensor(
            placement,
            TensorDescriptor(DataType::UINT8, {std::max<size_t>(lengthsToOffsetsPlan.temp_storage_bytes, 1)}));

        allFeatureInputTensorIds = {featureInputs[0]->getTensorId(), featureInputs[1]->getTensorId()};
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void cleanup() override {
        rowLengths = Tensor();
        scanTempStorage = Tensor();
        for (Event& event : forwardInputReadyEvents) event = Event();
        outputsReadyEvent = Event();
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

        streams[0].waitFor(streams[1], forwardInputReadyEvents[1]);

        RowPartitionRuntime outputPartition(featureOutputs[1].value(), outputDescriptor.getRowPartition());
        outputPartition.clearHostOffsets();
        outputPartition.clearHostActiveValueCount();
        outputPartition.clearHostMaxActiveRowLength();

        launchRaggedSequenceSliceRowLengths(
            featureInputs[1].value(), rowLengths, start, length, batchSize, streams[0]);
        rowPartitionLengthsToOffsets(
            lengthsToOffsetsPlan, scanTempStorage, rowLengths, featureOutputs[1].value(), streams[0]);
        launchRaggedSequenceSliceValues(featureInputs[0].value(),
                                        featureInputs[1].value(),
                                        featureOutputs[1].value(),
                                        featureOutputs[0].value(),
                                        start,
                                        length,
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
            throw std::logic_error("RaggedSequenceSlice received a gradient for its structural offsets output.");
        }
        if (!errorOutputs[0].has_value() || !previousLayers[0].has_value()) return;

        const uint64_t batchSize = outputDescriptor.getBatchSize();
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1 && resolvedValidExampleCount <= physicalBatchCapacity);

        launchRaggedSequenceSliceBackward(featureInputs[1].value(),
                                          featureOutputs[1].value(),
                                          errorInput.value(),
                                          errorOutputs[0].value(),
                                          start,
                                          length,
                                          batchSize,
                                          streams[0]);
        previousLayers[0].value()->backward(errorOutputs[0], resolvedValidExampleCount);
    }

    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        if (driverConnectionType < 0 || driverConnectionType > 1) {
            throw std::logic_error("RaggedSequenceSlice output connection type must be 0 (values) or 1 (offsets).");
        }
        const uint32_t outputIndex = static_cast<uint32_t>(driverConnectionType);
        if (nextLayers[outputIndex].has_value()) {
            throw std::logic_error("RaggedSequenceSlice output port was connected more than once without a fanout.");
        }
        ensureOutputAllocated(outputIndex);
        nextLayers[outputIndex] = nextLayer;

        const bool backPropagate = outputIndex == 0 && shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[outputIndex] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[outputIndex], streams[0], backPropagate, loaderConnectionType);
        if (outputIndex == 1 && errorInputs[outputIndex].has_value()) {
            throw std::logic_error("RaggedSequenceSlice structural offsets output cannot participate in autodiff.");
        }
        if (outputIndex == 0 && !errorInputs[0].has_value()) pruneUpstreamValueGradient();
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || connectionType > 1) {
            throw std::logic_error("RaggedSequenceSlice input connection type must be 0 (values) or 1 (offsets).");
        }
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("RaggedSequenceSlice input port was connected more than once.");
        }

        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        streams[inputIndex] = stream;
        if (inputIndex == 0 && backPropagateError && !isInferenceOnly()) {
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
            throw std::logic_error("RaggedSequenceSlice can replace only its values-output gradient.");
        }
        errorInputs[0] = newErrorInput;
        if (!newErrorInput.has_value()) pruneUpstreamValueGradient();
    }

   private:
    void ensureOutputAllocated(uint32_t outputIndex) {
        THOR_THROW_IF_FALSE(outputIndex < 2);
        if (featureOutputs[outputIndex].has_value()) return;
        std::optional<Tensor> firstInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(firstInput.has_value());
        const TensorPlacement placement = firstInput->getPlacement();
        const TensorDescriptor descriptor = outputIndex == 0 ? outputDescriptor.getValuesDescriptor()
                                                             : outputDescriptor.getOffsetsDescriptor();
        featureOutputs[outputIndex] = Tensor(placement, descriptor);
    }

    void validateConnectedInputs() const {
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value());
        if (featureInputs[0]->getPlacement() != featureInputs[1]->getPlacement()) {
            throw std::invalid_argument("RaggedSequenceSlice values and offsets must reside on one device.");
        }
        if (featureInputs[0]->getDescriptor() != inputDescriptor.getValuesDescriptor()) {
            throw std::invalid_argument("RaggedSequenceSlice values input does not match its declared descriptor.");
        }
        if (featureInputs[1]->getDescriptor() != inputDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("RaggedSequenceSlice offsets input does not match its declared descriptor.");
        }
    }

    void publishOutputHostPartition(RowPartitionRuntime& outputPartition) {
        RowPartitionRuntime inputPartition(featureInputs[1].value(), inputDescriptor.getRowPartition());
        const std::optional<std::vector<uint64_t>> inputHostOffsets = inputPartition.getHostOffsetsIfAvailable();
        if (!inputHostOffsets.has_value()) return;

        std::vector<uint64_t> outputHostOffsets(inputDescriptor.getBatchSize() + 1, 0);
        for (uint64_t row = 0; row < inputDescriptor.getBatchSize(); ++row) {
            const uint64_t rowBegin = inputHostOffsets->at(row);
            const uint64_t rowEnd = inputHostOffsets->at(row + 1);
            THOR_THROW_IF_FALSE(rowEnd >= rowBegin);
            const uint64_t rowLength = rowEnd - rowBegin;
            const uint64_t slicedLength = rowLength <= start ? 0 : std::min<uint64_t>(length, rowLength - start);
            THOR_THROW_IF_FALSE(outputHostOffsets[row] <=
                                std::numeric_limits<uint64_t>::max() - slicedLength);
            outputHostOffsets[row + 1] = outputHostOffsets[row] + slicedLength;
        }
        THOR_THROW_IF_FALSE(outputHostOffsets.back() <= outputDescriptor.getMaxTotalValues());
        outputPartition.setHostOffsets(std::move(outputHostOffsets));
    }

    void pruneUpstreamValueGradient() {
        if (!errorOutputs[0].has_value()) return;
        if (previousLayers[0].has_value()) previousLayers[0].value()->replaceErrorInput(errorOutputs[0], std::nullopt);
        errorOutputs[0] = std::nullopt;
    }

    uint64_t start = 0;
    uint64_t length = 0;
    RaggedTensorDescriptor inputDescriptor;
    RaggedTensorDescriptor outputDescriptor;

    Tensor rowLengths;
    Tensor scanTempStorage;
    RowPartitionLengthsToOffsetsPlan lengthsToOffsetsPlan;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    Event outputsReadyEvent;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
