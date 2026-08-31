#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/TensorOperations/Ragged/RaggedFilter.h"
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

// Physical R9C stable ragged filter. Input connection type 0 is packed feature
// values, 1 is a scalar BOOLEAN mask, and 2 is their shared canonical offsets.
// Output type 0 is compacted feature values and 1 is a newly produced canonical
// offsets tensor. Only feature values participate in autodiff.
class RaggedFilter : public MultiConnectionLayer {
   public:
    RaggedFilter(RaggedTensorDescriptor inputDescriptor,
                 RaggedTensorDescriptor maskDescriptor,
                 RaggedTensorDescriptor outputDescriptor)
        : inputDescriptor(std::move(inputDescriptor)),
          maskDescriptor(std::move(maskDescriptor)),
          outputDescriptor(std::move(outputDescriptor)) {
        if (this->inputDescriptor.getBatchSize() == 0) {
            throw std::invalid_argument("RaggedFilter requires a non-empty logical batch descriptor.");
        }
        if (this->inputDescriptor.getRowPartition() != this->maskDescriptor.getRowPartition()) {
            throw std::invalid_argument("RaggedFilter feature and mask descriptors must share one exact row partition.");
        }
        if (this->maskDescriptor.getValuesDataType() != DataType::BOOLEAN ||
            !this->maskDescriptor.getTrailingDimensions().empty()) {
            throw std::invalid_argument("RaggedFilter mask must be one BOOLEAN scalar per ragged token.");
        }
        if (this->inputDescriptor.getRowPartition() != this->outputDescriptor.getRowPartition() ||
            this->inputDescriptor.getValuesDataType() != this->outputDescriptor.getValuesDataType() ||
            this->inputDescriptor.getTrailingDimensions() != this->outputDescriptor.getTrailingDimensions()) {
            throw std::invalid_argument("RaggedFilter input/output descriptors are incompatible.");
        }

        previousLayers.resize(3);
        featureInputs.resize(3);
        errorOutputs.resize(3);
        streams.resize(3);
        forwardInputReadyEvents.resize(3);

        featureOutputs.resize(2);
        errorInputs.resize(2);
        nextLayers.resize(2);
    }

    ~RaggedFilter() override = default;

    std::string getType() override { return "RaggedFilter"; }

    std::optional<Tensor> createFeatureOutputTensor() override { THOR_UNREACHABLE(); }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == 3);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value() && featureInputs[2].has_value());
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

        allFeatureInputTensorIds = {
            featureInputs[0]->getTensorId(), featureInputs[1]->getTensorId(), featureInputs[2]->getTensorId()};
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
        streams[0].waitFor(streams[2], forwardInputReadyEvents[2]);

        RowPartitionRuntime outputPartition(featureOutputs[1].value(), outputDescriptor.getRowPartition());
        outputPartition.clearHostOffsets();
        outputPartition.clearHostActiveValueCount();
        outputPartition.clearHostMaxActiveRowLength();

        launchRaggedFilterRowLengths(featureInputs[1].value(), featureInputs[2].value(), rowLengths, batchSize, streams[0]);
        rowPartitionLengthsToOffsets(
            lengthsToOffsetsPlan, scanTempStorage, rowLengths, featureOutputs[1].value(), streams[0]);
        launchRaggedFilterValues(featureInputs[0].value(),
                                 featureInputs[1].value(),
                                 featureInputs[2].value(),
                                 featureOutputs[1].value(),
                                 featureOutputs[0].value(),
                                 batchSize,
                                 streams[0]);

        // Q depends on runtime mask data, not just structural input metadata.
        // Do not introduce an implicit device-to-host synchronization merely to
        // populate optional host row-partition caches.
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
            throw std::logic_error("RaggedFilter received a gradient for its structural offsets output.");
        }
        if (!errorOutputs[0].has_value() || !previousLayers[0].has_value()) return;

        const uint64_t batchSize = outputDescriptor.getBatchSize();
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1 && resolvedValidExampleCount <= physicalBatchCapacity);

        launchRaggedFilterBackward(featureInputs[1].value(),
                                   featureInputs[2].value(),
                                   featureOutputs[1].value(),
                                   errorInput.value(),
                                   errorOutputs[0].value(),
                                   batchSize,
                                   streams[0]);
        previousLayers[0].value()->backward(errorOutputs[0], resolvedValidExampleCount);
    }

    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        if (driverConnectionType < 0 || driverConnectionType > 1) {
            throw std::logic_error("RaggedFilter output connection type must be 0 (values) or 1 (offsets).");
        }
        const uint32_t outputIndex = static_cast<uint32_t>(driverConnectionType);
        if (nextLayers[outputIndex].has_value()) {
            throw std::logic_error("RaggedFilter output port was connected more than once without a fanout.");
        }
        ensureOutputAllocated(outputIndex);
        nextLayers[outputIndex] = nextLayer;

        const bool backPropagate = outputIndex == 0 && shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[outputIndex] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[outputIndex], streams[0], backPropagate, loaderConnectionType);
        if (outputIndex == 1 && errorInputs[outputIndex].has_value()) {
            throw std::logic_error("RaggedFilter structural offsets output cannot participate in autodiff.");
        }
        if (outputIndex == 0 && !errorInputs[0].has_value()) pruneUpstreamFeatureGradient();
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || connectionType > 2) {
            throw std::logic_error("RaggedFilter input connection type must be 0 (values), 1 (mask), or 2 (offsets).");
        }
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("RaggedFilter input port was connected more than once.");
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
            throw std::logic_error("RaggedFilter can replace only its values-output gradient.");
        }
        errorInputs[0] = newErrorInput;
        if (!newErrorInput.has_value()) pruneUpstreamFeatureGradient();
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
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value() && featureInputs[2].has_value());
        const TensorPlacement placement = featureInputs[0]->getPlacement();
        if (featureInputs[1]->getPlacement() != placement || featureInputs[2]->getPlacement() != placement) {
            throw std::invalid_argument("RaggedFilter values, mask, and offsets must reside on one device.");
        }
        if (featureInputs[0]->getDescriptor() != inputDescriptor.getValuesDescriptor()) {
            throw std::invalid_argument("RaggedFilter values input does not match its declared descriptor.");
        }
        if (featureInputs[1]->getDescriptor() != maskDescriptor.getValuesDescriptor()) {
            throw std::invalid_argument("RaggedFilter mask input does not match its declared descriptor.");
        }
        if (featureInputs[2]->getDescriptor() != inputDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("RaggedFilter offsets input does not match its declared descriptor.");
        }
    }

    void pruneUpstreamFeatureGradient() {
        if (!errorOutputs[0].has_value()) return;
        if (previousLayers[0].has_value()) previousLayers[0].value()->replaceErrorInput(errorOutputs[0], std::nullopt);
        errorOutputs[0] = std::nullopt;
    }

    RaggedTensorDescriptor inputDescriptor;
    RaggedTensorDescriptor maskDescriptor;
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
