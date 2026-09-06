#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceSlice.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Physical sequence-axis slice. Input connection type 0 is packed values,
// 1 is the source partition's device offsets, and 2 is the newly-created
// partition's Thor-managed device offsets. The physical layer produces packed
// values only; authoritative host state for the new partition is published by
// StampedNetwork before physical batch submission.
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

        previousLayers.resize(3);
        featureInputs.resize(3);
        errorOutputs.resize(3);
        streams.resize(3);
        forwardInputReadyEvents.resize(3);

        featureOutputs.resize(1);
        errorInputs.resize(1);
        nextLayers.resize(1);
    }

    ~RaggedSequenceSlice() override = default;

    std::string getType() override { return "RaggedSequenceSlice"; }

    std::optional<Tensor> createFeatureOutputTensor() override { THOR_UNREACHABLE(); }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == 3);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value() && featureInputs[2].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        validateConnectedInputs();
        ensureOutputAllocated();

        allFeatureInputTensorIds = {featureInputs[0]->getTensorId(), featureInputs[1]->getTensorId(), featureInputs[2]->getTensorId()};
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void cleanup() override {
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

        launchRaggedSequenceSliceValues(featureInputs[0].value(),
                                        featureInputs[1].value(),
                                        featureInputs[2].value(),
                                        featureOutputs[0].value(),
                                        start,
                                        length,
                                        batchSize,
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
            throw std::logic_error("RaggedSequenceSlice received an unknown output gradient.");
        }
        if (!errorOutputs[0].has_value() || !previousLayers[0].has_value()) return;

        const uint64_t batchSize = outputDescriptor.getBatchSize();
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1 && resolvedValidExampleCount <= physicalBatchCapacity);

        launchRaggedSequenceSliceBackward(featureInputs[1].value(),
                                          featureInputs[2].value(),
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
        if (driverConnectionType != 0) {
            throw std::logic_error("RaggedSequenceSlice has only one physical output: packed values.");
        }
        if (nextLayers[0].has_value()) {
            throw std::logic_error("RaggedSequenceSlice values output was connected more than once without a fanout.");
        }
        ensureOutputAllocated();
        nextLayers[0] = nextLayer;

        const bool backPropagate = shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[0] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[0], streams[0], backPropagate, loaderConnectionType);
        if (!errorInputs[0].has_value()) pruneUpstreamValueGradient();
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || connectionType > 2) {
            throw std::logic_error(
                "RaggedSequenceSlice physical input connection type must be 0 (values), 1 (source offsets), or 2 (output offsets).");
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
    void ensureOutputAllocated() {
        if (featureOutputs[0].has_value()) return;
        std::optional<Tensor> firstInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(firstInput.has_value());
        featureOutputs[0] = Tensor(firstInput->getPlacement(), outputDescriptor.getValuesDescriptor());
    }

    void validateConnectedInputs() const {
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value() && featureInputs[2].has_value());
        if (featureInputs[0]->getPlacement() != featureInputs[1]->getPlacement() ||
            featureInputs[0]->getPlacement() != featureInputs[2]->getPlacement()) {
            throw std::invalid_argument("RaggedSequenceSlice values and offsets must reside on one device.");
        }
        if (featureInputs[0]->getDescriptor() != inputDescriptor.getValuesDescriptor()) {
            throw std::invalid_argument("RaggedSequenceSlice values input does not match its declared descriptor.");
        }
        if (featureInputs[1]->getDescriptor() != inputDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("RaggedSequenceSlice source offsets input does not match its declared descriptor.");
        }
        if (featureInputs[2]->getDescriptor() != outputDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("RaggedSequenceSlice output offsets input does not match its declared descriptor.");
        }
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

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    Event outputsReadyEvent;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
