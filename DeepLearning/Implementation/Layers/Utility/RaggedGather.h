#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RaggedGather.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Physical R9D row-local gather. Input ports are source values, indices values,
// source offsets, and (when distinct) indices offsets. The values output uses
// the indices partition Q, but Q itself remains owned by the indices producer
// and is therefore not copied or emitted as a second structural output.
class RaggedGather : public MultiConnectionLayer {
   public:
    RaggedGather(RaggedTensorDescriptor sourceDescriptor,
                 RaggedTensorDescriptor indicesDescriptor,
                 RaggedTensorDescriptor outputDescriptor,
                 bool sharedOffsets)
        : sourceDescriptor(std::move(sourceDescriptor)),
          indicesDescriptor(std::move(indicesDescriptor)),
          outputDescriptor(std::move(outputDescriptor)),
          sharedOffsets(sharedOffsets) {
        if (this->sourceDescriptor.getBatchSize() == 0 ||
            this->sourceDescriptor.getBatchSize() != this->indicesDescriptor.getBatchSize()) {
            throw std::invalid_argument("RaggedGather source and indices must have the same non-zero batch size.");
        }
        if ((this->indicesDescriptor.getValuesDataType() != DataType::UINT32 &&
             this->indicesDescriptor.getValuesDataType() != DataType::UINT64) ||
            !this->indicesDescriptor.getTrailingDimensions().empty()) {
            throw std::invalid_argument("RaggedGather indices must be scalar UINT32 or UINT64 ragged values.");
        }
        if (this->outputDescriptor.getValuesDataType() != this->sourceDescriptor.getValuesDataType() ||
            this->outputDescriptor.getTrailingDimensions() != this->sourceDescriptor.getTrailingDimensions() ||
            this->outputDescriptor.getRowPartition() != this->indicesDescriptor.getRowPartition()) {
            throw std::invalid_argument("RaggedGather output must use source value geometry and indices partition Q.");
        }
        if (sharedOffsets && this->sourceDescriptor.getRowPartition() != this->indicesDescriptor.getRowPartition()) {
            throw std::invalid_argument("RaggedGather shared-offset construction requires the same source/indices partition.");
        }

        inputPortCount = sharedOffsets ? 3U : 4U;
        indicesOffsetsInputPort = sharedOffsets ? 2U : 3U;
        previousLayers.resize(inputPortCount);
        featureInputs.resize(inputPortCount);
        errorOutputs.resize(inputPortCount);
        streams.resize(inputPortCount);
        forwardInputReadyEvents.resize(inputPortCount);

        featureOutputs.resize(1);
        errorInputs.resize(1);
        nextLayers.resize(1);
    }

    ~RaggedGather() override = default;

    std::string getType() override { return "RaggedGather"; }

    std::optional<Tensor> createFeatureOutputTensor() override { THOR_UNREACHABLE(); }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == inputPortCount);
        for (const std::optional<Tensor>& input : featureInputs) THOR_THROW_IF_FALSE(input.has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        validateConnectedInputs();
        ensureOutputAllocated();

        allFeatureInputTensorIds.clear();
        for (const std::optional<Tensor>& input : featureInputs) allFeatureInputTensorIds.insert(input->getTensorId());
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
        THOR_THROW_IF_FALSE(running && featureInput.has_value());

        const uint64_t batchSize = sourceDescriptor.getBatchSize();
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

        launchRaggedGather(featureInputs[0].value(),
                           featureInputs[2].value(),
                           featureInputs[1].value(),
                           featureInputs[indicesOffsetsInputPort].value(),
                           featureOutputs[0].value(),
                           batchSize,
                           streams[0]);

        streams[0].putEvent(outputsReadyEvent);
        if (nextLayers[0].has_value()) {
            nextLayers[0].value()->forward(featureOutputs[0], validationPass, currentValidExampleCount);
        }
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!errorInput.has_value()) return;
        if (!errorInputs[0].has_value() || errorInput.value() != errorInputs[0].value()) {
            throw std::logic_error("RaggedGather received an unknown output gradient.");
        }
        if (!errorOutputs[0].has_value() || !previousLayers[0].has_value()) return;

        const uint64_t batchSize = sourceDescriptor.getBatchSize();
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1 && resolvedValidExampleCount <= physicalBatchCapacity);

        launchRaggedGatherBackward(featureInputs[2].value(),
                                   featureInputs[1].value(),
                                   featureInputs[indicesOffsetsInputPort].value(),
                                   errorInput.value(),
                                   errorOutputs[0].value(),
                                   batchSize,
                                   streams[0]);
        previousLayers[0].value()->backward(errorOutputs[0], resolvedValidExampleCount);
    }

    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        if (driverConnectionType != 0) throw std::logic_error("RaggedGather has only values output connection type 0.");
        if (nextLayers[0].has_value()) {
            throw std::logic_error("RaggedGather values output was connected more than once without a fanout.");
        }
        ensureOutputAllocated();
        nextLayers[0] = nextLayer;
        const bool backPropagate = shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[0] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[0], streams[0], backPropagate, loaderConnectionType);
        if (!errorInputs[0].has_value()) pruneUpstreamSourceGradient();
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || static_cast<uint32_t>(connectionType) >= inputPortCount) {
            throw std::logic_error("RaggedGather input connection type is out of range.");
        }
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("RaggedGather input port was connected more than once.");
        }
        if (inputIndex == 0 && backPropagateError && !isInferenceOnly()) {
            const DataType dtype = featureInput->getDataType();
            if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
                throw std::invalid_argument(
                    "RaggedGather training supports only FP16, BF16, and FP32 source values; inference may gather other dtypes.");
            }
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
            throw std::logic_error("RaggedGather can replace only its values-output gradient.");
        }
        errorInputs[0] = newErrorInput;
        if (!newErrorInput.has_value()) pruneUpstreamSourceGradient();
    }

   private:
    void ensureOutputAllocated() {
        if (featureOutputs[0].has_value()) return;
        std::optional<Tensor> firstInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(firstInput.has_value());
        featureOutputs[0] = Tensor(firstInput->getPlacement(), outputDescriptor.getValuesDescriptor());
    }

    void validateConnectedInputs() const {
        for (const std::optional<Tensor>& input : featureInputs) THOR_THROW_IF_FALSE(input.has_value());
        const TensorPlacement placement = featureInputs[0]->getPlacement();
        for (const std::optional<Tensor>& input : featureInputs) {
            if (input->getPlacement() != placement) {
                throw std::invalid_argument("RaggedGather source, indices, and partitions must reside on one device.");
            }
        }
        if (featureInputs[0]->getDescriptor() != sourceDescriptor.getValuesDescriptor()) {
            throw std::invalid_argument("RaggedGather source values input does not match its declared descriptor.");
        }
        if (featureInputs[1]->getDescriptor() != indicesDescriptor.getValuesDescriptor()) {
            throw std::invalid_argument("RaggedGather indices values input does not match its declared descriptor.");
        }
        if (featureInputs[2]->getDescriptor() != sourceDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("RaggedGather source offsets input does not match its declared descriptor.");
        }
        if (featureInputs[indicesOffsetsInputPort]->getDescriptor() != indicesDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("RaggedGather indices offsets input does not match its declared descriptor.");
        }
    }

    void pruneUpstreamSourceGradient() {
        if (!errorOutputs[0].has_value()) return;
        if (previousLayers[0].has_value()) previousLayers[0].value()->replaceErrorInput(errorOutputs[0], std::nullopt);
        errorOutputs[0] = std::nullopt;
    }

    RaggedTensorDescriptor sourceDescriptor;
    RaggedTensorDescriptor indicesDescriptor;
    RaggedTensorDescriptor outputDescriptor;
    bool sharedOffsets = false;
    uint32_t inputPortCount = 0;
    uint32_t indicesOffsetsInputPort = 3;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    Event outputsReadyEvent;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
