#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/TensorOperations/Ragged/RaggedDenseAdapters.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Physical R9E ragged -> padded-dense adapter. Input ports are packed values
// and canonical offsets. The single output is a normal dense tensor with
// physical shape [B, max_values_per_row, ...]. Padding is constant and therefore
// has no gradient; backward gathers only logical dense gradients back into the
// active packed prefix using the same canonical partition.
class RaggedToPaddedDense : public MultiConnectionLayer {
   public:
    RaggedToPaddedDense(RaggedTensorDescriptor inputDescriptor,
                        TensorDescriptor outputDescriptor,
                        double paddingValue)
        : inputDescriptor(std::move(inputDescriptor)),
          outputDescriptor(std::move(outputDescriptor)),
          paddingValue(paddingValue) {
        if (!this->inputDescriptor.hasMaxValuesPerRow()) {
            throw std::invalid_argument("RaggedToPaddedDense requires max_values_per_row on its ragged input descriptor.");
        }
        const std::vector<uint64_t> outputDims = this->outputDescriptor.getDimensions();
        if (outputDims.size() != this->inputDescriptor.getTrailingDimensions().size() + 2 ||
            outputDims[0] != this->inputDescriptor.getBatchSize() ||
            outputDims[1] != this->inputDescriptor.getMaxValuesPerRow() ||
            this->outputDescriptor.getDataType() != this->inputDescriptor.getValuesDataType() ||
            std::vector<uint64_t>(outputDims.begin() + 2, outputDims.end()) != this->inputDescriptor.getTrailingDimensions()) {
            throw std::invalid_argument("RaggedToPaddedDense output descriptor must be [B, max_values_per_row, ...trailing].");
        }

        previousLayers.resize(2);
        featureInputs.resize(2);
        errorOutputs.resize(2);
        streams.resize(2);
        forwardInputReadyEvents.resize(2);

        featureOutputs.resize(1);
        errorInputs.resize(1);
        nextLayers.resize(1);
    }

    ~RaggedToPaddedDense() override = default;

    std::string getType() override { return "RaggedToPaddedDense"; }

    std::optional<Tensor> createFeatureOutputTensor() override { THOR_UNREACHABLE(); }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == 2);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        validateConnectedInputs();
        ensureOutputAllocated();
        validationErrorBits = Tensor(featureInputs[0]->getPlacement(), TensorDescriptor(DataType::UINT32, {1}));
        allFeatureInputTensorIds = {featureInputs[0]->getTensorId(), featureInputs[1]->getTensorId()};
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void cleanup() override {
        validationErrorBits = Tensor();
        for (Event& event : forwardInputReadyEvents) event = Event();
        outputsReadyEvent = Event();
        MultiConnectionLayer::cleanup();
    }

    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(running && featureInput.has_value());
        const uint32_t validExamples = resolveBatchCardinality(runtimeBatchSize);

        auto waiting = stillWaitingForFeatureInputTensors.find(featureInput->getTensorId());
        THOR_THROW_IF_FALSE(waiting != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(waiting);
        if (!stillWaitingForFeatureInputTensors.empty()) return;
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;

        streams[0].waitFor(streams[1], forwardInputReadyEvents[1]);
        RaggedTensor ragged(featureInputs[0].value(), RowPartitionRuntime(featureInputs[1].value(), inputDescriptor.getRowPartition()));
        raggedToDense(ragged, featureOutputs[0].value(), paddingValue, validationErrorBits, streams[0]);

        streams[0].putEvent(outputsReadyEvent);
        if (nextLayers[0].has_value()) nextLayers[0].value()->forward(featureOutputs[0], validationPass, validExamples);
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!errorInput.has_value()) return;
        if (!errorInputs[0].has_value() || errorInput.value() != errorInputs[0].value()) {
            throw std::logic_error("RaggedToPaddedDense received an unknown output gradient.");
        }
        if (!errorOutputs[0].has_value() || !previousLayers[0].has_value()) return;

        const uint32_t validExamples = resolvedRuntimeBatch(runtimeBatchSize);
        static_cast<void>(raggedFromDense(errorInput.value(),
                                          featureInputs[1].value(),
                                          errorOutputs[0].value(),
                                          validationErrorBits,
                                          streams[0]));
        previousLayers[0].value()->backward(errorOutputs[0], validExamples);
    }

    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        if (driverConnectionType != 0) throw std::logic_error("RaggedToPaddedDense has only output connection type 0.");
        if (nextLayers[0].has_value()) {
            throw std::logic_error("RaggedToPaddedDense output was connected more than once without a fanout.");
        }
        ensureOutputAllocated();
        nextLayers[0] = nextLayer;
        const bool backPropagate = shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[0] = nextLayer->connectToPreviousLayer(this, featureOutputs[0], streams[0], backPropagate, loaderConnectionType);
        if (!errorInputs[0].has_value()) pruneUpstreamValuesGradient();
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || connectionType > 1) {
            throw std::logic_error("RaggedToPaddedDense input connection type must be 0 (values) or 1 (offsets).");
        }
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("RaggedToPaddedDense input port was connected more than once.");
        }
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        streams[inputIndex] = stream;
        if (inputIndex == 0 && backPropagateError && !isInferenceOnly()) {
            const DataType dtype = featureInput->getDataType();
            if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
                throw std::invalid_argument("RaggedToPaddedDense training supports only FP16, BF16, and FP32 values.");
            }
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
            throw std::logic_error("RaggedToPaddedDense can replace only its dense-output gradient.");
        }
        errorInputs[0] = newErrorInput;
        if (!newErrorInput.has_value()) pruneUpstreamValuesGradient();
    }

   private:
    uint32_t resolveBatchCardinality(uint32_t runtimeBatchSize) {
        const uint32_t resolved = resolvedRuntimeBatch(runtimeBatchSize);
        if (batchCardinalitySet) {
            THOR_THROW_IF_FALSE(currentValidExampleCount == resolved);
        } else {
            currentValidExampleCount = resolved;
            batchCardinalitySet = true;
        }
        return resolved;
    }

    uint32_t resolvedRuntimeBatch(uint32_t runtimeBatchSize) const {
        const uint64_t batchSize = inputDescriptor.getBatchSize();
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t capacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolved = runtimeBatchSize == 0 ? capacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolved >= 1 && resolved <= capacity);
        return resolved;
    }

    void ensureOutputAllocated() {
        if (featureOutputs[0].has_value()) return;
        std::optional<Tensor> firstInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(firstInput.has_value());
        featureOutputs[0] = Tensor(firstInput->getPlacement(), outputDescriptor);
    }

    void validateConnectedInputs() const {
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value());
        if (featureInputs[0]->getPlacement() != featureInputs[1]->getPlacement()) {
            throw std::invalid_argument("RaggedToPaddedDense values and offsets must reside on one device.");
        }
        if (featureInputs[0]->getDescriptor() != inputDescriptor.getValuesDescriptor()) {
            throw std::invalid_argument("RaggedToPaddedDense values input does not match its declared descriptor.");
        }
        if (featureInputs[1]->getDescriptor() != inputDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("RaggedToPaddedDense offsets input does not match its declared descriptor.");
        }
    }

    void pruneUpstreamValuesGradient() {
        if (!errorOutputs[0].has_value()) return;
        if (previousLayers[0].has_value()) previousLayers[0].value()->replaceErrorInput(errorOutputs[0], std::nullopt);
        errorOutputs[0] = std::nullopt;
    }

    RaggedTensorDescriptor inputDescriptor;
    TensorDescriptor outputDescriptor;
    double paddingValue = 0.0;
    Tensor validationErrorBits;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    Event outputsReadyEvent;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
