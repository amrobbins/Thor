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

// Physical R9E padded-dense -> ragged adapter. Input 0 is normal dense
// [B, padded_width, ...] data and input 1 is an already-canonical partition's
// offsets. The output creates only packed values and reuses that exact
// partition; no structural tensor is copied or regenerated.
class PaddedDenseToRagged : public MultiConnectionLayer {
   public:
    PaddedDenseToRagged(TensorDescriptor denseInputDescriptor,
                        RaggedTensorDescriptor partitionDescriptor,
                        RaggedTensorDescriptor outputDescriptor)
        : denseInputDescriptor(std::move(denseInputDescriptor)),
          partitionDescriptor(std::move(partitionDescriptor)),
          outputDescriptor(std::move(outputDescriptor)) {
        if (!this->partitionDescriptor.hasMaxValuesPerRow()) {
            throw std::invalid_argument("PaddedDenseToRagged requires max_values_per_row on partition_input.");
        }
        const std::vector<uint64_t> denseDims = this->denseInputDescriptor.getDimensions();
        if (denseDims.size() < 2 || denseDims[0] != this->partitionDescriptor.getBatchSize() ||
            denseDims[1] < this->partitionDescriptor.getMaxValuesPerRow()) {
            throw std::invalid_argument("PaddedDenseToRagged dense input must be [B, padded_width, ...] with padded_width >= max_values_per_row.");
        }
        if (this->outputDescriptor.getRowPartition() != this->partitionDescriptor.getRowPartition() ||
            this->outputDescriptor.getValuesDataType() != this->denseInputDescriptor.getDataType() ||
            this->outputDescriptor.getTrailingDimensions() !=
                std::vector<uint64_t>(denseDims.begin() + 2, denseDims.end())) {
            throw std::invalid_argument("PaddedDenseToRagged output must reuse partition_input and dense trailing value geometry.");
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

    ~PaddedDenseToRagged() override = default;

    std::string getType() override { return "PaddedDenseToRagged"; }

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
        static_cast<void>(raggedFromDense(featureInputs[0].value(),
                                          featureInputs[1].value(),
                                          featureOutputs[0].value(),
                                          validationErrorBits,
                                          streams[0]));

        if (nextLayers[0].has_value()) nextLayers[0].value()->forward(featureOutputs[0], validationPass, validExamples);
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!errorInput.has_value()) return;
        if (!errorInputs[0].has_value() || errorInput.value() != errorInputs[0].value()) {
            throw std::logic_error("PaddedDenseToRagged received an unknown output gradient.");
        }
        if (!errorOutputs[0].has_value() || !previousLayers[0].has_value()) return;

        const uint32_t validExamples = resolvedRuntimeBatch(runtimeBatchSize);
        RaggedTensor raggedGradient(errorInput.value(),
                                    RowPartitionRuntime(featureInputs[1].value(), outputDescriptor.getRowPartition()));
        raggedToDense(raggedGradient, errorOutputs[0].value(), 0.0, validationErrorBits, streams[0]);
        previousLayers[0].value()->backward(errorOutputs[0], validExamples);
    }

    void connectToNextLayer(Layer* nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        if (driverConnectionType != 0) throw std::logic_error("PaddedDenseToRagged has only values output connection type 0.");
        if (nextLayers[0].has_value()) {
            throw std::logic_error("PaddedDenseToRagged values output was connected more than once without a fanout.");
        }
        ensureOutputAllocated();
        nextLayers[0] = nextLayer;
        const bool backPropagate = shouldConnectToBackPropErrorIn() && !isBackPropStub();
        errorInputs[0] = nextLayer->connectToPreviousLayer(this, featureOutputs[0], streams[0], backPropagate, loaderConnectionType);
        if (!errorInputs[0].has_value()) pruneUpstreamDenseGradient();
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(Layer* previousLayer,
                                                  std::optional<Tensor> featureInput,
                                                  Stream stream,
                                                  bool backPropagateError,
                                                  int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || connectionType > 1) {
            throw std::logic_error("PaddedDenseToRagged input connection type must be 0 (dense values) or 1 (offsets).");
        }
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("PaddedDenseToRagged input port was connected more than once.");
        }
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        streams[inputIndex] = stream;
        if (inputIndex == 0 && backPropagateError && !isInferenceOnly()) {
            const DataType dtype = featureInput->getDataType();
            if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
                throw std::invalid_argument("PaddedDenseToRagged training supports only FP16, BF16, and FP32 values.");
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
            throw std::logic_error("PaddedDenseToRagged can replace only its values-output gradient.");
        }
        errorInputs[0] = newErrorInput;
        if (!newErrorInput.has_value()) pruneUpstreamDenseGradient();
    }

    // Dense adapter padding is structural capacity, not logical model payload.
    // Count only numerical values inside the authoritative active ragged prefix;
    // physical padding fill/zeroing and row-partition metadata are excluded.
    uint64_t logicalByteCountForward(uint64_t validExampleCount) override {
        const std::optional<uint64_t> activeValues = logicalActiveValues(validExampleCount);
        if (!activeValues.has_value() || featureInputs.size() < 2 || !featureInputs[0].has_value() ||
            featureOutputs.size() != 1 || !featureOutputs[0].has_value()) return 0;
        return logicalAdapterBytes(featureInputs[0].value(), denseLogicalValueCapacity(),
                                   featureOutputs[0].value(), outputDescriptor.getMaxTotalValues(), *activeValues,
                                   "PaddedDenseToRagged forward");
    }
    uint64_t logicalByteCountBackward(uint64_t validExampleCount) override {
        const std::optional<uint64_t> activeValues = logicalActiveValues(validExampleCount);
        if (!activeValues.has_value()) return 0;
        if (errorInputs.empty() || !errorInputs[0].has_value() || errorOutputs.empty() || !errorOutputs[0].has_value()) return 0;
        return logicalAdapterBytes(errorInputs[0].value(), outputDescriptor.getMaxTotalValues(),
                                   errorOutputs[0].value(), denseLogicalValueCapacity(), *activeValues,
                                   "PaddedDenseToRagged backward");
    }

   private:
    [[nodiscard]] std::optional<uint64_t> logicalActiveValues(uint64_t validExampleCount) const noexcept {
        try {
            if (featureInputs.size() < 2 || !featureInputs[1].has_value()) return std::nullopt;
            const Tensor& carrier = featureInputs[1].value();
            if (validExampleCount == 0) {
                const auto active = RowPartitionRuntime::getPublishedHostActiveValueCountIfAvailable(carrier);
                if (!active.has_value() || *active > outputDescriptor.getMaxTotalValues()) return std::nullopt;
                return active;
            }
            if (validExampleCount > partitionDescriptor.getBatchSize()) return std::nullopt;
            const auto active = RowPartitionRuntime::getPublishedHostOffsetIfAvailable(carrier, validExampleCount);
            if (!active.has_value() || *active > outputDescriptor.getMaxTotalValues()) return std::nullopt;
            return active;
        } catch (...) {
            return std::nullopt;
        }
    }

    [[nodiscard]] uint64_t denseLogicalValueCapacity() const noexcept {
        try {
            const std::vector<uint64_t> dims = denseInputDescriptor.getDimensions();
            if (dims.size() < 2 || dims[0] != partitionDescriptor.getBatchSize()) return 0;
            if (dims[0] != 0 && dims[1] > std::numeric_limits<uint64_t>::max() / dims[0]) return 0;
            return dims[0] * dims[1];
        } catch (...) {
            return 0;
        }
    }

    [[nodiscard]] static uint64_t logicalAdapterBytes(const Tensor& first,
                                                       uint64_t firstValueCapacity,
                                                       const Tensor& second,
                                                       uint64_t secondValueCapacity,
                                                       uint64_t activeValues,
                                                       const char* where) noexcept {
        try {
            if (firstValueCapacity == 0 || secondValueCapacity == 0 || activeValues > firstValueCapacity ||
                activeValues > secondValueCapacity) return 0;
            const uint64_t firstBytes = first.getArraySizeInBytes();
            const uint64_t secondBytes = second.getArraySizeInBytes();
            if (firstBytes % firstValueCapacity != 0 || secondBytes % secondValueCapacity != 0) return 0;
            const uint64_t firstBytesPerValue = firstBytes / firstValueCapacity;
            const uint64_t secondBytesPerValue = secondBytes / secondValueCapacity;
            if (firstBytesPerValue != 0 && activeValues > std::numeric_limits<uint64_t>::max() / firstBytesPerValue) return 0;
            if (secondBytesPerValue != 0 && activeValues > std::numeric_limits<uint64_t>::max() / secondBytesPerValue) return 0;
            const uint64_t firstActiveBytes = activeValues * firstBytesPerValue;
            const uint64_t secondActiveBytes = activeValues * secondBytesPerValue;
            if (secondActiveBytes > std::numeric_limits<uint64_t>::max() - firstActiveBytes) return 0;
            (void)where;
            return firstActiveBytes + secondActiveBytes;
        } catch (...) {
            return 0;
        }
    }
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
        const uint64_t batchSize = partitionDescriptor.getBatchSize();
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
        featureOutputs[0] = Tensor(firstInput->getPlacement(), outputDescriptor.getValuesDescriptor());
    }

    void validateConnectedInputs() const {
        THOR_THROW_IF_FALSE(featureInputs[0].has_value() && featureInputs[1].has_value());
        if (featureInputs[0]->getPlacement() != featureInputs[1]->getPlacement()) {
            throw std::invalid_argument("PaddedDenseToRagged dense values and offsets must reside on one device.");
        }
        if (featureInputs[0]->getDescriptor() != denseInputDescriptor) {
            throw std::invalid_argument("PaddedDenseToRagged dense input does not match its declared descriptor.");
        }
        if (featureInputs[1]->getDescriptor() != partitionDescriptor.getOffsetsDescriptor()) {
            throw std::invalid_argument("PaddedDenseToRagged offsets input does not match its declared descriptor.");
        }
    }

    void pruneUpstreamDenseGradient() {
        if (!errorOutputs[0].has_value()) return;
        if (previousLayers[0].has_value()) previousLayers[0].value()->replaceErrorInput(errorOutputs[0], std::nullopt);
        errorOutputs[0] = std::nullopt;
    }

    TensorDescriptor denseInputDescriptor;
    RaggedTensorDescriptor partitionDescriptor;
    RaggedTensorDescriptor outputDescriptor;
    Tensor validationErrorBits;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    std::vector<Event> forwardInputReadyEvents;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;
};

}  // namespace ThorImplementation
