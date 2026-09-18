#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/DistinctProducerStreamJoin.h"
#include "DeepLearning/Implementation/Layers/DistinctTargetStreamFanout.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Concatenate canonical rank-1 RaggedTensor packed values along a trailing
// value axis. The final input port is the shared managed [1] active-count
// carrier. CUDA no longer reads that scalar payload; its authoritative published
// host partition determines the active packed prefix and is propagated onward.
class RaggedConcatenate : public MultiConnectionLayer {
   public:
    RaggedConcatenate(unsigned int valuesAxis, uint32_t expectedValueInputs, uint64_t batchSize)
        : axis(valuesAxis), valueInputCount(expectedValueInputs), batchSize(batchSize) {
        if (valueInputCount < 2) throw std::invalid_argument("RaggedConcatenate requires at least two values inputs.");
        if (batchSize == 0) throw std::invalid_argument("RaggedConcatenate batch size must be positive.");
        partitionInputIndex = valueInputCount;
        const uint32_t totalInputCount = valueInputCount + 1;
        previousLayers.resize(totalInputCount);
        featureInputs.resize(totalInputCount);
        streams.resize(totalInputCount);
        errorOutputs.resize(totalInputCount);
        forwardInputReadyEvents.resize(totalInputCount);
    }

    ~RaggedConcatenate() override = default;

    std::string getType() override { return "RaggedConcatenate"; }

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (!featureInputs[0].has_value()) throw std::logic_error("RaggedConcatenate input[0] is missing.");
        const TensorDescriptor& reference = featureInputs[0]->getDescriptor();
        const auto& referenceDimensions = reference.getDimensions();
        if (referenceDimensions.size() < 2 || axis == 0 || axis >= referenceDimensions.size()) {
            throw std::logic_error("RaggedConcatenate axis must name a trailing packed-value dimension.");
        }
        const uint64_t capacityRows = referenceDimensions[0];
        uint64_t newAxisSize = 0;
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (!featureInputs[i].has_value()) throw std::logic_error("RaggedConcatenate values input is missing.");
            const TensorDescriptor& descriptor = featureInputs[i]->getDescriptor();
            const auto& dimensions = descriptor.getDimensions();
            if (descriptor.getDataType() != reference.getDataType() || dimensions.size() != referenceDimensions.size() ||
                dimensions[0] != capacityRows) {
                throw std::invalid_argument("RaggedConcatenate values inputs must share dtype, rank, and packed capacity.");
            }
            for (uint32_t d = 1; d < dimensions.size(); ++d) {
                if (d != axis && dimensions[d] != referenceDimensions[d]) {
                    throw std::invalid_argument("RaggedConcatenate non-concatenated trailing dimensions must match.");
                }
            }
            newAxisSize += dimensions[axis];
        }
        if (!featureInputs[partitionInputIndex].has_value())
            throw std::logic_error("RaggedConcatenate active-count carrier is missing.");
        const TensorDescriptor& activeCountDescriptor = featureInputs[partitionInputIndex]->getDescriptor();
        if (activeCountDescriptor.getDimensions() != std::vector<uint64_t>{1}) {
            throw std::invalid_argument("RaggedConcatenate active-count shape must be [1].");
        }
        if (activeCountDescriptor.getDataType() != DataType::UINT32 &&
            activeCountDescriptor.getDataType() != DataType::UINT64) {
            throw std::invalid_argument("RaggedConcatenate active count must use UINT32 or UINT64 storage.");
        }

        std::vector<uint64_t> outputDimensions = referenceDimensions;
        outputDimensions[axis] = newAxisSize;
        return Tensor(featureInputs[0]->getPlacement(), TensorDescriptor(reference.getDataType(), outputDimensions));
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1 && featureOutputs[0].has_value());
        THOR_THROW_IF_FALSE(nextLayers.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[partitionInputIndex].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        splitTensorErrorOutputMemoriesArray_d = nullptr;
        spanGeometryPerSplitTensor_d = nullptr;

        std::vector<void*> valuePointers(valueInputCount);
        for (uint32_t i = 0; i < valueInputCount; ++i) valuePointers[i] = featureInputs[i]->getMemPtr();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&splitTensorFeatureInputMemoriesArray_d), valueInputCount * sizeof(void*)));
        CUDA_CHECK(cudaMemcpyAsync(splitTensorFeatureInputMemoriesArray_d, valuePointers.data(), valueInputCount * sizeof(void*),
                                   cudaMemcpyHostToDevice, streams[0].getStream()));

        // Value-input connections and their Tensor backing allocations are fixed
        // once this layer is compiled, so the forward pointer table is static.
        // Keeping it device-resident avoids a per-batch H2D copy and callback.

        if (errorInputs[0].has_value()) {
            discardedErrorOutputs.resize(valueInputCount);
            std::vector<void*> errorPointers(valueInputCount);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&splitTensorErrorOutputMemoriesArray_d), valueInputCount * sizeof(void*)));
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (errorOutputs[i].has_value()) {
                    errorPointers[i] = errorOutputs[i]->getMemPtr();
                } else {
                    discardedErrorOutputs[i] = featureInputs[i]->clone();
                    errorPointers[i] = discardedErrorOutputs[i]->getMemPtr();
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(splitTensorErrorOutputMemoriesArray_d, errorPointers.data(), valueInputCount * sizeof(void*),
                                       cudaMemcpyHostToDevice, streams[0].getStream()));
        }

        std::vector<uint64_t> axisOffsets(valueInputCount + 1, 0);
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            axisOffsets[i + 1] = axisOffsets[i] + featureInputs[i]->getDescriptor().getDimensions()[axis];
        }
        const TensorDescriptor& outputDescriptor = featureOutputs[0]->getDescriptor();
        const auto& outputDimensions = outputDescriptor.getDimensions();
        uint64_t outerSlicesPerValue = 1;
        for (uint32_t d = 1; d < axis; ++d) outerSlicesPerValue *= outputDimensions[d];
        uint64_t innerElements = 1;
        for (uint32_t d = axis + 1; d < outputDimensions.size(); ++d) innerElements *= outputDimensions[d];
        const std::vector<RaggedConcatenateSpanGeometry> spanGeometry = buildRaggedConcatenateSpanGeometry(
            TensorDescriptor::getElementSizeInBytes(outputDescriptor.getDataType()),
            outerSlicesPerValue,
            innerElements,
            axisOffsets);
        CUDA_CHECK(cudaMalloc(&spanGeometryPerSplitTensor_d, spanGeometry.size() * sizeof(RaggedConcatenateSpanGeometry)));
        CUDA_CHECK(cudaMemcpyAsync(spanGeometryPerSplitTensor_d, spanGeometry.data(),
                                   spanGeometry.size() * sizeof(RaggedConcatenateSpanGeometry),
                                   cudaMemcpyHostToDevice, streams[0].getStream()));
        streams[0].synchronize();

        for (uint32_t i = 0; i < featureInputs.size(); ++i) allFeatureInputTensorIds.insert(featureInputs[i]->getTensorId());
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t runtimeBatchSize = 0) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        // runtimeBatchSize identifies the valid example prefix. The authoritative
        // host partition published on the managed active-count carrier supplies
        // its packed-value end; CUDA never rereads the scalar payload.
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount =
            runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        THOR_THROW_IF_FALSE(resolvedValidExampleCount <= physicalBatchCapacity);
        if (batchCardinalitySet) {
            THOR_THROW_IF_FALSE(currentValidExampleCount == resolvedValidExampleCount);
        } else {
            currentValidExampleCount = resolvedValidExampleCount;
            batchCardinalitySet = true;
        }

        auto it = stillWaitingForFeatureInputTensors.find(featureInput->getTensorId());
        THOR_THROW_IF_FALSE(it != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(it);
        if (!stillWaitingForFeatureInputTensors.empty()) return;
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;

        detail::waitForDistinctProducerStreams(streams[0], streams, forwardInputReadyEvents, 1);

        const TensorDescriptor& outputDescriptor = featureOutputs[0]->getDescriptor();
        const auto& outputDimensions = outputDescriptor.getDimensions();
        THOR_THROW_IF_FALSE(!outputDimensions.empty() && outputDimensions[0] > 0);
        const uint64_t elementsPerOutputValue = outputDescriptor.getTotalNumElements() / outputDimensions[0];
        uint64_t outerSlicesPerValue = 1;
        for (uint32_t d = 1; d < axis; ++d) outerSlicesPerValue *= outputDimensions[d];
        RowPartitionRuntime rowPartition = RowPartitionRuntime::fromHostStateCarrier(
            featureInputs[partitionInputIndex].value(), batchSize, outputDimensions[0]);
        const uint64_t activeRows = rowPartition.requireHostOffset(currentValidExampleCount);
        launchRaggedConcatenate(
            featureOutputs[0]->getMemPtr(),
            splitTensorFeatureInputMemoriesArray_d,
            outputDimensions[0],
            elementsPerOutputValue * TensorDescriptor::getElementSizeInBytes(outputDescriptor.getDataType()),
            outerSlicesPerValue,
            valueInputCount,
            spanGeometryPerSplitTensor_d,
            activeRows,
            streams[0]);

        rowPartition.publishHostStateTo(featureOutputs[0].value());
        nextLayers[0].value()->forward(featureOutputs[0], validationPass, currentValidExampleCount);
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        if (!errorInput.has_value()) return;
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolvedValidExampleCount =
            runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        THOR_THROW_IF_FALSE(resolvedValidExampleCount <= physicalBatchCapacity);
        if (splitTensorErrorOutputMemoriesArray_d != nullptr) {
            const TensorDescriptor& errorDescriptor = errorInput->getDescriptor();
            const auto& dimensions = errorDescriptor.getDimensions();
            const uint64_t elementsPerSourceValue = errorDescriptor.getTotalNumElements() / dimensions[0];
            uint64_t outerSlicesPerValue = 1;
            for (uint32_t d = 1; d < axis; ++d) outerSlicesPerValue *= dimensions[d];
            RowPartitionRuntime rowPartition = RowPartitionRuntime::fromHostStateCarrier(
                featureInputs[partitionInputIndex].value(), batchSize, dimensions[0]);
            const uint64_t activeRows = rowPartition.requireHostOffset(resolvedValidExampleCount);
            launchRaggedSplit(
                splitTensorErrorOutputMemoriesArray_d,
                errorInput->getMemPtr(),
                dimensions[0],
                elementsPerSourceValue * TensorDescriptor::getElementSizeInBytes(errorDescriptor.getDataType()),
                outerSlicesPerValue,
                valueInputCount,
                spanGeometryPerSplitTensor_d,
                activeRows,
                streams[0]);
        }

        ThorImplementation::detail::recordCompletionAndWaitOnDistinctTargetStreams(
            streams[0],
            backwardOutputsReadyEvent,
            valueInputCount,
            [](std::size_t) { return true; },
            [this](std::size_t i) -> const Stream& { return streams[i]; });
        for (uint32_t i = 0; i < valueInputCount; ++i) {
            if (previousLayers[i].has_value())
                previousLayers[i].value()->backward(errorOutputs[i], resolvedValidExampleCount);
        }
    }

    void cleanup() override {
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        ScopedGpu scopedGpu(featureInputs[0]->getPlacement().getDeviceNum());
        if (splitTensorFeatureInputMemoriesArray_d != nullptr) CUDA_CHECK(cudaFree(splitTensorFeatureInputMemoriesArray_d));
        if (splitTensorErrorOutputMemoriesArray_d != nullptr) CUDA_CHECK(cudaFree(splitTensorErrorOutputMemoriesArray_d));
        if (spanGeometryPerSplitTensor_d != nullptr) CUDA_CHECK(cudaFree(spanGeometryPerSplitTensor_d));
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        splitTensorErrorOutputMemoriesArray_d = nullptr;
        spanGeometryPerSplitTensor_d = nullptr;
        discardedErrorOutputs.clear();
        for (Event& event : forwardInputReadyEvents)
            event = Event();
        backwardOutputsReadyEvent = Event();
        MultiConnectionLayer::cleanup();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        (void)driverConnectionType;
        THOR_THROW_IF_FALSE(!running);
        nextLayers.push_back(nextLayer);
        featureOutputs.emplace_back(createFeatureOutputTensor());
        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams[0], shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType));
        if (!errorInputs.back().has_value()) {
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (errorOutputs[i].has_value() && previousLayers[i].has_value())
                    previousLayers[i].value()->replaceErrorInput(errorOutputs[i], std::nullopt);
                errorOutputs[i] = std::nullopt;
            }
        }
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream,
        bool backPropagateError, int connectionType) override {
        THOR_THROW_IF_FALSE(!running && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || static_cast<uint32_t>(connectionType) > partitionInputIndex)
            throw std::logic_error("RaggedConcatenate connection type is outside its declared input range.");
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value())
            throw std::logic_error("RaggedConcatenate input port was connected more than once.");

        streams[inputIndex] = stream;
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        if (inputIndex < valueInputCount && backPropagateError && !isInferenceOnly())
            errorOutputs[inputIndex] = featureInput->clone();
        else
            errorOutputs[inputIndex] = std::nullopt;
        ensureNoDeviceCrossing();
        return errorOutputs[inputIndex];
    }

    uint64_t logicalByteCountForward(uint64_t validExampleCount) override {
        const std::optional<uint64_t> activeRows = logicalActiveRows(validExampleCount);
        if (!activeRows.has_value()) return 0;

        try {
            uint64_t bytes = 0;
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (!featureInputs[i].has_value()) return 0;
                const std::optional<uint64_t> contribution = logicalBytesForActiveRows(featureInputs[i].value(), *activeRows);
                if (!contribution.has_value()) return 0;
                bytes = checkedLogicalByteAdd(bytes, *contribution, "RaggedConcatenate forward input");
            }
            if (featureOutputs.size() != 1 || !featureOutputs[0].has_value()) return 0;
            const std::optional<uint64_t> outputBytes = logicalBytesForActiveRows(featureOutputs[0].value(), *activeRows);
            if (!outputBytes.has_value()) return 0;
            return checkedLogicalByteAdd(bytes, *outputBytes, "RaggedConcatenate forward output");
        } catch (...) {
            // Logical-work telemetry is best-effort and must not fail submitted model work.
            return 0;
        }
    }

    uint64_t logicalByteCountBackward(uint64_t validExampleCount) override {
        const std::optional<uint64_t> activeRows = logicalActiveRows(validExampleCount);
        if (!activeRows.has_value()) return 0;

        try {
            if (errorInputs.empty() || !errorInputs[0].has_value()) return 0;
            const std::optional<uint64_t> inputBytes = logicalBytesForActiveRows(errorInputs[0].value(), *activeRows);
            if (!inputBytes.has_value()) return 0;
            uint64_t bytes = *inputBytes;
            // The physical split kernel may write discarded buffers for pruned
            // branches. Those are implementation artifacts, not authored logical
            // gradient results, so count only the error outputs that remain connected.
            for (uint32_t i = 0; i < valueInputCount; ++i) {
                if (!errorOutputs[i].has_value()) continue;
                const std::optional<uint64_t> outputBytes = logicalBytesForActiveRows(errorOutputs[i].value(), *activeRows);
                if (!outputBytes.has_value()) return 0;
                bytes = checkedLogicalByteAdd(bytes, *outputBytes, "RaggedConcatenate backward output");
            }
            return bytes;
        } catch (...) {
            return 0;
        }
    }

   private:
    [[nodiscard]] std::optional<uint64_t> logicalActiveRows(uint64_t validExampleCount) const noexcept {
        try {
            if (featureInputs.size() <= partitionInputIndex || featureInputs.empty() || !featureInputs[0].has_value() ||
                !featureInputs[partitionInputIndex].has_value()) {
                return std::nullopt;
            }
            const Tensor& carrier = featureInputs[partitionInputIndex].value();
            if (validExampleCount == 0) {
                const std::optional<uint64_t> active =
                    RowPartitionRuntime::getPublishedHostActiveValueCountIfAvailable(carrier);
                if (!active.has_value() || *active > featureInputs[0]->getDimensions()[0]) return std::nullopt;
                return active;
            }
            if (validExampleCount > batchSize) return std::nullopt;
            const std::optional<uint64_t> active =
                RowPartitionRuntime::getPublishedHostOffsetIfAvailable(carrier, validExampleCount);
            if (!active.has_value() || *active > featureInputs[0]->getDimensions()[0]) return std::nullopt;
            return active;
        } catch (...) {
            return std::nullopt;
        }
    }

    [[nodiscard]] static std::optional<uint64_t> logicalBytesForActiveRows(
        const Tensor& tensor, uint64_t activeRows) noexcept {
        try {
            const std::vector<uint64_t> dimensions = tensor.getDimensions();
            if (dimensions.empty() || dimensions[0] == 0 || activeRows > dimensions[0]) return std::nullopt;
            const uint64_t totalBytes = tensor.getArraySizeInBytes();
            if (totalBytes % dimensions[0] != 0) return std::nullopt;
            const uint64_t bytesPerRow = totalBytes / dimensions[0];
            if (bytesPerRow != 0 && activeRows > std::numeric_limits<uint64_t>::max() / bytesPerRow) {
                return std::nullopt;
            }
            return activeRows * bytesPerRow;
        } catch (...) {
            return std::nullopt;
        }
    }
    unsigned int axis;
    uint32_t valueInputCount;
    uint32_t partitionInputIndex;
    uint64_t batchSize;
    void **splitTensorFeatureInputMemoriesArray_d = nullptr;
    void **splitTensorErrorOutputMemoriesArray_d = nullptr;
    RaggedConcatenateSpanGeometry *spanGeometryPerSplitTensor_d = nullptr;
    std::vector<std::optional<Tensor>> discardedErrorOutputs;
    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;

    std::vector<Event> forwardInputReadyEvents;
    Event backwardOutputsReadyEvent;
};

}  // namespace ThorImplementation
