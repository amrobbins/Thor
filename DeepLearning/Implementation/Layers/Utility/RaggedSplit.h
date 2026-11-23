#pragma once

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/DistinctProducerStreamJoin.h"
#include "DeepLearning/Implementation/Layers/DistinctTargetStreamFanout.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"
#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

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

// Split canonical rank-1 RaggedTensor packed values along a trailing value axis into consecutive pieces that share the
// input's row partition: the reverse of RaggedConcatenate. Input 0 is the packed values, input 1 the shared managed
// [1] active-count carrier, whose published host partition gives the active packed prefix (CUDA does not read it).
// Output i is driver connection type i and has outputDataTypes[i], converted as it moves; the host partition is
// published on every piece. Backward, one concatenate converts the pieces' gradients into the input's; a piece that
// takes no gradient (its consumer sends none, or its type or the input's is integral) contributes zeros, read from a
// zero tensor of the input's type made at compile.
class RaggedSplit : public MultiConnectionLayer {
   public:
    RaggedSplit(unsigned int valuesAxis, std::vector<unsigned long> axisElements, uint64_t batchSize,
                std::vector<DataType> outputDataTypes)
        : axis(valuesAxis), axisElements(std::move(axisElements)), batchSize(batchSize),
          outputDataTypes(std::move(outputDataTypes)) {
        if (this->axisElements.size() < 2) throw std::invalid_argument("RaggedSplit requires at least two outputs.");
        if (this->outputDataTypes.size() != this->axisElements.size())
            throw std::invalid_argument("RaggedSplit takes one output data type per output.");
        for (const unsigned long elements : this->axisElements)
            if (elements == 0) throw std::invalid_argument("RaggedSplit outputs must have a positive extent on the axis.");
        if (batchSize == 0) throw std::invalid_argument("RaggedSplit batch size must be positive.");
        const size_t outputs = this->axisElements.size();
        previousLayers.resize(kInputs);
        featureInputs.resize(kInputs);
        errorOutputs.resize(kInputs);
        forwardInputReadyEvents.resize(kInputs);
        // The inputs' streams, then a stream for every output after the first (which runs on the values input's).
        streams.resize(kInputs + outputs - 1);
        featureOutputs.assign(outputs, std::nullopt);
        nextLayers.assign(outputs, std::nullopt);
        errorInputs.assign(outputs, std::nullopt);
        backwardInputReadyEvents.resize(outputs);
    }

    ~RaggedSplit() override = default;

    std::string getType() override { return "RaggedSplit"; }

    std::optional<Tensor> createFeatureOutputTensor() override {
        throw std::logic_error("RaggedSplit creates output i when connection type i is connected.");
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        const uint32_t outputs = static_cast<uint32_t>(axisElements.size());
        THOR_THROW_IF_FALSE(featureInputs[kValuesInput].has_value() && featureInputs[kPartitionInput].has_value());
        THOR_THROW_IF_FALSE(featureInputs[kValuesInput]->getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        for (uint32_t i = 0; i < outputs; ++i) {
            if (!featureOutputs[i].has_value() || !nextLayers[i].has_value())
                throw std::logic_error("RaggedSplit output " + std::to_string(i) + " is not connected.");
        }
        inputDataType = featureInputs[kValuesInput]->getDescriptor().getDataType();

        // No output gradient at all: nothing to send back, so the path upstream is pruned.
        if (numPresentTensors(errorInputs) == 0 && errorOutputs[kValuesInput].has_value()) {
            if (previousLayers[kValuesInput].has_value())
                previousLayers[kValuesInput].value()->replaceErrorInput(errorOutputs[kValuesInput], std::nullopt);
            errorOutputs[kValuesInput].reset();
        }

        // Tensors are fixed once this layer is compiled, so their pointers go to every launch as its parameters.
        outputMemories.assign(outputs, nullptr);
        for (uint32_t i = 0; i < outputs; ++i) outputMemories[i] = featureOutputs[i]->getMemPtr();
        backwardSourceMemories.clear();
        backwardSourceDataTypes.clear();
        zeroGradients.clear();
        if (errorOutputs[kValuesInput].has_value()) {
            backwardSourceMemories.assign(outputs, nullptr);
            backwardSourceDataTypes.assign(outputs, inputDataType);
            for (uint32_t i = 0; i < outputs; ++i) {
                if (errorInputs[i].has_value()) {
                    backwardSourceMemories[i] = errorInputs[i]->getMemPtr();
                    backwardSourceDataTypes[i] = errorInputs[i]->getDescriptor().getDataType();
                } else {
                    const TensorDescriptor zeroDescriptor(inputDataType, featureOutputs[i]->getDescriptor().getDimensions());
                    zeroGradients.push_back(Tensor::zeros(featureInputs[kValuesInput]->getPlacement(), zeroDescriptor, streams[0]));
                    backwardSourceMemories[i] = zeroGradients.back().getMemPtr();
                }
            }
        }

        std::vector<uint64_t> axisElementsPerOutput(axisElements.begin(), axisElements.end());
        geometry = buildRaggedTrailingConcatenateGeometry(featureInputs[kValuesInput]->getDescriptor().getDimensions(), axis,
                                                          axisElementsPerOutput);
        allFeatureInputTensorIds.clear();
        for (uint32_t i = 0; i < kInputs; ++i) allFeatureInputTensorIds.insert(featureInputs[i]->getTensorId());
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
        const uint32_t resolvedValidExampleCount = resolveValidExampleCount(runtimeBatchSize);
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

        detail::waitForDistinctProducerStreams(
            streams[0], kInputs, [](std::size_t i) { return i >= 1; },
            [this](std::size_t i) -> const Stream & { return streams[i]; },
            [this](std::size_t i) -> Event & { return forwardInputReadyEvents[i]; });

        const uint64_t activeRows = activeRowsFor(currentValidExampleCount);
        launchSplit(outputMemories, outputDataTypes, featureInputs[kValuesInput]->getMemPtr(), inputDataType,
                    geometry.outerSlicesFor(activeRows), geometry.spanElements, streams[0]);

        RowPartitionRuntime rowPartition = partitionFor();
        for (uint32_t i = 0; i < featureOutputs.size(); ++i) rowPartition.publishHostStateTo(featureOutputs[i].value());
        ThorImplementation::detail::recordCompletionAndWaitOnDistinctTargetStreams(
            streams[0],
            outputsReadyEvent,
            featureOutputs.size(),
            [](std::size_t) { return true; },
            [this](std::size_t i) -> const Stream & { return outputStream(static_cast<uint32_t>(i)); });
        const uint32_t validExamples = currentValidExampleCount;
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
        for (uint32_t i = 0; i < featureOutputs.size(); ++i)
            nextLayers[i].value()->forward(featureOutputs[i], validationPass, validExamples);
    }

    void backward(std::optional<Tensor> errorInput, uint32_t runtimeBatchSize = 0) override {
        if (!errorInput.has_value()) return;
        THOR_THROW_IF_FALSE(errorOutputs[kValuesInput].has_value());
        const uint32_t resolvedValidExampleCount = resolveValidExampleCount(runtimeBatchSize);

        // Every output that takes a gradient delivers it here; the last one to arrive runs the concatenate.
        THOR_THROW_IF_FALSE(stillWaitingForErrorInputTensors.count(errorInput->getTensorId()) == 1);
        stillWaitingForErrorInputTensors.erase(errorInput->getTensorId());
        if (!stillWaitingForErrorInputTensors.empty()) return;
        stillWaitingForErrorInputTensors = allErrorInputTensorIds;

        detail::waitForDistinctProducerStreams(
            streams[0], errorInputs.size(), [this](std::size_t i) { return errorInputs[i].has_value(); },
            [this](std::size_t i) -> const Stream & { return outputStream(static_cast<uint32_t>(i)); },
            [this](std::size_t i) -> Event & { return backwardInputReadyEvents[i]; });

        const uint64_t activeRows = activeRowsFor(resolvedValidExampleCount);
        launchConcatenate(errorOutputs[kValuesInput]->getMemPtr(), inputDataType, backwardSourceMemories, backwardSourceDataTypes,
                          geometry.outerSlicesFor(activeRows), geometry.spanElements, streams[0]);
        if (previousLayers[kValuesInput].has_value())
            previousLayers[kValuesInput].value()->backward(errorOutputs[kValuesInput], resolvedValidExampleCount);
    }

    void cleanup() override {
        outputMemories.clear();
        backwardSourceMemories.clear();
        backwardSourceDataTypes.clear();
        zeroGradients.clear();
        geometry = RaggedTrailingConcatenateGeometry{};
        for (Event &event : forwardInputReadyEvents) event = Event();
        for (Event &event : backwardInputReadyEvents) event = Event();
        outputsReadyEvent = Event();
        MultiConnectionLayer::cleanup();
    }

    // Connection type driverConnectionType is output driverConnectionType.
    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!running);
        if (driverConnectionType < 0 || static_cast<size_t>(driverConnectionType) >= axisElements.size())
            throw std::invalid_argument("RaggedSplit has no output " + std::to_string(driverConnectionType) + ".");
        const uint32_t output = static_cast<uint32_t>(driverConnectionType);
        if (featureOutputs[output].has_value())
            throw std::logic_error("RaggedSplit output " + std::to_string(output) + " is already connected.");
        if (!featureInputs[kValuesInput].has_value() || !featureInputs[kPartitionInput].has_value())
            throw std::logic_error("RaggedSplit outputs connect after its values and partition inputs.");

        const Tensor &values = featureInputs[kValuesInput].value();
        std::vector<unsigned long> dimensions = values.getDescriptor().getDimensions();
        if (dimensions.size() < 2 || axis == 0 || axis >= dimensions.size())
            throw std::logic_error("RaggedSplit axis must name a trailing packed-value dimension.");
        dimensions[axis] = axisElements[output];
        featureOutputs[output] = Tensor(values.getPlacement(), TensorDescriptor(outputDataTypes[output], dimensions));
        nextLayers[output] = nextLayer;
        errorInputs[output] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[output], outputStream(output),
            shouldConnectToBackPropErrorIn() && !isBackPropStub() && takesGradient(output), loaderConnectionType);
        if (errorInputs[output].has_value()) {
            THOR_THROW_IF_FALSE(errorInputs[output]->getDescriptor() == featureOutputs[output]->getDescriptor());
        }
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) override {
        THOR_THROW_IF_FALSE(!running && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType != static_cast<int>(kValuesInput) && connectionType != static_cast<int>(kPartitionInput))
            throw std::logic_error("RaggedSplit connection type is outside its declared input range.");
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value())
            throw std::logic_error("RaggedSplit input port was connected more than once.");

        streams[inputIndex] = stream;
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        if (inputIndex == kValuesInput) {
            for (size_t i = kInputs; i < streams.size(); ++i) streams[i] = Stream(featureInput->getPlacement().getDeviceNum());
        } else {
            const TensorDescriptor &carrier = featureInput->getDescriptor();
            if (carrier.getDimensions() != std::vector<uint64_t>{1} ||
                (carrier.getDataType() != DataType::UINT32 && carrier.getDataType() != DataType::UINT64))
                throw std::invalid_argument("RaggedSplit requires a UINT32 or UINT64 [1] active-count carrier.");
        }
        if (inputIndex == kValuesInput && backPropagateError && !isInferenceOnly() &&
            !TensorDescriptor::isIntegralType(featureInput->getDescriptor().getDataType()))
            errorOutputs[inputIndex] = featureInput->clone();
        else
            errorOutputs[inputIndex] = std::nullopt;
        ensureNoDeviceCrossing();
        return errorOutputs[inputIndex];
    }

    // One values input and several outputs, so an error input is not paired with an error output by index. When every
    // output is connected and none takes a gradient, the values input's gradient goes too.
    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {
        THOR_THROW_IF_FALSE(oldErrorInput.has_value());
        bool replaced = false;
        for (uint32_t i = 0; i < errorInputs.size(); ++i) {
            if (!errorInputs[i].has_value() || errorInputs[i].value() != oldErrorInput.value()) continue;
            replaced = true;
            allErrorInputTensorIds.erase(errorInputs[i]->getTensorId());
            errorInputs[i] = newErrorInput;
            if (errorInputs[i].has_value()) allErrorInputTensorIds.insert(errorInputs[i]->getTensorId());
        }
        THOR_THROW_IF_FALSE(replaced);
        stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        if (numPresentTensors(featureOutputs) == featureOutputs.size() && numPresentTensors(errorInputs) == 0 &&
            errorOutputs[kValuesInput].has_value()) {
            if (previousLayers[kValuesInput].has_value())
                previousLayers[kValuesInput].value()->replaceErrorInput(errorOutputs[kValuesInput], std::nullopt);
            errorOutputs[kValuesInput].reset();
        }
    }

    // The active packed values only: the input and every piece forward, every gradient and the input's backward.
    uint64_t logicalByteCountForward(uint64_t validExampleCount) override {
        const std::optional<uint64_t> activeRows = logicalActiveRows(validExampleCount);
        if (!activeRows.has_value()) return 0;
        try {
            if (!featureInputs[kValuesInput].has_value()) return 0;
            std::optional<uint64_t> bytes = logicalBytesForActiveRows(featureInputs[kValuesInput].value(), *activeRows);
            if (!bytes.has_value()) return 0;
            for (const std::optional<Tensor> &output : featureOutputs) {
                if (!output.has_value()) return 0;
                const std::optional<uint64_t> outputBytes = logicalBytesForActiveRows(output.value(), *activeRows);
                if (!outputBytes.has_value()) return 0;
                bytes = checkedLogicalByteAdd(*bytes, *outputBytes, "RaggedSplit forward output");
            }
            return *bytes;
        } catch (...) {
            // Logical-work telemetry is best-effort and must not fail submitted model work.
            return 0;
        }
    }

    uint64_t logicalByteCountBackward(uint64_t validExampleCount) override {
        const std::optional<uint64_t> activeRows = logicalActiveRows(validExampleCount);
        if (!activeRows.has_value()) return 0;
        try {
            if (!errorOutputs[kValuesInput].has_value()) return 0;
            std::optional<uint64_t> bytes = logicalBytesForActiveRows(errorOutputs[kValuesInput].value(), *activeRows);
            if (!bytes.has_value()) return 0;
            // A piece without a gradient is read from its zero tensor, of the input's type.
            for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
                if (!featureOutputs[i].has_value()) return 0;
                const TensorDescriptor source =
                    errorInputs[i].has_value()
                        ? errorInputs[i]->getDescriptor()
                        : TensorDescriptor(inputDataType, featureOutputs[i]->getDescriptor().getDimensions());
                const std::optional<uint64_t> sourceBytes = logicalBytesForActiveRows(source, *activeRows);
                if (!sourceBytes.has_value()) return 0;
                bytes = checkedLogicalByteAdd(*bytes, *sourceBytes, "RaggedSplit backward input");
            }
            return *bytes;
        } catch (...) {
            return 0;
        }
    }

   private:
    static constexpr uint32_t kValuesInput = 0;
    static constexpr uint32_t kPartitionInput = 1;
    static constexpr uint32_t kInputs = 2;

    const Stream &outputStream(uint32_t output) const { return output == 0 ? streams[0] : streams[kInputs + output - 1]; }

    // A gradient passes through the conversion only between floating-point types.
    bool takesGradient(uint32_t output) const {
        return !TensorDescriptor::isIntegralType(outputDataTypes[output]) &&
               !TensorDescriptor::isIntegralType(featureInputs[kValuesInput]->getDescriptor().getDataType());
    }

    uint32_t resolveValidExampleCount(uint32_t runtimeBatchSize) const {
        THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(batchSize);
        const uint32_t resolved = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
        THOR_THROW_IF_FALSE(resolved >= 1 && resolved <= physicalBatchCapacity);
        return resolved;
    }

    RowPartitionRuntime partitionFor() const {
        const auto &dimensions = featureInputs[kValuesInput]->getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty() && dimensions[0] > 0);
        return RowPartitionRuntime::fromHostStateCarrier(featureInputs[kPartitionInput].value(), batchSize, dimensions[0]);
    }

    // The packed values of the first validExamples rows, from the host partition published on the carrier.
    uint64_t activeRowsFor(uint32_t validExamples) const {
        const uint64_t activeRows = partitionFor().requireHostOffset(validExamples);
        THOR_THROW_IF_FALSE(activeRows <= featureInputs[kValuesInput]->getDescriptor().getDimensions()[0]);
        return activeRows;
    }

    [[nodiscard]] std::optional<uint64_t> logicalActiveRows(uint64_t validExampleCount) const noexcept {
        try {
            if (!featureInputs[kValuesInput].has_value() || !featureInputs[kPartitionInput].has_value()) return std::nullopt;
            const Tensor &carrier = featureInputs[kPartitionInput].value();
            const uint64_t capacity = featureInputs[kValuesInput]->getDimensions()[0];
            std::optional<uint64_t> active;
            if (validExampleCount == 0) {
                active = RowPartitionRuntime::getPublishedHostActiveValueCountIfAvailable(carrier);
            } else {
                if (validExampleCount > batchSize) return std::nullopt;
                active = RowPartitionRuntime::getPublishedHostOffsetIfAvailable(carrier, validExampleCount);
            }
            if (!active.has_value() || *active > capacity) return std::nullopt;
            return active;
        } catch (...) {
            return std::nullopt;
        }
    }

    [[nodiscard]] static std::optional<uint64_t> logicalBytesForActiveRows(const Tensor &tensor, uint64_t activeRows) noexcept {
        try {
            return logicalBytesForActiveRows(tensor.getDescriptor(), activeRows);
        } catch (...) {
            return std::nullopt;
        }
    }

    [[nodiscard]] static std::optional<uint64_t> logicalBytesForActiveRows(const TensorDescriptor &descriptor,
                                                                           uint64_t activeRows) noexcept {
        try {
            const std::vector<uint64_t> dimensions = descriptor.getDimensions();
            if (dimensions.empty() || dimensions[0] == 0 || activeRows > dimensions[0]) return std::nullopt;
            const uint64_t totalBytes = descriptor.getArraySizeInBytes();
            if (totalBytes % dimensions[0] != 0) return std::nullopt;
            const uint64_t bytesPerRow = totalBytes / dimensions[0];
            if (bytesPerRow != 0 && activeRows > std::numeric_limits<uint64_t>::max() / bytesPerRow) return std::nullopt;
            return activeRows * bytesPerRow;
        } catch (...) {
            return std::nullopt;
        }
    }

    unsigned int axis;
    std::vector<unsigned long> axisElements;
    uint64_t batchSize;
    std::vector<DataType> outputDataTypes;
    DataType inputDataType = DataType::FP32;

    std::vector<void *> outputMemories;
    std::vector<void *> backwardSourceMemories;
    std::vector<DataType> backwardSourceDataTypes;
    std::vector<Tensor> zeroGradients;
    RaggedTrailingConcatenateGeometry geometry;
    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;

    std::vector<Event> forwardInputReadyEvents;
    std::vector<Event> backwardInputReadyEvents;
    Event outputsReadyEvent;
};

}  // namespace ThorImplementation
