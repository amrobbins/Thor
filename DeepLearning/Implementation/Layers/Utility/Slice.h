#pragma once

#include "DeepLearning/Implementation/Layers/DistinctProducerStreamJoin.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/Copy/StridedCopy.h"
#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

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

// The entries [start, start + length) of one axis of a dense tensor (axis >= 1: the batch is axis 0), or of a trailing
// value axis of canonical rank-1 RaggedTensor packed values (axis >= 1: the packed values are axis 0). The output is
// dense, of outputDataType.
//
// Dense: one input, the batch's valid examples are moved. Ragged (raggedBatchSize > 0): input 0 is the packed values,
// input 1 the managed [1] active-count carrier, whose published host partition gives the active packed prefix (CUDA
// does not read it); the partition is published on the output.
//
// Forward: of the input's type, the dense copy (StridedCopy.h, planned for each count of active outer slices); of
// another type, one typed split that writes only the slice. Backward: one typed concatenate into the input's gradient,
// the slice's gradient converted to the input's type and the entries outside the slice written as zeros (null
// sources: nothing is read for them). A gradient passes only between floating-point types.
class Slice : public MultiConnectionLayer {
   public:
    Slice(unsigned int axis, uint64_t start, uint64_t length, DataType outputDataType, uint64_t raggedBatchSize = 0)
        : axis(axis), start(start), length(length), outputDataType(outputDataType), raggedBatchSize(raggedBatchSize),
          inputs(raggedBatchSize == 0 ? 1 : 2) {
        if (axis == 0) throw std::invalid_argument("Slice axis must not be the batch or packed values axis.");
        if (length == 0) throw std::invalid_argument("Slice length must be positive.");
        previousLayers.resize(inputs);
        featureInputs.resize(inputs);
        errorOutputs.resize(inputs);
        streams.resize(inputs);
        forwardInputReadyEvents.resize(inputs);
        featureOutputs.assign(1, std::nullopt);
        nextLayers.assign(1, std::nullopt);
        errorInputs.assign(1, std::nullopt);
    }

    ~Slice() override = default;

    std::string getType() override { return "Slice"; }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInputs[kValuesInput].has_value());
        const Tensor &input = featureInputs[kValuesInput].value();
        std::vector<unsigned long> dimensions = input.getDescriptor().getDimensions();
        if (dimensions.size() < 2 || axis >= dimensions.size())
            throw std::logic_error("Slice axis " + std::to_string(axis) + " is out of range for an input of rank " +
                                   std::to_string(dimensions.size()) + ".");
        if (start > dimensions[axis] || length > dimensions[axis] - start)
            throw std::logic_error("Slice [start, start + length) leaves its axis.");
        dimensions[axis] = length;
        return Tensor(input.getPlacement(), TensorDescriptor(outputDataType, dimensions));
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        for (uint32_t i = 0; i < inputs; ++i) THOR_THROW_IF_FALSE(featureInputs[i].has_value());
        THOR_THROW_IF_FALSE(featureOutputs[0].has_value() && nextLayers[0].has_value());
        const Tensor &input = featureInputs[kValuesInput].value();
        THOR_THROW_IF_FALSE(input.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        inputDataType = input.getDescriptor().getDataType();

        // No gradient from the output: nothing to send back, so the path upstream is pruned.
        if (!errorInputs[0].has_value() && errorOutputs[kValuesInput].has_value()) {
            if (previousLayers[kValuesInput].has_value())
                previousLayers[kValuesInput].value()->replaceErrorInput(errorOutputs[kValuesInput], std::nullopt);
            errorOutputs[kValuesInput].reset();
        }

        const std::vector<uint64_t> dimensions = input.getDescriptor().getDimensions();
        extent = dimensions[axis];
        innerElements = 1;
        for (size_t d = axis + 1; d < dimensions.size(); ++d)
            innerElements = checkedMultiply(innerElements, dimensions[d], "Slice inner geometry overflow.");
        outerSlicesPerRow = 1;
        for (size_t d = 1; d < axis; ++d)
            outerSlicesPerRow = checkedMultiply(outerSlicesPerRow, dimensions[d], "Slice outer geometry overflow.");

        // The pieces of an outer slice, in elements: before the slice, the slice, after it (empty ones left out).
        pieceSpanElements.clear();
        slicePiece = 0;
        if (start > 0) pieceSpanElements.push_back(checkedMultiply(start, innerElements, "Slice span overflow."));
        slicePiece = static_cast<uint32_t>(pieceSpanElements.size());
        pieceSpanElements.push_back(checkedMultiply(length, innerElements, "Slice span overflow."));
        if (start + length < extent)
            pieceSpanElements.push_back(checkedMultiply(extent - start - length, innerElements, "Slice span overflow."));

        forwardDestinations.assign(pieceSpanElements.size(), nullptr);
        forwardDestinations[slicePiece] = featureOutputs[0]->getMemPtr();
        forwardDataTypes.assign(pieceSpanElements.size(), inputDataType);
        forwardDataTypes[slicePiece] = outputDataType;
        backwardSources.clear();
        backwardDataTypes.clear();
        if (errorOutputs[kValuesInput].has_value()) {
            backwardSources.assign(pieceSpanElements.size(), nullptr);
            backwardSources[slicePiece] = errorInputs[0]->getMemPtr();
            backwardDataTypes.assign(pieceSpanElements.size(), inputDataType);
            backwardDataTypes[slicePiece] = errorInputs[0]->getDescriptor().getDataType();
        }
        copyPlan.reset();

        allFeatureInputTensorIds.clear();
        for (uint32_t i = 0; i < inputs; ++i) allFeatureInputTensorIds.insert(featureInputs[i]->getTensorId());
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        pendingValidExamples = 0;
    }

    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream, unsigned int) override {}

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        auto it = stillWaitingForFeatureInputTensors.find(featureInput->getTensorId());
        THOR_THROW_IF_FALSE(it != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(it);
        // Both inputs of a ragged slice carry the same batch.
        const uint32_t resolved = resolveValidExampleCount(batchSize);
        if (pendingValidExamples != 0) THOR_THROW_IF_FALSE(pendingValidExamples == resolved);
        pendingValidExamples = resolved;
        if (!stillWaitingForFeatureInputTensors.empty()) return;
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
        const uint32_t validExamples = pendingValidExamples;
        pendingValidExamples = 0;
        if (inputs > 1) {
            detail::waitForDistinctProducerStreams(
                streams[0], inputs, [](std::size_t i) { return i >= 1; },
                [this](std::size_t i) -> const Stream & { return streams[i]; },
                [this](std::size_t i) -> Event & { return forwardInputReadyEvents[i]; });
        }

        const uint64_t outerSlices = activeOuterSlices(validExamples);
        if (outerSlices != 0) {
            if (outputDataType == inputDataType) {
                const StridedSpan source(featureInputs[kValuesInput].value(), start * innerElements,
                                         {outerSlices, length * innerElements}, {extent * innerElements, 1});
                const StridedSpan destination(featureOutputs[0].value(), 0, {outerSlices, length * innerElements},
                                              {length * innerElements, 1});
                if (!copyPlan.has_value() || copyPlanOuterSlices != outerSlices) {
                    copyPlan = planCopyToDense(source, destination);
                    copyPlanOuterSlices = outerSlices;
                }
                launchCopyToDense(copyPlan.value(), source, destination, streams[0]);
            } else {
                launchSplit(forwardDestinations, forwardDataTypes, featureInputs[kValuesInput]->getMemPtr(), inputDataType,
                            outerSlices, pieceSpanElements, streams[0]);
            }
        }

        if (inputs > 1) partitionFor().publishHostStateTo(featureOutputs[0].value());
        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[0].value()->forward(featureOutputs[0], validationPass, validExamples);
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        if (!errorInput.has_value()) return;
        THOR_THROW_IF_FALSE(errorOutputs[kValuesInput].has_value());
        const uint32_t validExamples = resolveValidExampleCount(batchSize);
        const uint64_t outerSlices = activeOuterSlices(validExamples);
        if (outerSlices != 0) {
            launchConcatenate(errorOutputs[kValuesInput]->getMemPtr(), inputDataType, backwardSources, backwardDataTypes,
                              outerSlices, pieceSpanElements, streams[0]);
        }
        if (previousLayers[kValuesInput].has_value())
            previousLayers[kValuesInput].value()->backward(errorOutputs[kValuesInput], validExamples);
    }

    void cleanup() override {
        forwardDestinations.clear();
        forwardDataTypes.clear();
        backwardSources.clear();
        backwardDataTypes.clear();
        pieceSpanElements.clear();
        copyPlan.reset();
        for (Event &event : forwardInputReadyEvents) event = Event();
        MultiConnectionLayer::cleanup();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!running);
        if (driverConnectionType != 0) throw std::invalid_argument("Slice has one output.");
        if (nextLayers[0].has_value()) throw std::logic_error("Slice output is already connected.");
        for (uint32_t i = 0; i < inputs; ++i)
            if (!featureInputs[i].has_value()) throw std::logic_error("Slice output connects after its inputs.");
        featureOutputs[0] = createFeatureOutputTensor();
        nextLayers[0] = nextLayer;
        errorInputs[0] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[0], streams[0], shouldConnectToBackPropErrorIn() && !isBackPropStub() && takesGradient(),
            loaderConnectionType);
        if (errorInputs[0].has_value()) {
            THOR_THROW_IF_FALSE(errorInputs[0]->getDescriptor() == featureOutputs[0]->getDescriptor());
        }
        // No gradient comes back: the path upstream is pruned.
        if (!errorInputs[0].has_value() && errorOutputs[kValuesInput].has_value()) {
            if (previousLayers[kValuesInput].has_value())
                previousLayers[kValuesInput].value()->replaceErrorInput(errorOutputs[kValuesInput], std::nullopt);
            errorOutputs[kValuesInput].reset();
        }
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) override {
        THOR_THROW_IF_FALSE(!running && featureInput.has_value() && previousLayer != nullptr);
        if (connectionType < 0 || static_cast<uint32_t>(connectionType) >= inputs)
            throw std::logic_error("Slice connection type is outside its declared input range.");
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value())
            throw std::logic_error("Slice input port was connected more than once.");

        streams[inputIndex] = stream;
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        if (inputIndex == kPartitionInput) {
            const TensorDescriptor &carrier = featureInput->getDescriptor();
            if (carrier.getDimensions() != std::vector<uint64_t>{1} ||
                (carrier.getDataType() != DataType::UINT32 && carrier.getDataType() != DataType::UINT64))
                throw std::invalid_argument("Slice requires a UINT32 or UINT64 [1] active-count carrier.");
        }
        if (inputIndex == kValuesInput && backPropagateError && !isInferenceOnly() &&
            !TensorDescriptor::isIntegralType(featureInput->getDescriptor().getDataType()) && takesGradient())
            errorOutputs[inputIndex] = featureInput->clone();
        else
            errorOutputs[inputIndex] = std::nullopt;
        ensureNoDeviceCrossing();
        return errorOutputs[inputIndex];
    }

    // The valid examples (dense) or the active packed values (ragged) only: the slice read and written forward, its
    // gradient read and the input's gradient written backward.
    uint64_t logicalByteCountForward(uint64_t validExampleCount) override {
        try {
            const std::optional<uint64_t> outer = logicalOuterSlices(validExampleCount);
            if (!outer.has_value() || !featureInputs[kValuesInput].has_value()) return 0;
            const uint64_t sliceElements = checkedMultiply(*outer, length * innerElementsOf(), "Slice logical bytes");
            return checkedLogicalByteAdd(checkedMultiply(sliceElements, elementBytes(inputDataTypeOf()), "Slice logical bytes"),
                                         checkedMultiply(sliceElements, elementBytes(outputDataType), "Slice logical bytes"),
                                         "Slice forward");
        } catch (...) {
            // Logical-work telemetry is best-effort and must not fail submitted model work.
            return 0;
        }
    }

    uint64_t logicalByteCountBackward(uint64_t validExampleCount) override {
        try {
            const std::optional<uint64_t> outer = logicalOuterSlices(validExampleCount);
            if (!outer.has_value() || !errorOutputs[kValuesInput].has_value() || !errorInputs[0].has_value()) return 0;
            const uint64_t inner = innerElementsOf();
            const uint64_t written = checkedMultiply(checkedMultiply(*outer, extentOf() * inner, "Slice logical bytes"),
                                                     elementBytes(inputDataTypeOf()), "Slice logical bytes");
            const uint64_t read =
                checkedMultiply(checkedMultiply(*outer, length * inner, "Slice logical bytes"),
                                elementBytes(errorInputs[0]->getDescriptor().getDataType()), "Slice logical bytes");
            return checkedLogicalByteAdd(written, read, "Slice backward");
        } catch (...) {
            return 0;
        }
    }

   private:
    static constexpr uint32_t kValuesInput = 0;
    static constexpr uint32_t kPartitionInput = 1;

    // A gradient passes through the conversion only between floating-point types.
    bool takesGradient() const {
        if (TensorDescriptor::isIntegralType(outputDataType)) return false;
        return !featureInputs[kValuesInput].has_value() ||
               !TensorDescriptor::isIntegralType(featureInputs[kValuesInput]->getDescriptor().getDataType());
    }

    uint32_t resolveValidExampleCount(uint32_t batchSize) const {
        const uint64_t capacity = inputs > 1 ? raggedBatchSize : featureInputs[kValuesInput]->getDescriptor().getDimensions()[0];
        THOR_THROW_IF_FALSE(capacity >= 1 && capacity <= std::numeric_limits<uint32_t>::max());
        const uint32_t resolved = batchSize == 0 ? static_cast<uint32_t>(capacity) : batchSize;
        THOR_THROW_IF_FALSE(resolved >= 1 && resolved <= capacity);
        return resolved;
    }

    RowPartitionRuntime partitionFor() const {
        const auto &dimensions = featureInputs[kValuesInput]->getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty() && dimensions[0] > 0);
        return RowPartitionRuntime::fromHostStateCarrier(featureInputs[kPartitionInput].value(), raggedBatchSize, dimensions[0]);
    }

    // The outer slices of the valid examples (dense) or of the packed values of the first validExamples rows, from
    // the host partition published on the carrier (ragged).
    uint64_t activeOuterSlices(uint32_t validExamples) const {
        uint64_t rows = validExamples;
        if (inputs > 1) {
            rows = partitionFor().requireHostOffset(validExamples);
            THOR_THROW_IF_FALSE(rows <= featureInputs[kValuesInput]->getDescriptor().getDimensions()[0]);
        }
        return checkedMultiply(rows, outerSlicesPerRow, "Slice active outer-slice count overflow.");
    }

    [[nodiscard]] std::optional<uint64_t> logicalOuterSlices(uint64_t validExampleCount) const noexcept {
        try {
            if (!featureInputs[kValuesInput].has_value()) return std::nullopt;
            const std::vector<uint64_t> dimensions = featureInputs[kValuesInput]->getDescriptor().getDimensions();
            if (dimensions.size() <= axis || dimensions[0] == 0) return std::nullopt;
            uint64_t perRow = 1;
            for (size_t d = 1; d < axis; ++d) perRow = checkedMultiply(perRow, dimensions[d], "Slice logical bytes");
            std::optional<uint64_t> rows;
            if (inputs == 1) {
                rows = validExampleCount == 0 ? dimensions[0] : validExampleCount;
            } else {
                if (!featureInputs[kPartitionInput].has_value()) return std::nullopt;
                const Tensor &carrier = featureInputs[kPartitionInput].value();
                if (validExampleCount == 0) {
                    rows = RowPartitionRuntime::getPublishedHostActiveValueCountIfAvailable(carrier);
                } else {
                    if (validExampleCount > raggedBatchSize) return std::nullopt;
                    rows = RowPartitionRuntime::getPublishedHostOffsetIfAvailable(carrier, validExampleCount);
                }
            }
            if (!rows.has_value() || *rows > dimensions[0]) return std::nullopt;
            return checkedMultiply(*rows, perRow, "Slice logical bytes");
        } catch (...) {
            return std::nullopt;
        }
    }

    // Geometry read from the input, for the byte counts before compile.
    uint64_t innerElementsOf() const {
        const std::vector<uint64_t> dimensions = featureInputs[kValuesInput]->getDescriptor().getDimensions();
        uint64_t inner = 1;
        for (size_t d = axis + 1; d < dimensions.size(); ++d) inner = checkedMultiply(inner, dimensions[d], "Slice logical bytes");
        return inner;
    }
    uint64_t extentOf() const { return featureInputs[kValuesInput]->getDescriptor().getDimensions().at(axis); }
    DataType inputDataTypeOf() const { return featureInputs[kValuesInput]->getDescriptor().getDataType(); }
    static uint64_t elementBytes(DataType dataType) { return TensorDescriptor::getElementSizeInBytes(dataType); }

    static uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char *what) {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) throw std::overflow_error(what);
        return lhs * rhs;
    }

    unsigned int axis;
    uint64_t start;
    uint64_t length;
    DataType outputDataType;
    uint64_t raggedBatchSize;
    uint32_t inputs;
    DataType inputDataType = DataType::FP32;

    uint64_t extent = 0;
    uint64_t innerElements = 1;
    uint64_t outerSlicesPerRow = 1;
    std::vector<uint64_t> pieceSpanElements;
    uint32_t slicePiece = 0;
    std::vector<void *> forwardDestinations;
    std::vector<DataType> forwardDataTypes;
    std::vector<void *> backwardSources;
    std::vector<DataType> backwardDataTypes;
    // The dense copy's plan for the last count of active outer slices.
    std::optional<DenseCopyPlan> copyPlan;
    uint64_t copyPlanOuterSlices = 0;

    std::set<uint64_t> allFeatureInputTensorIds;
    std::set<uint64_t> stillWaitingForFeatureInputTensors;
    uint32_t pendingValidExamples = 0;
    std::vector<Event> forwardInputReadyEvents;
};

}  // namespace ThorImplementation
