#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <stdexcept>
#include <vector>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/DistinctProducerStreamJoin.h"
#include "DeepLearning/Implementation/Layers/DistinctTargetStreamFanout.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

namespace ThorImplementation {

/**
 * All tensors that are being split will have the same number of dimensions.
 * All dimensions other than the split axis will be of equal size among all tensors.
 *
 * Example 1:
 * axis = 0
 * The input tensor has dimensions [52][16]
 * axisElements = {32, 20}
 * Output tensor0 has dimensions [32][16]
 * Output tensor1 has dimensions [20][16]
 * Output tensor0's entries are from [0..31] [0..15]
 * Output tensor1's entries are from [32..51][0..15]
 *
 * Example 2:
 * axis = 1
 * The input tensor has dimensions [32][26]
 * axisElements = {16, 10}
 * Tensor0 has dimensions [32][16]
 * Tensor1 has dimensions [32][10]
 * Output tensor0's entries are from [0..31] [0..15]
 * Output tensor1's entries are from [0..31][16..25]
 *
 * Example 3:
 * axis = 1
 * axisElements = {8, 16, 8}
 * The input tensor has dimensions [64][32][128]
 * Tensor0 has dimensions [64] [8][128]
 * Tensor1 has dimensions [64][16][128]
 * Tensor2 has dimensions [64] [8][128]
 * Output tensor0's entries are from [0..63]  [0..7][0..127]
 * Output tensor1's entries are from [0..63] [8..23][0..127]
 * Output tensor2's entries are from [0..63][24..31][0..127]
 *
 * Output i has outputDataTypes[i], converted from the input's type as it moves (one launch writes every output). Output
 * i is the layer's connection (driver connection type) i, whatever order the outputs are connected in. Backward, one
 * concatenate converts each output's gradient to the input's type. An output that takes no gradient (its next layer
 * asks for none, or its type or the input's is integral) contributes zeros: its source is null, so the concatenate
 * writes zero bits into its span of the input's gradient and reads nothing for it.
 */
class Split : public MultiConnectionLayer {
   public:
    ~Split() override {}

    Split(unsigned int axis, std::vector<unsigned long> axisElements, std::vector<DataType> outputDataTypes)
        : axis(axis), axisElements(std::move(axisElements)), outputDataTypes(std::move(outputDataTypes)) {
        if (this->axisElements.empty()) throw std::invalid_argument("Split requires at least one output.");
        if (this->outputDataTypes.size() != this->axisElements.size())
            throw std::invalid_argument("Split takes one output data type per output.");
        for (const unsigned long elements : this->axisElements)
            if (elements == 0) throw std::invalid_argument("Split outputs must have a positive extent on the axis.");
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        throw std::logic_error("Split creates output i when connection type i is connected.");
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        const uint32_t numSplitTensors = static_cast<uint32_t>(axisElements.size());
        unsigned long totalAxisElements = 0;
        for (unsigned int i = 0; i < numSplitTensors; ++i)
            totalAxisElements += axisElements[i];
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        const std::vector<unsigned long> inputDimensions = featureInputs[0].value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() > axis);
        THOR_THROW_IF_FALSE(totalAxisElements == inputDimensions[axis]);
        THOR_THROW_IF_FALSE(featureOutputs.size() == numSplitTensors);
        THOR_THROW_IF_FALSE(nextLayers.size() == numSplitTensors);
        THOR_THROW_IF_FALSE(streams.size() == numSplitTensors);
        THOR_THROW_IF_FALSE(errorInputs.size() == numSplitTensors);
        for (unsigned int i = 0; i < numSplitTensors; ++i) {
            if (!featureOutputs[i].has_value() || !nextLayers[i].has_value())
                throw std::logic_error("Split output " + std::to_string(i) + " is not connected.");
            THOR_THROW_IF_FALSE(featureOutputs[i].value().getDescriptor().getDimensions()[axis] == axisElements[i]);
        }
        backwardInputReadyEvents.clear();
        backwardInputReadyEvents.resize(streams.size());
        outputsReadyEvent = Event();

        THOR_THROW_IF_FALSE(featureInputs[0].value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        inputDataType = featureInputs[0].value().getDescriptor().getDataType();

        // No output gradient at all: nothing to send back, so the path upstream is pruned.
        if (numPresentTensors(errorInputs) == 0 && errorOutputs[0].has_value()) {
            THOR_THROW_IF_FALSE(previousLayers[0].has_value());
            previousLayers[0].value()->replaceErrorInput(errorOutputs[0], std::nullopt);
            errorOutputs[0].reset();
        }

        // The tensors cannot change after compilation, so their pointers are fixed for the runnable layer and go to
        // every launch as its parameters.
        splitTensorFeatureOutputMemories.assign(numSplitTensors, nullptr);
        for (uint32_t i = 0; i < numSplitTensors; ++i)
            splitTensorFeatureOutputMemories[i] = featureOutputs[i].value().getMemPtr();

        backwardSourceMemories.clear();
        backwardSourceDataTypes.clear();
        if (errorOutputs[0].has_value()) {
            backwardSourceMemories.assign(numSplitTensors, nullptr);
            backwardSourceDataTypes.assign(numSplitTensors, inputDataType);
            for (uint32_t i = 0; i < numSplitTensors; ++i) {
                if (!errorInputs[i].has_value())
                    continue;
                backwardSourceMemories[i] = errorInputs[i].value().getMemPtr();
                backwardSourceDataTypes[i] = errorInputs[i].value().getDescriptor().getDataType();
            }
        }

        uint64_t innerElements = 1;
        for (uint32_t d = axis + 1; d < inputDimensions.size(); ++d)
            innerElements = checkedMultiply(innerElements, inputDimensions[d], "Split inner geometry overflow.");

        // A span is one output's elements of one outer slice: its extent on the axis times the elements after it.
        spanElementsPerSplitTensor.assign(numSplitTensors, 0);
        for (uint32_t i = 0; i < numSplitTensors; ++i)
            spanElementsPerSplitTensor[i] = checkedMultiply(axisElements[i], innerElements, "Split span geometry overflow.");
        THOR_THROW_IF_FALSE(totalAxisElements > 0 && innerElements > 0);

        outerSlicesPerBatch = 1;
        for (uint32_t d = 1; d < axis; ++d)
            outerSlicesPerBatch = checkedMultiply(outerSlicesPerBatch, inputDimensions[d], "Split outer geometry overflow.");
    }

    void infer(std::optional<Tensor> inputTensor,
               std::optional<Tensor> outputTensor,
               Stream stream,
               unsigned int connectionNumber) override {}

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream stream,
                  unsigned int connectionNumber) override {}

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(streams.size() == featureOutputs.size());
        THOR_THROW_IF_FALSE(nextLayers.size() == featureOutputs.size());

        const uint64_t activeOuterSlices = resolveActiveOuterSlices(featureInput.value().getDescriptor(), batchSize);
        launchSplit(splitTensorFeatureOutputMemories,
                    outputDataTypes,
                    featureInput.value().getMemPtr(),
                    inputDataType,
                    activeOuterSlices,
                    spanElementsPerSplitTensor,
                    streams[0]);

        ThorImplementation::detail::recordCompletionAndWaitOnDistinctTargetStreams(
            streams[0],
            outputsReadyEvent,
            streams,
            /*firstLogicalTarget=*/0);
        for (unsigned int i = 0; i < featureOutputs.size(); ++i)
            nextLayers[i].value()->forward(featureOutputs[i], validationPass, batchSize);
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        // Experimental - back propagation stops at empty error input
        if (!errorInput.has_value())
            return;
        THOR_THROW_IF_FALSE(errorOutputs[0].has_value());
        THOR_THROW_IF_FALSE(streams.size() == errorInputs.size());

        // Every output that takes a gradient delivers it here; the last one to arrive runs the concatenate.
        THOR_THROW_IF_FALSE(stillWaitingForErrorInputTensors.count(errorInput.value().getTensorId()) == 1);
        stillWaitingForErrorInputTensors.erase(errorInput.value().getTensorId());
        if (!stillWaitingForErrorInputTensors.empty())
            return;
        stillWaitingForErrorInputTensors = allErrorInputTensorIds;

        detail::waitForDistinctProducerStreams(streams[0], streams, backwardInputReadyEvents, 1);

        const uint64_t activeOuterSlices = resolveActiveOuterSlices(errorOutputs[0].value().getDescriptor(), batchSize);
        launchConcatenate(
            errorOutputs[0].value().getMemPtr(),
            inputDataType,
            backwardSourceMemories,
            backwardSourceDataTypes,
            activeOuterSlices,
            spanElementsPerSplitTensor,
            streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[0].value()->backward(errorOutputs[0], batchSize);
    }

    void cleanup() override {
        TensorPlacement placement = featureInputs[0].value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        splitTensorFeatureOutputMemories.clear();
        backwardSourceMemories.clear();
        backwardSourceDataTypes.clear();
        spanElementsPerSplitTensor.clear();
        outerSlicesPerBatch = 1;
        outputsReadyEvent = Event();
        backwardInputReadyEvents.clear();
    }

    // Connection type driverConnectionType is output driverConnectionType.
    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!running);
        THOR_THROW_IF_FALSE(featureInputs.size() == 1 && featureInputs[0].has_value());
        if (driverConnectionType < 0 || static_cast<size_t>(driverConnectionType) >= axisElements.size())
            throw std::invalid_argument("Split has no output " + std::to_string(driverConnectionType) + ".");
        const uint32_t output = static_cast<uint32_t>(driverConnectionType);
        if (featureOutputs[output].has_value())
            throw std::logic_error("Split output " + std::to_string(output) + " is already connected.");

        const Tensor &input = featureInputs[0].value();
        THOR_THROW_IF_FALSE(input.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        std::vector<unsigned long> dimensions = input.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(axis < dimensions.size());
        dimensions[axis] = axisElements[output];
        featureOutputs[output] = Tensor(input.getPlacement(), TensorDescriptor(outputDataTypes[output], dimensions));
        nextLayers[output] = nextLayer;
        errorInputs[output] = nextLayer->connectToPreviousLayer(
            this, featureOutputs[output], streams[output],
            shouldConnectToBackPropErrorIn() && !isBackPropStub() && takesGradient(output), loaderConnectionType);

        if (errorInputs[output].has_value()) {
            THOR_THROW_IF_FALSE(errorInputs[output].value().getDescriptor() == featureOutputs[output].value().getDescriptor());
            THOR_THROW_IF_FALSE(errorInputs[output].value().getPlacement() == featureOutputs[output].value().getPlacement());
        }
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        THOR_THROW_IF_FALSE(featureInputs.empty());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        const uint32_t numSplitTensors = static_cast<uint32_t>(axisElements.size());

        // Output 0 runs on the input's stream, every other output on its own.
        streams.clear();
        streams.push_back(stream);
        for (uint32_t i = 1; i < numSplitTensors; ++i) streams.emplace_back(featureInput.value().getPlacement().getDeviceNum());
        featureOutputs.assign(numSplitTensors, std::nullopt);
        nextLayers.assign(numSplitTensors, std::nullopt);
        errorInputs.assign(numSplitTensors, std::nullopt);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        if (backPropagateError && !isInferenceOnly() &&
            !TensorDescriptor::isIntegralType(featureInput.value().getDescriptor().getDataType()))
            errorOutputs.emplace_back(featureInput.value().clone());
        else
            errorOutputs.emplace_back(std::nullopt);

        if (errorOutputs.back().has_value()) {
            THOR_THROW_IF_FALSE(featureInputs.back().value().getDescriptor() == errorOutputs.back().value().getDescriptor());
            THOR_THROW_IF_FALSE(featureInputs.back().value().getPlacement() == errorOutputs.back().value().getPlacement());
        }
        ensureNoDeviceCrossing();

        return errorOutputs.back();
    }

    // One input and several outputs, so an error input is not paired with an error output by index. When every output
    // is connected and none takes a gradient, the input's gradient goes too (compile does the same for outputs that
    // never took one).
    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {
        THOR_THROW_IF_FALSE(oldErrorInput.has_value());
        bool replaced = false;
        for (uint32_t i = 0; i < errorInputs.size(); ++i) {
            if (!errorInputs[i].has_value() || errorInputs[i].value() != oldErrorInput.value()) continue;
            replaced = true;
            allErrorInputTensorIds.erase(errorInputs[i].value().getTensorId());
            errorInputs[i] = newErrorInput;
            if (errorInputs[i].has_value()) allErrorInputTensorIds.insert(errorInputs[i].value().getTensorId());
        }
        THOR_THROW_IF_FALSE(replaced);
        stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        if (numPresentTensors(featureOutputs) == featureOutputs.size() && numPresentTensors(errorInputs) == 0 &&
            !errorOutputs.empty() && errorOutputs[0].has_value()) {
            if (previousLayers[0].has_value()) previousLayers[0].value()->replaceErrorInput(errorOutputs[0], std::nullopt);
            errorOutputs[0].reset();
        }
    }

   private:
    static uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char *what) {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
            throw std::invalid_argument(what);
        return lhs * rhs;
    }

    // A gradient passes through the conversion only between floating-point types.
    bool takesGradient(uint32_t output) const {
        return !TensorDescriptor::isIntegralType(outputDataTypes[output]) &&
               !TensorDescriptor::isIntegralType(featureInputs[0].value().getDescriptor().getDataType());
    }

    uint64_t resolveActiveOuterSlices(const TensorDescriptor &descriptor, uint32_t batchSize) const {
        if (axis == 0) return 1;
        const std::vector<uint64_t> &dimensions = descriptor.getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty());
        THOR_THROW_IF_FALSE(dimensions.front() >= 1);
        THOR_THROW_IF_FALSE(dimensions.front() <= std::numeric_limits<uint32_t>::max());
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(dimensions.front());
        const uint32_t validExamples = batchSize == 0 ? physicalBatchCapacity : batchSize;
        THOR_THROW_IF_FALSE(validExamples >= 1);
        THOR_THROW_IF_FALSE(validExamples <= physicalBatchCapacity);
        return checkedMultiply(validExamples, outerSlicesPerBatch,
                               "Split active outer-slice count overflow.");
    }

    unsigned int axis;
    std::vector<unsigned long> axisElements;
    std::vector<DataType> outputDataTypes;
    DataType inputDataType = DataType::FP32;

    std::vector<void *> splitTensorFeatureOutputMemories;
    std::vector<void *> backwardSourceMemories;
    std::vector<DataType> backwardSourceDataTypes;
    std::vector<uint64_t> spanElementsPerSplitTensor;
    uint64_t outerSlicesPerBatch = 1;

    Event outputsReadyEvent;
    std::vector<Event> backwardInputReadyEvents;
};

}  // namespace ThorImplementation
