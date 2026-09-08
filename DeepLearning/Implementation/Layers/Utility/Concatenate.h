#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

namespace ThorImplementation {

/**
 * All tensors that are being concatenated need to have the same number of dimensions.
 * All dimensions other than the concatenation axis need to be of equal size among all tensors.
 *
 * Example 1:
 * axis = 0
 * Tensor0 has dimensions [32][16]
 * Tensor1 has dimensions [20][16]
 * The concatenated tensor has dimensions [52][16]
 * Tensor0's entries are in [0..31] [0..15]
 * Tensor1's entries are in [32..51][0..15]
 *
 * Example 2:
 * axis = 1
 * Tensor0 has dimensions [32][16]
 * Tensor1 has dimensions [32][10]
 * The concatenated tensor has dimensions [32][26]
 * Tensor0's entries are in [0..31] [0..15]
 * Tensor1's entries are in [0..31][16..25]
 *
 * Example 3:
 * axis = 1
 * Tensor0 has dimensions [64] [8][128]
 * Tensor1 has dimensions [64][16][128]
 * Tensor2 has dimensions [64] [8][128]
 * The concatenated tensor has dimensions [64][32][128]
 * Tensor0's entries are in [0..63]  [0..7][0..127]
 * Tensor1's entries are in [0..63] [8..23][0..127]
 * Tensor2's entries are in [0..63][24..31][0..127]
 */
class Concatenate : public MultiConnectionLayer {
   public:
    ~Concatenate() override {}

    Concatenate(unsigned int axis, uint32_t expectedNumInputs) {
        this->axis = (int)axis;
        this->expectedNumInputs = expectedNumInputs;
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        splitTensorErrorOutputMemoriesArray_d = nullptr;
        spanGeometryPerSplitTensor_d = nullptr;

        if (expectedNumInputs < 2)
            throw std::invalid_argument("Concatenate requires at least two declared inputs.");
        previousLayers.resize(expectedNumInputs);
        featureInputs.resize(expectedNumInputs);
        streams.resize(expectedNumInputs);
        errorOutputs.resize(expectedNumInputs);
        forwardInputReadyEvents.resize(expectedNumInputs);
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (featureInputs.size() <= 1) {
            THOR_THROW_LOGIC_ERROR("Concatenate requires at least two feature inputs, but received " +
                                   std::to_string(featureInputs.size()) + layerContext() + ".");
        }
        if (!featureInputs.front().has_value()) {
            THOR_THROW_LOGIC_ERROR("Concatenate input[0] is missing" + layerContext() +
                                   ". Every Concatenate input connection must provide a tensor.");
        }

        const TensorDescriptor &referenceDescriptor = featureInputs.front().value().getDescriptor();
        const std::vector<uint64_t> &referenceDimensions = referenceDescriptor.getDimensions();
        if (axis >= referenceDimensions.size()) {
            THOR_THROW_LOGIC_ERROR(
                "Concatenate physical axis " + std::to_string(axis) + " is out of range for input rank " +
                std::to_string(referenceDimensions.size()) + layerContext() + ". input_shapes=" + inputShapesToString() +
                ". Implementation-layer axes include the batch dimension; an API concatenation axis is stamped as axis + 1.");
        }

        const unsigned int numDimensions = referenceDimensions.size();
        uint64_t newAxisSize = referenceDimensions[axis];
        for (unsigned int i = 1; i < featureInputs.size(); ++i) {
            if (!featureInputs[i].has_value()) {
                THOR_THROW_LOGIC_ERROR("Concatenate input[" + std::to_string(i) + "] is missing" + layerContext() +
                                       ". Every Concatenate input connection must provide a tensor. input_shapes=" + inputShapesToString() +
                                       ".");
            }

            const TensorDescriptor &descriptor = featureInputs[i].value().getDescriptor();
            if (descriptor.getDataType() != referenceDescriptor.getDataType()) {
                THOR_THROW_LOGIC_ERROR("Concatenate data type mismatch between input[0] and input[" + std::to_string(i) + "]" +
                                       layerContext() + ". expected_data_type=" + referenceDescriptor.getElementTypeName() +
                                       ", actual_data_type=" + descriptor.getElementTypeName() + ", input_shapes=" + inputShapesToString() +
                                       ". Convert inputs to the same storage data type before concatenating them.");
            }

            const std::vector<uint64_t> &dimensions = descriptor.getDimensions();
            if (dimensions.size() != numDimensions) {
                THOR_THROW_LOGIC_ERROR("Concatenate rank mismatch at input[" + std::to_string(i) + "]" + layerContext() +
                                       ". physical_concatenation_axis=" + std::to_string(axis) +
                                       ", expected_rank_from_input_0=" + std::to_string(numDimensions) +
                                       ", actual_rank=" + std::to_string(dimensions.size()) + ", input_shapes=" + inputShapesToString() +
                                       ". All Concatenate inputs must have the same rank. "
                                       "After rank validation, every non-concatenation dimension must match.");
            }

            for (unsigned int j = 0; j < numDimensions; ++j) {
                if (j == axis)
                    continue;
                if (dimensions[j] != referenceDimensions[j]) {
                    THOR_THROW_LOGIC_ERROR(
                        "Concatenate input shape mismatch at input[" + std::to_string(i) + "], mismatched physical dimension " +
                        std::to_string(j) + layerContext() + ". physical_concatenation_axis=" + std::to_string(axis) +
                        ", expected_dimension=" + std::to_string(referenceDimensions[j]) +
                        ", actual_dimension=" + std::to_string(dimensions[j]) + ", input_shapes=" + inputShapesToString() +
                        ". All inputs must have identical dimensions except on physical axis " + std::to_string(axis) +
                        ". Check sequence/window lengths, preserved prefix dimensions, and the selected API concatenation axis "
                        "(the implementation axis includes the batch dimension).");
                }
            }
            newAxisSize += dimensions[axis];
        }

        std::vector<uint64_t> outputDimensions = referenceDimensions;
        outputDimensions[axis] = newAxisSize;
        TensorDescriptor outputDescriptor = TensorDescriptor(referenceDescriptor.getDataType(), outputDimensions);

        return Tensor(featureInputs[0].value().getPlacement(), outputDescriptor);
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        THOR_THROW_IF_FALSE(featureOutputs[0].has_value());
        THOR_THROW_IF_FALSE(nextLayers.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0].value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(!streams.empty());
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        const uint32_t numSplitTensors = static_cast<uint32_t>(featureInputs.size());

        // Pointer tables and static span geometry are uploaded on the same
        // non-blocking stream used by the copy kernels. The host vectors remain
        // alive until the final synchronization below.
        std::vector<void *> splitTensorFeatureInputMemoriesArray(numSplitTensors);
        std::vector<void *> splitTensorErrorOutputMemoriesArray;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&splitTensorFeatureInputMemoriesArray_d),
                              numSplitTensors * sizeof(void *)));
        for (uint32_t i = 0; i < numSplitTensors; ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            splitTensorFeatureInputMemoriesArray[i] = featureInputs[i].value().getMemPtr();
        }
        CUDA_CHECK(cudaMemcpyAsync(splitTensorFeatureInputMemoriesArray_d,
                                   splitTensorFeatureInputMemoriesArray.data(),
                                   numSplitTensors * sizeof(void *),
                                   cudaMemcpyHostToDevice,
                                   streams[0].getStream()));

        if (errorInputs[0].has_value()) {
            // Backpropagation through Concatenate may be intentionally sparse:
            // missing upstream destinations are backed by throwaway tensors so
            // Split can keep the same static span plan as forward.
            discardedErrorOutputs.resize(numSplitTensors);
            splitTensorErrorOutputMemoriesArray.resize(numSplitTensors);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&splitTensorErrorOutputMemoriesArray_d),
                                  numSplitTensors * sizeof(void *)));
            for (uint32_t i = 0; i < numSplitTensors; ++i) {
                if (errorOutputs[i].has_value()) {
                    splitTensorErrorOutputMemoriesArray[i] = errorOutputs[i].value().getMemPtr();
                } else {
                    THOR_THROW_IF_FALSE(featureInputs[i].has_value());
                    discardedErrorOutputs[i] = featureInputs[i].value().clone();
                    splitTensorErrorOutputMemoriesArray[i] = discardedErrorOutputs[i].value().getMemPtr();
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(splitTensorErrorOutputMemoriesArray_d,
                                       splitTensorErrorOutputMemoriesArray.data(),
                                       numSplitTensors * sizeof(void *),
                                       cudaMemcpyHostToDevice,
                                       streams[0].getStream()));
        }

        const TensorDescriptor &outputDescriptor = featureOutputs[0].value().getDescriptor();
        const std::vector<uint64_t> &outputDimensions = outputDescriptor.getDimensions();
        THOR_THROW_IF_FALSE(axis < outputDimensions.size());

        uint64_t innerElements = 1;
        for (uint32_t d = axis + 1; d < outputDimensions.size(); ++d)
            innerElements = checkedMultiply(innerElements, outputDimensions[d],
                                            "Concatenate inner geometry overflow.");

        std::vector<uint64_t> axisElementsPerSplitTensor(numSplitTensors);
        for (uint32_t i = 0; i < numSplitTensors; ++i)
            axisElementsPerSplitTensor[i] = featureInputs[i].value().getDescriptor().getDimensions()[axis];

        const std::vector<ConcatenateSpanGeometry> spanGeometry = buildConcatenateSpanGeometry(
            TensorDescriptor::getElementSizeInBytes(outputDescriptor.getDataType()),
            innerElements,
            axisElementsPerSplitTensor);
        THOR_THROW_IF_FALSE(spanGeometry.size() == numSplitTensors);
        CUDA_CHECK(cudaMalloc(&spanGeometryPerSplitTensor_d,
                              spanGeometry.size() * sizeof(ConcatenateSpanGeometry)));
        CUDA_CHECK(cudaMemcpyAsync(spanGeometryPerSplitTensor_d,
                                   spanGeometry.data(),
                                   spanGeometry.size() * sizeof(ConcatenateSpanGeometry),
                                   cudaMemcpyHostToDevice,
                                   streams[0].getStream()));

        packedSliceBytes = spanGeometry.back().packedOffsetBytes + spanGeometry.back().spanBytes;
        THOR_THROW_IF_FALSE(packedSliceBytes > 0);

        outerSlicesPerBatch = 1;
        for (uint32_t d = 1; d < axis; ++d)
            outerSlicesPerBatch = checkedMultiply(outerSlicesPerBatch, outputDimensions[d],
                                                  "Concatenate outer geometry overflow.");

        streams[0].synchronize();

        for (unsigned int i = 0; i < featureInputs.size(); ++i)
            allFeatureInputTensorIds.insert(featureInputs[i].value().getTensorId());
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
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

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        if (errorInput.has_value()) {
            const uint64_t activeOuterSlices = resolveActiveOuterSlices(errorInput.value().getDescriptor(), batchSize);
            launchSplit(splitTensorErrorOutputMemoriesArray_d,
                        errorInput.value().getMemPtr(),
                        activeOuterSlices,
                        static_cast<uint32_t>(errorOutputs.size()),
                        packedSliceBytes,
                        spanGeometryPerSplitTensor_d,
                        streams[0]);
        }

        streams[0].putEvent(backwardOutputsReadyEvent);
        previousLayers[0].value()->backward(errorOutputs[0], batchSize);
        for (unsigned int i = 1; i < errorOutputs.size(); ++i) {
            streams[i].waitEvent(backwardOutputsReadyEvent);
            previousLayers[i].value()->backward(errorOutputs[i], batchSize);
        }
    }

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        const std::vector<uint64_t> inputDimensions = featureInput.value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(!inputDimensions.empty());
        THOR_THROW_IF_FALSE(inputDimensions.front() >= 1);
        THOR_THROW_IF_FALSE(inputDimensions.front() <= std::numeric_limits<uint32_t>::max());
        const uint32_t physicalBatchCapacity = static_cast<uint32_t>(inputDimensions.front());
        const uint32_t resolvedValidExampleCount = batchSize == 0 ? physicalBatchCapacity : batchSize;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        THOR_THROW_IF_FALSE(resolvedValidExampleCount <= physicalBatchCapacity);

        if (axis == 0) {
            // Concatenating along the physical batch axis appends complete batches. A partial
            // input would place padding before valid examples from a later input, violating
            // Thor's valid-prefix batch contract. Until Concatenate compacts valid rows, only
            // full-capacity inputs are supported on axis zero.
            THOR_THROW_IF_FALSE(resolvedValidExampleCount == physicalBatchCapacity);
        } else if (batchCardinalitySet) {
            THOR_THROW_IF_FALSE(currentValidExampleCount == resolvedValidExampleCount);
        } else {
            currentValidExampleCount = resolvedValidExampleCount;
            batchCardinalitySet = true;
        }

        auto it = stillWaitingForFeatureInputTensors.find(featureInput.value().getTensorId());
        THOR_THROW_IF_FALSE(it != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(it);

        if (!stillWaitingForFeatureInputTensors.empty())
            return;

        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;

        for (unsigned int i = 1; i < featureInputs.size(); ++i)
            streams[0].waitFor(streams[i], forwardInputReadyEvents[i]);

        refreshFeatureInputMemoryArray(streams[0]);

        const uint64_t activeOuterSlices =
            axis == 0 ? 1 : checkedMultiply(currentValidExampleCount, outerSlicesPerBatch,
                                            "Concatenate active outer-slice count overflow.");
        launchConcatenate(
            featureOutputs[0].value().getMemPtr(),
            splitTensorFeatureInputMemoriesArray_d,
            activeOuterSlices,
            static_cast<uint32_t>(featureInputs.size()),
            packedSliceBytes,
            spanGeometryPerSplitTensor_d,
            streams[0]);

        uint32_t outputValidExampleCount = currentValidExampleCount;
        if (axis == 0) {
            const std::vector<uint64_t> outputDimensions = featureOutputs[0].value().getDescriptor().getDimensions();
            THOR_THROW_IF_FALSE(!outputDimensions.empty());
            THOR_THROW_IF_FALSE(outputDimensions.front() >= 1);
            THOR_THROW_IF_FALSE(outputDimensions.front() <= std::numeric_limits<uint32_t>::max());
            outputValidExampleCount = static_cast<uint32_t>(outputDimensions.front());
        }

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[0].value()->forward(featureOutputs[0], validationPass, outputValidExampleCount);
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void cleanup() override {
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        TensorPlacement placement = featureInputs[0].value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        if (splitTensorFeatureInputMemoriesArray_d != nullptr) {
            CUDA_CHECK(cudaFree(splitTensorFeatureInputMemoriesArray_d));
            splitTensorFeatureInputMemoriesArray_d = nullptr;
        }
        if (splitTensorErrorOutputMemoriesArray_d != nullptr) {
            CUDA_CHECK(cudaFree(splitTensorErrorOutputMemoriesArray_d));
            splitTensorErrorOutputMemoriesArray_d = nullptr;
        }
        discardedErrorOutputs.clear();
        if (spanGeometryPerSplitTensor_d != nullptr) {
            CUDA_CHECK(cudaFree(spanGeometryPerSplitTensor_d));
            spanGeometryPerSplitTensor_d = nullptr;
        }
        packedSliceBytes = 0;
        outerSlicesPerBatch = 1;
        for (Event& event : forwardInputReadyEvents)
            event = Event();
        backwardOutputsReadyEvent = Event();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!running);
        nextLayers.push_back(nextLayer);
        featureOutputs.emplace_back(createFeatureOutputTensor());
        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams[0], shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType));

        THOR_THROW_IF_FALSE(featureOutputs.back().has_value());
        if (errorInputs.back().has_value()) {
            THOR_THROW_IF_FALSE(errorInputs.back().value().getDescriptor() == errorInputs.front().value().getDescriptor());
            THOR_THROW_IF_FALSE(errorInputs.back().value().getDescriptor() == featureOutputs.back().value().getDescriptor());
            THOR_THROW_IF_FALSE(errorInputs.back().value().getPlacement() == errorInputs.front().value().getPlacement());
            THOR_THROW_IF_FALSE(errorInputs.back().value().getPlacement() == featureOutputs.back().value().getPlacement());
        }

        if (!errorInputs.back().has_value()) {
            for (uint32_t i = 0; i < errorOutputs.size(); ++i) {
                THOR_THROW_IF_FALSE(previousLayers[i].has_value());
                if (errorOutputs[i].has_value())
                    previousLayers[i].value()->replaceErrorInput(errorOutputs[i], std::nullopt);
            }
        }

        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) override {
        THOR_THROW_IF_FALSE(!running);
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(previousLayer != nullptr);

        if (connectionType < 0 || static_cast<uint32_t>(connectionType) >= expectedNumInputs) {
            throw std::logic_error("Concatenate input connection type " + std::to_string(connectionType) +
                                   " is outside the declared input range [0," +
                                   std::to_string(expectedNumInputs - 1) + "].");
        }
        const uint32_t inputIndex = static_cast<uint32_t>(connectionType);
        if (featureInputs[inputIndex].has_value() || previousLayers[inputIndex].has_value()) {
            throw std::logic_error("Concatenate input[" + std::to_string(inputIndex) +
                                   "] was connected more than once. Every Concatenate connection must carry its declared input port.");
        }

        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (!featureInputs[i].has_value())
                continue;
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == featureInputs[i].value().getPlacement());
        }

        streams[inputIndex] = stream;
        previousLayers[inputIndex] = previousLayer;
        featureInputs[inputIndex] = featureInput;
        if (backPropagateError && !isInferenceOnly())
            errorOutputs[inputIndex] = featureInput.value().clone();
        else
            errorOutputs[inputIndex] = std::nullopt;

        THOR_THROW_IF_FALSE(featureInputs[inputIndex].has_value());
        if (errorOutputs[inputIndex].has_value()) {
            THOR_THROW_IF_FALSE(featureInputs[inputIndex].value().getDescriptor() == errorOutputs[inputIndex].value().getDescriptor());
            THOR_THROW_IF_FALSE(featureInputs[inputIndex].value().getPlacement() == errorOutputs[inputIndex].value().getPlacement());
        }
        ensureNoDeviceCrossing();

        return errorOutputs[inputIndex];
    }

   private:
    uint32_t expectedNumInputs;

    static std::string dimensionsToString(const std::vector<uint64_t> &dimensions) {
        std::ostringstream out;
        out << '[';
        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            if (i != 0)
                out << ',';
            out << dimensions[i];
        }
        out << ']';
        return out.str();
    }

    std::string inputShapesToString() const {
        std::ostringstream out;
        out << '{';
        for (std::size_t i = 0; i < featureInputs.size(); ++i) {
            if (i != 0)
                out << ", ";
            out << "input[" << i << "]=";
            if (featureInputs[i].has_value())
                out << dimensionsToString(featureInputs[i].value().getDescriptor().getDimensions());
            else
                out << "<missing>";
        }
        out << '}';
        return out.str();
    }

    std::string layerContext() const {
        std::ostringstream out;
        out << " for Concatenate layer id=" << getId();
        if (!getName().empty())
            out << " name='" << getName() << '\'';
        return out.str();
    }

    struct FeatureInputMemoryArrayRefreshArgs : public HostFunctionArgsBase {
        std::vector<void *> splitTensorFeatureInputMemories;
    };

    static void releaseFeatureInputMemoryArrayRefresh(void *) {}

    void refreshFeatureInputMemoryArray(Stream stream) {
        THOR_THROW_IF_FALSE(splitTensorFeatureInputMemoriesArray_d != nullptr);

        const int numSplitTensors = featureInputs.size();
        auto refreshArgs = std::make_unique<FeatureInputMemoryArrayRefreshArgs>();
        refreshArgs->splitTensorFeatureInputMemories.resize(numSplitTensors);
        for (int i = 0; i < numSplitTensors; ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            refreshArgs->splitTensorFeatureInputMemories[i] = featureInputs[i].value().getMemPtr();
        }

        CUDA_CHECK(cudaMemcpyAsync(splitTensorFeatureInputMemoriesArray_d,
                                   refreshArgs->splitTensorFeatureInputMemories.data(),
                                   numSplitTensors * sizeof(void *),
                                   cudaMemcpyHostToDevice,
                                   stream));
        stream.enqueueHostFunction(&releaseFeatureInputMemoryArrayRefresh, std::move(refreshArgs));
    }

    static uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char *what) {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
            throw std::invalid_argument(what);
        return lhs * rhs;
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
                               "Concatenate active outer-slice count overflow.");
    }

    unsigned int axis;

    void **splitTensorFeatureInputMemoriesArray_d;
    void **splitTensorErrorOutputMemoriesArray_d;
    std::vector<std::optional<Tensor>> discardedErrorOutputs;
    ConcatenateSpanGeometry *spanGeometryPerSplitTensor_d;
    uint64_t packedSliceBytes = 0;
    uint64_t outerSlicesPerBatch = 1;

    std::set<unsigned long> allFeatureInputTensorIds;
    std::set<unsigned long> stillWaitingForFeatureInputTensors;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;

    std::vector<Event> forwardInputReadyEvents;
    Event backwardOutputsReadyEvent;
};

}  // namespace ThorImplementation
