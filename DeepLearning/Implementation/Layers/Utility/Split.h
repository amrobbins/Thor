#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
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
 */
class Split : public MultiConnectionLayer {
   public:
    ~Split() override {}

    Split(unsigned int axis, std::vector<unsigned long> axisElements) {
        this->axis = axis;
        this->axisElements = axisElements;
        THOR_THROW_IF_FALSE(!axisElements.empty());
        splitTensorErrorInputMemoriesArray_d = nullptr;
        splitTensorFeatureOutputMemoriesArray_d = nullptr;
        spanGeometryPerSplitTensor_d = nullptr;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        unsigned int connection = featureOutputs.size();
        THOR_THROW_IF_FALSE(connection < axisElements.size());
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());

        std::vector<unsigned long> dimensions = featureInputs[0].value().getDescriptor().getDimensions();
        dimensions[axis] = axisElements[connection];
        return Tensor(featureInputs[0].value().getPlacement(),
                      TensorDescriptor(featureInputs[0].value().getDescriptor().getDataType(), dimensions));
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        unsigned long totalAxisElements = 0;
        for (unsigned int i = 0; i < axisElements.size(); ++i)
            totalAxisElements += axisElements[i];
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        const std::vector<unsigned long> inputDimensions = featureInputs[0].value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() > axis);
        THOR_THROW_IF_FALSE(totalAxisElements == inputDimensions[axis]);
        THOR_THROW_IF_FALSE(featureOutputs.size() == axisElements.size());
        THOR_THROW_IF_FALSE(nextLayers.size() == featureOutputs.size());
        THOR_THROW_IF_FALSE(streams.size() == featureOutputs.size());
        backwardInputReadyEvents.clear();
        backwardInputReadyEvents.resize(streams.size());
        outputsReadyEvent = Event();
        for (unsigned int i = 0; i < axisElements.size(); ++i) {
            THOR_THROW_IF_FALSE(featureOutputs[i].has_value());
            THOR_THROW_IF_FALSE(featureOutputs[i].value().getDescriptor().getDimensions()[axis] == axisElements[i]);
        }

        THOR_THROW_IF_FALSE(featureInputs[0].value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        const uint32_t numSplitTensors = static_cast<uint32_t>(featureOutputs.size());
        THOR_THROW_IF_FALSE(errorInputs.size() == featureOutputs.size());

        const uint32_t numPresentErrorInputs = numPresentTensors(errorInputs);
        THOR_THROW_IF_FALSE(numPresentErrorInputs == errorInputs.size() || numPresentErrorInputs == 0);

        // Pointer tables and static span geometry are uploaded on the same
        // non-blocking execution stream and synchronized once before compile
        // returns, keeping pageable host metadata alive until DMA completes.
        std::vector<void *> splitTensorFeatureOutputMemoriesArray(numSplitTensors);
        std::vector<void *> splitTensorErrorInputMemoriesArray;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&splitTensorFeatureOutputMemoriesArray_d),
                              numSplitTensors * sizeof(void *)));
        for (uint32_t i = 0; i < numSplitTensors; ++i)
            splitTensorFeatureOutputMemoriesArray[i] = featureOutputs[i].value().getMemPtr();
        CUDA_CHECK(cudaMemcpyAsync(splitTensorFeatureOutputMemoriesArray_d,
                                   splitTensorFeatureOutputMemoriesArray.data(),
                                   numSplitTensors * sizeof(void *),
                                   cudaMemcpyHostToDevice,
                                   streams[0].getStream()));

        if (numPresentErrorInputs > 0) {
            splitTensorErrorInputMemoriesArray.resize(numSplitTensors);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&splitTensorErrorInputMemoriesArray_d),
                                  numSplitTensors * sizeof(void *)));
            for (uint32_t i = 0; i < numSplitTensors; ++i)
                splitTensorErrorInputMemoriesArray[i] = errorInputs[i].value().getMemPtr();
            CUDA_CHECK(cudaMemcpyAsync(splitTensorErrorInputMemoriesArray_d,
                                       splitTensorErrorInputMemoriesArray.data(),
                                       numSplitTensors * sizeof(void *),
                                       cudaMemcpyHostToDevice,
                                       streams[0].getStream()));
        } else {
            for (uint32_t i = 0; i < errorOutputs.size(); ++i) {
                THOR_THROW_IF_FALSE(previousLayers[i].has_value());
                if (errorOutputs[i].has_value())
                    previousLayers[i].value()->replaceErrorInput(errorOutputs[i], std::nullopt);
            }
        }

        const TensorDescriptor &inputDescriptor = featureInputs[0].value().getDescriptor();
        uint64_t innerElements = 1;
        for (uint32_t d = axis + 1; d < inputDimensions.size(); ++d)
            innerElements = checkedMultiply(innerElements, inputDimensions[d],
                                            "Split inner geometry overflow.");

        std::vector<uint64_t> axisElements64(axisElements.begin(), axisElements.end());
        const std::vector<ConcatenateSpanGeometry> spanGeometry = buildConcatenateSpanGeometry(
            TensorDescriptor::getElementSizeInBytes(inputDescriptor.getDataType()),
            innerElements,
            axisElements64);
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
            outerSlicesPerBatch = checkedMultiply(outerSlicesPerBatch, inputDimensions[d],
                                                  "Split outer geometry overflow.");

        streams[0].synchronize();
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
        launchSplit(splitTensorFeatureOutputMemoriesArray_d,
                    featureInput.value().getMemPtr(),
                    activeOuterSlices,
                    static_cast<uint32_t>(featureOutputs.size()),
                    packedSliceBytes,
                    spanGeometryPerSplitTensor_d,
                    streams[0]);

        streams[0].putEvent(outputsReadyEvent);
        nextLayers[0].value()->forward(featureOutputs[0], validationPass, batchSize);
        for (unsigned int i = 1; i < featureOutputs.size(); ++i) {
            streams[i].waitEvent(outputsReadyEvent);
            nextLayers[i].value()->forward(featureOutputs[i], validationPass, batchSize);
        }
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        // Experimental - back propagation stops at empty error input
        if (!errorInput.has_value())
            return;

        THOR_THROW_IF_FALSE(streams.size() == errorInputs.size());

        if (errorInputs.size() > 1) {
            THOR_THROW_IF_FALSE(stillWaitingForErrorInputTensors.count(errorInput.value().getTensorId()) == 1);
            stillWaitingForErrorInputTensors.erase(errorInput.value().getTensorId());
            if (!stillWaitingForErrorInputTensors.empty())
                return;

            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        }

        for (unsigned int i = 1; i < errorInputs.size(); ++i)
            streams[0].waitFor(streams[i], backwardInputReadyEvents[i]);

        const uint64_t activeOuterSlices = resolveActiveOuterSlices(errorOutputs[0].value().getDescriptor(), batchSize);
        launchConcatenate(
            errorOutputs[0].value().getMemPtr(),
            splitTensorErrorInputMemoriesArray_d,
            activeOuterSlices,
            static_cast<uint32_t>(errorInputs.size()),
            packedSliceBytes,
            spanGeometryPerSplitTensor_d,
            streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[0].value()->backward(errorOutputs[0], batchSize);
    }

    void cleanup() override {
        TensorPlacement placement = featureInputs[0].value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        if (splitTensorFeatureOutputMemoriesArray_d != nullptr) {
            CUDA_CHECK(cudaFree(splitTensorFeatureOutputMemoriesArray_d));
            splitTensorFeatureOutputMemoriesArray_d = nullptr;
        }
        if (splitTensorErrorInputMemoriesArray_d != nullptr) {
            CUDA_CHECK(cudaFree(splitTensorErrorInputMemoriesArray_d));
            splitTensorErrorInputMemoriesArray_d = nullptr;
        }
        if (spanGeometryPerSplitTensor_d != nullptr) {
            CUDA_CHECK(cudaFree(spanGeometryPerSplitTensor_d));
            spanGeometryPerSplitTensor_d = nullptr;
        }
        packedSliceBytes = 0;
        outerSlicesPerBatch = 1;
        outputsReadyEvent = Event();
        backwardInputReadyEvents.clear();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        // FIXME: Reuse MultiConnectionLayer connectToNextLayer and add any additional logic here if needed
        THOR_THROW_IF_FALSE(!running);
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);

        unsigned int connection = featureOutputs.size();
        THOR_THROW_IF_FALSE(connection < axisElements.size());

        TensorPlacement placement = featureInputs[0].value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

        featureOutputs.push_back(createFeatureOutputTensor());
        nextLayers.push_back(nextLayer);
        if (connection != 0)
            streams.emplace_back(placement.getDeviceNum());
        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams.back(), shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType));

        if (errorInputs.back().has_value()) {
            THOR_THROW_IF_FALSE(errorInputs.back().value().getDescriptor() == featureOutputs.back().value().getDescriptor());
            THOR_THROW_IF_FALSE(errorInputs.back().value().getPlacement() == errorInputs.front().value().getPlacement());
            THOR_THROW_IF_FALSE(errorInputs.back().value().getPlacement() == featureOutputs.back().value().getPlacement());
        }
        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);
        THOR_THROW_IF_FALSE(featureInputs.empty());
        THOR_THROW_IF_FALSE(featureInput.has_value());

        streams.push_back(stream);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        if (backPropagateError && !isInferenceOnly())
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

   private:
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
                               "Split active outer-slice count overflow.");
    }

    unsigned int axis;
    std::vector<unsigned long> axisElements;

    void **splitTensorErrorInputMemoriesArray_d;
    void **splitTensorFeatureOutputMemoriesArray_d;
    ConcatenateSpanGeometry *spanGeometryPerSplitTensor_d;
    uint64_t packedSliceBytes = 0;
    uint64_t outerSlicesPerBatch = 1;

    Event outputsReadyEvent;
    std::vector<Event> backwardInputReadyEvents;
};

}  // namespace ThorImplementation
