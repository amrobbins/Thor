#pragma once

#include <cstddef>
#include <optional>
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
        stridePerPackedTensorDimension_d = nullptr;
        stridePerSplitTensorDimension_d = nullptr;
        axisElementsPerSplitTensor_d = nullptr;
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
        // Ensure that the right amount of axis elements are specified and all output tensors were connected
        unsigned long totalAxisElements = 0;
        for (unsigned int i = 0; i < axisElements.size(); ++i)
            totalAxisElements += axisElements[i];
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        std::vector<unsigned long> inputDimensions = featureInputs[0].value().getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() > axis);
        THOR_THROW_IF_FALSE(totalAxisElements == inputDimensions[axis]);
        THOR_THROW_IF_FALSE(featureOutputs.size() == axisElements.size());
        THOR_THROW_IF_FALSE(nextLayers.size() == featureOutputs.size());
        THOR_THROW_IF_FALSE(streams.size() == featureOutputs.size());
        for (unsigned int i = 0; i < axisElements.size(); ++i) {
            THOR_THROW_IF_FALSE(featureOutputs[i].has_value());
            THOR_THROW_IF_FALSE(featureOutputs[i].value().getDescriptor().getDimensions()[axis] == axisElements[i]);
        }

        THOR_THROW_IF_FALSE(featureInputs[0].value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        const int numSplitTensors = static_cast<int>(featureOutputs.size());
        THOR_THROW_IF_FALSE(errorInputs.size() == featureOutputs.size());

        const uint32_t numPresentErrorInputs = numPresentTensors(errorInputs);
        THOR_THROW_IF_FALSE(numPresentErrorInputs == errorInputs.size() || numPresentErrorInputs == 0);

        // All metadata uploads are ordered on the same non-blocking stream that
        // launches Split/Concatenate. A synchronous cudaMemcpy from pageable host
        // memory is allowed to return after staging but before the device DMA has
        // completed, and Thor streams intentionally do not synchronize with the
        // legacy default stream. Keep the pageable host vectors alive and wait once
        // after enqueueing every upload so execution can never observe a partially
        // populated metadata buffer under GPU load.
        std::vector<void *> splitTensorFeatureOutputMemoriesArray(numSplitTensors);
        std::vector<void *> splitTensorErrorInputMemoriesArray;
        std::vector<long> axisElementsPerSplitTensor(numSplitTensors);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&splitTensorFeatureOutputMemoriesArray_d),
                              numSplitTensors * sizeof(void *)));
        for (int i = 0; i < numSplitTensors; ++i)
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
            for (int i = 0; i < numSplitTensors; ++i)
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

        for (int i = 0; i < numSplitTensors; ++i)
            axisElementsPerSplitTensor[i] = static_cast<long>(axisElements[i]);
        CUDA_CHECK(cudaMalloc(&axisElementsPerSplitTensor_d, numSplitTensors * sizeof(long)));
        CUDA_CHECK(cudaMemcpyAsync(axisElementsPerSplitTensor_d,
                                   axisElementsPerSplitTensor.data(),
                                   numSplitTensors * sizeof(long),
                                   cudaMemcpyHostToDevice,
                                   streams[0].getStream()));

        const unsigned int numDimensions = featureInputs.front().value().getDescriptor().getDimensions().size();
        std::vector<long> stridePerSplitTensorDimension(numDimensions * numSplitTensors);
        for (int t = 0; t < numSplitTensors; ++t) {
            stridePerSplitTensorDimension[t * numDimensions + (numDimensions - 1)] = 1;
            for (int d = static_cast<int>(numDimensions) - 2; d >= 0; --d) {
                stridePerSplitTensorDimension[t * numDimensions + d] =
                    stridePerSplitTensorDimension[t * numDimensions + (d + 1)] *
                    static_cast<long>(featureOutputs[t].value().getDescriptor().getDimensions()[d + 1]);
            }
        }
        CUDA_CHECK(cudaMalloc(&stridePerSplitTensorDimension_d,
                              numDimensions * numSplitTensors * sizeof(long)));
        CUDA_CHECK(cudaMemcpyAsync(stridePerSplitTensorDimension_d,
                                   stridePerSplitTensorDimension.data(),
                                   numDimensions * numSplitTensors * sizeof(long),
                                   cudaMemcpyHostToDevice,
                                   streams[0].getStream()));

        std::vector<long> stridePerPackedTensorDimension(inputDimensions.size());
        stridePerPackedTensorDimension.back() = 1;
        for (int i = static_cast<int>(inputDimensions.size()) - 2; i >= 0; --i) {
            stridePerPackedTensorDimension[i] =
                static_cast<long>(inputDimensions[i + 1]) * stridePerPackedTensorDimension[i + 1];
        }
        CUDA_CHECK(cudaMalloc(&stridePerPackedTensorDimension_d,
                              inputDimensions.size() * sizeof(long)));
        CUDA_CHECK(cudaMemcpyAsync(stridePerPackedTensorDimension_d,
                                   stridePerPackedTensorDimension.data(),
                                   inputDimensions.size() * sizeof(long),
                                   cudaMemcpyHostToDevice,
                                   streams[0].getStream()));

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

        launchSplit(splitTensorFeatureOutputMemoriesArray_d,
                    featureInput.value().getMemPtr(),
                    static_cast<std::size_t>(TensorDescriptor::getElementSizeInBytes(featureInput.value().getDescriptor().getDataType())),
                    featureInput.value().getDescriptor().getTotalNumElements(),
                    featureInput.value().getDescriptor().getDimensions().size(),
                    featureOutputs.size(),
                    axis,
                    axisElementsPerSplitTensor_d,
                    stridePerPackedTensorDimension_d,
                    stridePerSplitTensorDimension_d,
                    streams[0]);

        Event readyEvent = streams[0].putEvent();
        nextLayers[0].value()->forward(featureOutputs[0], validationPass, batchSize);
        for (unsigned int i = 1; i < featureOutputs.size(); ++i) {
            streams[i].waitEvent(readyEvent);
            nextLayers[i].value()->forward(featureOutputs[i], validationPass, batchSize);
        }
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        // Experimental - back propagation stops at empty error input
        if (!errorInput.has_value())
            return;

        THOR_THROW_IF_FALSE(streams.size() == errorInputs.size());

        if (errorInputs.size() > 1) {
            // Locked section
            std::unique_lock<std::mutex> lck(mtx);

            if (errorInput.has_value()) {
                THOR_THROW_IF_FALSE(stillWaitingForErrorInputTensors.count(errorInput.value().getTensorId()) == 1);
                stillWaitingForErrorInputTensors.erase(errorInput.value().getTensorId());
            }
            if (!stillWaitingForErrorInputTensors.empty())
                return;

            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        }

        for (unsigned int i = 1; i < errorInputs.size(); ++i)
            streams[0].waitEvent(streams[i].putEvent());

        launchConcatenate(
            errorOutputs[0].value().getMemPtr(),
            splitTensorErrorInputMemoriesArray_d,
            static_cast<std::size_t>(TensorDescriptor::getElementSizeInBytes(errorOutputs[0].value().getDescriptor().getDataType())),
            errorOutputs[0].value().getDescriptor().getTotalNumElements(),
            errorOutputs[0].value().getDescriptor().getDimensions().size(),
            errorInputs.size(),
            axis,
            axisElementsPerSplitTensor_d,
            stridePerPackedTensorDimension_d,
            stridePerSplitTensorDimension_d,
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
        if (axisElementsPerSplitTensor_d != nullptr) {
            CUDA_CHECK(cudaFree(axisElementsPerSplitTensor_d));
            axisElementsPerSplitTensor_d = nullptr;
        }
        if (stridePerPackedTensorDimension_d != nullptr) {
            CUDA_CHECK(cudaFree(stridePerPackedTensorDimension_d));
            stridePerPackedTensorDimension_d = nullptr;
        }
        if (stridePerSplitTensorDimension_d != nullptr) {
            CUDA_CHECK(cudaFree(stridePerSplitTensorDimension_d));
            stridePerSplitTensorDimension_d = nullptr;
        }
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
    unsigned int axis;
    std::vector<unsigned long> axisElements;

    void **splitTensorErrorInputMemoriesArray_d;
    void **splitTensorFeatureOutputMemoriesArray_d;
    long *stridePerPackedTensorDimension_d;
    long *stridePerSplitTensorDimension_d;
    long *axisElementsPerSplitTensor_d;

};

}  // namespace ThorImplementation
