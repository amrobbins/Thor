#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
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
    virtual ~Split() {}

    Split(unsigned int axis, std::vector<unsigned long> axisElements) {
        this->axis = axis;
        this->axisElements = axisElements;
        assert(!axisElements.empty());
        splitTensorErrorInputMemoriesArray_d = nullptr;
        splitTensorFeatureOutputMemoriesArray_d = nullptr;
        stridePerPackedTensorDimension_d = nullptr;
        stridePerSplitTensorDimension_d = nullptr;
        axisElementsPerSplitTensor_d = nullptr;
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        unsigned int connection = featureOutputs.size();
        assert(connection < axisElements.size());
        assert(featureInputs[0].isPresent());

        std::vector<unsigned long> dimensions = featureInputs[0].get().getDescriptor().getDimensions();
        dimensions[axis] = axisElements[connection];
        return Tensor(featureInputs[0].get().getPlacement(),
                      TensorDescriptor(featureInputs[0].get().getDescriptor().getDataType(), dimensions));
    }

    virtual void compileImpl() {
        MultiConnectionLayer::compileImpl();
        // Ensure that the right amount of axis elements are specified and all output tensors were connected
        unsigned long totalAxisElements = 0;
        for (unsigned int i = 0; i < axisElements.size(); ++i)
            totalAxisElements += axisElements[i];
        assert(featureInputs.size() == 1);
        assert(featureInputs[0].isPresent());
        std::vector<unsigned long> inputDimensions = featureInputs[0].get().getDescriptor().getDimensions();
        assert(inputDimensions.size() > axis);
        assert(totalAxisElements == inputDimensions[axis]);
        assert(featureOutputs.size() == axisElements.size());
        assert(nextLayers.size() == featureOutputs.size());
        for (unsigned int i = 0; i < axisElements.size(); ++i) {
            assert(featureOutputs[i].isPresent());
            assert(featureOutputs[i].get().getDescriptor().getDimensions()[axis] == axisElements[i]);
        }

        assert(featureInputs[0].get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].get().getPlacement().getDeviceNum());
        cudaError_t cudaStatus;
        int numSplitTensors = featureOutputs.size();
        assert(errorInputs.size() == featureOutputs.size());

        uint32_t numPresentErrorInputs = numPresentTensors(errorInputs);
        assert(numPresentErrorInputs == errorInputs.size() || numPresentErrorInputs == 0);

        cudaStatus = cudaMalloc(&splitTensorFeatureOutputMemoriesArray_d, numSplitTensors * sizeof(half *));
        assert(cudaStatus == cudaSuccess);
        half **splitTensorFeatureOutputMemoriesArray = new half *[numSplitTensors];
        for (int i = 0; i < numSplitTensors; ++i)
            splitTensorFeatureOutputMemoriesArray[i] = (half *)featureOutputs[i].get().getMemPtr();
        cudaStatus = cudaMemcpy(splitTensorFeatureOutputMemoriesArray_d,
                                splitTensorFeatureOutputMemoriesArray,
                                numSplitTensors * sizeof(half *),
                                cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
        delete[] splitTensorFeatureOutputMemoriesArray;

        if (numPresentErrorInputs > 0) {
            cudaStatus = cudaMalloc(&splitTensorErrorInputMemoriesArray_d, numSplitTensors * sizeof(half *));
            assert(cudaStatus == cudaSuccess);
            half **splitTensorErrorInputMemoriesArray = new half *[numSplitTensors];
            for (int i = 0; i < numSplitTensors; ++i) {
                splitTensorErrorInputMemoriesArray[i] = (half *)errorInputs[i].get().getMemPtr();
            }
            cudaStatus = cudaMemcpy(splitTensorErrorInputMemoriesArray_d,
                                    splitTensorErrorInputMemoriesArray,
                                    numSplitTensors * sizeof(half *),
                                    cudaMemcpyHostToDevice);
            assert(cudaStatus == cudaSuccess);
            delete[] splitTensorErrorInputMemoriesArray;
        } else {
            for (uint32_t i = 0; i < errorOutputs.size(); ++i) {
                assert(previousLayers[i].isPresent());
                if (errorOutputs[i].isPresent())
                    previousLayers[i].get()->replaceErrorInput(errorOutputs[i], Optional<Tensor>::empty());
            }
        }

        long *axisElementsPerSplitTensor = new long[numSplitTensors];
        for (int i = 0; i < numSplitTensors; ++i)
            axisElementsPerSplitTensor[i] = axisElements[i];
        cudaStatus = cudaMalloc(&axisElementsPerSplitTensor_d, numSplitTensors * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpy(axisElementsPerSplitTensor_d, axisElementsPerSplitTensor, numSplitTensors * sizeof(long), cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);

        unsigned int numDimensions = featureInputs.front().get().getDescriptor().getDimensions().size();
        long *stridePerSplitTensorDimension = new long[numDimensions * numSplitTensors];
        for (int t = 0; t < numSplitTensors; ++t) {
            stridePerSplitTensorDimension[t * numDimensions + (numDimensions - 1)] = 1;
            for (int d = numDimensions - 2; d >= 0; --d)
                stridePerSplitTensorDimension[t * numDimensions + d] = stridePerSplitTensorDimension[t * numDimensions + (d + 1)] *
                                                                       featureOutputs[t].get().getDescriptor().getDimensions()[d + 1];
        }
        cudaStatus = cudaMalloc(&stridePerSplitTensorDimension_d, numDimensions * numSplitTensors * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpy(stridePerSplitTensorDimension_d,
                                stridePerSplitTensorDimension,
                                numDimensions * numSplitTensors * sizeof(long),
                                cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);

        delete[] stridePerSplitTensorDimension;
        delete[] axisElementsPerSplitTensor;

        long *stridePerPackedTensorDimension = new long[inputDimensions.size()];
        stridePerPackedTensorDimension[inputDimensions.size() - 1] = 1;
        for (int i = (int)inputDimensions.size() - 2; i >= 0; --i)
            stridePerPackedTensorDimension[i] = inputDimensions[i + 1] * stridePerPackedTensorDimension[i + 1];
        cudaStatus = cudaMalloc(&stridePerPackedTensorDimension_d, inputDimensions.size() * sizeof(unsigned long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpy(stridePerPackedTensorDimension_d,
                                stridePerPackedTensorDimension,
                                inputDimensions.size() * sizeof(unsigned long),
                                cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
        delete[] stridePerPackedTensorDimension;
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {}

    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) {}

    virtual void forward(Optional<Tensor> featureInput, bool validationPass) {
        assert(featureInput.isPresent());

        launchSplit(splitTensorFeatureOutputMemoriesArray_d,
                    (half *)featureInput.get().getMemPtr(),
                    featureInput.get().getDescriptor().getTotalNumElements(),
                    featureInput.get().getDescriptor().getDimensions().size(),
                    featureOutputs.size(),
                    axis,
                    axisElementsPerSplitTensor_d,
                    stridePerPackedTensorDimension_d,
                    stridePerSplitTensorDimension_d,
                    streams[0]);

        Event readyEvent = streams[0].putEvent();
        nextLayers[0].get()->forward(featureOutputs[0], validationPass);
        for (unsigned int i = 1; i < featureOutputs.size(); ++i) {
            streams[i].waitEvent(readyEvent);
            nextLayers[i].get()->forward(featureOutputs[i], validationPass);
        }
    }

    virtual void backward(Optional<Tensor> errorInput) {
        // Experimental - back propagation stops at empty error input
        if (errorInput.isEmpty())
            return;

        if (errorInputs.size() > 1) {
            // Locked section
            std::unique_lock<std::mutex> lck(mtx);

            if (errorInput.isPresent()) {
                assert(stillWaitingForErrorInputTensors.count(errorInput.get().getTensorId()) == 1);
                stillWaitingForErrorInputTensors.erase(errorInput.get().getTensorId());
            }
            if (!stillWaitingForErrorInputTensors.empty())
                return;

            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        }

        for (unsigned int i = 1; i < errorInputs.size(); ++i)
            streams[0].waitEvent(streams[i].putEvent());

        launchConcatenate((half *)errorOutputs[0].get().getMemPtr(),
                          splitTensorErrorInputMemoriesArray_d,
                          errorOutputs[0].get().getDescriptor().getTotalNumElements(),
                          errorOutputs[0].get().getDescriptor().getDimensions().size(),
                          errorInputs.size(),
                          axis,
                          axisElementsPerSplitTensor_d,
                          stridePerPackedTensorDimension_d,
                          stridePerSplitTensorDimension_d,
                          streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[0].get()->backward(errorOutputs[0]);
    }

    virtual void cleanup() {
        cudaError_t cudaStatus;
        TensorPlacement placement = featureInputs[0].get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].get().getPlacement().getDeviceNum());
        if (splitTensorErrorInputMemoriesArray_d != nullptr) {
            cudaStatus = cudaFree(splitTensorErrorInputMemoriesArray_d);
            assert(cudaStatus == cudaSuccess);
            splitTensorErrorInputMemoriesArray_d = nullptr;
        }
        cudaStatus = cudaFree(axisElementsPerSplitTensor_d);
        assert(cudaStatus == cudaSuccess);
        axisElementsPerSplitTensor_d = nullptr;
        cudaStatus = cudaFree(stridePerPackedTensorDimension_d);
        assert(cudaStatus == cudaSuccess);
        stridePerPackedTensorDimension_d = nullptr;
    }

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        // FIXME: Reuse MultiConnectionLayer connectToNextLayer and add any additional logic here if needed
        assert(!running);
        assert(featureInputs.size() == 1);

        unsigned int connection = featureOutputs.size();
        assert(connection < axisElements.size());

        TensorPlacement placement = featureInputs[0].get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

        featureOutputs.push_back(createFeatureOutputTensor());
        nextLayers.push_back(nextLayer);
        if (connection != 0)
            streams.emplace_back(placement.getDeviceNum());
        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams.back(), shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType));

        if (errorInputs.back().isPresent()) {
            assert(errorInputs.back().get().getDescriptor() == featureOutputs.back().get().getDescriptor());
            assert(errorInputs.back().get().getPlacement() == errorInputs.front().get().getPlacement());
            assert(errorInputs.back().get().getPlacement() == featureOutputs.back().get().getPlacement());
        }
        ensureNoDeviceCrossing();
    }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        assert(!compiled);
        assert(featureInputs.empty());
        assert(featureInput.isPresent());

        streams.push_back(stream);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        if (backPropagateError && !isInferenceOnly())
            errorOutputs.emplace_back(featureInput.get().clone());
        else
            errorOutputs.emplace_back(Optional<Tensor>::empty());

        if (errorOutputs.back().isPresent()) {
            assert(featureInputs.back().get().getDescriptor() == errorOutputs.back().get().getDescriptor());
            assert(featureInputs.back().get().getPlacement() == errorOutputs.back().get().getPlacement());
        }
        ensureNoDeviceCrossing();

        return errorOutputs.back();
    }

   private:
    unsigned int axis;
    std::vector<unsigned long> axisElements;

    half **splitTensorErrorInputMemoriesArray_d;
    half **splitTensorFeatureOutputMemoriesArray_d;
    long *stridePerPackedTensorDimension_d;
    long *stridePerSplitTensorDimension_d;
    long *axisElementsPerSplitTensor_d;
};

}  // namespace ThorImplementation
