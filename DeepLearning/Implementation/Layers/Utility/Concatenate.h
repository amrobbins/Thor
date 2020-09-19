#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
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
    virtual ~Concatenate() {}

    Concatenate(unsigned int axis) {
        this->axis = (int)axis;
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        splitTensorErrorOutputMemoriesArray_d = nullptr;
        stridePerPackedTensorDimension_d = nullptr;
        stridePerSplitTensorDimension_d = nullptr;
        axisElementsPerSplitTensor_d = nullptr;
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInputs.size() > 1);
        for (unsigned int i = 1; i < featureInputs.size(); ++i) {
            assert(featureInputs[i].isPresent());
            assert(featureInputs[i].get().getDescriptor().getDataType() == featureInputs[0].get().getDescriptor().getDataType());
        }
        assert(featureInputs.front().isPresent());
        assert(axis < featureInputs.front().get().getDescriptor().getDimensions().size());
        unsigned int numDimensions = featureInputs.front().get().getDescriptor().getDimensions().size();
        unsigned long newAxisSize = featureInputs.front().get().getDescriptor().getDimensions()[axis];
        for (unsigned int i = 1; i < featureInputs.size(); ++i) {
            assert(featureInputs[i].get().getDescriptor().getDimensions().size() == numDimensions);
            for (unsigned int j = 0; j < numDimensions; ++j) {
                if (j == axis)
                    continue;
                assert(featureInputs[i].get().getDescriptor().getDimensions()[j] ==
                       featureInputs.front().get().getDescriptor().getDimensions()[j]);
            }
            newAxisSize += featureInputs[i].get().getDescriptor().getDimensions()[axis];
        }

        vector<unsigned long> outputDimensions = featureInputs.front().get().getDescriptor().getDimensions();
        outputDimensions[axis] = newAxisSize;
        TensorDescriptor outputDescriptor = TensorDescriptor(featureInputs.front().get().getDescriptor().getDataType(), outputDimensions);

        return Tensor(featureInputs[0].get().getPlacement(), outputDescriptor);
    }

    virtual void compile() {
        assert(featureOutputs.size() == 1);
        assert(featureOutputs[0].isPresent());
        assert(nextLayers.size() == 1);
        assert(featureInputs[0].isPresent());
        assert(featureInputs[0].get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].get().getPlacement().getDeviceNum());
        cudaError_t cudaStatus;
        int numSplitTensors = featureInputs.size();

        cudaStatus = cudaMalloc(&splitTensorFeatureInputMemoriesArray_d, numSplitTensors * sizeof(half *));
        assert(cudaStatus == cudaSuccess);
        half **splitTensorFeatureInputMemoriesArray = new half *[numSplitTensors];
        for (int i = 0; i < numSplitTensors; ++i) {
            assert(featureInputs[i].isPresent());
            splitTensorFeatureInputMemoriesArray[i] = (half *)featureInputs[i].get().getMemPtr();
        }
        cudaStatus = cudaMemcpy(splitTensorFeatureInputMemoriesArray_d,
                                splitTensorFeatureInputMemoriesArray,
                                numSplitTensors * sizeof(half *),
                                cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
        delete[] splitTensorFeatureInputMemoriesArray;

        cudaStatus = cudaMalloc(&splitTensorErrorOutputMemoriesArray_d, numPresentTensors(errorOutputs) * sizeof(half *));
        assert(cudaStatus == cudaSuccess);
        half **splitTensorErrorOutputMemoriesArray = new half *[numSplitTensors];
        for (int i = 0; i < numSplitTensors; ++i) {
            if (errorOutputs[i].isPresent())
                splitTensorErrorOutputMemoriesArray[i] = (half *)errorOutputs[i].get().getMemPtr();
        }
        cudaStatus = cudaMemcpy(splitTensorErrorOutputMemoriesArray_d,
                                splitTensorErrorOutputMemoriesArray,
                                numPresentTensors(errorOutputs) * sizeof(half *),
                                cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
        delete[] splitTensorErrorOutputMemoriesArray;

        long *axisElementsPerSplitTensor = new long[numSplitTensors];
        for (int i = 0; i < numSplitTensors; ++i)
            axisElementsPerSplitTensor[i] = featureInputs[i].get().getDescriptor().getDimensions()[axis];
        cudaStatus = cudaMalloc(&axisElementsPerSplitTensor_d, numSplitTensors * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpy(axisElementsPerSplitTensor_d, axisElementsPerSplitTensor, numSplitTensors * sizeof(long), cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);

        unsigned int numDimensions = featureInputs[0].get().getDescriptor().getDimensions().size();
        long *stridePerSplitTensorDimension = new long[numDimensions * numSplitTensors];
        for (unsigned int t = 0; t < featureInputs.size(); ++t) {
            stridePerSplitTensorDimension[t * numDimensions + (numDimensions - 1)] = 1;
            for (int d = numDimensions - 2; d >= 0; --d)
                stridePerSplitTensorDimension[t * numDimensions + d] = stridePerSplitTensorDimension[t * numDimensions + (d + 1)] *
                                                                       featureInputs[t].get().getDescriptor().getDimensions()[d + 1];
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

        vector<unsigned long> outputDimensions = featureOutputs[0].get().getDescriptor().getDimensions();
        long *stridePerPackedTensorDimension = new long[outputDimensions.size()];
        stridePerPackedTensorDimension[outputDimensions.size() - 1] = 1;
        for (int i = (int)outputDimensions.size() - 2; i >= 0; --i)
            stridePerPackedTensorDimension[i] = outputDimensions[i + 1] * stridePerPackedTensorDimension[i + 1];
        cudaStatus = cudaMalloc(&stridePerPackedTensorDimension_d, outputDimensions.size() * sizeof(unsigned long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpy(stridePerPackedTensorDimension_d,
                                stridePerPackedTensorDimension,
                                outputDimensions.size() * sizeof(unsigned long),
                                cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
        delete[] stridePerPackedTensorDimension;

        for (unsigned int i = 0; i < featureInputs.size(); ++i)
            allFeatureInputTensorIds.insert(featureInputs[i].get().getTensorId());
    }

    virtual void initialize() { stillWaitingForFeatureInputTensors = allFeatureInputTensorIds; }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {}

    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) {}

    virtual void backward(Optional<Tensor> errorInput) {
        assert(errorInput.isPresent());
        launchSplit(splitTensorErrorOutputMemoriesArray_d,
                    (half *)errorInput.get().getMemPtr(),
                    errorInput.get().getDescriptor().getTotalNumElements(),
                    errorInput.get().getDescriptor().getDimensions().size(),
                    numPresentTensors(errorOutputs),
                    axis,
                    axisElementsPerSplitTensor_d,
                    stridePerPackedTensorDimension_d,
                    stridePerSplitTensorDimension_d,
                    streams[0]);

        Event readyEvent = streams[0].putEvent();
        previousLayers[0].get()->backward(errorOutputs[0]);
        for (unsigned int i = 1; i < errorOutputs.size(); ++i) {
            streams[i].waitEvent(readyEvent);
            previousLayers[i].get()->backward(errorOutputs[i]);
        }
    }

    virtual void forward(Optional<Tensor> featureInput) {
        assert(featureInput.isPresent());
        auto it = stillWaitingForFeatureInputTensors.find(featureInput.get().getTensorId());
        assert(it != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(it);

        if (!stillWaitingForFeatureInputTensors.empty())
            return;

        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;

        for (unsigned int i = 1; i < featureInputs.size(); ++i)
            streams[0].waitEvent(streams[i].putEvent());

        launchConcatenate((half *)featureOutputs[0].get().getMemPtr(),
                          splitTensorFeatureInputMemoriesArray_d,
                          featureOutputs[0].get().getDescriptor().getTotalNumElements(),
                          featureOutputs[0].get().getDescriptor().getDimensions().size(),
                          featureInputs.size(),
                          axis,
                          axisElementsPerSplitTensor_d,
                          stridePerPackedTensorDimension_d,
                          stridePerSplitTensorDimension_d,
                          streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[0].get()->forward(featureOutputs[0]);
    }

    virtual void cleanup() {
        cudaError_t cudaStatus;
        assert(featureInputs[0].isPresent());
        TensorPlacement placement = featureInputs[0].get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].get().getPlacement().getDeviceNum());
        cudaStatus = cudaFree(splitTensorFeatureInputMemoriesArray_d);
        assert(cudaStatus == cudaSuccess);
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        cudaStatus = cudaFree(axisElementsPerSplitTensor_d);
        assert(cudaStatus == cudaSuccess);
        axisElementsPerSplitTensor_d = nullptr;
        cudaStatus = cudaFree(stridePerPackedTensorDimension_d);
        assert(cudaStatus == cudaSuccess);
        stridePerPackedTensorDimension_d = nullptr;
    }

    virtual void connectToNextLayer(Layer *nextLayer, int connectionType = 0) {
        assert(!running);
        nextLayers.push_back(nextLayer);
        featureOutputs.emplace_back(createFeatureOutputTensor());
        errorInputs.emplace_back(
            nextLayer->connectToPreviousLayer(this, featureOutputs.back(), streams[0], shouldConnectToBackPropErrorIn(), connectionType));

        assert(errorInputs.back().isPresent());
        assert(featureOutputs.back().isPresent());
        assert(errorInputs.back().get().getDescriptor() == errorInputs.front().get().getDescriptor());
        assert(errorInputs.back().get().getDescriptor() == featureOutputs.back().get().getDescriptor());
        assert(errorInputs.back().get().getPlacement() == errorInputs.front().get().getPlacement());
        assert(errorInputs.back().get().getPlacement() == featureOutputs.back().get().getPlacement());
        ensureNoDeviceCrossing();
    }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        assert(!running);
        assert(featureInput.isPresent());

        if (!featureInputs.empty()) {
            assert(featureInputs[0].isPresent());
            assert(featureInput.get().getPlacement() == featureInputs[0].get().getPlacement());
        }

        streams.push_back(stream);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        if (backPropagateError)
            errorOutputs.emplace_back(featureInput.get().clone());
        else
            errorOutputs.emplace_back(Optional<Tensor>::empty());

        assert(featureInputs.back().isPresent());
        assert(featureInputs.back().get().getPlacement() == featureInputs[0].get().getPlacement());
        if (errorOutputs.back().isPresent()) {
            assert(featureInputs.back().get().getDescriptor() == errorOutputs.back().get().getDescriptor());
            assert(featureInputs.back().get().getPlacement() == errorOutputs.back().get().getPlacement());
        }
        ensureNoDeviceCrossing();

        if (errorOutputs.back().isPresent()) {
            return errorOutputs.back();
        } else {
            return Tensor();
        }
    }

   private:
    unsigned int axis;

    half **splitTensorFeatureInputMemoriesArray_d;
    half **splitTensorErrorOutputMemoriesArray_d;
    long *stridePerPackedTensorDimension_d;
    long *stridePerSplitTensorDimension_d;
    long *axisElementsPerSplitTensor_d;

    set<unsigned long> allFeatureInputTensorIds;
    set<unsigned long> stillWaitingForFeatureInputTensors;
};

}  // namespace ThorImplementation
