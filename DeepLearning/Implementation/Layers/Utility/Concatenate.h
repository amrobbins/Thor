#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
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

    Concatenate(unsigned int axis) {
        this->axis = (int)axis;
        splitTensorFeatureInputMemoriesArray_d = nullptr;
        splitTensorErrorOutputMemoriesArray_d = nullptr;
        stridePerPackedTensorDimension_d = nullptr;
        stridePerSplitTensorDimension_d = nullptr;
        axisElementsPerSplitTensor_d = nullptr;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInputs.size() > 1);
        for (unsigned int i = 1; i < featureInputs.size(); ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            THOR_THROW_IF_FALSE(featureInputs[i].value().getDescriptor().getDataType() == featureInputs[0].value().getDescriptor().getDataType());
        }
        THOR_THROW_IF_FALSE(featureInputs.front().has_value());
        THOR_THROW_IF_FALSE(axis < featureInputs.front().value().getDescriptor().getDimensions().size());
        unsigned int numDimensions = featureInputs.front().value().getDescriptor().getDimensions().size();
        unsigned long newAxisSize = featureInputs.front().value().getDescriptor().getDimensions()[axis];
        for (unsigned int i = 1; i < featureInputs.size(); ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].value().getDescriptor().getDimensions().size() == numDimensions);
            for (unsigned int j = 0; j < numDimensions; ++j) {
                if (j == axis)
                    continue;
                THOR_THROW_IF_FALSE(featureInputs[i].value().getDescriptor().getDimensions()[j] ==
                       featureInputs.front().value().getDescriptor().getDimensions()[j]);
            }
            newAxisSize += featureInputs[i].value().getDescriptor().getDimensions()[axis];
        }

        std::vector<unsigned long> outputDimensions = featureInputs.front().value().getDescriptor().getDimensions();
        outputDimensions[axis] = newAxisSize;
        TensorDescriptor outputDescriptor = TensorDescriptor(featureInputs.front().value().getDescriptor().getDataType(), outputDimensions);

        return Tensor(featureInputs[0].value().getPlacement(), outputDescriptor);
    }

    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        THOR_THROW_IF_FALSE(featureOutputs[0].has_value());
        THOR_THROW_IF_FALSE(nextLayers.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        THOR_THROW_IF_FALSE(featureInputs[0].value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        cudaError_t cudaStatus;
        int numSplitTensors = featureInputs.size();

        cudaStatus = cudaMalloc(&splitTensorFeatureInputMemoriesArray_d, numSplitTensors * sizeof(half *));
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        half **splitTensorFeatureInputMemoriesArray = new half *[numSplitTensors];
        for (int i = 0; i < numSplitTensors; ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            splitTensorFeatureInputMemoriesArray[i] = (half *)featureInputs[i].value().getMemPtr();
        }
        cudaStatus = cudaMemcpy(splitTensorFeatureInputMemoriesArray_d,
                                splitTensorFeatureInputMemoriesArray,
                                numSplitTensors * sizeof(half *),
                                cudaMemcpyHostToDevice);
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        delete[] splitTensorFeatureInputMemoriesArray;

        if (errorInputs[0].has_value()) {
            cudaStatus = cudaMalloc(&splitTensorErrorOutputMemoriesArray_d, numPresentTensors(errorOutputs) * sizeof(half *));
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            half **splitTensorErrorOutputMemoriesArray = new half *[numSplitTensors];
            for (int i = 0; i < numSplitTensors; ++i) {
                if (errorOutputs[i].has_value())
                    splitTensorErrorOutputMemoriesArray[i] = (half *)errorOutputs[i].value().getMemPtr();
            }
            cudaStatus = cudaMemcpy(splitTensorErrorOutputMemoriesArray_d,
                                    splitTensorErrorOutputMemoriesArray,
                                    numPresentTensors(errorOutputs) * sizeof(half *),
                                    cudaMemcpyHostToDevice);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            delete[] splitTensorErrorOutputMemoriesArray;
        }

        long *axisElementsPerSplitTensor = new long[numSplitTensors];
        for (int i = 0; i < numSplitTensors; ++i)
            axisElementsPerSplitTensor[i] = featureInputs[i].value().getDescriptor().getDimensions()[axis];
        cudaStatus = cudaMalloc(&axisElementsPerSplitTensor_d, numSplitTensors * sizeof(long));
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpy(axisElementsPerSplitTensor_d, axisElementsPerSplitTensor, numSplitTensors * sizeof(long), cudaMemcpyHostToDevice);
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);

        unsigned int numDimensions = featureInputs[0].value().getDescriptor().getDimensions().size();
        long *stridePerSplitTensorDimension = new long[numDimensions * numSplitTensors];
        for (unsigned int t = 0; t < featureInputs.size(); ++t) {
            stridePerSplitTensorDimension[t * numDimensions + (numDimensions - 1)] = 1;
            for (int d = numDimensions - 2; d >= 0; --d)
                stridePerSplitTensorDimension[t * numDimensions + d] = stridePerSplitTensorDimension[t * numDimensions + (d + 1)] *
                                                                       featureInputs[t].value().getDescriptor().getDimensions()[d + 1];
        }
        cudaStatus = cudaMalloc(&stridePerSplitTensorDimension_d, numDimensions * numSplitTensors * sizeof(long));
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpy(stridePerSplitTensorDimension_d,
                                stridePerSplitTensorDimension,
                                numDimensions * numSplitTensors * sizeof(long),
                                cudaMemcpyHostToDevice);
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);

        delete[] stridePerSplitTensorDimension;
        delete[] axisElementsPerSplitTensor;

        std::vector<unsigned long> outputDimensions = featureOutputs[0].value().getDescriptor().getDimensions();
        long *stridePerPackedTensorDimension = new long[outputDimensions.size()];
        stridePerPackedTensorDimension[outputDimensions.size() - 1] = 1;
        for (int i = (int)outputDimensions.size() - 2; i >= 0; --i)
            stridePerPackedTensorDimension[i] = outputDimensions[i + 1] * stridePerPackedTensorDimension[i + 1];
        cudaStatus = cudaMalloc(&stridePerPackedTensorDimension_d, outputDimensions.size() * sizeof(unsigned long));
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpy(stridePerPackedTensorDimension_d,
                                stridePerPackedTensorDimension,
                                outputDimensions.size() * sizeof(unsigned long),
                                cudaMemcpyHostToDevice);
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        delete[] stridePerPackedTensorDimension;

        for (unsigned int i = 0; i < featureInputs.size(); ++i)
            allFeatureInputTensorIds.insert(featureInputs[i].value().getTensorId());
    }

    void initialize() override {
        MultiConnectionLayer::initialize();
        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) override {}

    void backProp(
        std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) override {}

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        if (errorInput.has_value()) {
            launchSplit(splitTensorErrorOutputMemoriesArray_d,
                        (half *)errorInput.value().getMemPtr(),
                        errorInput.value().getDescriptor().getTotalNumElements(),
                        errorInput.value().getDescriptor().getDimensions().size(),
                        numPresentTensors(errorOutputs),
                        axis,
                        axisElementsPerSplitTensor_d,
                        stridePerPackedTensorDimension_d,
                        stridePerSplitTensorDimension_d,
                        streams[0]);
        }

        Event readyEvent = streams[0].putEvent();
        previousLayers[0].value()->backward(errorOutputs[0], batchSize);
        for (unsigned int i = 1; i < errorOutputs.size(); ++i) {
            streams[i].waitEvent(readyEvent);
            previousLayers[i].value()->backward(errorOutputs[i], batchSize);
        }
    }

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        auto it = stillWaitingForFeatureInputTensors.find(featureInput.value().getTensorId());
        THOR_THROW_IF_FALSE(it != stillWaitingForFeatureInputTensors.end());
        stillWaitingForFeatureInputTensors.erase(it);

        if (!stillWaitingForFeatureInputTensors.empty())
            return;

        stillWaitingForFeatureInputTensors = allFeatureInputTensorIds;

        for (unsigned int i = 1; i < featureInputs.size(); ++i)
            streams[0].waitEvent(streams[i].putEvent());

        refreshFeatureInputMemoryArray(streams[0]);

        launchConcatenate((half *)featureOutputs[0].value().getMemPtr(),
                          splitTensorFeatureInputMemoriesArray_d,
                          featureOutputs[0].value().getDescriptor().getTotalNumElements(),
                          featureOutputs[0].value().getDescriptor().getDimensions().size(),
                          featureInputs.size(),
                          axis,
                          axisElementsPerSplitTensor_d,
                          stridePerPackedTensorDimension_d,
                          stridePerSplitTensorDimension_d,
                          streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[0].value()->forward(featureOutputs[0], validationPass);
    }

    void cleanup() override {
        cudaError_t cudaStatus;
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        TensorPlacement placement = featureInputs[0].value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        if (splitTensorFeatureInputMemoriesArray_d != nullptr) {
            cudaStatus = cudaFree(splitTensorFeatureInputMemoriesArray_d);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            splitTensorFeatureInputMemoriesArray_d = nullptr;
        }
        if (splitTensorErrorOutputMemoriesArray_d != nullptr) {
            cudaStatus = cudaFree(splitTensorErrorOutputMemoriesArray_d);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            splitTensorErrorOutputMemoriesArray_d = nullptr;
        }
        if (axisElementsPerSplitTensor_d != nullptr) {
            cudaStatus = cudaFree(axisElementsPerSplitTensor_d);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            axisElementsPerSplitTensor_d = nullptr;
        }
        if (stridePerPackedTensorDimension_d != nullptr) {
            cudaStatus = cudaFree(stridePerPackedTensorDimension_d);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            stridePerPackedTensorDimension_d = nullptr;
        }
        if (stridePerSplitTensorDimension_d != nullptr) {
            cudaStatus = cudaFree(stridePerSplitTensorDimension_d);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            stridePerSplitTensorDimension_d = nullptr;
        }
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
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!running);
        THOR_THROW_IF_FALSE(featureInput.has_value());

        if (!featureInputs.empty()) {
            THOR_THROW_IF_FALSE(featureInputs[0].has_value());
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == featureInputs[0].value().getPlacement());
        }

        streams.push_back(stream);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        if (backPropagateError && !isInferenceOnly())
            errorOutputs.emplace_back(featureInput.value().clone());
        else
            errorOutputs.emplace_back(std::nullopt);

        THOR_THROW_IF_FALSE(featureInputs.back().has_value());
        THOR_THROW_IF_FALSE(featureInputs.back().value().getPlacement() == featureInputs[0].value().getPlacement());
        if (errorOutputs.back().has_value()) {
            THOR_THROW_IF_FALSE(featureInputs.back().value().getDescriptor() == errorOutputs.back().value().getDescriptor());
            THOR_THROW_IF_FALSE(featureInputs.back().value().getPlacement() == errorOutputs.back().value().getPlacement());
        }
        ensureNoDeviceCrossing();

        return errorOutputs.back();
    }

   private:
    struct FeatureInputMemoryArrayRefreshArgs : public HostFunctionArgsBase {
        std::vector<half *> splitTensorFeatureInputMemories;
    };

    static void releaseFeatureInputMemoryArrayRefresh(void *) {}

    void refreshFeatureInputMemoryArray(Stream stream) {
        THOR_THROW_IF_FALSE(splitTensorFeatureInputMemoriesArray_d != nullptr);

        const int numSplitTensors = featureInputs.size();
        auto refreshArgs = std::make_unique<FeatureInputMemoryArrayRefreshArgs>();
        refreshArgs->splitTensorFeatureInputMemories.resize(numSplitTensors);
        for (int i = 0; i < numSplitTensors; ++i) {
            THOR_THROW_IF_FALSE(featureInputs[i].has_value());
            refreshArgs->splitTensorFeatureInputMemories[i] = (half *)featureInputs[i].value().getMemPtr();
        }

        cudaError_t cudaStatus = cudaMemcpyAsync(splitTensorFeatureInputMemoriesArray_d,
                                                 refreshArgs->splitTensorFeatureInputMemories.data(),
                                                 numSplitTensors * sizeof(half *),
                                                 cudaMemcpyHostToDevice,
                                                 stream);
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        stream.enqueueHostFunction(&releaseFeatureInputMemoryArrayRefresh, std::move(refreshArgs));
    }

    unsigned int axis;

    half **splitTensorFeatureInputMemoriesArray_d;
    half **splitTensorErrorOutputMemoriesArray_d;
    long *stridePerPackedTensorDimension_d;
    long *stridePerSplitTensorDimension_d;
    long *axisElementsPerSplitTensor_d;

    std::set<unsigned long> allFeatureInputTensorIds;
    std::set<unsigned long> stillWaitingForFeatureInputTensors;
};

}  // namespace ThorImplementation
