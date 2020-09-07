#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

// TensorFanout has a single input tensor that is connected to multiple output tensors.
// New streams are created for outputs 1+
class TensorFanout : public MultiConnectionLayer {
   public:
    virtual ~TensorFanout() {}

    virtual void connectToNextLayer(Layer *nextLayer, int connectionType = 0) {
        // There is no actual movement of data on the inputs,
        // instead streams are created to fork the execution.
        if (featureOutputs.empty()) {
            errorInputs.push_back(nextLayer->connectToPreviousLayer(this, featureInputs[0], streams[0], true, connectionType));
        } else {
            streams.emplace_back(streams[0].getGpuNum());
            errorInputs.push_back(nextLayer->connectToPreviousLayer(this, featureInputs[0], streams.back(), true, connectionType));
        }
        nextLayers.push_back(nextLayer);
    }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        // This special case layer can only be connected to on a single input feature tensor
        assert(featureInputs.empty());
        assert(featureInput.isPresent());
        assert(errorOutputs.empty());
        assert(streams.empty());
        assert(previousLayers.empty());
        featureInputs.push_back(featureInput);
        if (backPropagateError)
            errorOutputs.push_back(featureInput.get().clone());
        else
            errorOutputs.push_back(Optional<Tensor>::empty());
        streams.push_back(stream);
        previousLayers.push_back(previousLayer);

        return errorOutputs[0];
    }

    virtual Optional<Tensor> createFeatureOutputTensor() { return Tensor(); }

    // allocate anything needed for execution, choose optimal kernels, etc.
    virtual void compile() {
        assert(featureInputs.size() == 1);
        assert(featureInputs[0].isPresent());
        TensorPlacement placement = featureInputs[0].get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].get().getPlacement().getDeviceNum());
        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc(&errorInputArray_d, errorInputs.size() * sizeof(half *));
        assert(cudaStatus == cudaSuccess);

        half **errorInputArray = new half *[numPresentTensors(errorInputs)];
        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].isPresent()) {
                errorInputArray[i] = (half *)errorInputs[i].get().getMemPtr();
                allErrorInputTensorIds.insert(errorInputs[i].get().getTensorId());
            }
        }
        cudaStatus = cudaMemcpy(errorInputArray_d, errorInputArray, errorInputs.size() * sizeof(half *), cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
        delete[] errorInputArray;
    }

    // initialize weights using the configured initializer. In general set any initial values.
    virtual void initialize() {}

    // release any resources that are used for execution and need to be released
    virtual void cleanup() {
        assert(featureInputs.size() == 1);
        assert(featureInputs[0].isPresent());
        TensorPlacement placement = featureInputs[0].get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].get().getPlacement().getDeviceNum());
        cudaError_t cudaStatus = cudaFree(errorInputArray_d);
        assert(cudaStatus == cudaSuccess);
        errorInputArray_d = nullptr;
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {}
    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) {}

    virtual void forward(Optional<Tensor> featureInput) {
        assert(featureInput.isPresent());
        assert(featureInput.get() == featureInputs[0]);

        // Synchronize all streams at the point at which inputTensor is populated.
        Event inputReadyEvent = streams[0].putEvent();
        for (unsigned int i = 1; i < streams.size(); ++i)
            streams[i].waitEvent(inputReadyEvent);

        for (unsigned int i = 0; i < nextLayers.size(); ++i)
            nextLayers[i].get()->forward(featureInput);
    }

    virtual void backward(Optional<Tensor> errorInput) {
        if (errorInputs.size() > 1) {
            // Locked section
            unique_lock<mutex> lck(mtx);

            if (errorInput.isPresent()) {
                assert(stillWaitingForErrorInputTensors.count(errorInput.get().getTensorId()) == 1);
                stillWaitingForErrorInputTensors.erase(errorInput.get().getTensorId());
            } else {
                assert(stillWaitingForNumEmptyErrorInputConnections != 0);
                numEmptyErrorInputConnections -= 1;
            }
            if (!stillWaitingForErrorInputTensors.empty() || stillWaitingForNumEmptyErrorInputConnections != 0)
                return;

            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
            stillWaitingForNumEmptyErrorInputConnections = numEmptyErrorInputConnections;
        }

        for (unsigned int i = 1; i < errorInputs.size(); ++i)
            streams[0].waitEvent(streams[i].putEvent());

        sum((half *)errorOutputs[0].get().getMemPtr(),
            errorInputArray_d,
            errorInputs.size(),
            (int)errorOutputs[0].get().getDescriptor().getTotalNumElements(),
            streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[0].get()->backward(errorOutputs[0]);
    }

   protected:
    half **errorInputArray_d;
};
