#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

// TensorFanout has a single input tensor that is connected to multiple output tensors.
// New streams are created for outputs 1+
class TensorFanout : public MultiConnectionLayer {
   public:
    TensorFanout() {}

    virtual ~TensorFanout() {}

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        // If this is not the first connection
        if (errorInputs.size() == streams.size()) {
            streams.emplace_back(streams[0].getGpuNum());
        }
        // The set of errorInputs is checked during compile to make optimizations when possible
        errorInputs.push_back(nextLayer->connectToPreviousLayer(
            this, featureInputs[0], streams.back(), shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType));
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
        featureOutputs.push_back(featureInput);
        if (backPropagateError && !isInferenceOnly())
            errorOutputs.push_back(featureInput.get().clone());
        else
            errorOutputs.push_back(Optional<Tensor>::empty());
        streams.push_back(stream);
        previousLayers.push_back(previousLayer);

        return errorOutputs[0];
    }

    virtual Optional<Tensor> createFeatureOutputTensor() { return Optional<Tensor>::empty(); }

    // allocate anything needed for execution, choose optimal kernels, etc.
    virtual void compile() {
        assert(featureInputs.size() == 1);
        assert(featureInputs[0].isPresent());
        TensorPlacement placement = featureInputs[0].get().getPlacement();
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].get().getPlacement().getDeviceNum());
        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc(&errorInputArray_d, numPresentTensors(errorInputs) * sizeof(half *));
        assert(cudaStatus == cudaSuccess);

        if (numPresentTensors(errorInputs) > 0) {
            half **errorInputArray = new half *[numPresentTensors(errorInputs)];
            uint32_t j = 0;
            for (unsigned int i = 0; i < errorInputs.size(); ++i) {
                if (errorInputs[i].isPresent()) {
                    errorInputArray[j] = (half *)errorInputs[i].get().getMemPtr();
                    allErrorInputTensorIds.insert(errorInputs[i].get().getTensorId());
                    ++j;
                }
            }
            cudaStatus =
                cudaMemcpy(errorInputArray_d, errorInputArray, numPresentTensors(errorInputs) * sizeof(half *), cudaMemcpyHostToDevice);
            assert(cudaStatus == cudaSuccess);
            delete[] errorInputArray;
        }

        // When there is only one layer that back propagates (for example if the fanout is just to connect the tensor to a network output),
        // then the errorInput replaces the errorOutput so that back prop is a nop.
        // When there is no layer that back propagates, then prune the existing back prop path
        // The third option is that there are multiple present errorInputs, in that case they will be summed together and passed as errorOut
        if (numPresentTensors(errorInputs) == 1 && numPresentTensors(errorOutputs) == 1) {
            // Fuse
            assert(previousLayers[0].isPresent());
            previousLayers[0].get()->replaceErrorInput(errorOutputs[0], getFirstPresentTensor(errorInputs));
            errorOutputs[0] = getFirstPresentTensor(errorInputs);
        } else if (numPresentTensors(errorInputs) == 0 && numPresentTensors(errorOutputs) == 1) {
            // Prune
            Optional<Tensor> newErrorOutput = Optional<Tensor>::empty();
            assert(previousLayers[0].isPresent());
            previousLayers[0].get()->replaceErrorInput(errorOutputs[0], newErrorOutput);
            errorOutputs[0] = newErrorOutput;
        }
    }

    virtual void replaceErrorInput(Optional<Tensor> oldErrorInput, Optional<Tensor> newErrorInput) {
        assert(oldErrorInput.isPresent());
        bool replacementHappend = false;
        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].isEmpty() || errorInputs[i].get() != oldErrorInput.get())
                continue;
            replacementHappend = true;

            // During compile, tensorFanout will replace the errorInput of the previous layer when:
            // 1. There is just one populated errorInput to tensorFanout, in this case it can be fused.
            // 2. There are no populated errorInputs to tensorFanout, in this case the path can be pruned.
            errorInputs[i] = newErrorInput;
        }
        assert(replacementHappend);
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

    virtual void forward(Optional<Tensor> featureInput, bool validationPass) {
        assert(featureInput.isPresent());
        assert(featureInput.get() == featureInputs[0]);

        // Synchronize all streams at the point at which inputTensor is populated.
        Event inputReadyEvent = streams[0].putEvent();
        for (unsigned int i = 1; i < streams.size(); ++i)
            streams[i].waitEvent(inputReadyEvent);

        for (unsigned int i = 0; i < nextLayers.size(); ++i)
            nextLayers[i].get()->forward(featureInput, validationPass);
    }

    virtual void backward(Optional<Tensor> errorInput) {
        if (errorOutputs[0].isEmpty())
            return;

        // Experimental - back propagation stops at empty error input
        if (errorInput.isEmpty())
            return;

        if (numPresentTensors(errorInputs) == 1) {
            // NOP (errorInput[0] is connected to errorOutput[0])
            // Just sync streams and initiate back prop
            for (unsigned int i = 1; i < errorInputs.size(); ++i)
                streams[0].waitEvent(streams[i].putEvent());
            previousLayers[0].get()->backward(errorOutputs[0]);
            return;
        }

        if (errorInput.isPresent()) {
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

        sum((half *)errorOutputs[0].get().getMemPtr(),
            errorInputArray_d,
            numPresentTensors(errorInputs),
            (uint64_t)errorOutputs[0].get().getDescriptor().getTotalNumElements(),
            streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[0].get()->backward(errorOutputs[0]);
    }

    virtual uint32_t getDownStreamFanoutMultiplier() {
        uint32_t multiplier = 0;
        for (uint32_t i = 0; i < nextLayers.size(); ++i) {
            if (nextLayers[i].isPresent()) {
                multiplier += nextLayers[i].get()->getDownStreamFanoutMultiplier();
            }
        }
        if (multiplier == 0) {
            // return a 1 to avoid possible divide by 0
            multiplier = 1;
        }
        return multiplier;
    }

   protected:
    half **errorInputArray_d;
};

}  // namespace ThorImplementation
