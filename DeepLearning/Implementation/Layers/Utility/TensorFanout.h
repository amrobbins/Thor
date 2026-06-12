#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

#include <unordered_set>

namespace ThorImplementation {

// TensorFanout has a single input tensor that is connected to multiple output tensors.
// New streams are created for outputs 1+
class TensorFanout : public MultiConnectionLayer {
   public:
    TensorFanout() {}

    ~TensorFanout() override {}

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        // If this is not the first connection
        if (errorInputs.size() == streams.size()) {
            streams.emplace_back(streams[0].getGpuNum());
        }
        // The set of errorInputs is checked during compile to make optimizations when possible
        errorInputs.push_back(nextLayer->connectToPreviousLayer(
            this, featureInputs[0], streams.back(), shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType));
        nextLayers.push_back(nextLayer);
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        // This special case layer can only be connected to on a single input feature tensor
        THOR_THROW_IF_FALSE(featureInputs.empty());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(errorOutputs.empty());
        THOR_THROW_IF_FALSE(streams.empty());
        THOR_THROW_IF_FALSE(previousLayers.empty());
        featureInputs.push_back(featureInput);
        featureOutputs.push_back(featureInput);
        if (backPropagateError && !isInferenceOnly())
            errorOutputs.push_back(featureInput.value().clone());
        else
            errorOutputs.push_back(std::nullopt);
        streams.push_back(stream);
        previousLayers.push_back(previousLayer);

        return errorOutputs[0];
    }

    std::optional<Tensor> createFeatureOutputTensor() override { return std::nullopt; }

    // allocate anything needed for execution, choose optimal kernels, etc.
    void compileImpl() override {
        MultiConnectionLayer::compileImpl();
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        TensorPlacement placement = featureInputs[0].value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc(&errorInputArray_d, numPresentTensors(errorInputs) * sizeof(half *));
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);

        if (numPresentTensors(errorInputs) > 0) {
            half **errorInputArray = new half *[numPresentTensors(errorInputs)];
            uint32_t j = 0;
            for (unsigned int i = 0; i < errorInputs.size(); ++i) {
                if (errorInputs[i].has_value()) {
                    errorInputArray[j] = (half *)errorInputs[i].value().getMemPtr();
                    allErrorInputTensorIds.insert(errorInputs[i].value().getTensorId());
                    ++j;
                }
            }
            cudaStatus =
                cudaMemcpy(errorInputArray_d, errorInputArray, numPresentTensors(errorInputs) * sizeof(half *), cudaMemcpyHostToDevice);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
            delete[] errorInputArray;
        }

        // When there is only one layer that back propagates (for example if the fanout is just to connect the tensor to a network output),
        // then the errorInput replaces the errorOutput so that back prop is a nop.
        // When there is no layer that back propagates, then prune the existing back prop path
        // The third option is that there are multiple present errorInputs, in that case they will be summed together and passed as errorOut
        if (numPresentTensors(errorInputs) == 1 && numPresentTensors(errorOutputs) == 1) {
            // Fuse
            THOR_THROW_IF_FALSE(previousLayers[0].has_value());
            previousLayers[0].value()->replaceErrorInput(errorOutputs[0], getFirstPresentTensor(errorInputs));
            errorOutputs[0] = getFirstPresentTensor(errorInputs);
        } else if (numPresentTensors(errorInputs) == 0 && numPresentTensors(errorOutputs) == 1) {
            // Prune
            std::optional<Tensor> newErrorOutput = std::nullopt;
            THOR_THROW_IF_FALSE(previousLayers[0].has_value());
            previousLayers[0].value()->replaceErrorInput(errorOutputs[0], newErrorOutput);
            errorOutputs[0] = newErrorOutput;
        }
    }

    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {
        THOR_THROW_IF_FALSE(oldErrorInput.has_value());
        bool replacementHappend = false;
        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (!errorInputs[i].has_value() || errorInputs[i].value() != oldErrorInput.value())
                continue;
            replacementHappend = true;

            // During compile, tensorFanout will replace the errorInput of the previous layer when:
            // 1. There is just one populated errorInput to tensorFanout, in this case it can be fused.
            // 2. There are no populated errorInputs to tensorFanout, in this case the path can be pruned.
            errorInputs[i] = newErrorInput;
        }
        THOR_THROW_IF_FALSE(replacementHappend);
    }

    // release any resources that are used for execution and need to be released
    void cleanup() override {
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);
        THOR_THROW_IF_FALSE(featureInputs[0].has_value());
        TensorPlacement placement = featureInputs[0].value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        ScopedGpu scopedGpu(featureInputs[0].value().getPlacement().getDeviceNum());
        if (errorInputArray_d != nullptr) {
            cudaError_t cudaStatus = cudaFree(errorInputArray_d);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        }
        errorInputArray_d = nullptr;
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) override {}
    void backProp(
        std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) override {}

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value() == featureInputs[0]);

        // Synchronize all streams at the point at which inputTensor is populated.
        Event inputReadyEvent = streams[0].putEvent();
        for (unsigned int i = 1; i < streams.size(); ++i)
            streams[i].waitEvent(inputReadyEvent);

        std::unordered_set<Layer *> forwardedLayers;
        for (unsigned int i = 0; i < nextLayers.size(); ++i) {
            Layer *nextLayer = nextLayers[i].value();
            if (!forwardedLayers.insert(nextLayer).second) {
                continue;
            }
            nextLayer->forward(featureInput, validationPass, batchSize);
        }
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        if (!errorOutputs[0].has_value())
            return;

        // Experimental - back propagation stops at empty error input
        if (!errorInput.has_value())
            return;

        if (numPresentTensors(errorInputs) == 1) {
            // NOP (errorInput[0] is connected to errorOutput[0])
            // Just sync streams and initiate back prop
            for (unsigned int i = 1; i < errorInputs.size(); ++i)
                streams[0].waitEvent(streams[i].putEvent());
            previousLayers[0].value()->backward(errorOutputs[0], batchSize);
            return;
        }

        if (errorInput.has_value()) {
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

        sum((half *)errorOutputs[0].value().getMemPtr(),
            errorInputArray_d,
            numPresentTensors(errorInputs),
            (uint64_t)errorOutputs[0].value().getDescriptor().getTotalNumElements(),
            streams[0]);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[0].value()->backward(errorOutputs[0], batchSize);
    }

    uint32_t getDownStreamFanoutMultiplier() override {
        uint32_t multiplier = 0;
        for (uint32_t i = 0; i < nextLayers.size(); ++i) {
            if (nextLayers[i].has_value()) {
                multiplier += nextLayers[i].value()->getDownStreamFanoutMultiplier();
            }
        }
        if (multiplier == 0) {
            // return a 1 to avoid possible divide by 0
            multiplier = 1;
        }
        return multiplier;
    }

   protected:
    half **errorInputArray_d = nullptr;
};

}  // namespace ThorImplementation
