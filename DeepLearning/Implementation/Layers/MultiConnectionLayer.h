#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"

namespace ThorImplementation {

/**
 * A MultiConnectionLayer supports multiple input/output connection pairs.
 *
 */
class MultiConnectionLayer : public Layer {
   public:
    ~MultiConnectionLayer() override {}

    void compileImpl() override {
        Layer::compileImpl();

        numEmptyErrorInputConnections = 0;
        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].has_value())
                allErrorInputTensorIds.insert(errorInputs[i].value().getTensorId());
            else
                numEmptyErrorInputConnections += 1;
        }
    }

    void initialize() override {
        Layer::initialize();
        stillWaitingForErrorInputTensors = allErrorInputTensorIds;
    }

    // For situations where the error input should just pass through to the error output of the next layer,
    // this method is used to avoid duplicating the tensor and unnecessary data movement.
    // They may not have the same number of error inputs and error outputs, consider tensorFanout.
    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {
        THOR_THROW_IF_FALSE(oldErrorInput.has_value());
        bool replacementHappend = false;
        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (!errorInputs[i].has_value() || errorInputs[i].value() != oldErrorInput.value())
                continue;
            replacementHappend = true;

            if (errorOutputs[i].has_value()) {
                // 1. When it was populated but now should not be, then deallocate it
                // 2. When they are fused already they need to remain fused, and pass the message to check for this condition backward.
                if (!newErrorInput.has_value() || (errorOutputs[i].value() == errorInputs[i].value())) {
                    if (previousLayers[i].has_value())
                        previousLayers[i].value()->replaceErrorInput(errorOutputs[i], newErrorInput);
                    errorOutputs[i] = newErrorInput;
                }
                allErrorInputTensorIds.erase(errorInputs[i].value().getTensorId());
            }
            errorInputs[i] = newErrorInput;
            if (errorInputs[i].has_value())
                allErrorInputTensorIds.insert(errorInputs[i].value().getTensorId());
        }
        THOR_THROW_IF_FALSE(replacementHappend);
    }

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);

        unsigned int connectionNumber = 0;
        if (featureInput.has_value()) {
            for (; connectionNumber < featureInputs.size(); ++connectionNumber) {
                if (featureInputs[connectionNumber].has_value() && featureInput.value() == featureInputs[connectionNumber].value())
                    break;
            }
            THOR_THROW_IF_FALSE(connectionNumber != featureInputs.size());
        } else {
            THOR_THROW_IF_FALSE(featureInputs.size() - numPresentTensors(featureInputs) == 1);
            for (; connectionNumber < featureInputs.size(); ++connectionNumber) {
                if (!featureInputs[connectionNumber].has_value())
                    break;
            }
            THOR_THROW_IF_FALSE(connectionNumber != featureInputs.size());
        }

        infer(featureInputs[connectionNumber], featureOutputs[connectionNumber], streams[connectionNumber], connectionNumber);

        if (!nextLayers[connectionNumber].has_value())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[connectionNumber].value()->forward(featureOutputs[connectionNumber], validationPass, batchSize);
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);

        // Experimental - back propagation stops at empty error input
        if (!errorInput.has_value())
            return;

        unsigned int connectionNumber = 0;
        for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
            if (errorInputs[connectionNumber].has_value() && errorInput.value() == errorInputs[connectionNumber].value())
                break;
        }
        THOR_THROW_IF_FALSE(connectionNumber != errorInputs.size());

        THOR_THROW_IF_FALSE(stillWaitingForErrorInputTensors.count(errorInput.value().getTensorId()) == 1);
        stillWaitingForErrorInputTensors.erase(errorInput.value().getTensorId());

        backProp(featureInputs[connectionNumber],
                 errorInputs[connectionNumber],
                 errorOutputs[connectionNumber],
                 streams[connectionNumber],
                 connectionNumber);

        if (stillWaitingForErrorInputTensors.empty()) {
            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        }

        if (!previousLayers[connectionNumber].has_value())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].value()->backward(errorOutputs[connectionNumber], batchSize);
    }

    // Note: A featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
    std::optional<Tensor> createFeatureOutputTensor() override {
        // The default implementation just creates a clone of the corresponding feature input tensor,
        // this is the behavior of math layers etc that apply a function to the input tensor but do not reshape it.
        std::optional<Tensor> previouslyConnectedFeatureInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(previouslyConnectedFeatureInput.has_value());
        return previouslyConnectedFeatureInput.value().clone();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);

        nextLayers.push_back(nextLayer);
        if (nextLayer->hasFeatureInput())
            featureOutputs.emplace_back(createFeatureOutputTensor());
        else
            featureOutputs.emplace_back(std::nullopt);

        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams.back(), shouldConnectToBackPropErrorIn(), loaderConnectionType));

        uint32_t tensorSlot = errorInputs.size() - 1;
        if (errorInputs[tensorSlot].has_value()) {
            // Some logic would not function correctly if the same error input tensor were allowed to be connected multiple times,
            // so avoid that.
            for (uint32_t i = 0; i < errorInputs.size() - 1; ++i) {
                if (errorInputs[i].has_value())
                    THOR_THROW_IF_FALSE(errorInputs[i].value() != errorInputs.back().value());
            }
        } else if (previousLayers[tensorSlot].has_value() && errorOutputs[tensorSlot].has_value()) {
            // This layer is now being informed that this back propagation path is unused, so deallocate the tensor and inform the adjacent
            // layer in that path to do the same.
            previousLayers[tensorSlot].value()->replaceErrorInput(errorOutputs[tensorSlot], errorInputs[tensorSlot]);
            errorOutputs[tensorSlot].reset();
        }

        std::optional<Tensor> firstErrorInput = getFirstPresentTensor(errorInputs);
        if (firstErrorInput.has_value()) {
            std::optional<Tensor> lastErrorInput = getLastPresentTensor(errorInputs);
            THOR_THROW_IF_FALSE(firstErrorInput.value().getDescriptor() == lastErrorInput.value().getDescriptor());
            THOR_THROW_IF_FALSE(firstErrorInput.value().getPlacement() == lastErrorInput.value().getPlacement());

            std::optional<Tensor> lastFeatureOutput = getLastPresentTensor(featureOutputs);
            if (lastFeatureOutput.has_value()) {
                THOR_THROW_IF_FALSE(firstErrorInput.value().getDescriptor() == lastFeatureOutput.value().getDescriptor());
                THOR_THROW_IF_FALSE(firstErrorInput.value().getPlacement() == lastFeatureOutput.value().getPlacement());
            }
        }

        ensureNoDeviceCrossing();
    }

    virtual std::optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
        // backPropagateError allows the previous layer to specify that it does not support back propagation,
        // inferenceOnly means that even though back propagation may be supported, we are not using it since we are not training.
        if (backPropagateError && !isInferenceOnly())
            return getFirstPresentTensor(featureInputs).value().clone();
        else
            return std::nullopt;
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);

        std::optional<Tensor> previouslyConnectedFeatureInput = getFirstPresentTensor(featureInputs);
        if (previouslyConnectedFeatureInput.has_value() && featureInput.has_value()) {
            THOR_THROW_IF_FALSE(featureInput.value().getDescriptor() == previouslyConnectedFeatureInput.value().getDescriptor());
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == previouslyConnectedFeatureInput.value().getPlacement());
        }

        streams.push_back(stream);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        errorOutputs.emplace_back(createErrorOutputTensor(backPropagateError, errorOutputs.size()));

        std::optional<Tensor> lastFeatureInput = getLastPresentTensor(featureInputs);
        std::optional<Tensor> firstFeatureInput = getFirstPresentTensor(featureInputs);
        std::optional<Tensor> lastErrorOutput = getLastPresentTensor(errorOutputs);
        if (firstFeatureInput.has_value()) {
            THOR_THROW_IF_FALSE(lastFeatureInput.value().getDescriptor() == firstFeatureInput.value().getDescriptor());
            THOR_THROW_IF_FALSE(lastFeatureInput.value().getPlacement() == firstFeatureInput.value().getPlacement());
            if (lastErrorOutput.has_value()) {
                THOR_THROW_IF_FALSE(lastFeatureInput.value().getDescriptor() == lastErrorOutput.value().getDescriptor());
                THOR_THROW_IF_FALSE(lastFeatureInput.value().getPlacement() == lastErrorOutput.value().getPlacement());
            }
        } else if (lastErrorOutput.has_value()) {
            std::optional<Tensor> firstErrorOutput = getFirstPresentTensor(errorOutputs);
            THOR_THROW_IF_FALSE(lastErrorOutput.value().getDescriptor() == firstErrorOutput.value().getDescriptor());
            THOR_THROW_IF_FALSE(lastErrorOutput.value().getPlacement() == firstErrorOutput.value().getPlacement());
        }
        ensureNoDeviceCrossing();

        return errorOutputs.back();
    }

    virtual void ensureNoDeviceCrossing(std::optional<TensorPlacement> expectedPlacement = std::nullopt) {
        std::optional<Tensor> lastFeatureInput = getLastPresentTensor(featureInputs);
        std::optional<Tensor> lastErrorOutput = getLastPresentTensor(errorOutputs);
        std::optional<Tensor> lastErrorInput = getLastPresentTensor(errorInputs);
        std::optional<Tensor> lastFeatureOutput = getLastPresentTensor(featureOutputs);

        if (lastFeatureInput.has_value() && lastFeatureOutput.has_value())
            THOR_THROW_IF_FALSE(lastFeatureInput.value().getPlacement() == lastFeatureOutput.value().getPlacement());
        if (lastFeatureInput.has_value() && lastErrorInput.has_value())
            THOR_THROW_IF_FALSE(lastFeatureInput.value().getPlacement() == lastErrorInput.value().getPlacement());
        if (lastFeatureInput.has_value() && lastErrorOutput.has_value())
            THOR_THROW_IF_FALSE(lastFeatureInput.value().getPlacement() == lastErrorOutput.value().getPlacement());

        if (lastFeatureOutput.has_value() && lastErrorInput.has_value())
            THOR_THROW_IF_FALSE(lastFeatureOutput.value().getPlacement() == lastErrorInput.value().getPlacement());
        if (lastFeatureOutput.has_value() && lastErrorOutput.has_value())
            THOR_THROW_IF_FALSE(lastFeatureOutput.value().getPlacement() == lastErrorOutput.value().getPlacement());

        if (lastErrorInput.has_value() && lastErrorOutput.has_value())
            THOR_THROW_IF_FALSE(lastErrorInput.value().getPlacement() == lastErrorOutput.value().getPlacement());

        if (expectedPlacement.has_value()) {
            if (lastFeatureInput.has_value())
                THOR_THROW_IF_FALSE(lastFeatureInput.value().getPlacement() == expectedPlacement.value());
            if (lastFeatureOutput.has_value())
                THOR_THROW_IF_FALSE(lastFeatureOutput.value().getPlacement() == expectedPlacement.value());
            if (lastErrorInput.has_value())
                THOR_THROW_IF_FALSE(lastErrorInput.value().getPlacement() == expectedPlacement.value());
            if (lastErrorOutput.has_value())
                THOR_THROW_IF_FALSE(lastErrorOutput.value().getPlacement() == expectedPlacement.value());
        }
    }

    TensorPlacement getPlacement() override {
        std::optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        std::optional<Tensor> aFeatureOutput = getFirstPresentTensor(featureOutputs);
        std::optional<Tensor> anErrorInput = getFirstPresentTensor(errorInputs);
        std::optional<Tensor> anErrorOutput = getFirstPresentTensor(errorOutputs);

        if (anErrorInput.has_value()) {
            return anErrorInput.value().getPlacement();
        } else if (anErrorOutput.has_value()) {
            return anErrorOutput.value().getPlacement();
        } else if (aFeatureInput.has_value()) {
            return aFeatureInput.value().getPlacement();
        } else if (aFeatureOutput.has_value()) {
            return aFeatureOutput.value().getPlacement();
        } else {
            return TensorPlacement(TensorPlacement::MemDevices::CPU);
        }
    }

    bool isBackPropStub() override { return !getFirstPresentTensor(errorOutputs).has_value(); }

    virtual std::vector<std::optional<Tensor>> getFeatureInputs() { return featureInputs; }
    virtual std::vector<std::optional<Tensor>> getFeatureOutputs() { return featureOutputs; }
    virtual std::vector<std::optional<Tensor>> getErrorInputs() { return errorInputs; }
    virtual std::vector<std::optional<Tensor>> getErrorOutputs() { return errorOutputs; }
    virtual std::vector<std::optional<Layer *>> getNextLayers() { return nextLayers; }
    virtual std::vector<Stream> getStreams() { return streams; }

    // compute the fan in for one element of a batch
    uint64_t getFanIn() override { return 1; }

    // compute the fan out for one element of a batch
    uint64_t getFanOut() override {
        std::optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        std::optional<Tensor> aFeatureOutput = getFirstPresentTensor(featureOutputs);
        THOR_THROW_IF_FALSE(aFeatureInput.has_value());
        THOR_THROW_IF_FALSE(aFeatureOutput.has_value());

        uint64_t inElementsPerExample = aFeatureInput.value().getTotalNumElements() / aFeatureInput.value().getDimensions()[0];
        uint64_t outElementsPerExample = aFeatureOutput.value().getTotalNumElements() / aFeatureOutput.value().getDimensions()[0];
        return std::max<uint64_t>(1, outElementsPerExample / inElementsPerExample);
    }

    static std::optional<Tensor> getFirstPresentTensor(std::vector<std::optional<Tensor>> tensors) {
        for (auto it = tensors.begin(); it != tensors.end(); ++it) {
            if (it->has_value())
                return *it;
        }
        return std::nullopt;
    }

    static std::optional<Tensor> getLastPresentTensor(std::vector<std::optional<Tensor>> tensors) {
        for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
            if (it->has_value())
                return *it;
        }
        return std::nullopt;
    }

    static unsigned int numPresentTensors(std::vector<std::optional<Tensor>> tensors) {
        unsigned int numPresent = 0;
        for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
            if (it->has_value())
                numPresent += 1;
        }
        return numPresent;
    }

    std::optional<Tensor> getFeatureInput() override {
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);
        return featureInputs[0];
    }
    std::optional<Tensor> getFeatureOutput() override {
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }
    std::optional<Tensor> getErrorInput() override {
        THOR_THROW_IF_FALSE(errorInputs.size() == 1);
        return errorInputs[0];
    }
    std::optional<Tensor> getErrorOutput() override {
        THOR_THROW_IF_FALSE(errorOutputs.size() == 1);
        return errorOutputs[0];
    }
    std::optional<Layer *> getNextLayer() override {
        THOR_THROW_IF_FALSE(nextLayers.size() == 1);
        return nextLayers[0];
    }
    Stream getStream() override {
        THOR_THROW_IF_FALSE(streams.size() == 1);
        return streams[0];
    }

   protected:
    std::set<unsigned long> allErrorInputTensorIds;
    std::set<unsigned long> stillWaitingForErrorInputTensors;
    unsigned int numEmptyErrorInputConnections;

    std::vector<std::optional<Tensor>> featureInputs;
    std::vector<std::optional<Tensor>> featureOutputs;
    std::vector<std::optional<Tensor>> errorInputs;
    std::vector<std::optional<Tensor>> errorOutputs;
    std::vector<Stream> streams;
    std::vector<std::optional<Layer *>> nextLayers;
    std::vector<std::optional<Layer *>> previousLayers;

    virtual void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) = 0;

    virtual void backProp(
        std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) = 0;

   private:
    // Hide Layer's single instance members since they will not be used by classes derived from MultiConnectionLayer
    using Layer::errorInput;
    using Layer::errorOutput;
    using Layer::featureInput;
    using Layer::featureOutput;
    using Layer::nextLayer;
    using Layer::previousLayer;
    using Layer::stream;

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override { THOR_UNREACHABLE(); }
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override { THOR_UNREACHABLE(); }
    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override { THOR_UNREACHABLE(); }
};

}  // namespace ThorImplementation
