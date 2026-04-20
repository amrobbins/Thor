#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"

namespace ThorImplementation {

/**
 * A MultiConnectionLayer supports multiple input/output connection pairs.
 *
 */
class MultiConnectionLayer : public Layer {
   public:
    virtual ~MultiConnectionLayer() {}

    void compileImpl() override {
        Layer::compileImpl();

        numEmptyErrorInputConnections = 0;
        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].isPresent())
                allErrorInputTensorIds.insert(errorInputs[i].get().getTensorId());
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
    void replaceErrorInput(Optional<Tensor> oldErrorInput, Optional<Tensor> newErrorInput) override {
        assert(oldErrorInput.isPresent());
        bool replacementHappend = false;
        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].isEmpty() || errorInputs[i].get() != oldErrorInput.get())
                continue;
            replacementHappend = true;

            if (errorOutputs[i].isPresent()) {
                // 1. When it was populated but now should not be, then deallocate it
                // 2. When they are fused already they need to remain fused, and pass the message to check for this condition backward.
                if (newErrorInput.isEmpty() || (errorOutputs[i].get() == errorInputs[i].get())) {
                    if (previousLayers[i].isPresent())
                        previousLayers[i].get()->replaceErrorInput(errorOutputs[i], newErrorInput);
                    errorOutputs[i] = newErrorInput;
                }
                allErrorInputTensorIds.erase(errorInputs[i].get().getTensorId());
            }
            errorInputs[i] = newErrorInput;
            if (errorInputs[i].isPresent())
                allErrorInputTensorIds.insert(errorInputs[i].get().getTensorId());
        }
        assert(replacementHappend);
    }

    void forward(Optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        assert(running);

        unsigned int connectionNumber = 0;
        if (featureInput.isPresent()) {
            for (; connectionNumber < featureInputs.size(); ++connectionNumber) {
                if (featureInputs[connectionNumber].isPresent() && featureInput.get() == featureInputs[connectionNumber].get())
                    break;
            }
            assert(connectionNumber != featureInputs.size());
        } else {
            assert(featureInputs.size() - numPresentTensors(featureInputs) == 1);
            for (; connectionNumber < featureInputs.size(); ++connectionNumber) {
                if (featureInputs[connectionNumber].isEmpty())
                    break;
            }
            assert(connectionNumber != featureInputs.size());
        }

        infer(featureInputs[connectionNumber], featureOutputs[connectionNumber], streams[connectionNumber], connectionNumber);

        if (nextLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[connectionNumber].get()->forward(featureOutputs[connectionNumber], batchSize, validationPass);
    }

    void backward(Optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        assert(running);

        // Experimental - back propagation stops at empty error input
        if (errorInput.isEmpty())
            return;

        unsigned int connectionNumber = 0;
        for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
            if (errorInputs[connectionNumber].isPresent() && errorInput.get() == errorInputs[connectionNumber].get())
                break;
        }
        assert(connectionNumber != errorInputs.size());

        assert(stillWaitingForErrorInputTensors.count(errorInput.get().getTensorId()) == 1);
        stillWaitingForErrorInputTensors.erase(errorInput.get().getTensorId());

        backProp(featureInputs[connectionNumber],
                 errorInputs[connectionNumber],
                 errorOutputs[connectionNumber],
                 streams[connectionNumber],
                 connectionNumber);

        if (stillWaitingForErrorInputTensors.empty()) {
            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        }

        if (previousLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].get()->backward(errorOutputs[connectionNumber], batchSize);
    }

    // Note: A featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
    Optional<Tensor> createFeatureOutputTensor() override {
        // The default implementation just creates a clone of the corresponding feature input tensor,
        // this is the behavior of math layers etc that apply a function to the input tensor but do not reshape it.
        Optional<Tensor> previouslyConnectedFeatureInput = getFirstPresentTensor(featureInputs);
        assert(previouslyConnectedFeatureInput.isPresent());
        return previouslyConnectedFeatureInput.get().clone();
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        assert(!compiled);

        nextLayers.push_back(nextLayer);
        if (nextLayer->hasFeatureInput())
            featureOutputs.emplace_back(createFeatureOutputTensor());
        else
            featureOutputs.emplace_back(Optional<Tensor>::empty());

        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams.back(), shouldConnectToBackPropErrorIn(), loaderConnectionType));

        uint32_t tensorSlot = errorInputs.size() - 1;
        if (errorInputs[tensorSlot].isPresent()) {
            // Some logic would not function correctly if the same error input tensor were allowed to be connected multiple times,
            // so avoid that.
            for (uint32_t i = 0; i < errorInputs.size() - 1; ++i) {
                if (errorInputs[i].isPresent())
                    assert(errorInputs[i].get() != errorInputs.back().get());
            }
        } else if (previousLayers[tensorSlot].isPresent() && errorOutputs[tensorSlot].isPresent()) {
            // This layer is now being informed that this back propagation path is unused, so deallocate the tensor and inform the adjacent
            // layer in that path to do the same.
            previousLayers[tensorSlot].get()->replaceErrorInput(errorOutputs[tensorSlot], errorInputs[tensorSlot]);
            errorOutputs[tensorSlot].clear();
        }

        Optional<Tensor> firstErrorInput = getFirstPresentTensor(errorInputs);
        if (firstErrorInput.isPresent()) {
            Optional<Tensor> lastErrorInput = getLastPresentTensor(errorInputs);
            assert(firstErrorInput.get().getDescriptor() == lastErrorInput.get().getDescriptor());
            assert(firstErrorInput.get().getPlacement() == lastErrorInput.get().getPlacement());

            Optional<Tensor> lastFeatureOutput = getLastPresentTensor(featureOutputs);
            if (lastFeatureOutput.isPresent()) {
                assert(firstErrorInput.get().getDescriptor() == lastFeatureOutput.get().getDescriptor());
                assert(firstErrorInput.get().getPlacement() == lastFeatureOutput.get().getPlacement());
            }
        }

        ensureNoDeviceCrossing();
    }

    virtual Optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
        // backPropagateError allows the previous layer to specify that it does not support back propagation,
        // inferenceOnly means that even though back propagation may be supported, we are not using it since we are not training.
        if (backPropagateError && !isInferenceOnly())
            return getFirstPresentTensor(featureInputs).get().clone();
        else
            return Optional<Tensor>::empty();
    }

    Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        assert(!compiled);

        Optional<Tensor> previouslyConnectedFeatureInput = getFirstPresentTensor(featureInputs);
        if (previouslyConnectedFeatureInput.isPresent() && featureInput.isPresent()) {
            assert(featureInput.get().getDescriptor() == previouslyConnectedFeatureInput.get().getDescriptor());
            assert(featureInput.get().getPlacement() == previouslyConnectedFeatureInput.get().getPlacement());
        }

        streams.push_back(stream);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        errorOutputs.emplace_back(createErrorOutputTensor(backPropagateError, errorOutputs.size()));

        Optional<Tensor> lastFeatureInput = getLastPresentTensor(featureInputs);
        Optional<Tensor> firstFeatureInput = getFirstPresentTensor(featureInputs);
        Optional<Tensor> lastErrorOutput = getLastPresentTensor(errorOutputs);
        if (firstFeatureInput.isPresent()) {
            assert(lastFeatureInput.get().getDescriptor() == firstFeatureInput.get().getDescriptor());
            assert(lastFeatureInput.get().getPlacement() == firstFeatureInput.get().getPlacement());
            if (lastErrorOutput.isPresent()) {
                assert(lastFeatureInput.get().getDescriptor() == lastErrorOutput.get().getDescriptor());
                assert(lastFeatureInput.get().getPlacement() == lastErrorOutput.get().getPlacement());
            }
        } else if (lastErrorOutput.isPresent()) {
            Optional<Tensor> firstErrorOutput = getFirstPresentTensor(errorOutputs);
            assert(lastErrorOutput.get().getDescriptor() == firstErrorOutput.get().getDescriptor());
            assert(lastErrorOutput.get().getPlacement() == firstErrorOutput.get().getPlacement());
        }
        ensureNoDeviceCrossing();

        return errorOutputs.back();
    }

    virtual void ensureNoDeviceCrossing(Optional<TensorPlacement> expectedPlacement = Optional<TensorPlacement>::empty()) {
        Optional<Tensor> lastFeatureInput = getLastPresentTensor(featureInputs);
        Optional<Tensor> lastErrorOutput = getLastPresentTensor(errorOutputs);
        Optional<Tensor> lastErrorInput = getLastPresentTensor(errorInputs);
        Optional<Tensor> lastFeatureOutput = getLastPresentTensor(featureOutputs);

        if (lastFeatureInput.isPresent() && lastFeatureOutput.isPresent())
            assert(lastFeatureInput.get().getPlacement() == lastFeatureOutput.get().getPlacement());
        if (lastFeatureInput.isPresent() && lastErrorInput.isPresent())
            assert(lastFeatureInput.get().getPlacement() == lastErrorInput.get().getPlacement());
        if (lastFeatureInput.isPresent() && lastErrorOutput.isPresent())
            assert(lastFeatureInput.get().getPlacement() == lastErrorOutput.get().getPlacement());

        if (lastFeatureOutput.isPresent() && lastErrorInput.isPresent())
            assert(lastFeatureOutput.get().getPlacement() == lastErrorInput.get().getPlacement());
        if (lastFeatureOutput.isPresent() && lastErrorOutput.isPresent())
            assert(lastFeatureOutput.get().getPlacement() == lastErrorOutput.get().getPlacement());

        if (lastErrorInput.isPresent() && lastErrorOutput.isPresent())
            assert(lastErrorInput.get().getPlacement() == lastErrorOutput.get().getPlacement());

        if (expectedPlacement.isPresent()) {
            if (lastFeatureInput.isPresent())
                assert(lastFeatureInput.get().getPlacement() == expectedPlacement.get());
            if (lastFeatureOutput.isPresent())
                assert(lastFeatureOutput.get().getPlacement() == expectedPlacement.get());
            if (lastErrorInput.isPresent())
                assert(lastErrorInput.get().getPlacement() == expectedPlacement.get());
            if (lastErrorOutput.isPresent())
                assert(lastErrorOutput.get().getPlacement() == expectedPlacement.get());
        }
    }

    TensorPlacement getPlacement() override {
        Optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        Optional<Tensor> aFeatureOutput = getFirstPresentTensor(featureOutputs);
        Optional<Tensor> anErrorInput = getFirstPresentTensor(errorInputs);
        Optional<Tensor> anErrorOutput = getFirstPresentTensor(errorOutputs);

        if (anErrorInput.isPresent()) {
            return anErrorInput.get().getPlacement();
        } else if (anErrorOutput.isPresent()) {
            return anErrorOutput.get().getPlacement();
        } else if (aFeatureInput.isPresent()) {
            return aFeatureInput.get().getPlacement();
        } else if (aFeatureOutput.isPresent()) {
            return aFeatureOutput.get().getPlacement();
        } else {
            return TensorPlacement(TensorPlacement::MemDevices::CPU);
        }
    }

    bool isBackPropStub() override { return getFirstPresentTensor(errorOutputs).isEmpty(); }

    virtual std::vector<Optional<Tensor>> getFeatureInputs() { return featureInputs; }
    virtual std::vector<Optional<Tensor>> getFeatureOutputs() { return featureOutputs; }
    virtual std::vector<Optional<Tensor>> getErrorInputs() { return errorInputs; }
    virtual std::vector<Optional<Tensor>> getErrorOutputs() { return errorOutputs; }
    virtual std::vector<Optional<Layer *>> getNextLayers() { return nextLayers; }
    virtual std::vector<Stream> getStreams() { return streams; }

    // compute the fan in for one element of a batch
    uint64_t getFanIn() override { return 1; }

    // compute the fan out for one element of a batch
    uint64_t getFanOut() override {
        Optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        Optional<Tensor> aFeatureOutput = getFirstPresentTensor(featureOutputs);
        assert(aFeatureInput.isPresent());
        assert(aFeatureOutput.isPresent());

        uint64_t inElementsPerExample = aFeatureInput.get().getTotalNumElements() / aFeatureInput.get().getDimensions()[0];
        uint64_t outElementsPerExample = aFeatureOutput.get().getTotalNumElements() / aFeatureOutput.get().getDimensions()[0];
        return std::max<uint64_t>(1, outElementsPerExample / inElementsPerExample);
    }

    static Optional<Tensor> getFirstPresentTensor(std::vector<Optional<Tensor>> tensors) {
        for (auto it = tensors.begin(); it != tensors.end(); ++it) {
            if (it->isPresent())
                return *it;
        }
        return Optional<Tensor>::empty();
    }

    static Optional<Tensor> getLastPresentTensor(std::vector<Optional<Tensor>> tensors) {
        for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
            if (it->isPresent())
                return *it;
        }
        return Optional<Tensor>::empty();
    }

    static unsigned int numPresentTensors(std::vector<Optional<Tensor>> tensors) {
        unsigned int numPresent = 0;
        for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
            if (it->isPresent())
                numPresent += 1;
        }
        return numPresent;
    }

    virtual Optional<Tensor> getFeatureInput() {
        assert(featureInputs.size() == 1);
        return featureInputs[0];
    }
    virtual Optional<Tensor> getFeatureOutput() {
        assert(featureInputs.size() == 1);
        return featureInputs[0];
    }
    virtual Optional<Tensor> getErrorInput() {
        assert(featureInputs.size() == 1);
        return featureInputs[0];
    }
    virtual Optional<Tensor> getErrorOutput() {
        assert(featureInputs.size() == 1);
        return featureInputs[0];
    }
    virtual Optional<Layer *> getNextLayer() {
        assert(nextLayers.size() == 1);
        return nextLayers[0];
    }
    virtual Stream getStream() {
        assert(streams.size() == 1);
        return streams[0];
    }

   protected:
    std::set<unsigned long> allErrorInputTensorIds;
    std::set<unsigned long> stillWaitingForErrorInputTensors;
    unsigned int numEmptyErrorInputConnections;

    std::vector<Optional<Tensor>> featureInputs;
    std::vector<Optional<Tensor>> featureOutputs;
    std::vector<Optional<Tensor>> errorInputs;
    std::vector<Optional<Tensor>> errorOutputs;
    std::vector<Stream> streams;
    std::vector<Optional<Layer *>> nextLayers;
    std::vector<Optional<Layer *>> previousLayers;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) = 0;

    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) = 0;

   private:
    // Hide Layer's single instance members since they will not be used by classes derived from MultiConnectionLayer
    using Layer::errorInput;
    using Layer::errorOutput;
    using Layer::featureInput;
    using Layer::featureOutput;
    using Layer::nextLayer;
    using Layer::previousLayer;
    using Layer::stream;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) { assert(false); }
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) { assert(false); }
    virtual Optional<Tensor> createErrorOutputTensor(bool backPropagateError) { assert(false); }
};

}  // namespace ThorImplementation
