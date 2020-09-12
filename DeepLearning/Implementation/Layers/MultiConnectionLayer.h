#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"

/**
 * A MultiConnectionLayer supports multiple input/output connection pairs.
 *
 */
class MultiConnectionLayer : public Layer {
   public:
    virtual ~MultiConnectionLayer() {}

    virtual void parentCompile() {
        Layer::parentCompile();

        for (unsigned int i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].isPresent())
                allErrorInputTensorIds.insert(errorInputs[i].get().getTensorId());
            else
                numEmptyErrorInputConnections += 1;
        }
    }

    virtual void parentInitialize() {
        Layer::parentInitialize();

        stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        stillWaitingForNumEmptyErrorInputConnections = numEmptyErrorInputConnections;
    }

    virtual void forward(Optional<Tensor> featureInput) {
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
        nextLayers[connectionNumber].get()->forward(featureOutputs[connectionNumber]);
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);

        unsigned int connectionNumber = 0;
        if (errorInput.isPresent()) {
            for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
                if (errorInputs[connectionNumber].isPresent() && errorInput.get() == errorInputs[connectionNumber].get())
                    break;
            }
            assert(connectionNumber != errorInputs.size());
        } else {
            assert(errorInputs.size() - numPresentTensors(errorInputs) == 1);
            for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
                if (errorInputs[connectionNumber].isEmpty())
                    break;
            }
            assert(connectionNumber != errorInputs.size());
        }

        backProp(featureInputs[connectionNumber],
                 errorInputs[connectionNumber],
                 errorOutputs[connectionNumber],
                 streams[connectionNumber],
                 connectionNumber);

        if (errorInput.isPresent()) {
            assert(stillWaitingForErrorInputTensors.count(errorInput.get().getTensorId()) == 1);
            stillWaitingForErrorInputTensors.erase(errorInput.get().getTensorId());
        } else {
            assert(stillWaitingForNumEmptyErrorInputConnections != 0);
            stillWaitingForNumEmptyErrorInputConnections -= 1;
        }
        if (stillWaitingForErrorInputTensors.empty() && stillWaitingForNumEmptyErrorInputConnections == 0) {
            processedAllErrorInputs(streams[0].putEvent());
            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
            stillWaitingForNumEmptyErrorInputConnections = numEmptyErrorInputConnections;
        }

        if (previousLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].get()->backward(errorOutputs[connectionNumber]);
    }

    virtual void processedAllErrorInputs(Event allProcessedEvent) {}

    // Note: A featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
    virtual Optional<Tensor> createFeatureOutputTensor() {
        // The default implementation just creates a clone of the corresponding feature input tensor,
        // this is the behavior of math layers etc that apply a function to the input tensor but do not reshape it.
        Optional<Tensor> previouslyConnectedFeatureInput = getFirstPresentTensor(featureInputs);
        assert(previouslyConnectedFeatureInput.isPresent());
        return previouslyConnectedFeatureInput.get().clone();
    }

    virtual void connectToNextLayer(Layer *nextLayer, int connectionType = 0) {
        assert(!compiled);

        nextLayers.push_back(nextLayer);
        if (nextLayer->hasFeatureInput())
            featureOutputs.emplace_back(createFeatureOutputTensor());
        else
            featureOutputs.emplace_back(Optional<Tensor>::empty());

        errorInputs.emplace_back(nextLayer->connectToPreviousLayer(
            this, featureOutputs.back(), streams.back(), shouldConnectToBackPropErrorIn(), connectionType));

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

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        assert(!compiled);

        Optional<Tensor> previouslyConnectedFeatureInput = getFirstPresentTensor(featureInputs);
        if (previouslyConnectedFeatureInput.isPresent() && featureInput.isPresent()) {
            assert(featureInput.get().getDescriptor() == previouslyConnectedFeatureInput.get().getDescriptor());
            assert(featureInput.get().getPlacement() == previouslyConnectedFeatureInput.get().getPlacement());
        }

        streams.push_back(stream);

        previousLayers.push_back(previousLayer);
        featureInputs.emplace_back(featureInput);
        // backPropagateError allows the previous layer to specify that it does not support back propagation,
        // inferenceOnly means that even though back propagation may be supported, we are not using it since we are not training.
        if (backPropagateError && !isInferenceOnly())
            errorOutputs.emplace_back(featureInput.get().clone());
        else
            errorOutputs.emplace_back(Optional<Tensor>::empty());

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

    virtual void ensureNoDeviceCrossing() {
        Optional<Tensor> lastFeatureInput = getLastPresentTensor(featureInputs);
        Optional<Tensor> lastErrorOutput = getLastPresentTensor(errorOutputs);
        Optional<Tensor> lastErrorInput = getLastPresentTensor(errorInputs);
        Optional<Tensor> lastFeatureOutput = getLastPresentTensor(featureOutputs);

        if (lastFeatureInput.isPresent() && lastErrorOutput.isPresent())
            assert(lastFeatureInput.get().getPlacement() == lastErrorOutput.get().getPlacement());
        if (lastFeatureOutput.isPresent() && lastErrorInput.isPresent())
            assert(featureOutputs.back().get().getPlacement() == errorInputs.back().get().getPlacement());
    }

    virtual bool isBackPropStub() { return getFirstPresentTensor(errorOutputs).isEmpty(); }

    virtual vector<Optional<Tensor>> getFeatureInputs() { return featureInputs; }
    virtual vector<Optional<Tensor>> getFeatureOutputs() { return featureOutputs; }
    virtual vector<Optional<Tensor>> getErrorInputs() { return errorInputs; }
    virtual vector<Optional<Tensor>> getErrorOutputs() { return errorOutputs; }
    virtual vector<Optional<Layer *>> getNextLayers() { return nextLayers; }
    virtual vector<Stream> getStreams() { return streams; }

   protected:
    set<unsigned long> allErrorInputTensorIds;
    set<unsigned long> stillWaitingForErrorInputTensors;
    unsigned int numEmptyErrorInputConnections;
    unsigned int stillWaitingForNumEmptyErrorInputConnections;

    vector<Optional<Tensor>> featureInputs;
    vector<Optional<Tensor>> featureOutputs;
    vector<Optional<Tensor>> errorInputs;
    vector<Optional<Tensor>> errorOutputs;
    vector<Stream> streams;
    vector<Optional<Layer *>> nextLayers;
    vector<Optional<Layer *>> previousLayers;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) = 0;

    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber) = 0;

    Optional<Tensor> getFirstPresentTensor(vector<Optional<Tensor>> tensors) {
        for (auto it = tensors.begin(); it != tensors.end(); ++it) {
            if (it->isPresent())
                return *it;
        }
        return Optional<Tensor>::empty();
    }

    Optional<Tensor> getLastPresentTensor(vector<Optional<Tensor>> tensors) {
        for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
            if (it->isPresent())
                return *it;
        }
        return Optional<Tensor>::empty();
    }

    unsigned int numPresentTensors(vector<Optional<Tensor>> tensors) {
        unsigned int numPresent = 0;
        for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
            if (it->isPresent())
                numPresent += 1;
        }
        return numPresent;
    }

   private:
    // Hide Layer's single instance members since they will not be used by classes derived from MultiConnectionLayer
    using Layer::errorInput;
    using Layer::errorOutput;
    using Layer::featureInput;
    using Layer::featureOutput;
    using Layer::nextLayer;
    using Layer::previousLayer;
    using Layer::stream;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream){};
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream){};
    virtual Optional<Tensor> getFeatureInput() { return featureInput; }
    virtual Optional<Tensor> getFeatureOutput() { return featureOutput; }
    virtual Optional<Tensor> getErrorInput() { return errorInput; }
    virtual Optional<Tensor> getErrorOutput() { return errorOutput; }
    virtual Optional<Layer *> getNextLayer() { return nextLayer; }
    virtual Stream getStream() { return stream; }
};
