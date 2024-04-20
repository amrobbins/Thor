#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/Optional.h"

namespace ThorImplementation {

/**
 * A TrainableWeightsBiasesLayer has a trainable weights tensor and possibly a trainable biases tensor.
 *
 * Updates to weights and biases during training is the responsibility of the connected Optimizer.
 *
 */
class TrainableWeightsBiasesLayer : public MultiConnectionLayer {
   public:
    virtual ~TrainableWeightsBiasesLayer() { clearOptimizer(); }

    // stampedId defaults to -1 so that it is not necessary to set it to some value during testing
    // where there are not multiple stamped networks
    // All real id's will have positive values
    TrainableWeightsBiasesLayer(bool hasBias, int64_t stampedId = -1) : hasBias(hasBias), usingSharedWeights(false), stampedId(stampedId) {}

    struct SharedWeightsPackage {
        Tensor weights;
        Optional<Tensor> biases;

        std::vector<Tensor> otherSharedMem;
    };

    TrainableWeightsBiasesLayer(SharedWeightsPackage sharedWeightsPackage, int64_t stampedId = -1)
        : hasBias(sharedWeightsPackage.biases.isPresent()), usingSharedWeights(false), stampedId(stampedId) {
        weights = sharedWeightsPackage.weights;
        biases = sharedWeightsPackage.biases;
    }

    uint64_t getStampedId() { return stampedId; }

    virtual void createWeightsIfNecessary() = 0;

    virtual void assignProjectedWeightsTensor(Tensor projectedWeights, Optional<Tensor> projectedBiases) {
        this->projectedWeights = projectedWeights;
        this->projectedBiases = projectedBiases;
    }

    virtual Optional<Tensor> getProjectedWeightsTensor() { return projectedWeights; }

    virtual Optional<Tensor> getProjectedBiasesTensor() { return projectedBiases; }

    virtual void forward(Optional<Tensor> featureInput, bool validationPass) {
        assert(running);

        // Forward direction must wait for weights update to finish before inference is called
        assert(streams.size() > 0);

        if (!isInferenceOnly() && projectedWeights.isPresent() && !projectedWeightsInitialized) {
            projectedWeights.get().copyFromAsync(weights, streams[0]);
            if (projectedBiases.isPresent()) {
                assert(biases.isPresent());
                projectedBiases.get().copyFromAsync(biases, streams[0]);
            }
            Event copiedFinished = streams[0].putEvent();
            for (uint32_t i = 1; i < streams.size(); ++i)
                streams[i].waitEvent(copiedFinished);
            projectedWeightsInitialized = true;
        }

        if (optimizer.isPresent()) {
            Stream gradientUpdateStream = optimizer.get()->getGradientUpdateStream();
            Event gradientUpdateFinished = gradientUpdateStream.putEvent();
            for (uint32_t i = 0; i < streams.size(); ++i)
                streams[i].waitEvent(gradientUpdateFinished);
        }

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

        infer(
            featureInputs[connectionNumber], featureOutputs[connectionNumber], streams[connectionNumber], connectionNumber, validationPass);

        if (nextLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[connectionNumber].get()->forward(featureOutputs[connectionNumber], validationPass);
    }

    // Note: the setWeightsAndBiases is not used during optimization, it is there to support loading a trained model.
    Event setWeightsAndBiases(Tensor newWeights, Optional<Tensor> newBiases, Event dataReadyEvent) {
        Stream stream = streams[0];
        stream.waitEvent(dataReadyEvent);
        weights.copyFromAsync(newWeights, stream);
        Event weightsUpdatedEvent = stream.putEvent();
        Optional<Event> biasesUpdatedEvent;
        if (hasBias) {
            assert(newBiases.isPresent());
            biases.get().copyFromAsync(newBiases, stream);
            biasesUpdatedEvent = stream.putEvent();
        }
        for (unsigned int i = 0; i < streams.size(); ++i) {
            streams[i].waitEvent(weightsUpdatedEvent);
            if (biasesUpdatedEvent.isPresent())
                streams[i].waitEvent(biasesUpdatedEvent);
        }

        // This layer instance is finished with the newWeights tensor when the following event triggers
        return streams[0].putEvent();
    }

    virtual SharedWeightsPackage getSharedWeightsPackage() {
        SharedWeightsPackage sharedWeightsPackage;
        sharedWeightsPackage.weights = weights;
        sharedWeightsPackage.biases = biases;

        return sharedWeightsPackage;
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);

        // Experimental - back propagation stops at empty error input
        if (errorInput.isEmpty())
            return;

        // When receiving the first errorInput of the set, clear the errorOutput and gradients.
        // For the other errorInputs, accumulate the values.
        bool accumulateValues = stillWaitingForErrorInputTensors != allErrorInputTensorIds;

        // Using the errorInput tensor, determine which connection backward is being called for.
        unsigned int connectionNumber = 0;
        for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
            if (errorInputs[connectionNumber].isPresent() && errorInput.get() == errorInputs[connectionNumber].get())
                break;
        }
        assert(connectionNumber != errorInputs.size());

        assert(optimizer.isPresent());
        optimizer.get()->getGradientUpdateStream().waitEvent(streams[connectionNumber].putEvent());

        assert(stillWaitingForErrorInputTensors.count(errorInput.get().getTensorId()) == 1);
        stillWaitingForErrorInputTensors.erase(errorInput.get().getTensorId());

        backProp(featureInputs[connectionNumber],
                 errorInputs[connectionNumber],
                 errorOutputs[connectionNumber],
                 // Since they all update a single gradients tensor, gradient updates must run sequentially to one another.
                 streams[connectionNumber],
                 connectionNumber,
                 accumulateValues);

        if (stillWaitingForErrorInputTensors.empty()) {
            stillWaitingForErrorInputTensors = allErrorInputTensorIds;
        }

        if (previousLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].get()->backward(errorOutputs[connectionNumber]);
    }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        Optional<Tensor> errorOutput =
            MultiConnectionLayer::connectToPreviousLayer(previousLayer, featureInput, stream, backPropagateError, connectionType);
        createWeightsIfNecessary();
        return errorOutput;
    }

    virtual void parentCompile() {
        MultiConnectionLayer::parentCompile();

        if (!isInferenceOnly()) {
            assert(!featureInputs.empty());
        }
    }

    // errorInput must be ready on data stream when calling computeWeightsGradient
    // gradientUpdateStream will first synchronize with data stream.
    // weightsGradient and biasesGradient will be ready at end of gradientUpdateStream
    virtual void computeWeightsGradient(Optional<Tensor> weightsGradient,
                                        Optional<Tensor> biasesGradient,
                                        Optional<Tensor> featureIn,
                                        Optional<Tensor> errorIn,
                                        Stream gradientUpdateStream,
                                        bool accumulateGradient) = 0;

    virtual Tensor getWeights() { return weights; }
    virtual Optional<Tensor> getBiases() { return biases; }

    // If an optimizer is set, it will not be replaced even if setOptimizer() is called again, which allows the user to override
    // the network level default optimizer for particular layers.
    // To replace an optimizer after it has been attached, you need to call setOptimizer again with an empty Optional
    virtual void setOptimizer(Optional<std::shared_ptr<Optimizer>> newOptimizer) {
        if (newOptimizer.isEmpty()) {
            this->optimizer.clear();
            this->optimizerShared.clear();
        } else if (this->optimizer.isEmpty()) {
            assert(optimizerShared.isEmpty());
            this->optimizerShared = newOptimizer;
            this->optimizer = newOptimizer.get().get();
        }
    }

    virtual Optional<std::shared_ptr<Optimizer>> getOptimizer() { return optimizerShared; }

    void clearOptimizer() {
        optimizer.clear();
        assert(optimizer.isEmpty());
    }

   protected:
    const bool hasBias;
    const bool usingSharedWeights;

    Tensor weights;
    Optional<Tensor> biases;

    // Note: not virtual, layers need to implement infer(..., weights, biases)
    void infer(
        Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber, bool validationPass) {
        if (projectedWeights.isPresent() && !isInferenceOnly() && !validationPass)
            infer(inputTensor, outputTensor, stream, connectionNumber, projectedWeights, projectedBiases);
        else
            infer(inputTensor, outputTensor, stream, connectionNumber, weights, biases);
    }

    virtual void infer(Optional<Tensor> inputTensor,
                       Optional<Tensor> outputTensor,
                       Stream stream,
                       unsigned int connectionNumber,
                       Tensor weights,
                       Optional<Tensor> biases) = 0;

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) = 0;

   protected:
    Optional<std::shared_ptr<Optimizer>> optimizerShared;
    Optional<Optimizer *> optimizer;

    Optional<Tensor> projectedWeights;
    Optional<Tensor> projectedBiases;
    bool projectedWeightsInitialized = false;

   private:
    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber){};

    // stampedId is used to identify which layers correspond to which other layers across multiple stamps of the same network.
    int64_t stampedId;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) {
        assert(false);
    }
};

}  // namespace ThorImplementation
