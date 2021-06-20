#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

namespace ThorImplementation {

/**
 * A TrainableWeightsBiasesLayer has a trainable weights tensor and possibly a trainable biases tensor.
 *
 * TrainableWeightsBiasesLayer support multiple connections, in that case the gradient will accumulate
 * as the sum of the gradients from the backProp step of each connection.
 *
 */
class TrainableWeightsBiasesLayer : public MultiConnectionLayer {
   public:
    virtual ~TrainableWeightsBiasesLayer() {}

    TrainableWeightsBiasesLayer(bool hasBias)
        : hasBias(hasBias), usingSharedWeights(false), weightUpdateCallback(nullptr), clearGradientAccumulator(true) {}

    struct SharedWeightsPackage {
        Tensor weights;
        Optional<Tensor> weightsGradient;
        Optional<Tensor> biases;
        Optional<Tensor> biasesGradient;

        Stream gradientUpdateStream;

        vector<Tensor> otherSharedMem;
    };

    TrainableWeightsBiasesLayer(SharedWeightsPackage sharedWeightsPackage)
        : hasBias(sharedWeightsPackage.biases.isPresent()),
          usingSharedWeights(false),
          weightUpdateCallback(nullptr),
          clearGradientAccumulator(true) {
        weights = sharedWeightsPackage.weights;
        weightsGradient = sharedWeightsPackage.weightsGradient;
        biases = sharedWeightsPackage.biases;
        biasesGradient = sharedWeightsPackage.biasesGradient;

        gradientUpdateStream = sharedWeightsPackage.gradientUpdateStream;
    }

    virtual void createWeightsIfNecessary() = 0;

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

        createWeightsIfNecessary();

        return errorOutputs.back();
    }

    Event updateWeightsAndBiases(Tensor newWeights, Optional<Tensor> newBiases, Event dataReadyEvent) {
        clearGradientAccumulator = true;
        Stream stream = gradientUpdateStream;
        if (!stream.isInitialized())
            stream = streams[0];
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

    Event updateWeightsAndBiasesWithScaledGradient() {
        assert(gradientUpdateStream.isInitialized());
        applyGradients(gradientUpdateStream, weights, weightsGradient, biases, biasesGradient);
        Event gradientAppliedEvent = gradientUpdateStream.putEvent();
        return gradientAppliedEvent;
    }

    virtual Stream getGradientUpdateStream() { return gradientUpdateStream; }

    virtual SharedWeightsPackage getSharedWeightsPackage() {
        SharedWeightsPackage sharedWeightsPackage;
        sharedWeightsPackage.weights = weights;
        sharedWeightsPackage.weightsGradient = weightsGradient;
        sharedWeightsPackage.biases = biases;
        sharedWeightsPackage.biasesGradient = biasesGradient;

        sharedWeightsPackage.gradientUpdateStream = gradientUpdateStream;

        return sharedWeightsPackage;
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
                 // Since they all update a single gradients tensor, gradient updates must run sequentially to one another.
                 gradientUpdateStream,
                 streams[connectionNumber],
                 connectionNumber,
                 !clearGradientAccumulator);
        clearGradientAccumulator = false;

        if (errorInputs.size() > 1) {
            if (errorInput.isPresent()) {
                assert(stillWaitingForErrorInputTensors.count(errorInput.get().getTensorId()) == 1);
                stillWaitingForErrorInputTensors.erase(errorInput.get().getTensorId());
            } else {
                assert(stillWaitingForNumEmptyErrorInputConnections != 0);
                stillWaitingForNumEmptyErrorInputConnections -= 1;
            }
            if (stillWaitingForErrorInputTensors.empty() && stillWaitingForNumEmptyErrorInputConnections == 0) {
                stillWaitingForErrorInputTensors = allErrorInputTensorIds;
                stillWaitingForNumEmptyErrorInputConnections = numEmptyErrorInputConnections;
            }
        }

        if (previousLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].get()->backward(errorOutputs[connectionNumber]);
    }

    virtual void parentCompile() {
        MultiConnectionLayer::parentCompile();

        if (!isInferenceOnly()) {
            assert(!featureInputs.empty());
            if (!usingSharedWeights) {
                // gradient upate streams have low priority so that this type of parallel work tends to build up
                gradientUpdateStream = Stream(featureInputs[0].get().getPlacement().getMemDevice(), Stream::Priority::LOW);
            }
        }
    }

    // Default implementation simply updates weights by learningRate*gradient, does not apply momentum or anything else.
    virtual void applyGradients(
        Stream stream, Tensor weights, Tensor weightsGradient, Optional<Tensor> biases, Optional<Tensor> biasesGradient) {
        assert(learningRate.isPresent());

        if (weights.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 &&
            weightsGradient.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            sumScale((half *)weights.getMemPtr(),
                     (half *)weights.getMemPtr(),
                     (half *)weightsGradient.getMemPtr(),
                     -1.0f * learningRate,  // subtract the gradient, scaled by the learning rate, from the weights
                     weights.getDescriptor().getTotalNumElements(),
                     stream);
        } else if (weights.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 &&
                   weightsGradient.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            sumScale((half *)weights.getMemPtr(),
                     (half *)weights.getMemPtr(),
                     (float *)weightsGradient.getMemPtr(),
                     -1.0f * learningRate,  // subtract the gradient, scaled by the learning rate, from the weights
                     weights.getDescriptor().getTotalNumElements(),
                     stream);
        } else if (weights.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 &&
                   weightsGradient.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            sumScale((float *)weights.getMemPtr(),
                     (float *)weights.getMemPtr(),
                     (float *)weightsGradient.getMemPtr(),
                     -1.0f * learningRate,  // subtract the gradient, scaled by the learning rate, from the weights
                     weights.getDescriptor().getTotalNumElements(),
                     stream);
        } else {
            assert(false);
        }
        if (hasBias) {
            assert(biases.isPresent());
            assert(biasesGradient.isPresent());
            if (biases.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 &&
                biasesGradient.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                sumScale((half *)biases.get().getMemPtr(),
                         (half *)biases.get().getMemPtr(),
                         (half *)biasesGradient.get().getMemPtr(),
                         -1.0f * learningRate,  // subtract the gradient, scaled by the learning rate, from the biases
                         biases.get().getDescriptor().getTotalNumElements(),
                         stream);
            } else if (biases.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP16 &&
                       biasesGradient.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                sumScale((half *)biases.get().getMemPtr(),
                         (half *)biases.get().getMemPtr(),
                         (float *)biasesGradient.get().getMemPtr(),
                         -1.0f * learningRate,  // subtract the gradient, scaled by the learning rate, from the biases
                         biases.get().getDescriptor().getTotalNumElements(),
                         stream);
            } else if (biases.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32 &&
                       biasesGradient.get().getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                sumScale((float *)biases.get().getMemPtr(),
                         (float *)biases.get().getMemPtr(),
                         (float *)biasesGradient.get().getMemPtr(),
                         -1.0f * learningRate,  // subtract the gradient, scaled by the learning rate, from the biases
                         biases.get().getDescriptor().getTotalNumElements(),
                         stream);
            } else {
                assert(false);
            }
        }

        clearGradientAccumulator = true;
    }

    // Callback can be used to directly apply the gradients to the weights,
    // or to copy the gradients to another device for accumulation for example.
    virtual void setCallBackWhenGradientsReady(void (*weightUpdateCallback)(
        Event readyEvent, Optional<Tensor> weights, Optional<Tensor> gradients, Optional<Tensor> biases, Optional<Tensor> biasesGradient)) {
        assert(!compiled);
        this->weightUpdateCallback = weightUpdateCallback;
    }

    virtual void setLearningRate(float learningRate) { this->learningRate = learningRate; }
    virtual float getLearningRate() { return learningRate; }

    virtual Tensor getWeights() { return weights; }
    virtual Optional<Tensor> getBiases() { return biases; }
    virtual Optional<Tensor> getWeightsGradient() { return weightsGradient; }
    virtual Optional<Tensor> getBiasesGradient() { return biasesGradient; }

   protected:
    const bool hasBias;
    const bool usingSharedWeights;
    Optional<float> learningRate;

    Tensor weights;
    Optional<Tensor> weightsGradient;
    Optional<Tensor> biases;
    Optional<Tensor> biasesGradient;

    Stream gradientUpdateStream;

    void (*weightUpdateCallback)(
        Event readyEvent, Optional<Tensor> weights, Optional<Tensor> gradients, Optional<Tensor> biases, Optional<Tensor> biasesGradient);

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream gradientStream,
                          Stream dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) = 0;

    bool clearGradientAccumulator;

   private:
    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber){};
};

}  // namespace ThorImplementation
