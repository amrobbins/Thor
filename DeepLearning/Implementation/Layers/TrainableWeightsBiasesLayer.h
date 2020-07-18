#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"

/**
 * A TrainableWeightsBiasesLayer has a trainable weights tensor and possibly a trainable biases tensor.
 *
 * TrainableWeightsBiasesLayer support multiple connections, in that case the gradient will accumulate
 * as the sum of the gradients from the backProp step of each connection.
 *
 */
class TrainableWeightsBiasesLayer : public MultiConnectionLayer {
   public:
    TrainableWeightsBiasesLayer(bool inferenceOnly, bool hasBias, Optional<float> learningRate)
        : inferenceOnly(inferenceOnly), hasBias(hasBias), weightUpdateCallback(nullptr) {
        if (!inferenceOnly)
            assert(learningRate.isPresent());
        this->learningRate = learningRate;
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

        if (errorInputs.size() > 1) {
            unique_lock<mutex> lck(mtx);

            bool firstBackProp = (stillWaitingForErrorInputTensors == allErrorInputTensorIds) &&
                                 (stillWaitingForNumEmptyErrorInputConnections == numEmptyErrorInputConnections);
            bool accumulateGradient = !firstBackProp;

            backProp(featureInputs[connectionNumber],
                     errorInputs[connectionNumber],
                     errorOutputs[connectionNumber],
                     // Since they all update a single gradients tensor, gradient updates must run sequentially to one another.
                     gradientUpdateStream,
                     streams[connectionNumber],
                     connectionNumber,
                     accumulateGradient);

            if (errorInput.isPresent()) {
                assert(stillWaitingForErrorInputTensors.count(errorInput.get().getTensorId()) == 1);
                stillWaitingForErrorInputTensors.erase(errorInput.get().getTensorId());
            } else {
                assert(stillWaitingForNumEmptyErrorInputConnections != 0);
                numEmptyErrorInputConnections -= 1;
            }
            if (stillWaitingForErrorInputTensors.empty() && stillWaitingForNumEmptyErrorInputConnections == 0) {
                processedAllErrorInputs(gradientUpdateStream);
                stillWaitingForErrorInputTensors = allErrorInputTensorIds;
                stillWaitingForNumEmptyErrorInputConnections = numEmptyErrorInputConnections;
            }
        }

        if (previousLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].get()->backward(errorOutputs[connectionNumber]);
    }

    virtual void parentInitialize() {
        MultiConnectionLayer::parentInitialize();

        if (weightsGradient.isPresent()) {
            assert(!featureInputs.empty());
            gradientUpdateStream = Stream(featureInputs[0].get().getPlacement().getMemDevice());
        }
    }

    virtual void processedAllErrorInputs(Stream stream) {
        applyGradients(stream, weights, weightsGradient, biases, biasesGradient);
        Event gradientAppliedEvent = stream.putEvent();

        assert(weightUpdateCallback != nullptr);
        weightUpdateCallback(gradientAppliedEvent, weights, weightsGradient, biases, biasesGradient);
    }

    // Default implementation simply updates weights by learningRate*gradient, does not apply momentum or anything else.
    virtual void applyGradients(
        Stream stream, Tensor weights, Tensor weightsGradient, Optional<Tensor> biases, Optional<Tensor> biasesGradient) {
        sumScale((half*)weights.getMemPtr(),
                 (half*)weights.getMemPtr(),
                 (half*)weightsGradient.getMemPtr(),
                 learningRate,
                 weights.getDescriptor().getTotalNumElements(),
                 stream);
        if (hasBias) {
            assert(biases.isPresent());
            assert(biasesGradient.isPresent());
            sumScale((half*)biases.get().getMemPtr(),
                     (half*)biases.get().getMemPtr(),
                     (half*)biasesGradient.get().getMemPtr(),
                     learningRate,
                     biases.get().getDescriptor().getTotalNumElements(),
                     stream);
        }
    }

    // Callback can be used to directly apply the gradients to the weights,
    // or to copy the gradients to another device for accumulation for example.
    virtual void setCallBackWhenGradientsReady(void (*weightUpdateCallback)(
        Event readyEvent, Optional<Tensor> weights, Optional<Tensor> gradients, Optional<Tensor> biases, Optional<Tensor> biasesGradient)) {
        assert(!compiled);
        this->weightUpdateCallback = weightUpdateCallback;
    }

    virtual void setLearningRate(float learningRate) { this->learningRate = learningRate; }

    virtual Optional<Tensor> getWeights() { return weights; }
    virtual Optional<Tensor> getBiases() { return biases; }

   protected:
    const bool inferenceOnly;
    const bool hasBias;
    float learningRate;

    Tensor weights;
    Optional<Tensor> weightsGradient;
    Optional<Tensor> biases;
    Optional<Tensor> biasesGradient;

    Optional<Stream> gradientUpdateStream;

    void (*weightUpdateCallback)(
        Event readyEvent, Optional<Tensor> weights, Optional<Tensor> gradients, Optional<Tensor> biases, Optional<Tensor> biasesGradient);

    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream gradientStream,
                          Optional<Stream> dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) = 0;

   private:
    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber){};
};
