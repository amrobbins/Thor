#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"

#include "omp.h"

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

    TrainableWeightsBiasesLayer(bool hasBias) : hasBias(hasBias), weightUpdateCallback(nullptr) { clearGradientAccumulator = true; }

    Event updateWeightsAndBiases(Tensor newWeights, Optional<Tensor> newBiases, Event dataReadyEvent) {
        clearGradientAccumulator = true;
        Event weightsUpdatedEvent = weights.copyFromAsync(newWeights, dataReadyEvent);
        Optional<Event> biasesUpdatedEvent;
        if (hasBias) {
            assert(newBiases.isPresent());
            biasesUpdatedEvent = biases.get().copyFromAsync(newBiases, dataReadyEvent);
        }
        for (unsigned int i = 0; i < streams.size(); ++i) {
            streams[i].waitEvent(weightsUpdatedEvent);
            if (biasesUpdatedEvent.isPresent())
                streams[i].waitEvent(biasesUpdatedEvent);
        }

        // This layer instance is done with the updated weights memory when the following event triggers
        return streams[0].putEvent();
    }

    Event updateWeightsAndBiasesWithScaledGradient() {
        assert(gradientUpdateStream.isPresent());
        applyGradients(gradientUpdateStream, weights, weightsGradient, biases, biasesGradient);
        Event gradientAppliedEvent = gradientUpdateStream.get().putEvent();
        return gradientAppliedEvent;
    }

    virtual Optional<Stream> getGradientUpdateStream() { return gradientUpdateStream; }

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
                numEmptyErrorInputConnections -= 1;
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

        if (!inferenceOnly) {
            assert(!featureInputs.empty());
            gradientUpdateStream = Stream(featureInputs[0].get().getPlacement().getMemDevice());
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
            } else {
                assert(false);
            }
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

    virtual Tensor getWeights() { return weights; }
    virtual Optional<Tensor> getBiases() { return biases; }
    virtual Optional<Tensor> getWeightsGradient() { return weightsGradient; }
    virtual Optional<Tensor> getBiasesGradient() { return biasesGradient; }

    class Initializer {
       public:
        virtual ~Initializer() {}

        virtual void initializeWeights(TrainableWeightsBiasesLayer *trainableWeightsBiasesLayer) const = 0;
        virtual void initializeBiases(TrainableWeightsBiasesLayer *trainableWeightsBiasesLayer) const = 0;
    };

    class UniformRandomInitializer : public Initializer {
       public:
        UniformRandomInitializer(float maxValue, float minValue) : maxValue(maxValue), minValue(minValue) {
            assert(isfinite(maxValue));
            assert(isfinite(minValue));
            assert(maxValue >= minValue);
        }

        virtual void initializeWeights(TrainableWeightsBiasesLayer *trainableWeightsBiasesLayer) const {
            initialize(trainableWeightsBiasesLayer, trainableWeightsBiasesLayer->weights);
        }

        virtual void initializeBiases(TrainableWeightsBiasesLayer *trainableWeightsBiasesLayer) const {
            assert(trainableWeightsBiasesLayer->hasBias);
            initialize(trainableWeightsBiasesLayer, trainableWeightsBiasesLayer->biases);
        }

       private:
        virtual void initialize(TrainableWeightsBiasesLayer *trainableWeightsBiasesLayer, Tensor weights) const {
            Tensor buffer = weights.clone(TensorPlacement::MemDevices::CPU);

            std::uniform_real_distribution<float> distribution(minValue, maxValue);
            std::hash<int> threadNumHash;

            uint64_t totalNumWeights = weights.getDescriptor().getTotalNumElements();
            half *bufferMem = (half *)buffer.getMemPtr();
            int numProcessors = omp_get_num_procs();
            if (numProcessors > 1)
                numProcessors -= 1;
            int maxDesiredProcessors = (totalNumWeights + 99999) / 100000;
            if (numProcessors > maxDesiredProcessors)
                numProcessors = maxDesiredProcessors;
            assert(numProcessors >= 1);
            omp_set_num_threads(numProcessors);
            uint64_t weightsPerThread = (totalNumWeights + (numProcessors - 1)) / numProcessors;
#pragma omp parallel
            {
                int threadNum = omp_get_thread_num();
                std::default_random_engine generator(time(NULL) * threadNumHash(threadNum));
                uint64_t threadEnd = (threadNum + 1) * weightsPerThread;
                if (totalNumWeights < threadEnd)
                    threadEnd = totalNumWeights;
                for (uint64_t i = threadNum * weightsPerThread; i < threadEnd; ++i) {
                    bufferMem[i] = (half)distribution(generator);
                }
            }

            weights.copyFromAsync(buffer, trainableWeightsBiasesLayer->streams[0]);

            if (trainableWeightsBiasesLayer->streams.size() > 1) {
                Event weightsUpdatedEvent = trainableWeightsBiasesLayer->streams[0].putEvent();
                for (unsigned int i = 1; i < trainableWeightsBiasesLayer->streams.size(); ++i) {
                    trainableWeightsBiasesLayer->streams[i].waitEvent(weightsUpdatedEvent);
                }
            }
        }

       private:
        float maxValue;
        float minValue;
    };

    void initializeWeights(const Initializer *initializer) {
        assert(compiled);

        const UniformRandomInitializer *uniformRandomInitializer = dynamic_cast<const UniformRandomInitializer *>(initializer);

        if (uniformRandomInitializer) {
            uniformRandomInitializer->initializeWeights(this);
            return;
        }

        assert(false);
    }

    void initializeBiases(const Initializer *initializer) {
        assert(compiled);
        assert(hasBias);

        const UniformRandomInitializer *uniformRandomInitializer = dynamic_cast<const UniformRandomInitializer *>(initializer);

        if (uniformRandomInitializer) {
            uniformRandomInitializer->initializeBiases(this);
            return;
        }

        assert(false);
    }

   protected:
    const bool hasBias;
    Optional<float> learningRate;

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

    bool clearGradientAccumulator;

   private:
    virtual void backProp(
        Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream, unsigned int connectionNumber){};
};
