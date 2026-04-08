#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "Utilities/Common/Optional.h"

namespace ThorImplementation {

/**
 *  Simplifying assumptions for this round:
 *     1. A layer is stamped only once.
 *         If there were more than one stamp, I would have to accumulate into a master gradient and then distribute the weights.
 */
// : public MultiConnectionLayer
class TrainableLayer : public Parameterizable {
   public:
    virtual ~TrainableLayer() = default;

    // All real stamped id's will have positive values
    TrainableLayer(int64_t stampedId = -1) : stampedId(stampedId) {}

    virtual void compileImpl() {
        // MultiConnectionLayer::compileImpl();

        std::unordered_set<uint64_t> dataStreamsSet;
        for (Stream dataStream : streams) {
            uint64_t dataStreamId = dataStream.getId();
            auto [it, inserted] = dataStreamsSet.insert(dataStreamId);
            if (inserted) {
                // first time seeing this stream
                uniqueDataStreams.push_back(dataStream);
            }
        }

        Optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        assert(aFeatureInput.isPresent());

        for (const auto &parameter : parameters) {
            if (!parameter->isTrainable())
                continue;
            if (gradientUpdateStream.isEmpty()) {
                gradientUpdateStream = Stream::getNextGradientUpdateStream(aFeatureInput.get().getPlacement().getDeviceNum());
                break;
            }
        }
        for (const auto &parameter : parameters) {
            parameter->compile(aFeatureInput, gradientUpdateStream, isInferenceOnly(), getFanIn(), getFanOut());
        }

        numBackwardConnections = 0;
        for (const auto &errorInput : errorInputs) {
            if (errorInput.isPresent())
                numBackwardConnections += 1;
        }
    }

   public:
    // FIXME: Temporarily moved here from layer
    bool inferenceOnly = false;
    virtual bool isInferenceOnly() { return inferenceOnly; }
    bool running = true;
    static bool isBackPropStub() { return false; }
    std::vector<Optional<Tensor>> featureInputs;
    std::vector<Optional<Tensor>> featureOutputs;
    std::vector<Optional<Tensor>> errorInputs;
    std::vector<Optional<Tensor>> errorOutputs;
    std::vector<Stream> streams;
    std::vector<Optional<Layer *>> nextLayers;
    std::vector<Optional<Layer *>> previousLayers;
    static Optional<Tensor> getFirstPresentTensor(std::vector<Optional<Tensor>> tensors) {
        for (auto it = tensors.begin(); it != tensors.end(); ++it) {
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
    // FIXME: Temporarily moved here from layer

    // compute the fan in for one element of a batch
    virtual uint64_t getFanIn() {
        uint64_t totalFanIn = 0;
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (featureInputs[i].isPresent()) {
                Tensor aFeatureInput = featureInputs[i];
                std::vector<uint64_t> inputDimensions = aFeatureInput.getDescriptor().getDimensions();
                uint64_t fanIn = 1;
                for (uint32_t j = 1; j < inputDimensions.size(); ++j) {
                    fanIn *= inputDimensions[j];
                }
                totalFanIn += fanIn;
            }
        }
        if (totalFanIn == 0) {
            // return a 1 to avoid possible divide by 0
            totalFanIn = 1;
        }
        return totalFanIn;

        Optional<Tensor> anyFeatureInput = getFirstPresentTensor(featureInputs);
        assert(anyFeatureInput.isPresent());
        std::vector<uint64_t> inputDimensions = anyFeatureInput.get().getDescriptor().getDimensions();
        uint64_t fanIn = 1;
        for (uint32_t i = 1; i < inputDimensions.size(); ++i) {
            fanIn *= inputDimensions[i];
        }
        return fanIn;
    }

    // compute the fan out for one element of a batch
    virtual uint64_t getFanOut() {
        uint64_t totalFanOut = 0;
        for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
            if (featureOutputs[i].isPresent()) {
                Tensor aFeatureOutput = featureOutputs[i];
                std::vector<uint64_t> outputDimensions = aFeatureOutput.getDescriptor().getDimensions();
                uint64_t fanOut = 1;
                for (uint32_t j = 1; j < outputDimensions.size(); ++j)
                    fanOut *= outputDimensions[j];
                if (nextLayers[i].isPresent())
                    fanOut *= nextLayers[i].get()->getDownStreamFanoutMultiplier();
                totalFanOut += fanOut;
            }
        }
        if (totalFanOut == 0) {
            // return a 1 to avoid possible divide by 0
            totalFanOut = 1;
        }
        return totalFanOut;
    }

   protected:
    virtual void forward(uint32_t connectionNumber, bool isValidation) {
        assert(running);

        if (isStartOfForward) {
            if (weightsAreUpToDateEvent.isPresent()) {
                for (const Stream &dataStream : uniqueDataStreams) {
                    // All data streams must block forward until the single gradient stream is done updating weights.
                    dataStream.waitEvent(weightsAreUpToDateEvent);
                }
            }
            weightsAreUpToDateEvent.clear();
            isStartOfForward = false;
            isStartOfBackward = true;
        }

        if (nextLayers[connectionNumber].isEmpty())
            return;

        // Compute feature output on the data stream
        computeFeatureOut(connectionNumber);

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        // FIXME: Put back
        // nextLayers[connectionNumber].get()->forward(connectionNumber, isValidation);
    }

    // Weights are up-to-date by end of data stream
    virtual void computeFeatureOut(uint32_t connectionNumber) = 0;

    virtual void backward(uint32_t connectionNumber, uint32_t batchSize) {
        assert(running);

        bool clearGradientFirst = false;
        if (isStartOfBackward) {
            clearGradientFirst = true;
            isStartOfBackward = false;
        }

        // Compute output error gradient using current weights, on the data stream
        if (!isBackPropStub() && previousLayers[connectionNumber].isPresent()) {
            Optional<Event> errorOutHasBeenComputedEvent = computeErrorOut(connectionNumber);
            if (errorOutHasBeenComputedEvent.isPresent())
                errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent);
        }

        // Accumulate gradient for the weights per this connection
        accumulateWeightsGradient(connectionNumber, clearGradientFirst);

        numBackwardConnectionsMade += 1;
        bool gradientComplete = false;
        if (numBackwardConnectionsMade == numBackwardConnections) {
            gradientComplete = true;
            numBackwardConnectionsMade = 0;
        }
        assert(numBackwardConnectionsMade < numBackwardConnections);

        if (gradientComplete) {
            weightsAreUpToDateEvent.clear();

            // Weights cannot be updated until errorOut has been computed.
            if (gradientUpdateStream.isPresent()) {
                // Gradient update stream is present iff there is at least 1 trainable parameter
                for (const Event &eOutComputedEvent : errorOutHasBeenComputedEvents) {
                    gradientUpdateStream.get().waitEvent(eOutComputedEvent);
                }

                // Update weights
                // gradientIn is accumulated, for each weights vector, on its gradient stream.
                // each weights vector is updated on its gradient stream
                bool anyTrainingEnabled = false;
                for (const auto &parameter : parameters) {
                    if (!parameter->isTrainingEnabled())
                        continue;
                    anyTrainingEnabled = true;
                    const std::shared_ptr<Optimizer> &optimizer = parameter->getOptimizer();
                    assert(optimizer != nullptr);  // Training is enabled so it needs to have an optimizer.
                    optimizer->updateWeights(batchSize);
                }
                if (anyTrainingEnabled) {
                    weightsAreUpToDateEvent = gradientUpdateStream.get().putEvent();
                }
            }
            errorOutHasBeenComputedEvents.clear();
            isStartOfForward = true;
        }

        if (previousLayers[connectionNumber].isEmpty())
            return;

        // Propagate output error gradient to the previous layer, on the data stream.        // Expecting to get tail-recursion optimization
        // of -O3 so that stack space does not build up here.
        // FIXME: put back
        // previousLayers[connectionNumber].get()->backward(connectionNumber);
    }

    // Error in is up-to-date by the end of the data stream.
    // Gradient update stream must wait for that.
    virtual void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) = 0;

    // Error in is up-to-date by the end of the data stream.
    virtual Optional<Event> computeErrorOut(uint32_t connectionNumber) = 0;

    // FIXME: Weights created in parameter.compile. Gradients created when parameter.compile calls optimizer.compile()
    // virtual void createWeights(Tensor featureInput) = 0;
    // virtual void createGradients() = 0;

   public:
    // Setters/Getters
    uint64_t getStampedId() { return stampedId; }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        // FIXME: put back
        // Optional<Tensor> errorOutput =
        //     MultiConnectionLayer::connectToPreviousLayer(previousLayer, featureInput, stream, backPropagateError, connectionType);
        // return errorOutput;
        return Optional<Tensor>::empty();
    }

    virtual std::string getLayerType() = 0;

    // FIXME: Use parameters instead
    // virtual std::unordered_map<std::string, Tensor> getWeights() = 0;
    // // All optimizers the same:
    // virtual void setOptimizer(std::shared_ptr<Optimizer> newOptimizer) = 0;
    // // Per weight optimizer:
    // virtual void setOptimizer(std::string, std::shared_ptr<Optimizer> newOptimizer) = 0;
    //
    // virtual bool hasOptimizer() { return !optimizers.empty(); }
    // virtual std::shared_ptr<Optimizer> getOptimizer(std::string weightsName) = 0;
    // virtual std::unordered_map<std::string, std::shared_ptr<Optimizer>> getOptimizers() = 0;
    //
    // virtual void setInitializer(std::shared_ptr<ThorImplementation::Initializer> initializer) = 0;
    // virtual void setInitializer(std::string weightsName, std::shared_ptr<ThorImplementation::Initializer> initializer) = 0;
    // virtual std::vector<Event> initialize() = 0;
    // virtual std::vector<Event> initialize(std::string weightsName) = 0;
    // FIXME: Put back when fits
    // std::vector<Event> initialize(std::vector<std::shared_ptr<Initializer>> initializers) {
    //     for (const auto &initializer : initializers) {
    //         initializer->initialize();
    //     }
    // }

    // FIXME: just this comment. Now parameter owns optimizer and initializer side of things.
    virtual std::vector<std::string> getParameterNames() {
        std::vector<std::string> parameterNames;
        for (const auto &parameter : parameters) {
            parameterNames.push_back(parameter->getName());
        }
        return parameterNames;
    }
    virtual std::unordered_map<std::string, std::shared_ptr<Parameter>> getParameters() {
        std::unordered_map<std::string, std::shared_ptr<Parameter>> allParams;
        for (const auto &parameter : parameters)
            allParams[parameter->getName()] = parameter;
        return allParams;
    }
    virtual std::shared_ptr<Parameter> getParameter(const std::string &parameterName) {
        if (!parameterIndexByName.contains(parameterName))
            throw std::runtime_error("Do not have a parameter by the name of " + parameterName);
        return parameters[parameterIndexByName[parameterName]];
    }

   protected:
    // FIXME: Since inherits Parameterizable, it gets parameters
    // std::vector<std::shared_ptr<Parameter>> parameters;
    // std::vector<std::shared_ptr<Optimizer>> optimizers; FIXME: parameters own optimizers and initializers
    // std::vector<Initializer> weightsInitializers;

    std::vector<Stream> uniqueDataStreams;
    Optional<Stream> gradientUpdateStream;
    std::vector<Event> errorOutHasBeenComputedEvents;
    Optional<Event> weightsAreUpToDateEvent;

    bool isStartOfForward = true;
    bool isStartOfBackward = false;

    uint32_t numBackwardConnectionsMade = 0;
    uint32_t numBackwardConnections = 0;

   private:
    // stampedId is used to identify which layers correspond to which other layers across multiple stamps of the same network.
    const int64_t stampedId;
};

}  // namespace ThorImplementation
