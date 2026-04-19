#pragma once

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
class TrainableLayer : public MultiConnectionLayer, public Parameterizable {
   public:
    ~TrainableLayer() override = default;

    // All real stamped id's will have positive values
    TrainableLayer(const TensorPlacement &placement, bool inferenceOnly, int64_t stampedId = -1)
        : placement(placement), inferenceOnly(inferenceOnly), stampedId(stampedId) {}

    void attachGradientUpdateStream() {
        if (gradientUpdateStream.isPresent())
            return;
        for (const auto &parameter : parameters) {
            if (!parameter->isTrainable())
                continue;
            if (gradientUpdateStream.isEmpty()) {
                gradientUpdateStream = Stream::getNextGradientUpdateStream(placement.getDeviceNum());
                break;
            }
        }
    }

    void setOptimizer(const std::string &parameterName, const std::shared_ptr<Optimizer> &optimizer) {
        assert(parameterIndexByName.contains(parameterName));
        parameters[parameterIndexByName[parameterName]]->setOptimizer(optimizer);
    }
    void compileImpl() override {
        // MultiConnectionLayer::compileImpl();

        // Ensure state is clear
        isStartOfForward = true;
        isStartOfBackward = false;
        numBackwardConnectionsMade = 0;
        errorOutHasBeenComputedEvents.clear();
        weightsAreUpToDateEvent.clear();
        // It is assumed gradient update streams are best distributed via round-robin on first compile,
        // so once one is assigned, never clear it.

        std::unordered_set<uint64_t> dataStreamsSet;
        uniqueDataStreams.clear();
        for (Stream &dataStream : streams) {
            uint64_t dataStreamId = dataStream.getId();
            auto [it, inserted] = dataStreamsSet.insert(dataStreamId);
            if (inserted) {
                // first time seeing this stream
                uniqueDataStreams.push_back(dataStream);
            }
        }

        // Compile happens after all inputs and outputs are connected
        Optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        assert(aFeatureInput.isPresent());
        assert(aFeatureInput.get().getPlacement() == placement);

        numBackwardConnections = 0;
        for (const auto &errorInput : errorInputs) {
            if (errorInput.isPresent())
                numBackwardConnections += 1;
        }
    }

    virtual uint64_t flopCountForward() = 0;
    virtual uint64_t flopCountBackward() = 0;

   public:
    // FIXME: Temporarily moved here from layer
    // bool inferenceOnly = false;
    // virtual bool isInferenceOnly() { return inferenceOnly; }
    // bool running = true;
    // virtual bool isBackPropStub() { return false; }
    // std::vector<Optional<Tensor>> featureInputs;
    // std::vector<Optional<Tensor>> featureOutputs;
    // std::vector<Optional<Tensor>> errorInputs;
    // std::vector<Optional<Tensor>> errorOutputs;
    // std::vector<Stream> streams;
    // std::vector<Optional<TrainableLayer *>> nextLayers;
    // std::vector<Optional<TrainableLayer *>> previousLayers;
    // static Optional<Tensor> getFirstPresentTensor(const std::vector<Optional<Tensor>> &tensors) {
    //     for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    //         if (it->isPresent())
    //             return *it;
    //     }
    //     return Optional<Tensor>::empty();
    // }
    // static unsigned int numPresentTensors(const std::vector<Optional<Tensor>> &tensors) {
    //     unsigned int numPresent = 0;
    //     for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
    //         if (it->isPresent())
    //             numPresent += 1;
    //     }
    //     return numPresent;
    // }
    // static Optional<Tensor> getLastPresentTensor(std::vector<Optional<Tensor>> tensors) {
    //     for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
    //         if (it->isPresent())
    //             return *it;
    //     }
    //     return Optional<Tensor>::empty();
    // }
    // virtual void ensureNoDeviceCrossing() {
    //     Optional<Tensor> lastFeatureInput = getLastPresentTensor(featureInputs);
    //     Optional<Tensor> lastErrorOutput = getLastPresentTensor(errorOutputs);
    //     Optional<Tensor> lastErrorInput = getLastPresentTensor(errorInputs);
    //     Optional<Tensor> lastFeatureOutput = getLastPresentTensor(featureOutputs);
    //
    //     if (lastFeatureInput.isPresent() && lastFeatureOutput.isPresent())
    //         assert(lastFeatureInput.get().getPlacement() == lastFeatureOutput.get().getPlacement());
    //     if (lastFeatureInput.isPresent() && lastErrorInput.isPresent())
    //         assert(lastFeatureInput.get().getPlacement() == lastErrorInput.get().getPlacement());
    //     if (lastFeatureInput.isPresent() && lastErrorOutput.isPresent())
    //         assert(lastFeatureInput.get().getPlacement() == lastErrorOutput.get().getPlacement());
    //
    //     if (lastFeatureOutput.isPresent() && lastErrorInput.isPresent())
    //         assert(lastFeatureOutput.get().getPlacement() == lastErrorInput.get().getPlacement());
    //     if (lastFeatureOutput.isPresent() && lastErrorOutput.isPresent())
    //         assert(lastFeatureOutput.get().getPlacement() == lastErrorOutput.get().getPlacement());
    //
    //     if (lastErrorInput.isPresent() && lastErrorOutput.isPresent())
    //         assert(lastErrorInput.get().getPlacement() == lastErrorOutput.get().getPlacement());
    // }
    // virtual Optional<Tensor> createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    //     // backPropagateError allows the previous layer to specify that it does not support back propagation,
    //     // inferenceOnly means that even though back propagation may be supported, we are not using it since we are not training.
    //     if (backPropagateError && !isInferenceOnly())
    //         return getFirstPresentTensor(featureInputs).get().clone();
    //     else
    //         return Optional<Tensor>::empty();
    // }
    // virtual Optional<Tensor> connectToPreviousLayer(
    //     TrainableLayer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
    //     assert(!compiled);
    //
    //     Optional<Tensor> previouslyConnectedFeatureInput = getFirstPresentTensor(featureInputs);
    //     if (previouslyConnectedFeatureInput.isPresent() && featureInput.isPresent()) {
    //         assert(featureInput.get().getDescriptor() == previouslyConnectedFeatureInput.get().getDescriptor());
    //         assert(featureInput.get().getPlacement() == previouslyConnectedFeatureInput.get().getPlacement());
    //     }
    //
    //     streams.push_back(stream);
    //
    //     previousLayers.push_back(previousLayer);
    //     featureInputs.emplace_back(featureInput);
    //     errorOutputs.emplace_back(createErrorOutputTensor(backPropagateError, errorOutputs.size()));
    //
    //     Optional<Tensor> lastFeatureInput = getLastPresentTensor(featureInputs);
    //     Optional<Tensor> firstFeatureInput = getFirstPresentTensor(featureInputs);
    //     Optional<Tensor> lastErrorOutput = getLastPresentTensor(errorOutputs);
    //     if (firstFeatureInput.isPresent()) {
    //         assert(lastFeatureInput.get().getDescriptor() == firstFeatureInput.get().getDescriptor());
    //         assert(lastFeatureInput.get().getPlacement() == firstFeatureInput.get().getPlacement());
    //         if (lastErrorOutput.isPresent()) {
    //             assert(lastFeatureInput.get().getDescriptor() == lastErrorOutput.get().getDescriptor());
    //             assert(lastFeatureInput.get().getPlacement() == lastErrorOutput.get().getPlacement());
    //         }
    //     } else if (lastErrorOutput.isPresent()) {
    //         Optional<Tensor> firstErrorOutput = getFirstPresentTensor(errorOutputs);
    //         assert(lastErrorOutput.get().getDescriptor() == firstErrorOutput.get().getDescriptor());
    //         assert(lastErrorOutput.get().getPlacement() == firstErrorOutput.get().getPlacement());
    //     }
    //     ensureNoDeviceCrossing();
    //
    //     return errorOutputs.back();
    // }
    // FIXME: Temporarily moved here from layer

   public:
    void forward(Optional<Tensor> featureInput, bool isValidation, uint32_t batchSize = 0) override {
        assert(running);

        unsigned int connectionNumber = 0;
        for (; connectionNumber < featureInputs.size(); ++connectionNumber) {
            if (featureInputs[connectionNumber].isPresent() && featureInput.get() == featureInputs[connectionNumber].get())
                break;
        }
        assert(connectionNumber != featureInputs.size());

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

        // Compute feature output on the data stream
        computeFeatureOut(connectionNumber);

        if (nextLayers[connectionNumber].isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[connectionNumber].get()->forward(featureOutputs[connectionNumber], batchSize, isValidation);
    }

    // Weights are up-to-date by end of data stream
    virtual void computeFeatureOut(uint32_t connectionNumber) = 0;

    void backward(Optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        assert(running);

        unsigned int connectionNumber = 0;
        for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
            if (errorInputs[connectionNumber].isPresent() && errorInput.get() == errorInputs[connectionNumber].get())
                break;
        }
        assert(connectionNumber != errorInputs.size());

        bool clearGradientFirst = false;
        if (isStartOfBackward) {
            clearGradientFirst = true;
            isStartOfBackward = false;
        }

        // Record the point at which the incoming error tensor is ready on the data stream.
        // The gradient-update stream can wait on this and then run parameter-gradient work
        // without waiting for the later error-out backward kernel to finish.
        Optional<Event> errorInputReadyEvent = Optional<Event>::empty();
        if (gradientUpdateStream.isPresent() && errorInputs[connectionNumber].isPresent()) {
            errorInputReadyEvent = streams[connectionNumber].putEvent();
        }

        // Compute output error gradient using current weights, on the data stream.
        if ((!isBackPropStub() && previousLayers[connectionNumber].isPresent()) || wGradFusedWithEOutGrad) {
            Optional<Event> errorOutHasBeenComputedEvent = computeErrorOut(connectionNumber);
            if (errorOutHasBeenComputedEvent.isPresent())
                errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent);
        }

        if (!errorInputs[connectionNumber].isPresent())
            return;

        if (gradientUpdateStream.isPresent() && errorInputReadyEvent.isPresent()) {
            gradientUpdateStream.get().waitEvent(errorInputReadyEvent);
        }

        // Accumulate gradient for the weights per this connection.
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

        // Propagate output error gradient to the previous layer, on the data stream.
        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].get()->backward(errorOutputs[connectionNumber], batchSize);
    }

    // Error in is up-to-date by the end of the data stream.
    // Gradient update stream must wait for that.
    virtual void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) = 0;

    // Error in is up-to-date by the end of the data stream.
    virtual Optional<Event> computeErrorOut(uint32_t connectionNumber) = 0;

   public:
    // Setters/Getters
    uint64_t getStampedId() const { return stampedId; }  // FIXME: Move to layer

    virtual std::string getLayerType() = 0;
    Optional<Stream> getGradientUpdateStream() const { return gradientUpdateStream; }

    // The following functionality is handled by parameterizable.
    // virtual std::vector<std::string> getParameterNames() {
    //     std::vector<std::string> parameterNames;
    //     for (const auto &parameter : parameters) {
    //         parameterNames.push_back(parameter->getName());
    //     }
    //     return parameterNames;
    // }
    // virtual std::unordered_map<std::string, std::shared_ptr<Parameter>> getParameters() {
    //     std::unordered_map<std::string, std::shared_ptr<Parameter>> allParams;
    //     for (const auto &parameter : parameters)
    //         allParams[parameter->getName()] = parameter;
    //     return allParams;
    // }
    // virtual std::shared_ptr<Parameter> getParameter(const std::string &parameterName) {
    //     if (!parameterIndexByName.contains(parameterName))
    //         throw std::runtime_error("Do not have a parameter by the name of " + parameterName);
    //     return parameters[parameterIndexByName[parameterName]];
    // }

   protected:
    std::vector<Stream> uniqueDataStreams;
    Optional<Stream> gradientUpdateStream;
    std::vector<Event> errorOutHasBeenComputedEvents;
    Optional<Event> weightsAreUpToDateEvent;

    bool isStartOfForward = true;
    bool isStartOfBackward = false;

    uint32_t numBackwardConnectionsMade = 0;
    uint32_t numBackwardConnections = 0;

    bool wGradFusedWithEOutGrad = false;
    bool compiled = false;
    TensorPlacement placement;

    bool inferenceOnly;

   private:
    // stampedId is used to identify which layers correspond to which other layers across multiple stamps of the same network.
    const int64_t stampedId;

   private:
    // Using a pattern that includes gradient updates, rather than this one that is wired through MultiConnectionLayer:
    //     Also these could be updated to take only connection number, that would be provided by forward(...) or backward(...)
    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) override {
        assert(false);
    }
    void backProp(Optional<Tensor> dataIn,
                  Optional<Tensor> errorIn,
                  Optional<Tensor> errorOut,
                  Stream stream,
                  unsigned int connectionNumber) override {
        assert(false);
    }
};

}  // namespace ThorImplementation
