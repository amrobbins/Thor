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

   private:
    class UnsupportedBackwardImplementation : public std::logic_error {
       public:
        explicit UnsupportedBackwardImplementation(const std::string &message) : std::logic_error(message) {}
    };

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
        if (errorInputs[connectionNumber].isPresent()) {
            if (gradientUpdateStream.isPresent()) {
                errorInputReadyEvent = streams[connectionNumber].putEvent();
            }

            if (backwardGradientMode != BackwardGradientMode::Fused) {
                // Compute output error gradient using current weights, on the data stream.
                if ((!isBackPropStub() && previousLayers[connectionNumber].isPresent())) {
                    // Weights cannot be updated until all error outputs have been computed, so record the event marking that.
                    try {
                        Optional<Event> errorOutHasBeenComputedEvent = computeErrorOut(connectionNumber);
                        if (errorOutHasBeenComputedEvent.isPresent())
                            errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent);

                        // Gradient accumulation needs error input, so gradient stream waits for this connection's
                        // error input to be ready.
                        if (gradientUpdateStream.isPresent() && errorInputReadyEvent.isPresent()) {
                            gradientUpdateStream.get().waitEvent(errorInputReadyEvent);
                        }

                        // Accumulate gradient for the weights per this connection, on the gradient stream.
                        accumulateWeightsGradient(connectionNumber, clearGradientFirst);
                    } catch (const UnsupportedBackwardImplementation &) {
                        backwardGradientMode = BackwardGradientMode::Fused;
                    }
                }
            }

            // backwardGradientMode can mutate in the block above.
            if (backwardGradientMode == BackwardGradientMode::Fused) {
                if (!isBackPropStub() && previousLayers[connectionNumber].isPresent()) {
                    try {
                        // Gradient accumulation needs error input, so gradient stream waits for this connection's
                        // error input to be ready.
                        if (gradientUpdateStream.isPresent() && errorInputReadyEvent.isPresent()) {
                            gradientUpdateStream.get().waitEvent(errorInputReadyEvent);
                        }

                        // Weights cannot be updated until all error outputs have been computed, so record the event marking that.
                        Optional<Event> errorOutHasBeenComputedEvent =
                            computeErrorOutAccumulateWeightsGradienFused(connectionNumber, clearGradientFirst);
                        if (errorOutHasBeenComputedEvent.isPresent())
                            errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent);
                    } catch (const UnsupportedBackwardImplementation &) {
                        throw std::runtime_error(
                            getLayerType() +
                            " must implement either:\n"
                            "  (1) both computeErrorOut(...) and accumulateWeightsGradient(...)\n"
                            "or\n"
                            "  (2) computeErrorOutAccumulateWeightsGradienFused(...).\n"
                            "With the preference, for performance, being (1) when fusing provides no benefit (e.g. separate kernels "
                            "anyway, like matmuls).\n"
                            "The performance preference is (2) when a single kernel can be launched to compute both (e.g. an elementwise "
                            "or broadcast equation).\n");
                    }
                }
            }
        }

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

    // The following abstract methods need to be implemented, and then the layer will work.

    // Weights are up-to-date by end of data stream
    virtual void computeFeatureOut(uint32_t connectionNumber) = 0;

    // Error in is up-to-date by the end of the data stream.
    virtual Optional<Event> computeErrorOut(uint32_t connectionNumber) {
        throw UnsupportedBackwardImplementation("computeErrorOut(...) not implemented.");
    }

    // Error in is up-to-date by the end of the gradient stream.
    // Gradient accumulation must be performed on the gradient stream, for serialization.
    virtual void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
        throw UnsupportedBackwardImplementation("accumulateWeightsGradient(...) not implemented.");
    }

    // Error in is up-to-date by the end of the gradient stream.
    // Gradient accumulation must be performed on the gradient stream, for serialization.
    virtual Optional<Event> computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber, bool clearWeightsGradientFirstIfFused) {
        throw UnsupportedBackwardImplementation("computeErrorOutAccumulateWeightsGradienFused(...) not implemented.");
    }

    // Return the name of the layer type
    virtual std::string getLayerType() = 0;
    virtual uint64_t flopCountForward() = 0;
    virtual uint64_t flopCountBackward() = 0;

   public:
    // Setters/Getters
    uint64_t getStampedId() const { return stampedId; }  // FIXME: Move to layer
    Optional<Stream> getGradientUpdateStream() const { return gradientUpdateStream; }

   protected:
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

    enum class BackwardGradientMode : uint8_t { Unknown, Unfused, Fused };

    BackwardGradientMode backwardGradientMode = BackwardGradientMode::Unknown;

   private:
    // Using a pattern that includes gradient updates, rather than this one that is wired through MultiConnectionLayer:
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
