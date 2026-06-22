#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

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
        : placement(placement), stampedId(stampedId) {
        setConstructForInferenceOnly(inferenceOnly);
    }

    void setOptimizer(const std::string &parameterName, const std::shared_ptr<Optimizer> &optimizer) {
        THOR_THROW_IF_FALSE(parameterIndexByName.contains(parameterName));
        parameters[parameterIndexByName[parameterName]]->setOptimizer(optimizer);
    }
    void compileImpl() override {
        // MultiConnectionLayer::compileImpl();

        // Ensure state is clear
        isStartOfForward = true;
        isStartOfBackward = false;
        numBackwardConnectionsMade = 0;
        errorOutHasBeenComputedEvents.clear();
        weightsAreUpToDateEvent.reset();
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
        std::optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(aFeatureInput.has_value());
        THOR_THROW_IF_FALSE(aFeatureInput.value().getPlacement() == placement);

        refreshNumBackwardConnectionsFromCurrentErrorInputs();
    }

    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {
        MultiConnectionLayer::replaceErrorInput(oldErrorInput, newErrorInput);

        // Dynamic active-loss-root pruning can remove a downstream gradient edge after compile.
        // TrainableLayer uses numBackwardConnections to decide when all gradients for the
        // current batch have arrived and when it may apply the optimizer update.  If this
        // count remains at the compile-time value, the layer waits forever for the pruned
        // branch and the next forward observes stale weights.
        refreshNumBackwardConnectionsFromCurrentErrorInputs();
    }

   private:
    void refreshNumBackwardConnectionsFromCurrentErrorInputs() {
        numBackwardConnections = 0;
        for (const auto &errorInput : errorInputs) {
            if (errorInput.has_value())
                numBackwardConnections += 1;
        }
    }

    class UnsupportedBackwardImplementation : public std::logic_error {
       public:
        explicit UnsupportedBackwardImplementation(const std::string &message) : std::logic_error(message) {}
    };

   public:
    void forward(std::optional<Tensor> featureInput, bool isValidation, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);

        unsigned int connectionNumber = 0;
        for (; connectionNumber < featureInputs.size(); ++connectionNumber) {
            if (featureInputs[connectionNumber].has_value() && featureInput.value() == featureInputs[connectionNumber].value())
                break;
        }
        THOR_THROW_IF_FALSE(connectionNumber != featureInputs.size());

        if (isStartOfForward) {
            if (weightsAreUpToDateEvent.has_value()) {
                for (const Stream &dataStream : uniqueDataStreams) {
                    // All data streams must block forward until the single gradient stream is done updating weights.
                    dataStream.waitEvent(weightsAreUpToDateEvent.value());
                }
            }
            weightsAreUpToDateEvent.reset();
            isStartOfForward = false;
            isStartOfBackward = true;
        }

        // Compute feature output on the data stream
        computeFeatureOut(connectionNumber);

        if (!nextLayers[connectionNumber].has_value())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayers[connectionNumber].value()->forward(featureOutputs[connectionNumber], isValidation, batchSize);
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);

        unsigned int connectionNumber = 0;
        for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
            if (errorInputs[connectionNumber].has_value() && errorInput.value() == errorInputs[connectionNumber].value())
                break;
        }
        THOR_THROW_IF_FALSE(connectionNumber != errorInputs.size());

        bool clearGradientFirst = false;
        if (isStartOfBackward) {
            clearGradientFirst = true;
            isStartOfBackward = false;
        }

        // Record the point at which the incoming error tensor is ready on the data stream.
        // The gradient-update stream can wait on this and then run parameter-gradient work
        // without waiting for the later error-out backward kernel to finish.
        std::optional<Event> errorInputReadyEvent = std::nullopt;
        if (errorInputs[connectionNumber].has_value()) {
            if (gradientUpdateStream.has_value()) {
                errorInputReadyEvent = streams[connectionNumber].putEvent();
            }

            if (backwardGradientMode != BackwardGradientMode::Fused) {
                // Compute output error gradient using current weights, on the data stream.
                if ((!isBackPropStub() && previousLayers[connectionNumber].has_value())) {
                    // Weights cannot be updated until all error outputs have been computed, so record the event marking that.
                    try {
                        std::optional<Event> errorOutHasBeenComputedEvent = computeErrorOut(connectionNumber);
                        if (errorOutHasBeenComputedEvent.has_value())
                            errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent.value());

                        // Gradient accumulation needs error input, so gradient stream waits for this connection's
                        // error input to be ready.
                        if (gradientUpdateStream.has_value() && errorInputReadyEvent.has_value()) {
                            gradientUpdateStream.value().waitEvent(errorInputReadyEvent.value());
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
                if (!isBackPropStub() && previousLayers[connectionNumber].has_value()) {
                    try {
                        // Gradient accumulation needs error input, so gradient stream waits for this connection's
                        // error input to be ready.
                        if (gradientUpdateStream.has_value() && errorInputReadyEvent.has_value()) {
                            gradientUpdateStream.value().waitEvent(errorInputReadyEvent.value());
                        }

                        // Weights cannot be updated until all error outputs have been computed, so record the event marking that.
                        std::optional<Event> errorOutHasBeenComputedEvent =
                            computeErrorOutAccumulateWeightsGradienFused(connectionNumber, clearGradientFirst);
                        if (errorOutHasBeenComputedEvent.has_value())
                            errorOutHasBeenComputedEvents.push_back(errorOutHasBeenComputedEvent.value());
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
        THOR_THROW_IF_FALSE(numBackwardConnectionsMade < numBackwardConnections);

        if (gradientComplete) {
            weightsAreUpToDateEvent.reset();

            // Weights cannot be updated until errorOut has been computed.
            if (gradientUpdateStream.has_value()) {
                // Gradient update stream is present iff there is at least 1 trainable parameter
                for (const Event &eOutComputedEvent : errorOutHasBeenComputedEvents) {
                    gradientUpdateStream.value().waitEvent(eOutComputedEvent);
                }

                // Update weights
                // gradientIn is accumulated, for each weights vector, on its gradient stream.
                // each weights vector is updated on its gradient stream
                bool anyWeightsUpdated = false;
                for (const auto &parameter : parameters) {
                    anyWeightsUpdated |= parameter->applyGradient(batchSize * numBackwardConnections);
                }
                if (anyWeightsUpdated) {
                    weightsAreUpToDateEvent = gradientUpdateStream.value().putEvent();
                }
            }
            errorOutHasBeenComputedEvents.clear();
            isStartOfForward = true;
        }

        if (!previousLayers[connectionNumber].has_value())
            return;

        // Propagate output error gradient to the previous layer, on the data stream.
        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayers[connectionNumber].value()->backward(errorOutputs[connectionNumber], batchSize);
    }

    // The following abstract methods need to be implemented, and then the layer will work.

    // Weights are up-to-date by end of data stream
    virtual void computeFeatureOut(uint32_t connectionNumber) = 0;

    // Error in is up-to-date by the end of the data stream.
    virtual std::optional<Event> computeErrorOut(uint32_t connectionNumber) {
        throw UnsupportedBackwardImplementation("computeErrorOut(...) not implemented.");
    }

    // Error in is up-to-date by the end of the gradient stream.
    // Gradient accumulation must be performed on the gradient stream, for serialization.
    virtual void accumulateWeightsGradient(uint32_t connectionNumber, bool clearGradientFirst) {
        throw UnsupportedBackwardImplementation("accumulateWeightsGradient(...) not implemented.");
    }

    // Error in is up-to-date by the end of the gradient stream.
    // Gradient accumulation must be performed on the gradient stream, for serialization.
    virtual std::optional<Event> computeErrorOutAccumulateWeightsGradienFused(uint32_t connectionNumber, bool clearWeightsGradientFirstIfFused) {
        throw UnsupportedBackwardImplementation("computeErrorOutAccumulateWeightsGradienFused(...) not implemented.");
    }

    // Return the name of the layer type
    virtual std::string getLayerType() = 0;
    virtual uint64_t flopCountForward() = 0;
    virtual uint64_t flopCountBackward() = 0;

   public:
    // Setters/Getters
    uint64_t getStampedId() const { return stampedId; }  // FIXME: Move to layer
    std::optional<Stream> getGradientUpdateStream() const { return gradientUpdateStream; }

   protected:
    virtual PhysicalParameter::StorageContext buildParameterStorageContext() const {
        std::optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
        THOR_THROW_IF_FALSE(aFeatureInput.has_value());

        std::vector<Tensor> connectedFeatureInputs;
        connectedFeatureInputs.reserve(featureInputs.size());
        for (const auto &featureInput : featureInputs) {
            if (featureInput.has_value())
                connectedFeatureInputs.push_back(featureInput.value());
        }

        return PhysicalParameter::StorageContext(aFeatureInput.value());
    }

    void attachGradientUpdateStream() {
        if (gradientUpdateStream.has_value())
            return;
        for (const auto &parameter : parameters) {
            if (!parameter->isTrainable())
                continue;
            if (!gradientUpdateStream.has_value()) {
                gradientUpdateStream = Stream::getNextGradientUpdateStream(placement.getDeviceNum());
                break;
            }
        }
    }

   protected:
    std::vector<Stream> uniqueDataStreams;
    std::optional<Stream> gradientUpdateStream;
    std::vector<Event> errorOutHasBeenComputedEvents;
    std::optional<Event> weightsAreUpToDateEvent;

    bool isStartOfForward = true;
    bool isStartOfBackward = false;

    uint32_t numBackwardConnectionsMade = 0;
    uint32_t numBackwardConnections = 0;

    bool compiled = false;
    TensorPlacement placement;

   private:
    // stampedId is used to identify which layers correspond to which other layers across multiple stamps of the same network.
    // So it is the id of the API layer.
    const int64_t stampedId;

    enum class BackwardGradientMode : uint8_t { Unknown, Unfused, Fused };

    BackwardGradientMode backwardGradientMode = BackwardGradientMode::Unknown;

   private:
    // Using a pattern that includes gradient updates, rather than this one that is wired through MultiConnectionLayer:
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream, unsigned int connectionNumber) override {
        THOR_UNREACHABLE();
    }
    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream stream,
                  unsigned int connectionNumber) override {
        THOR_UNREACHABLE();
    }
};

}  // namespace ThorImplementation
