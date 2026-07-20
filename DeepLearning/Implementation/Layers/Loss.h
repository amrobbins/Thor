#pragma once

#include <optional>
#include <stdexcept>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"

namespace ThorImplementation {

/**
 * A Loss layer may have 1 input and 1 output, the output of a loss layer is the loss tensor.
 *
 * Loss layers do not connect an errorInput from the next layer, so they are a point at which
 * back propagation will terminate if connected at the output to a back-propagable layer.
 *
 * Loss layers compute loss elementwise:
 *   For categorical losses, this is one loss per batch item per class.
 *   For numerical losses, this is one loss per batch item per output.
 *
 * When a reduced form is needed for reporting, a LossShaper interprets a row-major loss tensor
 * [B, D1, ..., Dn] as the zero-copy flattened view [B, C], where C = D1 * ... * Dn, and reports:
 *   1. BATCH: one scalar [1, 1], equal to the sum of all losses divided by B.
 *   2. CLASSWISE: [1, C], with each flattened non-batch position averaged over B.
 *   3. ELEMENTWISE: [B, 1], with all C losses summed independently for each batch item.
 *   4. RAW: the unchanged elementwise loss tensor.
 *
 * Consequently, BATCH is the mean across the batch of each example's summed loss; it is not the
 * mean across every scalar element unless C == 1.
 *
 * featureInput: The predictions
 *   For categorical losses, the predictions should represent a probability distribution (i.e. they sum to 1.0), this is achieved
 *   by processing the predictions through a SoftMax layer, the output of which is sent as the input to the categorical loss.
 * labelsInput: ground truth labels
 * featureOutput: The elementwise loss
 * errorOutput: The loss gradient, scaled by Loss::lossScalingFactor
 */
class Loss : public Layer {
   public:
    Loss(DataType lossDataType) {
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
        this->lossDataType = lossDataType;
    }

    // Note: featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone(lossDataType);
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);

        THOR_THROW_IF_FALSE(!this->nextLayer.has_value());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = std::nullopt;

        // Losses are the origin of the back prop path and will fill the errorOutput directly.
        errorInput = std::nullopt;
        nextLayer->connectToPreviousLayer(
            this, featureOutput, stream, shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

        ensureNoDeviceCrossing();
    }

    // The feature input to this layer is the likelihood predictions per batch item per classification class
    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) override {
        if (connectionType == (int)ConnectionType::FORWARD_BACKWARD) {
            return connectToPredictionsInputLayer(previousLayer, featureInput, stream, backPropagateError);
        } else if (connectionType == (int)ConnectionType::LABELS) {
            return connectToLabelsInputLayer(previousLayer, featureInput, stream);
        } else {
            THOR_UNREACHABLE();
        }
    }

    virtual std::optional<Tensor> connectToPredictionsInputLayer(Layer *predictionsInputLayer,
                                                                 std::optional<Tensor> featureInput,
                                                                 Stream stream,
                                                                 bool backPropagateError) {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions().size() >= 1);
        THOR_THROW_IF_FALSE(!this->featureInput.has_value());

        if (labelsInput.has_value()) {
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelsInput.value().getPlacement());
        }

        // Losses are gradient origins during training.  The upstream layer may currently appear to be a
        // back-prop stub when another non-training consumer, such as a NetworkOutput used for stats/debug
        // reporting, was connected first.  Still allocate the loss error output for training so later
        // prediction-side connections can receive the gradient that the loss computes internally.
        const bool createTrainingErrorOutput = backPropagateError || !isInferenceOnly();

        // Allocates this->featureInput and this->errorOutput.
        Layer::connectToPreviousLayer(predictionsInputLayer, featureInput, stream, createTrainingErrorOutput);

        return errorOutput;
    }

    virtual std::optional<Tensor> connectToLabelsInputLayer(Layer *labelsLayer, std::optional<Tensor> labels, Stream labelsStream) {
        THOR_THROW_IF_FALSE(!this->labelsInput.has_value());

        THOR_THROW_IF_FALSE(labels.has_value());

        if (this->featureInput.has_value()) {
            THOR_THROW_IF_FALSE(this->featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(this->featureInput.value().getPlacement() == labels.value().getPlacement());
        }

        this->labelsInput = labels;
        this->labelsStream = labelsStream;

        // Labels do not cause back propagation
        return std::nullopt;
    }

    ~Loss() override {}

    std::vector<Event> getSynchronizeEvents() override {
        std::vector<Event> events;
        std::set<uint64_t> synchronizedStreamIds;
        appendSynchronizeEvent(events, synchronizedStreamIds, stream);
        appendSynchronizeEvent(events, synchronizedStreamIds, labelsStream);
        return events;
    }

    void initialize() override {
        Layer::initialize();
        featureInputReceived = false;
        labelsReceived = false;
        currentBatchSize = 0;
    }

    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize = 0) override {
        const bool emitDiagnostics = layerSubmitDiagnosticsActive();
        const auto totalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        uint64_t arrivalMicros = 0;
        uint64_t inferMicros = 0;
        uint64_t labelsWaitMicros = 0;
        THOR_THROW_IF_FALSE(running);
        if (!isInferenceOnly()) {
            THOR_THROW_IF_FALSE(labelsStream.isInitialized());
            THOR_THROW_IF_FALSE(labelsInput.has_value());
            THOR_THROW_IF_FALSE(errorOutput.has_value());
            THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        }
        if (labelsStream.isInitialized()) {
            THOR_THROW_IF_FALSE(labelsInput.has_value());
            THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        }
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());

        // After all inputs have been received an empty input tensor is sent to indicate
        // that the layer is ready to perform the forward pass.
        if (inputTensor.has_value()) {
            const auto arrivalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            if (batchSize != 0)
                currentBatchSize = batchSize;
            if (inputTensor.value() == featureInput.value())
                forwardFeatures(inputTensor.value(), validationPass);
            else if (inputTensor.value() == labelsInput.value())
                forwardLabels(inputTensor.value(), validationPass);
            else
                THOR_UNREACHABLE();
            if (emitDiagnostics) {
                arrivalMicros = layerSubmitDiagnosticElapsedMicros(arrivalStart, layerSubmitDiagnosticNow());
            }
        } else {
            THOR_THROW_IF_FALSE(!inputTensor.has_value());
            THOR_THROW_IF_FALSE(featureInputReceived);
            THOR_THROW_IF_FALSE(labelsReceived);
            featureInputReceived = false;
            labelsReceived = false;

            const auto inferStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            infer(featureInput, featureOutput, stream);
            if (emitDiagnostics) {
                inferMicros = layerSubmitDiagnosticElapsedMicros(inferStart, layerSubmitDiagnosticNow());
            }

            // Labels stream waits for infer to finish
            const auto labelsWaitStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            labelsStream.waitEvent(stream.putEvent());
            if (emitDiagnostics) {
                labelsWaitMicros = layerSubmitDiagnosticElapsedMicros(labelsWaitStart, layerSubmitDiagnosticNow());
            }
        }
        if (emitDiagnostics) {
            emitLayerSubmitDiagnostic("loss_forward",
                                      layerSubmitDiagnosticLabel("Loss", getId(), getName()),
                                      getId(),
                                      layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                                      {{"arrival_us", arrivalMicros},
                                       {"infer_us", inferMicros},
                                       {"labels_wait_us", labelsWaitMicros},
                                       {"input_present", inputTensor.has_value() ? 1UL : 0UL}});
        }
    }

    virtual void forwardFeatures(Tensor featureInput, bool validationPass) {
        THOR_THROW_IF_FALSE(this->featureInput.value() == featureInput);

        THOR_THROW_IF_FALSE(featureInputReceived == false);
        featureInputReceived = true;

        advanceDataIfReady(validationPass);
    }

    virtual void forwardLabels(Tensor labelsInput, bool validationPass) {
        THOR_THROW_IF_FALSE(this->labelsInput.value() == labelsInput);

        THOR_THROW_IF_FALSE(labelsReceived == false);
        labelsReceived = true;

        advanceDataIfReady(validationPass);
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        const bool emitDiagnostics = layerSubmitDiagnosticsActive();
        const auto totalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        uint64_t backPropMicros = 0;
        uint64_t labelsWaitMicros = 0;
        uint64_t upstreamMicros = 0;
        THOR_THROW_IF_FALSE(running);
        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsStream.isInitialized());
        THOR_THROW_IF_FALSE(!errorInput.has_value());

        if (errorOutput.has_value()) {
            const auto backPropStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            backProp(labelsInput, featureInput, errorOutput, stream);
            if (emitDiagnostics) {
                backPropMicros = layerSubmitDiagnosticElapsedMicros(backPropStart, layerSubmitDiagnosticNow());
            }
            // Labels stream waits for backProp to finish
            const auto labelsWaitStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            labelsStream.waitEvent(stream.putEvent());
            if (emitDiagnostics) {
                labelsWaitMicros = layerSubmitDiagnosticElapsedMicros(labelsWaitStart, layerSubmitDiagnosticNow());
            }
        }

        if (previousLayer.has_value()) {
            const uint32_t effectiveBatchSize = batchSize != 0 ? batchSize : currentBatchSize;
            const auto upstreamStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
            // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
            previousLayer.value()->backward(errorOutput, effectiveBatchSize);
            if (emitDiagnostics) {
                upstreamMicros = layerSubmitDiagnosticElapsedMicros(upstreamStart, layerSubmitDiagnosticNow());
            }
        }

        if (emitDiagnostics) {
            emitLayerSubmitDiagnostic("loss_backward",
                                      layerSubmitDiagnosticLabel("Loss", getId(), getName()),
                                      getId(),
                                      layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                                      {{"backprop_us", backPropMicros},
                                       {"labels_wait_us", labelsWaitMicros},
                                       {"upstream_us", upstreamMicros}});
        }
    }

    void ensureNoDeviceCrossing() override {
        if (featureInput.has_value()) {
            if (labelsInput.has_value())
                THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == featureInput.value().getPlacement());
            if (featureOutput.has_value())
                THOR_THROW_IF_FALSE(featureOutput.value().getPlacement() == featureInput.value().getPlacement());
            if (errorOutput.has_value())
                THOR_THROW_IF_FALSE(errorOutput.value().getPlacement() == featureInput.value().getPlacement());
        }
    }

    void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) override {}

    virtual std::optional<Tensor> getLabelsInput() { return labelsInput; }
    virtual std::optional<Tensor> getPredictionsInput() { return getFeatureInput(); }
    // When a loss shaper is present, that will provide batch loss etc. Loss::getLossOutput() provides raw loss
    virtual std::optional<Tensor> getLossOutput() { return featureOutput; }

    void setTrainingActive(bool active) {
        if (active && trainingBackpropPathPruned) {
            throw std::logic_error("Cannot reactivate a loss after its inactive training backprop path was pruned for this placed network.");
        }
        trainingActive = active;
    }
    bool isTrainingActive() const { return trainingActive; }

    // A CustomLoss can optimistically fuse its prediction-gradient expression into
    // the layer that produced the predictions while the graph is still being
    // connected.  Non-API utility layers such as TensorFanout may later discover
    // during compile that the fused path is not valid for the final downstream
    // backprop topology.  This hook lets that utility layer invalidate the loss-side
    // fused state before the loss compiles, so the loss materializes its ordinary
    // gradient tensor instead of leaving an unpopulated errorOutput in the graph.
    virtual void notifyFusedGradientUnregisteredFromDrivingLayer(const Tensor& predictions) {
        (void)predictions;
    }

    virtual void pruneTrainingBackpropPathIfInactive() {
        if (trainingActive || trainingBackpropPathPruned || isInferenceOnly() || !errorOutput.has_value()) {
            return;
        }
        if (previousLayer.has_value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, std::nullopt);
        }
        trainingBackpropPathPruned = true;
    }

    enum class ConnectionType { FORWARD_BACKWARD = 4289, LABELS };

    enum class LossType { BATCH = (int)ConnectionType::LABELS + 1027, CLASSWISE, ELEMENTWISE, RAW };

    static float getLossScalingFactor() { return lossScalingFactor; }

   protected:
    std::optional<Tensor> labelsInput;
    DataType lossDataType;

    // FIXME: only const for now for convenience
    static constexpr float lossScalingFactor = 32;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;
    bool trainingActive = true;
    bool trainingBackpropPathPruned = false;
    uint32_t currentBatchSize = 0;

    virtual void advanceDataIfReady(bool validationPass) {
        if (featureInputReceived && labelsReceived) {
            // DataStream waits for labels to arrive
            stream.waitEvent(labelsStream.putEvent());
            forward(std::nullopt, validationPass);
        } else {
            return;
        }

        if (nextLayer.has_value())
            nextLayer.value()->forward(featureOutput, validationPass, currentBatchSize);

        if (isInferenceOnly() || validationPass || !trainingActive)
            return;

        // Initiate back propagation
        THOR_THROW_IF_FALSE(previousLayer.has_value());
        backward(std::nullopt, currentBatchSize);
    }
};

}  // namespace ThorImplementation
