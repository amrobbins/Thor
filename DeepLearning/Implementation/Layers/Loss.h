#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

/**
 * A Loss layer may have 1 input and 1 output, the output of a loss layer is the loss tensor.
 *
 * Loss layers do not connect an errorInput from the next layer, so they are a point at which
 * back propagation will terminate if connected at the output to a back-propagable layer.
 *
 * Loss layers compute the loss element wise:
 *   For categorical losses, this is one loss per batch item per class
 *   For numerical losses, this is one loss per batch item per output (usually numerical losses have just one output though)
 * When this is not the final form of the desired loss for reporting purposes, a LossShaper is attached to the output of the loss layer.
 * The loss shaper can report loss in the following ways:
 *   For categorical losses:
 *     1. batchLoss (a single scalar)
 *     2. classwise loss (a scalar for each class)
 *   For numerical losses:
 *     1. batchLoss (a single scalar per output)
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

        // Allocates this->featureInput and this->errorOutput
        Layer::connectToPreviousLayer(predictionsInputLayer, featureInput, stream, backPropagateError);

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

    void initialize() override {
        Layer::initialize();
        featureInputReceived = false;
        labelsReceived = false;
    }

    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize = 0) override {
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
            if (inputTensor.value() == featureInput.value())
                forwardFeatures(inputTensor.value(), validationPass);
            else if (inputTensor.value() == labelsInput.value())
                forwardLabels(inputTensor.value(), validationPass);
            else
                THOR_UNREACHABLE();
        } else {
            THOR_THROW_IF_FALSE(!inputTensor.has_value());
            THOR_THROW_IF_FALSE(featureInputReceived);
            THOR_THROW_IF_FALSE(labelsReceived);
            featureInputReceived = false;
            labelsReceived = false;

            infer(featureInput, featureOutput, stream);

            // Labels stream waits for infer to finish
            labelsStream.waitEvent(stream.putEvent());
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
        THOR_THROW_IF_FALSE(running);
        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsStream.isInitialized());
        THOR_THROW_IF_FALSE(!errorInput.has_value());

        if (errorOutput.has_value()) {
            backProp(labelsInput, featureInput, errorOutput, stream);
            // Labels stream waits for backProp to finish
            labelsStream.waitEvent(stream.putEvent());
        }

        if (!previousLayer.has_value())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayer.value()->backward(errorOutput);
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

    enum class ConnectionType { FORWARD_BACKWARD = 4289, LABELS };

    enum class LossType { BATCH = (int)ConnectionType::LABELS + 1027, CLASSWISE, ELEMENTWISE, RAW };

    static float getLossScalingFactor() { return lossScalingFactor; }

   protected:
    std::optional<Tensor> labelsInput;
    DataType lossDataType;

    // FIXME: only const for now for convenience
    static constexpr float lossScalingFactor = 4;  // 32;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;

    virtual void advanceDataIfReady(bool validationPass) {
        if (featureInputReceived && labelsReceived) {
            // DataStream waits for labels to arrive
            stream.waitEvent(labelsStream.putEvent());
            forward(std::nullopt, validationPass);
        } else {
            return;
        }

        if (nextLayer.has_value())
            nextLayer.value()->forward(featureOutput, validationPass);

        if (isInferenceOnly() || validationPass)
            return;

        // Initiate back propagation
        THOR_THROW_IF_FALSE(previousLayer.has_value());
        backward(std::nullopt);
    }
};

}  // namespace ThorImplementation
