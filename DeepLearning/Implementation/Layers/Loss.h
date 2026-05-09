#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/Common/Optional.h"

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
    Loss(TensorDescriptor::DataType lossDataType) {
        THOR_THROW_IF_FALSE(lossDataType == TensorDescriptor::DataType::FP16 || lossDataType == TensorDescriptor::DataType::FP32);
        this->lossDataType = lossDataType;
    }

    // Note: featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        return featureInput.get().clone(lossDataType);
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);

        THOR_THROW_IF_FALSE(this->nextLayer.isEmpty());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = Optional<Tensor>::empty();

        // Losses are the origin of the back prop path and will fill the errorOutput directly.
        errorInput = Optional<Tensor>::empty();
        nextLayer->connectToPreviousLayer(
            this, featureOutput, stream, shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

        ensureNoDeviceCrossing();
    }

    // The feature input to this layer is the likelihood predictions per batch item per classification class
    Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) override {
        if (connectionType == (int)ConnectionType::FORWARD_BACKWARD) {
            return connectToPredictionsInputLayer(previousLayer, featureInput, stream, backPropagateError);
        } else if (connectionType == (int)ConnectionType::LABELS) {
            return connectToLabelsInputLayer(previousLayer, featureInput, stream);
        } else {
            THOR_UNREACHABLE();
        }
    }

    virtual Optional<Tensor> connectToPredictionsInputLayer(Layer *predictionsInputLayer,
                                                            Optional<Tensor> featureInput,
                                                            Stream stream,
                                                            bool backPropagateError) {
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDimensions().size() >= 1);
        THOR_THROW_IF_FALSE(this->featureInput.isEmpty());

        if (labelsInput.isPresent()) {
            THOR_THROW_IF_FALSE(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(featureInput.get().getPlacement() == labelsInput.get().getPlacement());
        }

        // Allocates this->featureInput and this->errorOutput
        Layer::connectToPreviousLayer(predictionsInputLayer, featureInput, stream, backPropagateError);

        return errorOutput;
    }

    virtual Optional<Tensor> connectToLabelsInputLayer(Layer *labelsLayer, Optional<Tensor> labels, Stream labelsStream) {
        THOR_THROW_IF_FALSE(this->labelsInput.isEmpty());

        THOR_THROW_IF_FALSE(labels.isPresent());

        if (this->featureInput.isPresent()) {
            THOR_THROW_IF_FALSE(this->featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(this->featureInput.get().getPlacement() == labels.get().getPlacement());
        }

        this->labelsInput = labels;
        this->labelsStream = labelsStream;

        // Labels do not cause back propagation
        return Optional<Tensor>::empty();
    }

    ~Loss() override {}

    void initialize() override {
        Layer::initialize();
        featureInputReceived = false;
        labelsReceived = false;
    }

    void forward(Optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (!isInferenceOnly()) {
            THOR_THROW_IF_FALSE(labelsStream.isInitialized());
            THOR_THROW_IF_FALSE(labelsInput.isPresent());
            THOR_THROW_IF_FALSE(errorOutput.isPresent());
            THOR_THROW_IF_FALSE(errorOutput.get().isInitialized());
        }
        if (labelsStream.isInitialized()) {
            THOR_THROW_IF_FALSE(labelsInput.isPresent());
            THOR_THROW_IF_FALSE(labelsInput.get().isInitialized());
        }
        THOR_THROW_IF_FALSE(featureOutput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.isPresent());

        // After all inputs have been received an empty input tensor is sent to indicate
        // that the layer is ready to perform the forward pass.
        if (inputTensor.isPresent()) {
            if (inputTensor.get() == featureInput.get())
                forwardFeatures(inputTensor, validationPass);
            else if (inputTensor.get() == labelsInput.get())
                forwardLabels(inputTensor, validationPass);
            else
                THOR_UNREACHABLE();
        } else {
            THOR_THROW_IF_FALSE(inputTensor.isEmpty());
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
        THOR_THROW_IF_FALSE(this->featureInput.get() == featureInput);

        THOR_THROW_IF_FALSE(featureInputReceived == false);
        featureInputReceived = true;

        advanceDataIfReady(validationPass);
    }

    virtual void forwardLabels(Tensor labelsInput, bool validationPass) {
        THOR_THROW_IF_FALSE(this->labelsInput.get() == labelsInput);

        THOR_THROW_IF_FALSE(labelsReceived == false);
        labelsReceived = true;

        advanceDataIfReady(validationPass);
    }

    void backward(Optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        THOR_THROW_IF_FALSE(labelsInput.isPresent());
        THOR_THROW_IF_FALSE(labelsInput.get().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.isPresent());
        THOR_THROW_IF_FALSE(errorOutput.get().isInitialized());
        THOR_THROW_IF_FALSE(labelsStream.isInitialized());
        THOR_THROW_IF_FALSE(errorInput.isEmpty());

        if (errorOutput.isPresent()) {
            backProp(labelsInput, featureInput, errorOutput, stream);
            // Labels stream waits for backProp to finish
            labelsStream.waitEvent(stream.putEvent());
        }

        if (previousLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayer.get()->backward(errorOutput);
    }

    void ensureNoDeviceCrossing() override {
        if (featureInput.isPresent()) {
            if (labelsInput.isPresent())
                THOR_THROW_IF_FALSE(labelsInput.get().getPlacement() == featureInput.get().getPlacement());
            if (featureOutput.isPresent())
                THOR_THROW_IF_FALSE(featureOutput.get().getPlacement() == featureInput.get().getPlacement());
            if (errorOutput.isPresent())
                THOR_THROW_IF_FALSE(errorOutput.get().getPlacement() == featureInput.get().getPlacement());
        }
    }

    void replaceErrorInput(Optional<Tensor> oldErrorInput, Optional<Tensor> newErrorInput) override {}

    virtual Optional<Tensor> getLabelsInput() { return labelsInput; }
    virtual Optional<Tensor> getPredictionsInput() { return getFeatureInput(); }
    // When a loss shaper is present, that will provide batch loss etc. Loss::getLossOutput() provides raw loss
    virtual Optional<Tensor> getLossOutput() { return featureOutput; }

    enum class ConnectionType { FORWARD_BACKWARD = 4289, LABELS };

    enum class LossType { BATCH = (int)ConnectionType::LABELS + 1027, CLASSWISE, ELEMENTWISE, RAW };

    static float getLossScalingFactor() { return lossScalingFactor; }

   protected:
    Optional<Tensor> labelsInput;
    TensorDescriptor::DataType lossDataType;

    // FIXME: only const for now for convenience
    static constexpr float lossScalingFactor = 4;  // 32;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;

    virtual void advanceDataIfReady(bool validationPass) {
        if (featureInputReceived && labelsReceived) {
            // DataStream waits for labels to arrive
            stream.waitEvent(labelsStream.putEvent());
            forward(Optional<Tensor>::empty(), stream);
        } else {
            return;
        }

        if (nextLayer.isPresent())
            nextLayer.get()->forward(featureOutput, validationPass);

        if (isInferenceOnly() || validationPass)
            return;

        // Initiate back propagation
        THOR_THROW_IF_FALSE(previousLayer.isPresent());
        backward(Optional<Tensor>().empty());
    }
};

}  // namespace ThorImplementation
