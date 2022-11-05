#pragma once

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
    Loss() {}

    // The feature input to this layer is the (unnormalized) likelihood predictions per batch item per classification class
    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) {
        if (connectionType == (int)ConnectionType::FORWARD_BACKWARD) {
            return connectToPredictionsInputLayer(previousLayer, featureInput, stream, backPropagateError);
        } else if (connectionType == (int)ConnectionType::LABELS) {
            return connectToLabelsInputLayer(previousLayer, featureInput, stream);
        } else {
            assert(false);
        }
    }

    virtual Optional<Tensor> connectToPredictionsInputLayer(Layer *predictionsInputLayer,
                                                            Optional<Tensor> featureInput,
                                                            Stream stream,
                                                            bool backPropagateError) {
        assert(featureInput.isPresent());
        assert(featureInput.get().getDescriptor().getDimensions().size() >= 1);
        assert(this->featureInput.isEmpty());

        if (labelsInput.isPresent()) {
            assert(featureInput.get().getDescriptor().getDimensions() == labelsInput.get().getDescriptor().getDimensions());
            assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            assert(featureInput.get().getPlacement() == labelsInput.get().getPlacement());
        }

        // Allocates this->featureInput and this->errorOutput
        Layer::connectToPreviousLayer(predictionsInputLayer, featureInput, stream, backPropagateError);

        return errorOutput;
    }

    virtual Optional<Tensor> connectToLabelsInputLayer(Layer *labelsLayer, Optional<Tensor> labels, Stream labelsStream) {
        assert(this->labelsInput.isEmpty());

        assert(labels.isPresent());

        if (this->featureInput.isPresent()) {
            assert(this->featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            assert(this->featureInput.get().getPlacement() == labels.get().getPlacement());
        }

        this->labelsInput = labels;
        this->labelsStream = labelsStream;

        // Labels do not cause back propagation
        return Optional<Tensor>::empty();
    }

    virtual ~Loss() {}

    virtual void initialize() {
        featureInputReceived = false;
        labelsReceived = false;
    }

    virtual void forward(Optional<Tensor> inputTensor, bool validationPass) {
        assert(running);
        if (!isInferenceOnly()) {
            assert(labelsStream.isInitialized());
            assert(labelsInput.isPresent());
            assert(errorOutput.isPresent());
            assert(errorOutput.get().isInitialized());
        }
        if (labelsStream.isInitialized()) {
            assert(labelsInput.isPresent());
            assert(labelsInput.get().isInitialized());
        }
        assert(featureOutput.isPresent());
        assert(featureInput.isPresent());

        // After all inputs have been received an empty input tensor is sent to indicate
        // that the layer is ready to perform the forward pass.
        if (inputTensor.isPresent()) {
            if (inputTensor.get() == featureInput.get())
                forwardFeatures(inputTensor, validationPass);
            else if (inputTensor.get() == labelsInput.get())
                forwardLabels(inputTensor, validationPass);
            else
                assert(false);
        } else {
            assert(inputTensor.isEmpty());
            assert(featureInputReceived);
            assert(labelsReceived);
            featureInputReceived = false;
            labelsReceived = false;

            infer(featureInput, featureOutput, stream);

            // Labels stream waits for infer to finish
            labelsStream.waitEvent(stream.putEvent());
        }
    }

    virtual void forwardFeatures(Tensor featureInput, bool validationPass) {
        assert(this->featureInput.get() == featureInput);

        assert(featureInputReceived == false);
        featureInputReceived = true;

        advanceDataIfReady(validationPass);
    }

    virtual void forwardLabels(Tensor labelsInput, bool validationPass) {
        assert(this->labelsInput.get() == labelsInput);

        assert(labelsReceived == false);
        labelsReceived = true;

        advanceDataIfReady(validationPass);
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);
        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(errorOutput.isPresent());
        assert(errorOutput.get().isInitialized());
        assert(labelsStream.isInitialized());
        assert(errorInput.isEmpty());

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

    virtual void ensureNoDeviceCrossing() {
        if (featureInput.isPresent()) {
            if (labelsInput.isPresent())
                assert(labelsInput.get().getPlacement() == featureInput.get().getPlacement());
            if (featureOutput.isPresent())
                assert(featureOutput.get().getPlacement() == featureInput.get().getPlacement());
            if (errorOutput.isPresent())
                assert(errorOutput.get().getPlacement() == featureInput.get().getPlacement());
        }
    }

    virtual Optional<Tensor> getLabelsInput() { return labelsInput; }
    virtual Optional<Tensor> getLossOutput() {
        if (batchLossOutput.isPresent())
            return batchLossOutput;
        else if (elementwiseLossOutput.isPresent())
            return elementwiseLossOutput;
        else
            assert(false);
    }

    enum class ConnectionType { FORWARD_BACKWARD = 4289, LABELS };

    enum class LossType { BATCH = (int)ConnectionType::LABELS + 1027, CLASSWISE, ELEMENTWISE, RAW };

    static uint32_t getLossScalingFactor() { return lossScalingFactor; }

   protected:
    Optional<Layer *> lossOutputLayer;

    Optional<Tensor> labelsInput;
    Optional<Tensor> elementwiseLossOutput;
    Optional<Tensor> batchLossOutput;

    // FIXME: only const for now for convenience
    static constexpr uint32_t lossScalingFactor = 32;
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
        assert(previousLayer.isPresent());
        backward(Optional<Tensor>().empty());
    }
};

}  // namespace ThorImplementation
