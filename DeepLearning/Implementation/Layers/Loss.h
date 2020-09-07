#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.h"

/**
 * A Loss layer may have 1 input and 1 output, the output of a loss layer is the loss tensor.
 *
 * Loss layers do not connect an errorInput from the next layer, so they are a point at which
 * back propagation will terminate if connected at the output to a back-propagable layer.
 *
 * The loss tensor give the loss per element of the batch, so it is a one dimensions array of
 * batchSize elements, the batch loss is the summation of the elements of the loss tensor.
 *
 * featureInput: The unnormalized predictions
 * labelsInput: ground truth labels
 * featureOutput: The normalized predictions (i.e. softmax applied)
 * errorOutput: The loss (per batch item per class, sum it for scalar loss), this value is scaled by lossScalingFactor
 */
// FIXME: inferenceOnly support
//        for inference only, this layer acts as a softmax layer, does not connect labels and does not compute loss and does not
//        backpropagate loss.
class Loss : public Layer {
   public:
    Loss(float lossScalingFactor = 1.0f) { this->lossScalingFactor = lossScalingFactor; }

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
        assert(featureInput.get().getDescriptor().getDimensions().size() >= 2);
        assert(this->featureInput.isEmpty());

        if (labelsInput.isPresent()) {
            assert(featureInput.get().getDescriptor().getDimensions() == labelsInput.get().getDescriptor().getDimensions());
            assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            assert(featureInput.get().getPlacement() == labelsInput.get().getPlacement());
        }

        // Allocates this->featureInput and this->errorOutput
        Layer::connectToPreviousLayer(predictionsInputLayer, featureInput, stream, backPropagateError);

        // Allocate scaling factor tensor
        Tensor lossScalingFactorTensorCpu =
            Tensor(TensorPlacement::MemDevices::CPU, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        ((float *)lossScalingFactorTensorCpu.getMemPtr())[0] = lossScalingFactor;
        lossScalingFactorTensor = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        lossScalingFactorTensor.copyFromAsync(lossScalingFactorTensorCpu, stream);

        return errorOutput;
    }

    virtual Optional<Tensor> connectToLabelsInputLayer(Layer *labelsLayer, Optional<Tensor> labels, Stream labelsStream) {
        assert(this->labelsInput.isEmpty());

        assert(labels.isPresent());

        if (this->featureInput.isPresent()) {
            assert(this->featureInput.get().getDescriptor().getDimensions() == labels.get().getDescriptor().getDimensions());
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

    // The output of this layer is the normalized predictions, per batch item per classification class
    // (i.e. softmax applied to the input predictions)
    virtual void connectToNextLayer(Layer *nextLayer, int connectionType) {
        if (connectionType == (int)ConnectionType::PREDICTIONS) {
            connectToPredictionOutputLayer(nextLayer);
        } else if (connectionType == (int)ConnectionType::LOSS) {
            connectToLossOutputLayer(nextLayer);
        } else {
            assert(false);
        }
    }

    virtual void connectToPredictionOutputLayer(Layer *predictionsOutputLayer) {
        assert(!compiled);
        assert(this->nextLayer.isEmpty());
        assert(featureOutput.isEmpty());

        this->nextLayer = predictionsOutputLayer;
        featureOutput = createFeatureOutputTensor();

        predictionsOutputLayer->connectToPreviousLayer(this, featureOutput, stream, false);

        ensureNoDeviceCrossing();
    }

    virtual void connectToLossOutputLayer(Layer *lossOutputLayer) {
        assert(!compiled);
        assert(this->lossOutputLayer.isEmpty());

        this->lossOutputLayer = lossOutputLayer;
        lossOutputLayer->connectToPreviousLayer(this, errorOutput, labelsStream, false);

        ensureNoDeviceCrossing();
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput.get().clone(TensorDescriptor::DataType::FP32);
    }

    virtual void forward(Optional<Tensor> inputTensor) {
        assert(running);
        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(errorOutput.isPresent());
        assert(errorOutput.get().isInitialized());
        assert(labelsStream.isInitialized());
        assert(inputTensor.isPresent());
        assert(featureOutput.isPresent());
        assert(featureInput.isPresent());

        assert(featureInputReceived == false);
        featureInputReceived = true;
        assert(inputTensor.get() == featureInput.get());

        advanceDataIfReady();
    }

    virtual void forwardLabels(Tensor labelsInput) {
        assert(this->labelsInput.get() == labelsInput);

        assert(labelsReceived == false);
        labelsReceived = true;

        if (stream != labelsStream)
            stream.waitEvent(labelsStream.putEvent());

        advanceDataIfReady();
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);
        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(errorOutput.isPresent());
        assert(errorOutput.get().isInitialized());
        assert(labelsStream.isInitialized());
        assert(errorInput.isEmpty());

        if (errorOutput.isPresent())
            backProp(labelsInput, featureOutput, errorOutput, stream);

        if (previousLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayer.get()->backward(errorOutput);
    }

    virtual void ensureNoDeviceCrossing() {
        if (featureInput.isPresent() && errorOutput.isPresent())
            assert(featureInput.get().getPlacement() == errorOutput.get().getPlacement());
        if (featureInput.isPresent()) {
            if (labelsInput.isPresent())
                assert(labelsInput.get().getPlacement() == featureInput.get().getPlacement());
            assert(errorOutput.get().getPlacement() == featureInput.get().getPlacement());
        }
    }

    virtual void computeLoss(Tensor labels, Tensor featureOutput, Tensor errorOutput, Stream dataStream, Stream labelsStream) = 0;

    enum class ConnectionType { FORWARD_BACKWARD = 5, LABELS, PREDICTIONS, LOSS };

   protected:
    Optional<Layer *> lossOutputLayer;

    Optional<Tensor> labelsInput;

    float lossScalingFactor;
    Tensor lossScalingFactorTensor;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;

    virtual void advanceDataIfReady() {
        if (featureInputReceived && labelsReceived) {
            infer(featureInput, featureOutput, stream);
            if (errorOutput.isPresent())
                computeLoss(labelsInput, featureOutput, errorOutput, stream, stream);

            featureInputReceived = false;
            labelsReceived = false;
        } else {
            return;
        }

        if (nextLayer.isEmpty())
            return;

        if (nextLayer.isPresent())
            nextLayer.get()->forward(featureOutput);
        if (lossOutputLayer.isPresent())
            lossOutputLayer.get()->forward(errorOutput);

        if (inferenceOnly)
            return;

        // Initiate back propagation
        assert(previousLayer.isPresent());
        previousLayer.get()->backward(errorOutput);
    }
};
