#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.h"

namespace ThorImplementation {

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
    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType, int loaderConnectionType = 0) {
        if (driverConnectionType == (int)ConnectionType::PREDICTIONS) {
            connectToPredictionOutputLayer(nextLayer, loaderConnectionType);
        } else if (driverConnectionType == (int)ConnectionType::LOSS) {
            connectToLossOutputLayer(nextLayer, loaderConnectionType);
        } else {
            assert(false);
        }
    }

    virtual void connectToPredictionOutputLayer(Layer *predictionsOutputLayer, int loaderConnectionType) {
        assert(!compiled);
        assert(this->nextLayer.isEmpty());
        assert(featureOutput.isEmpty());

        this->nextLayer = predictionsOutputLayer;
        featureOutput = createFeatureOutputTensor();

        // Predictions don't need labels to compute.
        // Inputs to a layer must all be connected before any outputs of a layer are connected, so that the layer knows how to create the
        // outputs. Backpropagation does require lables to compute. So when labelsStream is present, use it so that offloading predictions
        // does not delay backpropagation, otherwise use the data stream.
        predictionsOutputLayer->connectToPreviousLayer(
            this, featureOutput, labelsStream.isPresent() ? labelsStream.get() : stream, false, loaderConnectionType);

        ensureNoDeviceCrossing();
    }

    virtual void connectToLossOutputLayer(Layer *lossOutputLayer, int loaderConnectionType) {
        assert(!compiled);
        assert(this->lossOutputLayer.isEmpty());
        assert(featureInput.isPresent());

        this->lossOutputLayer = lossOutputLayer;

        // Allocate loss output tensor
        // FIXME: I want to output loss per batch item per class from loss layers, then another layer can be used
        //        to reduce this when desired. I want to be able to offer loss per class, and this needs the unreduced loss.
        uint64_t batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        lossOutput = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));

        lossOutputLayer->connectToPreviousLayer(this, lossOutput, labelsStream, false, loaderConnectionType);

        ensureNoDeviceCrossing();
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput.get().clone(TensorDescriptor::DataType::FP32);
    }

    virtual void forward(Optional<Tensor> inputTensor) {
        assert(running);
        if (!isInferenceOnly()) {
            assert(labelsStream.isPresent());
            assert(labelsInput.isPresent());
            assert(errorOutput.isPresent());
            assert(errorOutput.get().isInitialized());
        }
        if (labelsStream.isPresent()) {
            assert(labelsStream.get().isInitialized());
            assert(labelsInput.get().isInitialized());
        }
        assert(featureOutput.isPresent());
        assert(featureInput.isPresent());
        assert(inputTensor.isPresent());

        if (inputTensor.get() == featureInput.get())
            forwardFeatures(inputTensor);
        else if (inputTensor.get() == labelsInput.get())
            forwardLabels(inputTensor);
        else
            assert(false);
    }

    virtual void forwardFeatures(Tensor featureInput) {
        assert(this->featureInput.get() == featureInput);

        assert(featureInputReceived == false);
        featureInputReceived = true;

        advanceDataIfReady();
    }

    virtual void forwardLabels(Tensor labelsInput) {
        assert(this->labelsInput.get() == labelsInput);

        assert(labelsReceived == false);
        labelsReceived = true;

        advanceDataIfReady();
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);
        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(errorOutput.isPresent());
        assert(errorOutput.get().isInitialized());
        assert(labelsStream.isPresent());
        assert(labelsStream.get().isInitialized());
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
            if (errorOutput.isPresent())
                assert(errorOutput.get().getPlacement() == featureInput.get().getPlacement());
        }
    }

    virtual Optional<Tensor> getLabelsInput() { return labelsInput; }
    virtual Optional<Tensor> getLossOutput() { return lossOutput; }

    virtual void computeLoss(Tensor labels, Tensor normalizedPredictionsOut, Tensor loss, Stream stream) = 0;
    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) = 0;

    enum class ConnectionType { FORWARD_BACKWARD = 5, LABELS, PREDICTIONS, LOSS };

   protected:
    Optional<Layer *> lossOutputLayer;

    Optional<Tensor> labelsInput;
    Optional<Tensor> lossOutput;

    float lossScalingFactor;
    Tensor lossScalingFactorTensor;
    Optional<Stream> labelsStream;

    bool featureInputReceived;
    bool labelsReceived;

    virtual void advanceDataIfReady() {
        if (featureInputReceived && labelsReceived) {
            // Normalize predictions
            infer(featureInput, featureOutput, stream);

            // DataStream waits for labels to arrive,
            // Labels stream waits for infer to finish
            if (labelsStream.isPresent() && stream != labelsStream.get()) {
                stream.waitEvent(labelsStream.get().putEvent());
                labelsStream.get().waitEvent(stream.putEvent());
            }

            // Compute loss, for forward direction
            if (lossOutput.isPresent())
                computeLoss(labelsInput, featureOutput, lossOutput, labelsStream);

            // Compute loss gradient, for backward direction
            if (errorOutput.isPresent())
                computeLossGradient(labelsInput, featureOutput, errorOutput, stream);

            featureInputReceived = false;
            labelsReceived = false;
        } else {
            return;
        }

        if (nextLayer.isPresent())
            nextLayer.get()->forward(featureOutput);
        if (lossOutputLayer.isPresent())
            lossOutputLayer.get()->forward(lossOutput);

        if (isInferenceOnly())
            return;

        // Initiate back propagation
        assert(previousLayer.isPresent());
        previousLayer.get()->backward(errorOutput);
    }
};

}  // namespace ThorImplementation
