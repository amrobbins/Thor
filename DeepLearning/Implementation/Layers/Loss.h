#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Arithmetic/Scale.h"
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
        assert(featureInput.get().getDescriptor().getDimensions().size() >= 2);
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

    // The output of this layer is the normalized predictions, per batch item per classification class
    // (i.e. softmax applied to the input predictions)
    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType, int loaderConnectionType = 0) {
        if (driverConnectionType == (int)ConnectionType::PREDICTIONS) {
            connectToPredictionOutputLayer(nextLayer, loaderConnectionType);
        } else if (driverConnectionType == (int)ConnectionType::BATCH_LOSS) {
            assert(lossOutputLayer.isEmpty());
            connectToBatchLossOutputLayer(nextLayer, loaderConnectionType);
        } else if (driverConnectionType == (int)ConnectionType::CLASSWISE_PER_ELEMENT_LOSS) {
            assert(lossOutputLayer.isEmpty());
            // FIXME: implement
            assert(false);
        } else if (driverConnectionType == (int)ConnectionType::CLASSWISE_LOSS) {
            assert(lossOutputLayer.isEmpty());
            // FIXME: implement
            assert(false);
        } else if (driverConnectionType == (int)ConnectionType::ELEMENTWISE_LOSS) {
            assert(lossOutputLayer.isEmpty());
            connectToElementwiseLossOutputLayer(nextLayer, loaderConnectionType);
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
            this, featureOutput, labelsStream.isInitialized() ? labelsStream : stream, false, loaderConnectionType);

        ensureNoDeviceCrossing();
    }

    virtual void connectToElementwiseLossOutputLayer(Layer *lossOutputLayer, int loaderConnectionType) {
        assert(!compiled);
        assert(this->lossOutputLayer.isEmpty());
        assert(featureInput.isPresent());
        assert(elementwiseLossOutput.isEmpty());
        assert(batchLossOutput.isEmpty());

        this->lossOutputLayer = lossOutputLayer;

        uint64_t batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        elementwiseLossOutput = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
        lossOutputLayer->connectToPreviousLayer(this, elementwiseLossOutput, labelsStream, false, loaderConnectionType);

        ensureNoDeviceCrossing();
    }

    virtual void connectToBatchLossOutputLayer(Layer *lossOutputLayer, int loaderConnectionType) {
        assert(!compiled);
        assert(this->lossOutputLayer.isEmpty());
        assert(featureInput.isPresent());
        assert(featureInput.get().getPlacement() == TensorPlacement::MemDevices::GPU);
        assert(elementwiseLossOutput.isEmpty());
        assert(batchLossOutput.isEmpty());

        this->lossOutputLayer = lossOutputLayer;

        uint64_t batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        elementwiseLossOutput = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
        batchLossOutput = Tensor(elementwiseLossOutput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        lossOutputLayer->connectToPreviousLayer(this, batchLossOutput, labelsStream, false, loaderConnectionType);

        ensureNoDeviceCrossing();
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput.get().clone(TensorDescriptor::DataType::FP32);
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
            assert(labelsInput.get().isInitialized());
        }
        assert(featureOutput.isPresent());
        assert(featureInput.isPresent());
        assert(inputTensor.isPresent());

        if (inputTensor.get() == featureInput.get())
            forwardFeatures(inputTensor, validationPass);
        else if (inputTensor.get() == labelsInput.get())
            forwardLabels(inputTensor, validationPass);
        else
            assert(false);
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

        if (errorOutput.isPresent())
            backProp(labelsInput, featureOutput, errorOutput, stream);

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

    virtual void computeElementwiseLoss(Tensor labels, Tensor normalizedPredictionsOut, Tensor loss, Stream stream) = 0;
    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) = 0;

    enum class ConnectionType {
        FORWARD_BACKWARD = 5,
        LABELS,
        PREDICTIONS,
        BATCH_LOSS,
        CLASSWISE_PER_ELEMENT_LOSS,
        CLASSWISE_LOSS,
        ELEMENTWISE_LOSS
    };

   protected:
    Optional<Layer *> lossOutputLayer;

    Optional<Tensor> labelsInput;
    Optional<Tensor> elementwiseLossOutput;
    Optional<Tensor> batchLossOutput;

    uint32_t lossScalingFactor = 1;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;

    virtual void advanceDataIfReady(bool validationPass) {
        if (featureInputReceived && labelsReceived) {
            // Normalize predictions
            infer(featureInput, featureOutput, stream);

            // DataStream waits for labels to arrive,
            // Labels stream waits for infer to finish
            if (labelsStream.isInitialized() && stream != labelsStream) {
                stream.waitEvent(labelsStream.putEvent());
                labelsStream.waitEvent(stream.putEvent());
            }

            // Compute loss, for forward direction
            if (elementwiseLossOutput.isPresent()) {
                computeElementwiseLoss(labelsInput, featureOutput, elementwiseLossOutput, labelsStream);
            }

            if (batchLossOutput.isPresent()) {
                assert(elementwiseLossOutput.isPresent());
                computeBatchLoss(elementwiseLossOutput, batchLossOutput, labelsStream);
            }

            // Compute loss gradient, for backward direction
            if (!(isInferenceOnly() || validationPass)) {
                if (errorOutput.isPresent())
                    computeLossGradient(labelsInput, featureOutput, errorOutput, stream);
            }

            featureInputReceived = false;
            labelsReceived = false;
        } else {
            return;
        }

        if (nextLayer.isPresent())
            nextLayer.get()->forward(featureOutput, validationPass);
        if (lossOutputLayer.isPresent()) {
            if (batchLossOutput.isPresent())
                lossOutputLayer.get()->forward(batchLossOutput, validationPass);
            else if (elementwiseLossOutput.isPresent())
                lossOutputLayer.get()->forward(elementwiseLossOutput, validationPass);
            else
                assert(false);
        }
        if (isInferenceOnly() || validationPass)
            return;

        // Initiate back propagation
        assert(previousLayer.isPresent());
        previousLayer.get()->backward(errorOutput);
    }

    void computeBatchLoss(Tensor loss, Tensor batchLoss, Stream stream) {
        assert(loss.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(batchLoss.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        uint64_t batchSize = featureInput.get().getDescriptor().getDimensions()[0];

        if (loss.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            launchSumManyToOne((float *)loss.getMemPtr(), (float *)batchLoss.getMemPtr(), batchSize, 1, false, false, stream);
        } else if (loss.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            launchSumManyToOne((half *)loss.getMemPtr(), (half *)batchLoss.getMemPtr(), batchSize, 1, false, false, stream);
        } else {
            assert(false);
        }
    }
};

}  // namespace ThorImplementation
