#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Arithmetic/Scale.h"

namespace ThorImplementation {

/**
 * A metric layer may have 1 input and 1 output, the output of a metric layer is the a tensor that contains a c-style
 * string that describes the metric value.
 *
 * Metric layers do not connect an errorInput from the next layer, so they are a point at which
 * back propagation will terminate if connected at the output to a back-propagable layer.
 *
 * featureInput: The prediction probabilities
 * labelsInput: ground truth labels
 * featureOutput: The string that holds the metric value
 * errorOutput: not created
 *
 * Usually you will connect a metric to a NetworkOutput.
 */
class Metric : public Layer {
   public:
    Metric() {}

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) {
        if (connectionType == (int)ConnectionType::FORWARD) {
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

        // Allocates this->featureInput and sets this->errorOutput to empty
        Layer::connectToPreviousLayer(predictionsInputLayer, featureInput, stream, false);

        // Metrics do not back propagate
        return Optional<Tensor>::empty();
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

        // Metrics do not back propagate
        return Optional<Tensor>::empty();
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        if (isInferenceOnly()) {
            return Optional<Tensor>::empty();
        } else {
            assert(featureInput.isPresent());
            return Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {1024}));
        }
    }

    virtual ~Metric() {}

    virtual void initialize() {
        predictionsInputReceived = false;
        labelsReceived = false;
    }

    virtual void forward(Optional<Tensor> inputTensor, bool validationPass) {
        assert(running);
        if (isInferenceOnly())
            return;

        assert(labelsStream.isInitialized());
        assert(labelsInput.isPresent());
        assert(errorOutput.isPresent());
        assert(errorOutput.get().isInitialized());
        assert(labelsStream.isInitialized());
        assert(labelsInput.get().isInitialized());
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

        assert(predictionsInputReceived == false);
        predictionsInputReceived = true;

        advanceDataIfReady(validationPass);
    }

    virtual void forwardLabels(Tensor labelsInput, bool validationPass) {
        assert(this->labelsInput.get() == labelsInput);

        assert(labelsReceived == false);
        labelsReceived = true;

        advanceDataIfReady(validationPass);
    }

    virtual void backward(Optional<Tensor> errorInput) { assert(false); }

    virtual void ensureNoDeviceCrossing() {
        if (featureInput.isPresent()) {
            if (labelsInput.isPresent())
                assert(labelsInput.get().getPlacement() == featureInput.get().getPlacement());
            if (featureOutput.isPresent())
                assert(featureOutput.get().getPlacement() == featureInput.get().getPlacement());
        }
    }

    virtual Optional<Tensor> getLabelsInput() { return labelsInput; }

    virtual void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) = 0;

    /**
     * Warning:
     *
     * resizeMetricBuffer causes synchronization and should be done maybe one time if 1024 bytes is not enough for the metric's output
     */
    virtual void resizeMetricBuffer(uint64_t numBytes) {
        assert(featureOutput.isPresent());

        stream.synchronize();
        featureOutput.get().resize({numBytes});
    }

    enum class ConnectionType { FORWARD = 12, LABELS, METRIC };

   protected:
    Optional<Tensor> labelsInput;
    Stream labelsStream;

    bool predictionsInputReceived;
    bool labelsReceived;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        // Metrics use computeMetric(...) instead, due to different parameter requirements.
    }

    virtual void advanceDataIfReady(bool validationPass) {
        if (predictionsInputReceived && labelsReceived) {
            // DataStream waits for labels to arrive,
            stream.waitEvent(labelsStream.putEvent());

            computeMetric(labelsInput, featureInput, featureOutput, stream);

            predictionsInputReceived = false;
            labelsReceived = false;
        } else {
            return;
        }

        if (nextLayer.isPresent())
            nextLayer.get()->forward(featureOutput, validationPass);
    }
};

}  // namespace ThorImplementation
