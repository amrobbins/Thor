#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

/**
 * A metric layer has a predictions input, a labels input and a metric output.
 *
 * A metric must implement toDisplayString(Tensor metric_h), that carries a host tensor with the metric output,
 * and returns a descriptive string representing the metric.
 *
 * Metric layers do not connect an errorInput from the next layer, so they are a point at which
 * back propagation will terminate if connected at the output to a back-propagable layer (which would be a legal but
 * unusual use of a metric).
 *
 * featureInput: The prediction probabilities
 * labelsInput: ground truth labels
 * featureOutput: The value of the metric
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
            return connectToFeatureInputLayer(previousLayer, featureInput, stream, backPropagateError);
        } else if (connectionType == (int)ConnectionType::LABELS) {
            return connectToLabelsInputLayer(previousLayer, featureInput, stream);
        } else {
            assert(false);
        }
    }

    virtual Optional<Tensor> connectToFeatureInputLayer(Layer *featureInputLayer,
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
        Layer::connectToPreviousLayer(featureInputLayer, featureInput, stream, false);

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
            return Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        }
    }

    virtual std::string toDisplayString(Tensor metric_h) = 0;

    virtual ~Metric() {}

    virtual void initialize() {
        featureInputReceived = false;
        labelsReceived = false;
    }

    virtual void forward(Optional<Tensor> inputTensor, bool validationPass) {
        assert(running);
        if (isInferenceOnly())
            return;

        assert(labelsStream.isInitialized());
        assert(labelsInput.isPresent());
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

    enum class ConnectionType { FORWARD = 12, LABELS, METRIC };

   protected:
    Optional<Tensor> labelsInput;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        // Metrics use computeMetric(...) instead, due to different parameter requirements.
    }

    virtual void advanceDataIfReady(bool validationPass) {
        if (featureInputReceived && labelsReceived) {
            // DataStream waits for labels to arrive,
            stream.waitEvent(labelsStream.putEvent());

            computeMetric(labelsInput, featureInput, featureOutput, stream);

            featureInputReceived = false;
            labelsReceived = false;
        } else {
            return;
        }

        if (nextLayer.isPresent())
            nextLayer.get()->forward(featureOutput, validationPass);
    }

    virtual void backProp(Optional<Tensor>, Optional<Tensor>, Optional<Tensor>, Stream) { assert(false); }
};

}  // namespace ThorImplementation
