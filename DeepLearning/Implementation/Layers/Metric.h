#pragma once

#include "DeepLearning/Implementation/ThorError.h"

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

    Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) override {
        if (connectionType == (int)ConnectionType::FORWARD) {
            return connectToFeatureInputLayer(previousLayer, featureInput, stream, backPropagateError);
        } else if (connectionType == (int)ConnectionType::LABELS) {
            return connectToLabelsInputLayer(previousLayer, featureInput, stream);
        } else {
            THOR_UNREACHABLE();
        }
    }

    virtual Optional<Tensor> connectToFeatureInputLayer(Layer *featureInputLayer,
                                                        Optional<Tensor> featureInput,
                                                        Stream stream,
                                                        bool backPropagateError) {
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDimensions().size() >= 2);
        THOR_THROW_IF_FALSE(this->featureInput.isEmpty());

        if (labelsInput.isPresent()) {
            THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDimensions() == labelsInput.get().getDescriptor().getDimensions());
            THOR_THROW_IF_FALSE(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(featureInput.get().getPlacement() == labelsInput.get().getPlacement());
        }

        // Allocates this->featureInput and sets this->errorOutput to empty
        Layer::connectToPreviousLayer(featureInputLayer, featureInput, stream, false);

        // Metrics do not back propagate
        return Optional<Tensor>::empty();
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

        // Metrics do not back propagate
        return Optional<Tensor>::empty();
    }

    Optional<Tensor> createFeatureOutputTensor() override {
        if (isInferenceOnly()) {
            return Optional<Tensor>::empty();
        } else {
            THOR_THROW_IF_FALSE(featureInput.isPresent());
            return Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        }
    }

    virtual std::string toDisplayString(Tensor metric_h) = 0;

    ~Metric() override {}

    void initialize() override {
        Layer::initialize();
        featureInputReceived = false;
        labelsReceived = false;
    }

    void forward(Optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);
        if (isInferenceOnly())
            return;

        THOR_THROW_IF_FALSE(labelsStream.isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.isPresent());
        THOR_THROW_IF_FALSE(labelsStream.isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.get().isInitialized());
        THOR_THROW_IF_FALSE(featureOutput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(inputTensor.isPresent());

        if (inputTensor.get() == featureInput.get())
            forwardFeatures(inputTensor, validationPass);
        else if (inputTensor.get() == labelsInput.get())
            forwardLabels(inputTensor, validationPass);
        else
            THOR_UNREACHABLE();
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

    void backward(Optional<Tensor> errorInput, uint32_t batchSize = 0) override { THOR_UNREACHABLE(); }

    void ensureNoDeviceCrossing() override {
        if (featureInput.isPresent()) {
            if (labelsInput.isPresent())
                THOR_THROW_IF_FALSE(labelsInput.get().getPlacement() == featureInput.get().getPlacement());
            if (featureOutput.isPresent())
                THOR_THROW_IF_FALSE(featureOutput.get().getPlacement() == featureInput.get().getPlacement());
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

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
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

    void backProp(Optional<Tensor>, Optional<Tensor>, Optional<Tensor>, Stream) override { THOR_UNREACHABLE(); }
};

}  // namespace ThorImplementation
