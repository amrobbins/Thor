#pragma once

#include <optional>
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

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) override {
        if (connectionType == (int)ConnectionType::FORWARD) {
            return connectToFeatureInputLayer(previousLayer, featureInput, stream, backPropagateError);
        } else if (connectionType == (int)ConnectionType::LABELS) {
            THOR_THROW_IF_FALSE(requiresLabelsInput());
            return connectToLabelsInputLayer(previousLayer, featureInput, stream);
        } else {
            THOR_UNREACHABLE();
        }
    }

    virtual bool requiresLabelsInput() const { return true; }

    virtual std::optional<Tensor> connectToFeatureInputLayer(Layer *featureInputLayer,
                                                        std::optional<Tensor> featureInput,
                                                        Stream stream,
                                                        bool backPropagateError) {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions().size() >= 2);
        THOR_THROW_IF_FALSE(!this->featureInput.has_value());

        if (labelsInput.has_value()) {
            THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == labelsInput.value().getDescriptor().getDimensions());
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelsInput.value().getPlacement());
        }

        // Allocates this->featureInput and sets this->errorOutput to empty
        Layer::connectToPreviousLayer(featureInputLayer, featureInput, stream, false);

        // Metrics do not back propagate
        return std::nullopt;
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

        // Metrics do not back propagate
        return std::nullopt;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        // Metrics are forward-only, but they are still meaningful during
        // inference/evaluation. TrainingRuns uses inference-only composed
        // evaluator networks to report graph metrics, so a metric output must
        // be materialized even when the placed network is inference-only.
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::FP32, {1}));
    }

    virtual std::string toDisplayString(Tensor metric_h) = 0;

    ~Metric() override {}

    void initialize() override {
        Layer::initialize();
        featureInputReceived = false;
        labelsReceived = !requiresLabelsInput();
    }

    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(running);

        if (requiresLabelsInput()) {
            THOR_THROW_IF_FALSE(labelsStream.isInitialized());
            THOR_THROW_IF_FALSE(labelsInput.has_value());
            THOR_THROW_IF_FALSE(labelsStream.isInitialized());
            THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        }
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(inputTensor.has_value());

        if (inputTensor.value() == featureInput.value())
            forwardFeatures(inputTensor.value(), validationPass);
        else if (requiresLabelsInput() && labelsInput.has_value() && inputTensor.value() == labelsInput.value())
            forwardLabels(inputTensor.value(), validationPass);
        else
            THOR_UNREACHABLE();
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

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override { THOR_UNREACHABLE(); }

    void ensureNoDeviceCrossing() override {
        if (featureInput.has_value()) {
            if (labelsInput.has_value())
                THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == featureInput.value().getPlacement());
            if (featureOutput.has_value())
                THOR_THROW_IF_FALSE(featureOutput.value().getPlacement() == featureInput.value().getPlacement());
        }
    }

    virtual std::optional<Tensor> getLabelsInput() { return labelsInput; }

    virtual void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) = 0;

    enum class ConnectionType { FORWARD = 12, LABELS, METRIC };

   protected:
    std::optional<Tensor> labelsInput;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        // Metrics use computeMetric(...) instead, due to different parameter requirements.
    }

    virtual void advanceDataIfReady(bool validationPass) {
        const bool ready = featureInputReceived && (!requiresLabelsInput() || labelsReceived);
        if (!ready)
            return;

        if (requiresLabelsInput()) {
            // DataStream waits for labels to arrive,
            stream.waitEvent(labelsStream.putEvent());
            computeMetric(labelsInput.value(), featureInput.value(), featureOutput.value(), stream);
            labelsReceived = false;
        } else {
            computeMetric(featureInput.value(), featureInput.value(), featureOutput.value(), stream);
            labelsReceived = true;
        }

        featureInputReceived = false;

        if (nextLayer.has_value())
            nextLayer.value()->forward(featureOutput, validationPass);
    }

    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream) override { THOR_UNREACHABLE(); }
};

}  // namespace ThorImplementation
