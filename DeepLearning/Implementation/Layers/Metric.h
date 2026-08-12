#pragma once

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Metrics/MetricAggregation.h"

namespace ThorImplementation {

struct MetricBatchStatisticTensors {
    Thor::MetricAggregation aggregation = Thor::MetricAggregation::MEAN_BY_EXAMPLE;
    std::optional<Tensor> numerator;
    std::optional<Tensor> denominator;
    Event readyEvent;
};

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

    std::vector<Stream> getProcessingStreams() override {
        std::vector<Stream> processingStreams;
        if (stream.isInitialized())
            processingStreams.push_back(stream);
        if (labelsStream.isInitialized())
            processingStreams.push_back(labelsStream);
        return processingStreams;
    }

    std::vector<Event> getSynchronizeEvents() override {
        std::vector<Event> events;
        std::set<uint64_t> synchronizedStreamIds;
        appendSynchronizeEvent(events, synchronizedStreamIds, stream);
        appendSynchronizeEvent(events, synchronizedStreamIds, labelsStream);
        return events;
    }

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

    // Every declared metric exposes its aggregation contract here. Ratio
    // metrics additionally expose hidden sufficient-statistic tensors. Those
    // tensors are slot-local so queued submissions cannot overwrite statistics
    // that are still owned by a completion callback.
    virtual void preallocateMetricStatisticSlots(uint32_t numSlots) {
        THOR_THROW_IF_FALSE(numSlots >= 1);
    }
    virtual void setActiveMetricStatisticSlot(uint32_t slotIndex) { (void)slotIndex; }
    virtual std::optional<MetricBatchStatisticTensors> getMetricBatchStatisticTensorsForSlot(uint32_t slotIndex) const {
        (void)slotIndex;
        return std::nullopt;
    }
    virtual void extendMetricStatisticWritableEventForSlot(uint32_t slotIndex, Event event) {
        (void)slotIndex;
        (void)event;
    }

    ~Metric() override {}

    void initialize() override {
        Layer::initialize();
        featureInputReceived = false;
        labelsReceived = !requiresLabelsInput();
        currentValidExampleCount = 0;
        batchCardinalitySet = false;
    }

    void forward(std::optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount = 0) override {
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

        recordBatchCardinality(validExampleCount);
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

    virtual void computeMetric(
        Tensor labels, Tensor predictions, Tensor metric, Stream stream, uint32_t validExampleCount) = 0;

    enum class ConnectionType { FORWARD = 12, LABELS, METRIC };

   protected:
    std::optional<Tensor> labelsInput;
    Stream labelsStream;

    bool featureInputReceived;
    bool labelsReceived;
    uint32_t currentValidExampleCount = 0;
    bool batchCardinalitySet = false;

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        // Metrics use computeMetric(...) instead, due to different parameter requirements.
    }

    uint32_t getPhysicalBatchCapacity() const {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        const std::vector<uint64_t> dimensions = featureInput.value().getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty());
        THOR_THROW_IF_FALSE(dimensions.front() >= 1);
        THOR_THROW_IF_FALSE(dimensions.front() <= std::numeric_limits<uint32_t>::max());
        return static_cast<uint32_t>(dimensions.front());
    }

    void recordBatchCardinality(uint32_t validExampleCount) {
        const uint32_t physicalBatchCapacity = getPhysicalBatchCapacity();
        const uint32_t resolvedValidExampleCount =
            validExampleCount == 0 ? physicalBatchCapacity : validExampleCount;
        THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1);
        THOR_THROW_IF_FALSE(resolvedValidExampleCount <= physicalBatchCapacity);
        if (batchCardinalitySet) {
            if (currentValidExampleCount != resolvedValidExampleCount) {
                throw std::logic_error("Metric inputs for one batch must carry the same valid-example count.");
            }
            return;
        }
        currentValidExampleCount = resolvedValidExampleCount;
        batchCardinalitySet = true;
    }

    virtual void advanceDataIfReady(bool validationPass) {
        const bool ready = featureInputReceived && (!requiresLabelsInput() || labelsReceived);
        if (!ready)
            return;

        THOR_THROW_IF_FALSE(batchCardinalitySet);
        if (requiresLabelsInput()) {
            // The metric stream must wait until this batch's labels/weights have
            // arrived, and the labels stream must in turn wait until the metric has
            // finished consuming them before it may overwrite the statically
            // connected labels tensor for a later queued batch. Loss layers use the
            // same two-way stream handshake.
            stream.waitEvent(labelsStream.putEvent());
            computeMetric(labelsInput.value(), featureInput.value(), featureOutput.value(), stream, currentValidExampleCount);
            labelsStream.waitEvent(stream.putEvent());
            labelsReceived = false;
        } else {
            computeMetric(featureInput.value(), featureInput.value(), featureOutput.value(), stream, currentValidExampleCount);
            labelsReceived = true;
        }

        featureInputReceived = false;
        batchCardinalitySet = false;

        if (nextLayer.has_value())
            nextLayer.value()->forward(featureOutput, validationPass, currentValidExampleCount);
    }

    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream) override { THOR_UNREACHABLE(); }
};

}  // namespace ThorImplementation
