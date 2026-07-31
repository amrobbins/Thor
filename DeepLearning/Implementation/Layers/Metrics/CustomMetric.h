#pragma once

#include "DeepLearning/Api/BatchValidity.h"
#include "DeepLearning/Api/Layers/Metrics/MetricAggregation.h"
#include "DeepLearning/Implementation/Layers/Metric.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ThorImplementation {

class CustomMetric : public Metric {
   public:
    CustomMetric(DynamicExpression expr,
                 std::string predictionsName = "predictions",
                 std::string labelsName = "labels",
                 std::string metricName = "metric",
                 std::string displayName = "Metric",
                 Thor::MetricAggregation aggregation = Thor::MetricAggregation::MEAN_BY_EXAMPLE,
                 std::optional<std::string> batchValidityMaskName = std::nullopt);

    ~CustomMetric() override = default;

    bool requiresLabelsInput() const override { return !labelsName.empty(); }
    bool supportsPartialBatches() const override { return batchValidityMaskName.has_value(); }
    std::optional<Tensor> connectToFeatureInputLayer(Layer* featureInputLayer,
                                                     std::optional<Tensor> featureInput,
                                                     Stream stream,
                                                     bool backPropagateError) override;
    std::optional<Tensor> createFeatureOutputTensor() override;
    void compileImpl() override;
    void cleanup() override;
    void computeMetric(Tensor labels,
                       Tensor predictions,
                       Tensor metric,
                       Stream stream,
                       uint32_t validExampleCount) override;
    std::string toDisplayString(Tensor metric_h) override;

    void preallocateMetricStatisticSlots(uint32_t numSlots) override;
    void setActiveMetricStatisticSlot(uint32_t slotIndex) override;
    std::optional<MetricBatchStatisticTensors> getMetricBatchStatisticTensorsForSlot(uint32_t slotIndex) const override;
    void extendMetricStatisticWritableEventForSlot(uint32_t slotIndex, Event event) override;
    std::vector<Event> getSynchronizeEvents() override;

    std::string getType() override { return "CustomMetric"; }

   private:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    struct OutputDescriptor {
        std::vector<uint64_t> dimensions;
        DataType dataType = DataType::FP32;
    };

    struct RatioStatisticSlot {
        Tensor numeratorHost;
        Tensor denominatorHost;
        Tensor numeratorBuffer;
        Tensor denominatorBuffer;
        Event bufferReadyEvent;
        Event readyEvent;
        Event writableEvent;
    };

    TensorMap buildMetricInputs() const;
    TensorMap buildMetricOutputs() const;
    void validateMetricOutputNames(const std::vector<std::string>& outputNames) const;
    std::unordered_map<std::string, OutputDescriptor> inferMetricOutputDescriptors() const;
    std::vector<std::string> expectedMetricOutputNames() const;
    bool isRatioMetric() const { return aggregation == Thor::MetricAggregation::RATIO; }
    void allocateRatioStatisticSlots(uint32_t numSlots);
    void captureRatioStatistics();
    void requireRatioStatisticSlot(uint32_t slotIndex) const;

    DynamicExpression metricExpression;
    std::string predictionsName;
    std::string labelsName;
    std::string metricName;
    std::string displayName;
    Thor::MetricAggregation aggregation;
    std::optional<std::string> batchValidityMaskName;
    Tensor batchValidityMask;
    Tensor ratioNumerator;
    Tensor ratioDenominator;

    std::shared_ptr<PreparedDynamicExpression> metricPrepared;
    std::shared_ptr<StampedExecutionPlan> metricStamped;
    std::function<void(Stream&)> metricPreRunHook;

    uint32_t activeMetricStatisticSlot = 0;
    std::vector<RatioStatisticSlot> ratioStatisticSlots;
    std::optional<Stream> ratioStatisticDownloadStream;
};

}  // namespace ThorImplementation
