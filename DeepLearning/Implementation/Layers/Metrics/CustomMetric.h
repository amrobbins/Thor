#pragma once

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
                 std::string displayName = "Metric");

    ~CustomMetric() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    void compileImpl() override;
    void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) override;
    std::string toDisplayString(Tensor metric_h) override;

    std::string getType() override { return "CustomMetric"; }

   private:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    TensorMap buildMetricInputs() const;
    TensorMap buildMetricOutputs() const;
    void validateMetricOutputNames(const std::vector<std::string>& outputNames) const;
    std::pair<std::vector<uint64_t>, DataType> inferMetricOutputDescriptor() const;

    DynamicExpression metricExpression;
    std::string predictionsName;
    std::string labelsName;
    std::string metricName;
    std::string displayName;

    std::shared_ptr<PreparedDynamicExpression> metricPrepared;
    std::shared_ptr<StampedExecutionPlan> metricStamped;
    std::function<void(Stream&)> metricPreRunHook;
};

}  // namespace ThorImplementation
