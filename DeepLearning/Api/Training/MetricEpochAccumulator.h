#pragma once

#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace Thor {

/**
 * Combines exact per-batch metric statistics according to the metric's declared
 * aggregation contract.
 *
 * Loss aggregation intentionally remains separate: graph losses continue to use
 * their existing valid-example-weighted mean semantics.
 */
class MetricEpochAccumulator {
   public:
    explicit MetricEpochAccumulator(MetricAggregation aggregation);

    void add(const MetricBatchStat& statistic);

    [[nodiscard]] MetricAggregation getAggregation() const { return aggregation; }
    [[nodiscard]] bool empty() const { return !hasValue; }
    [[nodiscard]] std::optional<double> value() const;
    [[nodiscard]] std::optional<MetricBatchStat> statistic() const;

   private:
    MetricAggregation aggregation;
    bool hasValue = false;
    uint64_t totalValidExamples = 0;
    long double accumulatedNumerator = 0.0L;
    long double accumulatedDenominator = 0.0L;
    double extremum = 0.0;
};

/**
 * Owns one MetricEpochAccumulator per reported metric name. This keeps the
 * aggregation contract check and the conversion back to display/statistic maps
 * identical across queued training, Trainer observers, and composed evaluation.
 */
class MetricEpochAccumulatorMap {
   public:
    void clear() { accumulators.clear(); }
    [[nodiscard]] bool empty() const { return accumulators.empty(); }
    [[nodiscard]] bool contains(const std::string& metricName) const {
        return accumulators.contains(metricName);
    }

    void registerMetric(const std::string& metricName, MetricAggregation aggregation);
    void add(const std::string& metricName, const MetricBatchStat& statistic);

    [[nodiscard]] std::optional<double> value(const std::string& metricName) const;
    [[nodiscard]] std::optional<MetricBatchStat> statistic(const std::string& metricName) const;
    [[nodiscard]] std::unordered_map<std::string, double> values() const;
    [[nodiscard]] std::unordered_map<std::string, MetricBatchStat> statistics() const;

   private:
    std::unordered_map<std::string, MetricEpochAccumulator> accumulators{};
};

/**
 * Returns the exact statistic attached to a reported metric. Scalar reports that
 * are not backed by a declared Metric retain Thor's historical mean-by-example
 * behavior.
 */
[[nodiscard]] MetricBatchStat resolveMetricBatchStat(const TrainingStatsSnapshot& snapshot,
                                                      const std::string& metricName,
                                                      double reportedValue);

}  // namespace Thor
