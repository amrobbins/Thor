#pragma once

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>

namespace Thor {

// Reserved expression outputs used to carry sufficient statistics for ratio metrics.
// They are internal execution outputs and never become public API tensors or NetworkOutputs.
inline constexpr const char* METRIC_AGGREGATION_NUMERATOR_NAME = "__thor_metric_aggregation_numerator";
inline constexpr const char* METRIC_AGGREGATION_DENOMINATOR_NAME = "__thor_metric_aggregation_denominator";

/** Defines how a metric's scalar batch result combines across an epoch. */
enum class MetricAggregation {
    MEAN_BY_EXAMPLE,
    SUM,
    MIN,
    MAX,
    RATIO,
};

inline const char* metricAggregationName(MetricAggregation aggregation) {
    switch (aggregation) {
        case MetricAggregation::MEAN_BY_EXAMPLE:
            return "MEAN_BY_EXAMPLE";
        case MetricAggregation::SUM:
            return "SUM";
        case MetricAggregation::MIN:
            return "MIN";
        case MetricAggregation::MAX:
            return "MAX";
        case MetricAggregation::RATIO:
            return "RATIO";
    }
    throw std::invalid_argument("Unknown MetricAggregation value.");
}

inline MetricAggregation metricAggregationFromString(const std::string& value) {
    if (value == "MEAN_BY_EXAMPLE")
        return MetricAggregation::MEAN_BY_EXAMPLE;
    if (value == "SUM")
        return MetricAggregation::SUM;
    if (value == "MIN")
        return MetricAggregation::MIN;
    if (value == "MAX")
        return MetricAggregation::MAX;
    if (value == "RATIO")
        return MetricAggregation::RATIO;
    throw std::invalid_argument("Unknown MetricAggregation value: " + value);
}

inline void to_json(nlohmann::json& j, const MetricAggregation& aggregation) {
    j = metricAggregationName(aggregation);
}

inline void from_json(const nlohmann::json& j, MetricAggregation& aggregation) {
    aggregation = metricAggregationFromString(j.get<std::string>());
}

}  // namespace Thor
