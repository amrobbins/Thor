#include "DeepLearning/Api/Training/MetricEpochAccumulator.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Thor {

MetricEpochAccumulator::MetricEpochAccumulator(MetricAggregation aggregation)
    : aggregation(aggregation) {}

void MetricEpochAccumulator::add(const MetricBatchStat& statistic) {
    THOR_THROW_IF_FALSE(statistic.aggregation == aggregation);
    THOR_THROW_IF_FALSE(statistic.validExamples > 0);
    const bool hasNumerator = statistic.numerator.has_value();
    const bool hasDenominator = statistic.denominator.has_value();
    THOR_THROW_IF_FALSE(hasNumerator == hasDenominator);
    if (aggregation == MetricAggregation::RATIO) {
        THOR_THROW_IF_FALSE(hasNumerator);
        const double numerator = statistic.numerator.value();
        const double denominator = statistic.denominator.value();
        const double expectedValue = denominator == 0.0 ? 0.0 : numerator / denominator;
        // Ratio metric outputs and their sufficient statistics are FP32 scalars.
        // Permit ordinary FP32 rounding, but reject a public display scalar that
        // describes a different ratio than the statistics used for aggregation.
        const bool valuesMatch =
            (std::isnan(statistic.value) && std::isnan(expectedValue)) ||
            (std::isinf(statistic.value) && std::isinf(expectedValue) &&
             std::signbit(statistic.value) == std::signbit(expectedValue)) ||
            (std::isfinite(statistic.value) && std::isfinite(expectedValue) &&
             std::abs(statistic.value - expectedValue) <=
                 32.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
                     std::max({1.0, std::abs(statistic.value), std::abs(expectedValue)}));
        THOR_THROW_IF_FALSE(valuesMatch);
    } else {
        THOR_THROW_IF_FALSE(!hasNumerator);
    }

    totalValidExamples += statistic.validExamples;
    switch (aggregation) {
        case MetricAggregation::MEAN_BY_EXAMPLE:
            accumulatedNumerator +=
                static_cast<long double>(statistic.value) *
                static_cast<long double>(statistic.validExamples);
            accumulatedDenominator +=
                static_cast<long double>(statistic.validExamples);
            hasValue = true;
            return;
        case MetricAggregation::SUM:
            accumulatedNumerator += static_cast<long double>(statistic.value);
            hasValue = true;
            return;
        case MetricAggregation::MIN:
            if (!hasValue) {
                extremum = statistic.value;
            } else if (!std::isnan(extremum)) {
                extremum = std::isnan(statistic.value)
                               ? statistic.value
                               : std::min(extremum, statistic.value);
            }
            hasValue = true;
            return;
        case MetricAggregation::MAX:
            if (!hasValue) {
                extremum = statistic.value;
            } else if (!std::isnan(extremum)) {
                extremum = std::isnan(statistic.value)
                               ? statistic.value
                               : std::max(extremum, statistic.value);
            }
            hasValue = true;
            return;
        case MetricAggregation::RATIO:
            THOR_THROW_IF_FALSE(statistic.numerator.has_value());
            THOR_THROW_IF_FALSE(statistic.denominator.has_value());
            accumulatedNumerator +=
                static_cast<long double>(statistic.numerator.value());
            accumulatedDenominator +=
                static_cast<long double>(statistic.denominator.value());
            hasValue = true;
            return;
    }
    THOR_UNREACHABLE();
}

std::optional<double> MetricEpochAccumulator::value() const {
    if (!hasValue) {
        return std::nullopt;
    }

    switch (aggregation) {
        case MetricAggregation::MEAN_BY_EXAMPLE:
            THOR_THROW_IF_FALSE(accumulatedDenominator > 0.0L);
            return static_cast<double>(
                accumulatedNumerator / accumulatedDenominator);
        case MetricAggregation::SUM:
            return static_cast<double>(accumulatedNumerator);
        case MetricAggregation::MIN:
        case MetricAggregation::MAX:
            return extremum;
        case MetricAggregation::RATIO:
            return accumulatedDenominator == 0.0L
                       ? 0.0
                       : static_cast<double>(
                             accumulatedNumerator / accumulatedDenominator);
    }
    THOR_UNREACHABLE();
}

std::optional<MetricBatchStat> MetricEpochAccumulator::statistic() const {
    const std::optional<double> combinedValue = value();
    if (!combinedValue.has_value()) {
        return std::nullopt;
    }

    MetricBatchStat result;
    result.aggregation = aggregation;
    result.value = combinedValue.value();
    result.validExamples = totalValidExamples;
    if (aggregation == MetricAggregation::RATIO) {
        result.numerator = static_cast<double>(accumulatedNumerator);
        result.denominator = static_cast<double>(accumulatedDenominator);
    }
    return result;
}

void MetricEpochAccumulatorMap::registerMetric(const std::string& metricName,
                                                MetricAggregation aggregation) {
    THOR_THROW_IF_FALSE(!metricName.empty());
    THOR_THROW_IF_FALSE(
        accumulators.try_emplace(metricName, aggregation).second);
}

void MetricEpochAccumulatorMap::add(const std::string& metricName,
                                    const MetricBatchStat& statistic) {
    THOR_THROW_IF_FALSE(!metricName.empty());
    auto [it, inserted] = accumulators.try_emplace(
        metricName, statistic.aggregation);
    (void)inserted;
    it->second.add(statistic);
}

std::optional<double> MetricEpochAccumulatorMap::value(
    const std::string& metricName) const {
    const auto it = accumulators.find(metricName);
    return it == accumulators.end() ? std::optional<double>{} : it->second.value();
}

std::optional<MetricBatchStat> MetricEpochAccumulatorMap::statistic(
    const std::string& metricName) const {
    const auto it = accumulators.find(metricName);
    return it == accumulators.end() ? std::optional<MetricBatchStat>{}
                                    : it->second.statistic();
}

std::unordered_map<std::string, double> MetricEpochAccumulatorMap::values() const {
    std::unordered_map<std::string, double> result;
    result.reserve(accumulators.size());
    for (const auto& [name, accumulator] : accumulators) {
        const std::optional<double> combined = accumulator.value();
        if (combined.has_value()) {
            result.emplace(name, combined.value());
        }
    }
    return result;
}

std::unordered_map<std::string, MetricBatchStat>
MetricEpochAccumulatorMap::statistics() const {
    std::unordered_map<std::string, MetricBatchStat> result;
    result.reserve(accumulators.size());
    for (const auto& [name, accumulator] : accumulators) {
        const std::optional<MetricBatchStat> combined = accumulator.statistic();
        if (combined.has_value()) {
            result.emplace(name, combined.value());
        }
    }
    return result;
}

MetricBatchStat resolveMetricBatchStat(const TrainingStatsSnapshot& snapshot,
                                       const std::string& metricName,
                                       double reportedValue) {
    const auto exact = snapshot.metricBatchStats.find(metricName);
    if (exact != snapshot.metricBatchStats.end()) {
        MetricBatchStat statistic = exact->second;
        THOR_THROW_IF_FALSE(statistic.validExamples > 0);
        THOR_THROW_IF_FALSE(
            statistic.value == reportedValue ||
            (std::isnan(statistic.value) && std::isnan(reportedValue)));
        return statistic;
    }

    THOR_THROW_IF_FALSE(snapshot.validExamplesInBatch > 0);
    MetricBatchStat fallback;
    fallback.aggregation = MetricAggregation::MEAN_BY_EXAMPLE;
    fallback.value = reportedValue;
    fallback.validExamples = snapshot.validExamplesInBatch;
    return fallback;
}

}  // namespace Thor
