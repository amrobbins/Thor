#include "DeepLearning/Api/Training/MetricEpochAccumulator.h"

#include "gtest/gtest.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

using namespace Thor;

namespace {

MetricBatchStat batchStat(MetricAggregation aggregation,
                          double value,
                          uint64_t validExamples,
                          std::optional<double> numerator = std::nullopt,
                          std::optional<double> denominator = std::nullopt,
                          bool hasContribution = true,
                          bool zeroDenominatorMeansNoContribution = false) {
    MetricBatchStat statistic;
    statistic.aggregation = aggregation;
    statistic.value = value;
    statistic.validExamples = validExamples;
    statistic.numerator = numerator;
    statistic.denominator = denominator;
    statistic.hasContribution = hasContribution;
    statistic.zeroDenominatorMeansNoContribution = zeroDenominatorMeansNoContribution;
    return statistic;
}

}  // namespace

TEST(MetricEpochAccumulator, CombinesUnevenTailBatchesByDeclaredContract) {
    MetricEpochAccumulator mean(MetricAggregation::MEAN_BY_EXAMPLE);
    mean.add(batchStat(MetricAggregation::MEAN_BY_EXAMPLE, 2.5, 4));
    mean.add(batchStat(MetricAggregation::MEAN_BY_EXAMPLE, 10.0, 2));
    ASSERT_TRUE(mean.value().has_value());
    EXPECT_DOUBLE_EQ(mean.value().value(), 5.0);

    MetricEpochAccumulator sum(MetricAggregation::SUM);
    sum.add(batchStat(MetricAggregation::SUM, 10.0, 4));
    sum.add(batchStat(MetricAggregation::SUM, 20.0, 2));
    ASSERT_TRUE(sum.value().has_value());
    EXPECT_DOUBLE_EQ(sum.value().value(), 30.0);

    MetricEpochAccumulator minimum(MetricAggregation::MIN);
    minimum.add(batchStat(MetricAggregation::MIN, 1.0, 4));
    minimum.add(batchStat(MetricAggregation::MIN, -3.0, 2));
    ASSERT_TRUE(minimum.value().has_value());
    EXPECT_DOUBLE_EQ(minimum.value().value(), -3.0);

    MetricEpochAccumulator maximum(MetricAggregation::MAX);
    maximum.add(batchStat(MetricAggregation::MAX, 4.0, 4));
    maximum.add(batchStat(MetricAggregation::MAX, 11.0, 2));
    ASSERT_TRUE(maximum.value().has_value());
    EXPECT_DOUBLE_EQ(maximum.value().value(), 11.0);

    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    ratio.add(batchStat(MetricAggregation::RATIO, 5.0, 4, 10.0, 2.0));
    ratio.add(batchStat(MetricAggregation::RATIO, 10.0, 2, 90.0, 9.0));
    ASSERT_TRUE(ratio.value().has_value());
    EXPECT_NEAR(ratio.value().value(), 100.0 / 11.0, 1e-12);
}


TEST(MetricEpochAccumulator, R10LRaggedMeanAggregatesActiveScalarsAcrossUnequalAndEmptyBatches) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);

    // Two active scalars with mean 5.0.
    ratio.add(batchStat(MetricAggregation::RATIO, 5.0, 4, 10.0, 2.0));
    // A logically valid all-empty ragged batch contributes zero active scalars.
    ratio.add(batchStat(MetricAggregation::RATIO, 0.0, 4, 0.0, 0.0));
    // Eighteen active scalars with mean 10.0. The exact epoch mean is
    // (10 + 180) / (2 + 18) = 9.5, while averaging the two nonempty
    // batch means would incorrectly produce 7.5.
    ratio.add(batchStat(MetricAggregation::RATIO, 10.0, 2, 180.0, 18.0));

    ASSERT_TRUE(ratio.value().has_value());
    EXPECT_DOUBLE_EQ(ratio.value().value(), 9.5);
    const std::optional<MetricBatchStat> combined = ratio.statistic();
    ASSERT_TRUE(combined.has_value());
    ASSERT_TRUE(combined->numerator.has_value());
    ASSERT_TRUE(combined->denominator.has_value());
    EXPECT_DOUBLE_EQ(combined->numerator.value(), 190.0);
    EXPECT_DOUBLE_EQ(combined->denominator.value(), 20.0);
    EXPECT_EQ(combined->validExamples, 10u);
}


TEST(MetricEpochAccumulator, R10MExtremaIgnoreNoContributionBatches) {
    MetricEpochAccumulator minimum(MetricAggregation::MIN);
    minimum.add(batchStat(MetricAggregation::MIN, 4.0, 3));
    minimum.add(batchStat(MetricAggregation::MIN, 0.0, 5, std::nullopt, std::nullopt, false));
    minimum.add(batchStat(MetricAggregation::MIN, 6.0, 2));
    ASSERT_TRUE(minimum.value().has_value());
    // A fake zero for the empty middle batch would incorrectly win here.
    EXPECT_DOUBLE_EQ(minimum.value().value(), 4.0);

    MetricEpochAccumulator maximum(MetricAggregation::MAX);
    maximum.add(batchStat(MetricAggregation::MAX, -7.0, 3));
    maximum.add(batchStat(MetricAggregation::MAX, 0.0, 5, std::nullopt, std::nullopt, false));
    maximum.add(batchStat(MetricAggregation::MAX, -3.0, 2));
    ASSERT_TRUE(maximum.value().has_value());
    EXPECT_DOUBLE_EQ(maximum.value().value(), -3.0);

    // The extrema value ignores the empty batch, while validExamples still
    // records all logical rows processed in the population.
    ASSERT_TRUE(minimum.statistic().has_value());
    EXPECT_EQ(minimum.statistic()->validExamples, 10u);
    ASSERT_TRUE(maximum.statistic().has_value());
    EXPECT_EQ(maximum.statistic()->validExamples, 10u);
}

TEST(MetricEpochAccumulator, R10MAllEmptyExtremaPopulationHasNoResult) {
    MetricEpochAccumulator minimum(MetricAggregation::MIN);
    minimum.add(batchStat(MetricAggregation::MIN, 0.0, 4, std::nullopt, std::nullopt, false));
    minimum.add(batchStat(MetricAggregation::MIN, 0.0, 2, std::nullopt, std::nullopt, false));
    EXPECT_TRUE(minimum.empty());
    EXPECT_FALSE(minimum.value().has_value());
    EXPECT_FALSE(minimum.statistic().has_value());

    MetricEpochAccumulator maximum(MetricAggregation::MAX);
    maximum.add(batchStat(MetricAggregation::MAX, 0.0, 4, std::nullopt, std::nullopt, false));
    EXPECT_TRUE(maximum.empty());
    EXPECT_FALSE(maximum.value().has_value());
    EXPECT_FALSE(maximum.statistic().has_value());
}

TEST(MetricEpochAccumulator, R10MNoContributionIsExtremaOnly) {
    MetricEpochAccumulator sum(MetricAggregation::SUM);
    EXPECT_THROW(sum.add(batchStat(MetricAggregation::SUM, 0.0, 4, std::nullopt, std::nullopt, false)),
                 std::logic_error);
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    EXPECT_THROW(ratio.add(batchStat(MetricAggregation::RATIO, 0.0, 4, 0.0, 0.0, false)),
                 std::logic_error);
}

TEST(MetricEpochAccumulator, R10NRatioNoContributionRequiresMetricOptIn) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    EXPECT_NO_THROW(ratio.add(
        batchStat(MetricAggregation::RATIO, 0.0, 4, 0.0, 0.0, false, true)));
    EXPECT_FALSE(ratio.value().has_value());
}

TEST(MetricEpochAccumulator, RatioRetainsNumeratorFromZeroDenominatorBatch) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    ratio.add(batchStat(MetricAggregation::RATIO, 0.0, 4, 10.0, 0.0));
    ratio.add(batchStat(MetricAggregation::RATIO, 5.0, 2, 20.0, 4.0));

    ASSERT_TRUE(ratio.value().has_value());
    EXPECT_DOUBLE_EQ(ratio.value().value(), 7.5);
    const std::optional<MetricBatchStat> combined = ratio.statistic();
    ASSERT_TRUE(combined.has_value());
    ASSERT_TRUE(combined->numerator.has_value());
    ASSERT_TRUE(combined->denominator.has_value());
    EXPECT_DOUBLE_EQ(combined->numerator.value(), 30.0);
    EXPECT_DOUBLE_EQ(combined->denominator.value(), 4.0);
    EXPECT_EQ(combined->validExamples, 6u);
}

TEST(MetricEpochAccumulator, RatioReturnsZeroWhenSignedWeightsCancel) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    ratio.add(batchStat(MetricAggregation::RATIO, 2.0, 4, 4.0, 2.0));
    ratio.add(batchStat(MetricAggregation::RATIO, 3.0, 2, -6.0, -2.0));

    ASSERT_TRUE(ratio.value().has_value());
    EXPECT_DOUBLE_EQ(ratio.value().value(), 0.0);
    const std::optional<MetricBatchStat> combined = ratio.statistic();
    ASSERT_TRUE(combined.has_value());
    EXPECT_DOUBLE_EQ(combined->numerator.value(), -2.0);
    EXPECT_DOUBLE_EQ(combined->denominator.value(), 0.0);
}

TEST(MetricEpochAccumulator, R10NOptedInRatioHasNoResultWhenEpochWeightCancelsToZero) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    ratio.add(batchStat(MetricAggregation::RATIO, 2.0, 4, 4.0, 2.0, true, true));
    ratio.add(batchStat(MetricAggregation::RATIO, 3.0, 2, -6.0, -2.0, true, true));

    EXPECT_FALSE(ratio.value().has_value());
    EXPECT_FALSE(ratio.statistic().has_value());
}

TEST(MetricEpochAccumulator, RejectsMissingRatioStatisticsAndAggregationChanges) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    EXPECT_THROW(
        ratio.add(batchStat(MetricAggregation::RATIO, 1.0, 1)),
        std::logic_error);

    MetricEpochAccumulator sum(MetricAggregation::SUM);
    EXPECT_THROW(
        sum.add(batchStat(MetricAggregation::MAX, 1.0, 1)),
        std::logic_error);
}

TEST(MetricEpochAccumulator, UnrecognizedScalarFallsBackToMeanByExample) {
    TrainingStatsSnapshot snapshot;
    snapshot.validExamplesInBatch = 2;

    const MetricBatchStat statistic =
        resolveMetricBatchStat(snapshot, "legacy_scalar", 3.5);
    EXPECT_EQ(statistic.aggregation, MetricAggregation::MEAN_BY_EXAMPLE);
    EXPECT_DOUBLE_EQ(statistic.value, 3.5);
    EXPECT_EQ(statistic.validExamples, 2u);
    EXPECT_FALSE(statistic.numerator.has_value());
    EXPECT_FALSE(statistic.denominator.has_value());
}

TEST(MetricEpochAccumulator, RejectsStatisticsThatDoNotMatchAggregationShape) {
    MetricEpochAccumulator sum(MetricAggregation::SUM);
    EXPECT_THROW(
        sum.add(batchStat(MetricAggregation::SUM, 3.0, 2, 3.0, 1.0)),
        std::logic_error);

    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    EXPECT_THROW(
        ratio.add(batchStat(MetricAggregation::RATIO, 3.0, 2, 3.0, std::nullopt)),
        std::logic_error);
    EXPECT_THROW(
        ratio.add(batchStat(MetricAggregation::RATIO, 3.0, 2, std::nullopt, 1.0)),
        std::logic_error);
}

TEST(MetricEpochAccumulatorMap, CombinesNamedMetricsAndExportsExactStatistics) {
    MetricEpochAccumulatorMap accumulators;
    accumulators.registerMetric("sum", MetricAggregation::SUM);
    accumulators.registerMetric("ratio", MetricAggregation::RATIO);

    accumulators.add("sum", batchStat(MetricAggregation::SUM, 4.0, 4));
    accumulators.add("sum", batchStat(MetricAggregation::SUM, 2.0, 2));
    accumulators.add("ratio", batchStat(MetricAggregation::RATIO, 0.0, 4, 10.0, 0.0));
    accumulators.add("ratio", batchStat(MetricAggregation::RATIO, 5.0, 2, 20.0, 4.0));

    const auto values = accumulators.values();
    ASSERT_EQ(values.size(), 2u);
    EXPECT_DOUBLE_EQ(values.at("sum"), 6.0);
    EXPECT_DOUBLE_EQ(values.at("ratio"), 7.5);

    const auto statistics = accumulators.statistics();
    ASSERT_EQ(statistics.size(), 2u);
    EXPECT_EQ(statistics.at("sum").validExamples, 6u);
    EXPECT_FALSE(statistics.at("sum").numerator.has_value());
    EXPECT_FALSE(statistics.at("sum").denominator.has_value());
    EXPECT_EQ(statistics.at("ratio").validExamples, 6u);
    ASSERT_TRUE(statistics.at("ratio").numerator.has_value());
    ASSERT_TRUE(statistics.at("ratio").denominator.has_value());
    EXPECT_DOUBLE_EQ(statistics.at("ratio").numerator.value(), 30.0);
    EXPECT_DOUBLE_EQ(statistics.at("ratio").denominator.value(), 4.0);
}

TEST(MetricEpochAccumulatorMap, RejectsDuplicateRegistrationAndContractChanges) {
    MetricEpochAccumulatorMap accumulators;
    accumulators.registerMetric("metric", MetricAggregation::SUM);
    EXPECT_THROW(
        accumulators.registerMetric("metric", MetricAggregation::SUM),
        std::logic_error);
    EXPECT_THROW(
        accumulators.add(
            "metric", batchStat(MetricAggregation::MAX, 1.0, 1)),
        std::logic_error);
}

TEST(MetricEpochAccumulator, ExactStatisticMustMatchReportedMetricValue) {
    TrainingStatsSnapshot snapshot;
    snapshot.validExamplesInBatch = 2;
    snapshot.metricBatchStats.emplace(
        "sum", batchStat(MetricAggregation::SUM, 3.0, 2));

    EXPECT_NO_THROW((void)resolveMetricBatchStat(snapshot, "sum", 3.0));
    EXPECT_THROW((void)resolveMetricBatchStat(snapshot, "sum", 4.0), std::logic_error);
}

TEST(MetricEpochAccumulator, RatioRequiresPublicValueToMatchSufficientStatistics) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);

    EXPECT_THROW(
        ratio.add(batchStat(MetricAggregation::RATIO, 4.0, 2, 10.0, 2.0)),
        std::logic_error);
}

TEST(MetricEpochAccumulator, RatioAcceptsFp32ScaleRoundingDifference) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);
    const float numerator = 10.0f;
    const float denominator = 3.0f;
    const float displayedValue = numerator / denominator;

    EXPECT_NO_THROW(ratio.add(batchStat(MetricAggregation::RATIO,
                                        static_cast<double>(displayedValue),
                                        2,
                                        static_cast<double>(numerator),
                                        static_cast<double>(denominator))));
}

TEST(MetricEpochAccumulator, R10NWeightedMeanAggregatesGlobalWeightedSufficientStatistics) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);

    // Local weighted means 2 and 10 with very different denominator mass.
    ratio.add(batchStat(MetricAggregation::RATIO, 2.0, 4, 2.0, 1.0, true, true));
    // A zero-weight ragged WeightedMean batch is explicitly no-contribution.
    // It retains logical-row population accounting without making an undefined
    // local 0/0 ratio part of the epoch sufficient statistics.
    ratio.add(batchStat(MetricAggregation::RATIO, 0.0, 5, 0.0, 0.0, false, true));
    ratio.add(batchStat(MetricAggregation::RATIO, 10.0, 2, 900.0, 90.0, true, true));

    ASSERT_TRUE(ratio.value().has_value());
    EXPECT_NEAR(ratio.value().value(), 902.0 / 91.0, 1e-12);
    // Averaging the two nonzero local ratios would incorrectly produce 6.
    EXPECT_NE(ratio.value().value(), 6.0);
    const std::optional<MetricBatchStat> combined = ratio.statistic();
    ASSERT_TRUE(combined.has_value());
    EXPECT_DOUBLE_EQ(combined->numerator.value(), 902.0);
    EXPECT_DOUBLE_EQ(combined->denominator.value(), 91.0);
    EXPECT_EQ(combined->validExamples, 11u);
}

TEST(MetricEpochAccumulator, R10NZeroWeightRatioPopulationHasNoResultUntilAContributingBatchArrives) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);

    ratio.add(batchStat(MetricAggregation::RATIO, 0.0, 4, 0.0, 0.0, false, true));
    EXPECT_FALSE(ratio.value().has_value());
    EXPECT_FALSE(ratio.statistic().has_value());

    ratio.add(batchStat(MetricAggregation::RATIO, 3.0, 2, 15.0, 5.0, true, true));
    ASSERT_TRUE(ratio.value().has_value());
    EXPECT_DOUBLE_EQ(ratio.value().value(), 3.0);
    const std::optional<MetricBatchStat> combined = ratio.statistic();
    ASSERT_TRUE(combined.has_value());
    EXPECT_DOUBLE_EQ(combined->numerator.value(), 15.0);
    EXPECT_DOUBLE_EQ(combined->denominator.value(), 5.0);
    EXPECT_EQ(combined->validExamples, 6u);
}

TEST(MetricEpochAccumulator, R10OAccuracyAggregatesCorrectAndTokenCountsAcrossUnequalBatches) {
    MetricEpochAccumulator ratio(MetricAggregation::RATIO);

    // Batch-local accuracies are 1/2 and 9/10. Averaging them would produce
    // 0.7, but the correct population accuracy is 10/12.
    ratio.add(batchStat(MetricAggregation::RATIO, 0.5, 4, 1.0, 2.0));
    ratio.add(batchStat(MetricAggregation::RATIO, 0.9, 4, 9.0, 10.0));

    ASSERT_TRUE(ratio.value().has_value());
    EXPECT_NEAR(ratio.value().value(), 10.0 / 12.0, 1e-12);
    const std::optional<MetricBatchStat> combined = ratio.statistic();
    ASSERT_TRUE(combined.has_value());
    ASSERT_TRUE(combined->numerator.has_value());
    ASSERT_TRUE(combined->denominator.has_value());
    EXPECT_DOUBLE_EQ(combined->numerator.value(), 10.0);
    EXPECT_DOUBLE_EQ(combined->denominator.value(), 12.0);
}
