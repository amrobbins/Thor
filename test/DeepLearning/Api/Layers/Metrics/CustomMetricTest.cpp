#include "DeepLearning/Api/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace Thor;
namespace Impl = ThorImplementation;
using json = nlohmann::json;

namespace {

Impl::DynamicExpression makeSerializableMseMetricExpression(const std::string& predictionsName = "predictions",
                                                            const std::string& labelsName = "labels",
                                                            const std::string& metricName = "metric") {
    Impl::Expression predictions = Impl::Expression::input(predictionsName, DataType::FP32, DataType::FP32);
    Impl::Expression labels = Impl::Expression::input(labelsName, DataType::FP32, DataType::FP32);
    Impl::Expression diff = predictions - labels;
    Impl::Expression metric = (diff * diff).reduce_mean({0, 1}, {0}, DataType::FP32);
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{metricName, metric}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeSerializablePartialBatchMseMetricExpression() {
    Impl::Expression predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    Impl::Expression labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    Impl::Expression validity = Impl::Expression::input(Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
    Impl::Expression diff = predictions - labels;
    Impl::Expression numerator = ((diff * diff) * validity).reduce_sum({0, 1}, {0}, DataType::FP32);
    Impl::Expression denominator = validity.reduce_sum({0, 1}, {0}, DataType::FP32);
    Impl::Expression metric = numerator / denominator;
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"metric", metric}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}


Impl::DynamicExpression makeSerializableRatioMetricExpression(
    const std::string& metricName = "metric",
    bool includeNumerator = true,
    bool includeDenominator = true,
    bool scalarNumerator = true,
    DataType statisticDataType = DataType::FP32,
    bool scalarMetric = true) {
    Impl::Expression predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    Impl::Expression labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    Impl::Expression weightedValues = predictions * labels;
    Impl::Expression numerator = scalarNumerator
                                     ? weightedValues.reduce_sum({0, 1}, {0}, DataType::FP32)
                                     : weightedValues;
    Impl::Expression denominator = labels.reduce_sum({0, 1}, {0}, DataType::FP32);
    Impl::Expression scalarNumeratorForMetric =
        scalarNumerator ? numerator : numerator.reduce_sum({0, 1}, {0}, DataType::FP32);
    Impl::Expression metric = scalarMetric ? scalarNumeratorForMetric / denominator : predictions;
    if (statisticDataType != DataType::FP32) {
        numerator = numerator.cast(statisticDataType);
        denominator = denominator.cast(statisticDataType);
    }

    std::vector<std::pair<std::string, Impl::Expression>> outputs{{metricName, metric}};
    if (includeNumerator)
        outputs.emplace_back(Thor::METRIC_AGGREGATION_NUMERATOR_NAME, numerator);
    if (includeDenominator)
        outputs.emplace_back(Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, denominator);
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs(outputs));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}


Impl::DynamicExpression makePassThroughRatioMetricExpression() {
    Impl::Expression numerator = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    Impl::Expression denominator = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    Impl::Expression metric = numerator / denominator;
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({
        {"metric", metric},
        {Thor::METRIC_AGGREGATION_NUMERATOR_NAME, numerator},
        {Thor::METRIC_AGGREGATION_DENOMINATOR_NAME, denominator},
    }));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeNonSerializableMseMetricExpression() {
    return Impl::DynamicExpression({"predictions", "labels"},
                                   {"metric"},
                                   [](const Impl::DynamicExpression::TensorMap& inputs,
                                      const Impl::DynamicExpression::TensorMap& outputs,
                                      Stream& stream) -> Impl::DynamicExpressionBuild {
                                       auto predictions = Impl::Expression::input("predictions");
                                       auto labels = Impl::Expression::input("labels");
                                       auto diff = predictions - labels;
                                       auto metric = (diff * diff).reduce_mean({0, 1}, {0}, DataType::FP32);
                                       auto expressionOutputs = Impl::Expression::outputs({{"metric", metric}});
                                       return Impl::DynamicExpressionBuild{
                                           std::make_shared<Impl::FusedEquation>(
                                               Impl::FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum())),
                                           inputs,
                                           {},
                                           outputs,
                                           {}};
                                   });
}

}  // namespace

TEST(CustomMetricApi, BuildsAndSerializesExpressionBackedMetric) {
    Network network("custom_metric_builds");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomMetric customMetric = CustomMetric::Builder()
                                    .network(network)
                                    .expression(makeSerializableMseMetricExpression())
                                    .predictions(predictions)
                                    .labels(labels)
                                    .aggregation(MetricAggregation::MEAN_BY_EXAMPLE)
                                    .displayName("MSE")
                                    .build();

    ASSERT_TRUE(customMetric.isInitialized());
    ASSERT_EQ(customMetric.getPredictions(), predictions);
    ASSERT_EQ(customMetric.getLabels(), labels);
    ASSERT_EQ(customMetric.getMetric().getDataType(), DataType::FP32);
    ASSERT_EQ(customMetric.getMetric().getDimensions(), vector<uint64_t>({1}));
    ASSERT_EQ(customMetric.getPredictionsName(), string("predictions"));
    ASSERT_EQ(customMetric.getLabelsName(), string("labels"));
    ASSERT_EQ(customMetric.getMetricName(), string("metric"));
    ASSERT_EQ(customMetric.getDisplayName(), string("MSE"));
    ASSERT_EQ(customMetric.getAggregation(), MetricAggregation::MEAN_BY_EXAMPLE);
    ASSERT_FALSE(customMetric.usesBatchValidity());

    shared_ptr<Layer> cloneLayer = customMetric.clone();
    CustomMetric* clone = dynamic_cast<CustomMetric*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());
    ASSERT_EQ(clone->getMetric().getDataType(), DataType::FP32);
    ASSERT_EQ(clone->getMetric().getDimensions(), vector<uint64_t>({1}));
    ASSERT_EQ(clone->getAggregation(), MetricAggregation::MEAN_BY_EXAMPLE);
    ASSERT_FALSE(clone->usesBatchValidity());

    json metricJson = customMetric.architectureJson();
    ASSERT_EQ(metricJson.at("factory").get<string>(), Layer::Factory::Metric.value());
    ASSERT_EQ(metricJson.at("layer_type").get<string>(), string("custom_metric"));
    ASSERT_EQ(metricJson.at("predictions_name").get<string>(), string("predictions"));
    ASSERT_EQ(metricJson.at("labels_name").get<string>(), string("labels"));
    ASSERT_EQ(metricJson.at("metric_name").get<string>(), string("metric"));
    ASSERT_EQ(metricJson.at("display_name").get<string>(), string("MSE"));
    ASSERT_EQ(metricJson.at("aggregation").get<MetricAggregation>(), MetricAggregation::MEAN_BY_EXAMPLE);
    ASSERT_FALSE(metricJson.at("uses_batch_validity").get<bool>());
    ASSERT_FALSE(metricJson.contains("supports_partial_batches"));
    ASSERT_TRUE(metricJson.contains("expression"));
    ASSERT_FALSE(metricJson.at("expression").is_null());
}

TEST(CustomMetricApi, BuilderInfersNonDefaultExpressionNames) {
    Network network("custom_metric_infers_names");
    Tensor predictions(DataType::FP32, {5});
    Tensor labels(DataType::FP32, {5});

    CustomMetric customMetric = CustomMetric::Builder()
                                    .network(network)
                                    .expression(makeSerializableMseMetricExpression("y_hat", "target", "mse"))
                                    .predictions(predictions)
                                    .labels(labels)
                                    .aggregation(MetricAggregation::MEAN_BY_EXAMPLE)
                                    .build();

    ASSERT_EQ(customMetric.getPredictionsName(), string("y_hat"));
    ASSERT_EQ(customMetric.getLabelsName(), string("target"));
    ASSERT_EQ(customMetric.getMetricName(), string("mse"));
    ASSERT_EQ(customMetric.getMetric().getDimensions(), vector<uint64_t>({1}));
}

TEST(CustomMetricApi, RejectsMetricTensorDescriptorMismatch) {
    Network network("custom_metric_rejects_metric_mismatch");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});
    Tensor wrongMetric(DataType::FP32, {2});

    EXPECT_THROW(CustomMetric::Builder()
                     .network(network)
                     .expression(makeSerializableMseMetricExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .aggregation(MetricAggregation::MEAN_BY_EXAMPLE)
                     .metricTensor(wrongMetric)
                     .build(),
                 std::runtime_error);
}

TEST(CustomMetricApi, RejectsSavingNonSerializableBuilderExpression) {
    Network network("custom_metric_rejects_nonserializable_expression");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomMetric customMetric = CustomMetric::Builder()
                                    .network(network)
                                    .expression(makeNonSerializableMseMetricExpression())
                                    .predictions(predictions)
                                    .labels(labels)
                                    .aggregation(MetricAggregation::MEAN_BY_EXAMPLE)
                                    .build();

    EXPECT_THROW(customMetric.architectureJson(), std::runtime_error);
}

TEST(CustomMetricApi, BatchValidityMaskUseIsExplicitAndSerialized) {
    Network network("custom_metric_partial_batches");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomMetric customMetric = CustomMetric::Builder()
                                    .network(network)
                                    .expression(makeSerializablePartialBatchMseMetricExpression())
                                    .predictions(predictions)
                                    .labels(labels)
                                    .aggregation(MetricAggregation::MEAN_BY_EXAMPLE)
                                    .usesBatchValidity()
                                    .build();

    ASSERT_TRUE(customMetric.usesBatchValidity());
    json metricJson = customMetric.architectureJson();
    ASSERT_TRUE(metricJson.at("uses_batch_validity").get<bool>());
}

TEST(CustomMetricApi, BatchValidityMaskDeclarationRequiresValidityMaskInput) {
    Network network("custom_metric_missing_partial_batch_mask");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    EXPECT_THROW(CustomMetric::Builder()
                     .network(network)
                     .expression(makeSerializableMseMetricExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .aggregation(MetricAggregation::MEAN_BY_EXAMPLE)
                     .usesBatchValidity()
                     .build(),
                 std::runtime_error);
}

TEST(CustomMetricApi, AggregationIsRequired) {
    Network network("custom_metric_requires_aggregation");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    EXPECT_THROW(CustomMetric::Builder()
                     .network(network)
                     .expression(makeSerializableMseMetricExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .build(),
                 std::logic_error);
}

TEST(CustomMetricApi, AggregationAndValidityMaskFieldsAreStrictDuringDeserialization) {
    Network network("custom_metric_strict_deserialization");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomMetric customMetric = CustomMetric::Builder()
                                    .network(network)
                                    .expression(makeSerializableMseMetricExpression())
                                    .predictions(predictions)
                                    .labels(labels)
                                    .aggregation(MetricAggregation::SUM)
                                    .build();
    json metricJson = customMetric.architectureJson();

    json missingAggregation = metricJson;
    missingAggregation.erase("aggregation");
    EXPECT_THROW(CustomMetric::deserialize(missingAggregation, &network), std::exception);

    json missingValidityDeclaration = metricJson;
    missingValidityDeclaration.erase("uses_batch_validity");
    EXPECT_THROW(CustomMetric::deserialize(missingValidityDeclaration, &network), std::exception);

    json deprecatedAlias = metricJson;
    deprecatedAlias["supports_partial_batches"] = false;
    EXPECT_THROW(CustomMetric::deserialize(deprecatedAlias, &network), std::runtime_error);
}

TEST(MetricAggregationApi, SerializesStrictStringValues) {
    const std::vector<MetricAggregation> aggregations{
        MetricAggregation::MEAN_BY_EXAMPLE,
        MetricAggregation::SUM,
        MetricAggregation::MIN,
        MetricAggregation::MAX,
        MetricAggregation::RATIO,
    };
    for (MetricAggregation aggregation : aggregations) {
        json serialized = aggregation;
        EXPECT_EQ(serialized.get<std::string>(), metricAggregationName(aggregation));
        EXPECT_EQ(serialized.get<MetricAggregation>(), aggregation);
    }
    EXPECT_THROW(metricAggregationFromString("mean"), std::invalid_argument);
    EXPECT_THROW(json("UNKNOWN").get<MetricAggregation>(), std::invalid_argument);
}


TEST(CustomMetricApi, RatioAggregationRequiresNumeratorAndDenominatorOutputs) {
    Network network("custom_metric_ratio_requires_statistics");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    EXPECT_THROW(CustomMetric::Builder()
                     .network(network)
                     .expression(makeSerializableMseMetricExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .aggregation(MetricAggregation::RATIO)
                     .build(),
                 std::runtime_error);

    EXPECT_THROW(CustomMetric::Builder()
                     .network(network)
                     .expression(makeSerializableRatioMetricExpression("metric", true, false))
                     .predictions(predictions)
                     .labels(labels)
                     .aggregation(MetricAggregation::RATIO)
                     .build(),
                 std::runtime_error);

    EXPECT_THROW(CustomMetric::Builder()
                     .network(network)
                     .expression(makeSerializableRatioMetricExpression("metric", false, true))
                     .predictions(predictions)
                     .labels(labels)
                     .aggregation(MetricAggregation::RATIO)
                     .build(),
                 std::runtime_error);
}

TEST(CustomMetricApi, NonRatioAggregationRejectsReservedStatisticOutputs) {
    Network network("custom_metric_non_ratio_rejects_statistics");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    EXPECT_THROW(CustomMetric::Builder()
                     .network(network)
                     .expression(makeSerializableRatioMetricExpression())
                     .predictions(predictions)
                     .labels(labels)
                     .aggregation(MetricAggregation::MEAN_BY_EXAMPLE)
                     .build(),
                 std::runtime_error);
}

TEST(CustomMetricApi, RatioAggregationInfersOnlyPublicMetricOutputName) {
    Network network("custom_metric_ratio_infers_public_name");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomMetric metric = CustomMetric::Builder()
                              .network(network)
                              .expression(makeSerializableRatioMetricExpression("weighted_ratio"))
                              .predictions(predictions)
                              .labels(labels)
                              .aggregation(MetricAggregation::RATIO)
                              .build();

    EXPECT_EQ(metric.getMetricName(), "weighted_ratio");
    EXPECT_EQ(metric.getMetric().getDimensions(), std::vector<uint64_t>({1}));
    EXPECT_EQ(metric.getAggregation(), MetricAggregation::RATIO);
}

TEST(CustomMetricApi, RatioAggregationInfersPassThroughStatisticDTypes) {
    Network network("custom_metric_ratio_passthrough_dtype");
    Tensor numerator(DataType::FP32, {1});
    Tensor denominator(DataType::FP32, {1});

    CustomMetric metric = CustomMetric::Builder()
                              .network(network)
                              .expression(makePassThroughRatioMetricExpression())
                              .predictions(numerator)
                              .labels(denominator)
                              .aggregation(MetricAggregation::RATIO)
                              .build();

    EXPECT_EQ(metric.getMetric().getDataType(), DataType::FP32);
    EXPECT_EQ(metric.getMetric().getDimensions(), vector<uint64_t>({1, 1}));
}

TEST(CustomMetricApi, RatioStatisticsMustBeScalarFp32) {
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    {
        Network network("custom_metric_ratio_scalar_statistics");
        EXPECT_THROW(CustomMetric::Builder()
                         .network(network)
                         .expression(makeSerializableRatioMetricExpression("metric", true, true, false))
                         .predictions(predictions)
                         .labels(labels)
                         .aggregation(MetricAggregation::RATIO)
                         .build(),
                     std::runtime_error);
    }

    {
        Network network("custom_metric_ratio_fp32_statistics");
        EXPECT_THROW(CustomMetric::Builder()
                         .network(network)
                         .expression(makeSerializableRatioMetricExpression("metric", true, true, true, DataType::FP16))
                         .predictions(predictions)
                         .labels(labels)
                         .aggregation(MetricAggregation::RATIO)
                         .build(),
                     std::runtime_error);
    }

    {
        Network network("custom_metric_ratio_scalar_metric");
        EXPECT_THROW(CustomMetric::Builder()
                         .network(network)
                         .expression(makeSerializableRatioMetricExpression(
                             "metric", true, true, true, DataType::FP32, false))
                         .predictions(predictions)
                         .labels(labels)
                         .aggregation(MetricAggregation::RATIO)
                         .build(),
                     std::runtime_error);
    }
}

TEST(CustomMetricApi, RatioSerializationContainsContractButNotRuntimeStatisticState) {
    Network network("custom_metric_ratio_serialization_boundary");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    CustomMetric metric = CustomMetric::Builder()
                              .network(network)
                              .expression(makeSerializableRatioMetricExpression("weighted_ratio"))
                              .predictions(predictions)
                              .labels(labels)
                              .aggregation(MetricAggregation::RATIO)
                              .build();

    const json architecture = metric.architectureJson();
    EXPECT_EQ(architecture.at("aggregation").get<MetricAggregation>(),
              MetricAggregation::RATIO);
    EXPECT_TRUE(architecture.contains("expression"));
    EXPECT_FALSE(architecture.contains("active_metric_statistic_slot"));
    EXPECT_FALSE(architecture.contains("ratio_statistic_slots"));
    EXPECT_FALSE(architecture.contains("ratio_statistic_download_stream"));
    EXPECT_FALSE(architecture.contains("numerator_host"));
    EXPECT_FALSE(architecture.contains("denominator_host"));
    EXPECT_FALSE(architecture.contains("ready_event"));
    EXPECT_FALSE(architecture.contains("writable_event"));

    std::shared_ptr<Layer> clonedLayer = metric.clone();
    CustomMetric* clonedMetric = dynamic_cast<CustomMetric*>(clonedLayer.get());
    ASSERT_NE(clonedMetric, nullptr);
    EXPECT_EQ(clonedMetric->getAggregation(), MetricAggregation::RATIO);
    EXPECT_EQ(clonedMetric->getMetricName(), "weighted_ratio");
    const json clonedArchitecture = clonedMetric->architectureJson();
    EXPECT_EQ(clonedArchitecture.at("aggregation").get<MetricAggregation>(),
              MetricAggregation::RATIO);
    EXPECT_EQ(clonedArchitecture.at("expression"), architecture.at("expression"));
}
