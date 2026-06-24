#include "DeepLearning/Api/Layers/Metrics/LossMetric.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Loss/LossExpression.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace Thor;
using json = nlohmann::json;

namespace {

void expectLossMetricBuildsAndSerializes(LossMetric::Formula formula, const string& formulaName) {
    Network network("loss_metric_" + formulaName);
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    LossMetric metric = LossMetric::Builder().network(network).predictions(predictions).labels(labels).formula(formula).build();

    ASSERT_TRUE(metric.isInitialized());
    ASSERT_TRUE(metric.requiresLabels());
    ASSERT_EQ(metric.getPredictions(), predictions);
    ASSERT_EQ(metric.getLabels(), labels);
    ASSERT_EQ(metric.getFormula(), formula);
    ASSERT_EQ(metric.getMetric().getDataType(), DataType::FP32);
    ASSERT_EQ(metric.getMetric().getDimensions(), vector<uint64_t>({1}));

    shared_ptr<Layer> cloneLayer = metric.clone();
    LossMetric* clone = dynamic_cast<LossMetric*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());
    ASSERT_EQ(clone->getFormula(), formula);

    json metricJson = metric.architectureJson();
    ASSERT_EQ(metricJson.at("factory").get<string>(), Layer::Factory::Metric.value());
    ASSERT_EQ(metricJson.at("layer_type").get<string>(), string("loss_metric"));
    ASSERT_EQ(metricJson.at("formula").get<string>(), formulaName);
    ASSERT_EQ(metricJson.at("reduction").get<string>(), string("batch"));
    ASSERT_TRUE(metricJson.contains("predictions"));
    ASSERT_TRUE(metricJson.contains("labels"));
    ASSERT_TRUE(metricJson.contains("metric"));
    ASSERT_EQ(metricJson.at("predictions_name").get<string>(), string("predictions"));
    ASSERT_EQ(metricJson.at("labels_name").get<string>(), string("labels"));
    ASSERT_EQ(metricJson.at("metric_name").get<string>(), string("metric"));
}

}  // namespace

TEST(LossMetricApi, MeanSquaredErrorBuildsAndSerializes) {
    expectLossMetricBuildsAndSerializes(LossMetric::Formula::MEAN_SQUARED_ERROR, "mean_squared_error");
}

TEST(LossMetricApi, MeanAbsoluteErrorBuildsAndSerializes) {
    expectLossMetricBuildsAndSerializes(LossMetric::Formula::MEAN_ABSOLUTE_ERROR, "mean_absolute_error");
}

TEST(LossMetricApi, MeanAbsolutePercentageErrorBuildsAndSerializes) {
    Network network("loss_metric_mape");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    LossMetric metric = LossMetric::Builder()
                            .network(network)
                            .predictions(predictions)
                            .labels(labels)
                            .meanAbsolutePercentageError(0.01f, 250.0f)
                            .displayName("MAPE")
                            .build();

    ASSERT_TRUE(metric.isInitialized());
    ASSERT_EQ(metric.getFormula(), LossMetric::Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR);
    ASSERT_FLOAT_EQ(metric.getEpsilon(), 0.01f);
    ASSERT_FLOAT_EQ(metric.getMaxMagnitude(), 250.0f);
    ASSERT_EQ(metric.getDisplayName(), string("MAPE"));
    ASSERT_EQ(metric.getMetric().getDimensions(), vector<uint64_t>({1}));

    json metricJson = metric.architectureJson();
    ASSERT_EQ(metricJson.at("formula").get<string>(), string("mean_absolute_percentage_error"));
    ASSERT_FLOAT_EQ(metricJson.at("epsilon").get<float>(), 0.01f);
    ASSERT_FLOAT_EQ(metricJson.at("max_magnitude").get<float>(), 250.0f);
    ASSERT_EQ(metricJson.at("display_name").get<string>(), string("MAPE"));
}


TEST(LossMetricApi, BatchLossMetricExpressionKeepsExpectedNamesWhenOptionsAreMoved) {
    ThorImplementation::LossExpression::Options options;
    options.formula = LossMetric::Formula::MEAN_ABSOLUTE_ERROR;
    options.predictionsName = "predictions";
    options.labelsName = "labels";
    options.lossName = "metric";

    ThorImplementation::DynamicExpression expression =
        ThorImplementation::LossExpression::makeBatchLossMetricExpression(std::move(options));

    ASSERT_EQ(expression.getExpectedInputNames(), vector<string>({"predictions", "labels"}));
    ASSERT_EQ(expression.getExpectedOutputNames(), vector<string>({"metric"}));
}

TEST(LossMetricApi, RejectsShapeMismatch) {
    Network network("loss_metric_shape_mismatch");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {4});

    EXPECT_THROW(LossMetric::Builder().network(network).predictions(predictions).labels(labels).build(), std::exception);
}
