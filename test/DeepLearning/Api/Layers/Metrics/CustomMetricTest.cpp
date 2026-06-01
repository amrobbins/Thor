#include "DeepLearning/Api/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <string>
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
                                           std::make_shared<Impl::FusedEquation>(Impl::FusedEquation::compile(
                                               expressionOutputs.physicalOutputs(), stream.getGpuNum())),
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

    shared_ptr<Layer> cloneLayer = customMetric.clone();
    CustomMetric* clone = dynamic_cast<CustomMetric*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());
    ASSERT_EQ(clone->getMetric().getDataType(), DataType::FP32);
    ASSERT_EQ(clone->getMetric().getDimensions(), vector<uint64_t>({1}));

    json metricJson = customMetric.architectureJson();
    ASSERT_EQ(metricJson.at("factory").get<string>(), Layer::Factory::Metric.value());
    ASSERT_EQ(metricJson.at("layer_type").get<string>(), string("custom_metric"));
    ASSERT_EQ(metricJson.at("predictions_name").get<string>(), string("predictions"));
    ASSERT_EQ(metricJson.at("labels_name").get<string>(), string("labels"));
    ASSERT_EQ(metricJson.at("metric_name").get<string>(), string("metric"));
    ASSERT_EQ(metricJson.at("display_name").get<string>(), string("MSE"));
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
                                    .build();

    EXPECT_THROW(customMetric.architectureJson(), std::runtime_error);
}
