#include "DeepLearning/Api/Layers/Metrics/ReductionMetrics.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace Thor;
using json = nlohmann::json;

namespace {

constexpr std::array<DataType, 5> supportedReductionMetricDTypes{
    DataType::FP8_E4M3,
    DataType::FP8_E5M2,
    DataType::FP16,
    DataType::BF16,
    DataType::FP32,
};

template <typename MetricT>
void expectUnaryMetricSupportsDType(DataType dtype, const string& metricName) {
    Network network(metricName + "_supports_" + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    Tensor values(dtype, {3});

    typename MetricT::Builder builder;
    MetricT metric = builder.network(network).values(values).build();

    ASSERT_TRUE(metric.isInitialized());
    ASSERT_EQ(metric.getValues().getDataType(), dtype);
    ASSERT_EQ(metric.getMetric().getDataType(), DataType::FP32);
}

template <typename MetricT>
void expectUnaryMetricBuildsAndSerializes(const string& expectedLayerType) {
    Network network(expectedLayerType + "_metric_builds");
    Tensor values(DataType::FP32, {3});

    typename MetricT::Builder builder;
    MetricT metric = builder.network(network).values(values).build();

    ASSERT_TRUE(metric.isInitialized());
    ASSERT_FALSE(metric.requiresLabels());
    ASSERT_EQ(metric.getValues(), values);
    ASSERT_EQ(metric.getMetric().getDataType(), DataType::FP32);
    ASSERT_EQ(metric.getMetric().getDimensions(), vector<uint64_t>({1}));

    shared_ptr<Layer> cloneLayer = metric.clone();
    MetricT* clone = dynamic_cast<MetricT*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());

    json metricJson = metric.architectureJson();
    ASSERT_EQ(metricJson.at("factory").get<string>(), Layer::Factory::Metric.value());
    ASSERT_EQ(metricJson.at("layer_type").get<string>(), expectedLayerType);
    ASSERT_TRUE(metricJson.contains("values"));
    ASSERT_FALSE(metricJson.contains("labels"));
    ASSERT_FALSE(metricJson.contains("predictions"));
    ASSERT_TRUE(metricJson.contains("metric"));
}

}  // namespace

TEST(ReductionMetricApi, MeanBuildsAndSerializes) { expectUnaryMetricBuildsAndSerializes<Mean>("mean"); }

TEST(ReductionMetricApi, SumBuildsAndSerializes) { expectUnaryMetricBuildsAndSerializes<Sum>("sum"); }

TEST(ReductionMetricApi, MinBuildsAndSerializes) { expectUnaryMetricBuildsAndSerializes<Min>("min"); }

TEST(ReductionMetricApi, MaxBuildsAndSerializes) { expectUnaryMetricBuildsAndSerializes<Max>("max"); }

TEST(ReductionMetricApi, WeightedMeanBuildsAndSerializes) {
    Network network("weighted_mean_metric_builds");
    Tensor values(DataType::FP32, {3});
    Tensor weights(DataType::FP32, {3});

    WeightedMean metric = WeightedMean::Builder().network(network).values(values).weights(weights).build();

    ASSERT_TRUE(metric.isInitialized());
    ASSERT_TRUE(metric.requiresLabels());
    ASSERT_EQ(metric.getValues(), values);
    ASSERT_EQ(metric.getWeights(), weights);
    ASSERT_EQ(metric.getMetric().getDataType(), DataType::FP32);
    ASSERT_EQ(metric.getMetric().getDimensions(), vector<uint64_t>({1}));

    shared_ptr<Layer> cloneLayer = metric.clone();
    WeightedMean* clone = dynamic_cast<WeightedMean*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());

    json metricJson = metric.architectureJson();
    ASSERT_EQ(metricJson.at("factory").get<string>(), Layer::Factory::Metric.value());
    ASSERT_EQ(metricJson.at("layer_type").get<string>(), string("weighted_mean"));
    ASSERT_TRUE(metricJson.contains("values"));
    ASSERT_TRUE(metricJson.contains("weights"));
    ASSERT_TRUE(metricJson.contains("metric"));
    ASSERT_FALSE(metricJson.contains("labels"));
    ASSERT_FALSE(metricJson.contains("predictions"));
}

TEST(ReductionMetricApi, UnaryMetricsSupportThorFloatingStorageDTypes) {
    for (DataType dtype : supportedReductionMetricDTypes) {
        expectUnaryMetricSupportsDType<Mean>(dtype, "mean");
        expectUnaryMetricSupportsDType<Sum>(dtype, "sum");
        expectUnaryMetricSupportsDType<Min>(dtype, "min");
        expectUnaryMetricSupportsDType<Max>(dtype, "max");
    }
}

TEST(ReductionMetricApi, UnaryMetricsRejectUnsupportedStorageDTypes) {
    for (DataType dtype : {DataType::FP64, DataType::INT32, DataType::BOOLEAN}) {
        Network network("mean_rejects_" + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
        Tensor values(dtype, {3});
        EXPECT_THROW(Mean::Builder().network(network).values(values).build(), std::runtime_error);
    }
}

TEST(ReductionMetricApi, WeightedMeanSupportsThorFloatingStorageDTypes) {
    for (DataType dtype : supportedReductionMetricDTypes) {
        Network network("weighted_mean_supports_" + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
        Tensor values(dtype, {3});
        Tensor weights(dtype, {3});

        WeightedMean metric = WeightedMean::Builder().network(network).values(values).weights(weights).build();

        ASSERT_TRUE(metric.isInitialized());
        ASSERT_EQ(metric.getValues().getDataType(), dtype);
        ASSERT_EQ(metric.getWeights().getDataType(), dtype);
        ASSERT_EQ(metric.getMetric().getDataType(), DataType::FP32);
    }
}

TEST(ReductionMetricApi, WeightedMeanRejectsUnsupportedStorageDTypes) {
    Network valuesNetwork("weighted_mean_rejects_fp64_values");
    Tensor fp64Values(DataType::FP64, {3});
    Tensor fp32Weights(DataType::FP32, {3});
    EXPECT_THROW(WeightedMean::Builder().network(valuesNetwork).values(fp64Values).weights(fp32Weights).build(),
                 std::runtime_error);

    Network weightsNetwork("weighted_mean_rejects_int_weights");
    Tensor fp32Values(DataType::FP32, {3});
    Tensor intWeights(DataType::INT32, {3});
    EXPECT_THROW(WeightedMean::Builder().network(weightsNetwork).values(fp32Values).weights(intWeights).build(),
                 std::runtime_error);
}

TEST(ReductionMetricApi, WeightedMeanRejectsShapeMismatch) {
    Network network("weighted_mean_metric_rejects_shape_mismatch");
    Tensor values(DataType::FP32, {3});
    Tensor weights(DataType::FP32, {4});

    EXPECT_THROW(WeightedMean::Builder().network(network).values(values).weights(weights).build(), std::exception);
}
