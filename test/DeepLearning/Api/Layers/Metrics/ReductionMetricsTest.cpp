#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Metrics/ReductionMetrics.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <map>
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
void expectUnaryMetricBuildsAndSerializes(const string& expectedLayerType, MetricAggregation expectedAggregation) {
    Network network(expectedLayerType + "_metric_builds");
    Tensor values(DataType::FP32, {3});

    typename MetricT::Builder builder;
    MetricT metric = builder.network(network).values(values).build();

    ASSERT_TRUE(metric.isInitialized());
    ASSERT_FALSE(metric.requiresLabels());
    ASSERT_EQ(metric.getValues(), values);
    ASSERT_EQ(metric.getMetric().getDataType(), DataType::FP32);
    ASSERT_EQ(metric.getMetric().getDimensions(), vector<uint64_t>({1}));
    ASSERT_EQ(metric.getAggregation(), expectedAggregation);

    shared_ptr<Layer> cloneLayer = metric.clone();
    MetricT* clone = dynamic_cast<MetricT*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());

    json metricJson = metric.architectureJson();
    ASSERT_EQ(metricJson.at("factory").get<string>(), Layer::Factory::Metric.value());
    ASSERT_EQ(metricJson.at("layer_type").get<string>(), expectedLayerType);
    ASSERT_EQ(metricJson.at("aggregation").get<MetricAggregation>(), expectedAggregation);
    ASSERT_TRUE(metricJson.contains("values"));
    ASSERT_FALSE(metricJson.contains("labels"));
    ASSERT_FALSE(metricJson.contains("predictions"));
    ASSERT_TRUE(metricJson.contains("metric"));
}

}  // namespace

TEST(ReductionMetricApi, MeanBuildsAndSerializes) {
    expectUnaryMetricBuildsAndSerializes<Mean>("mean", MetricAggregation::MEAN_BY_EXAMPLE);
}

TEST(ReductionMetricApi, SumBuildsAndSerializes) { expectUnaryMetricBuildsAndSerializes<Sum>("sum", MetricAggregation::SUM); }

TEST(ReductionMetricApi, MinBuildsAndSerializes) { expectUnaryMetricBuildsAndSerializes<Min>("min", MetricAggregation::MIN); }

TEST(ReductionMetricApi, MaxBuildsAndSerializes) { expectUnaryMetricBuildsAndSerializes<Max>("max", MetricAggregation::MAX); }

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
    ASSERT_EQ(metric.getAggregation(), MetricAggregation::RATIO);

    shared_ptr<Layer> cloneLayer = metric.clone();
    WeightedMean* clone = dynamic_cast<WeightedMean*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());

    json metricJson = metric.architectureJson();
    ASSERT_EQ(metricJson.at("factory").get<string>(), Layer::Factory::Metric.value());
    ASSERT_EQ(metricJson.at("layer_type").get<string>(), string("weighted_mean"));
    ASSERT_EQ(metricJson.at("aggregation").get<MetricAggregation>(), MetricAggregation::RATIO);
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


TEST(ReductionMetricApi, PlacedNetworkExposesSlotLocalWeightedMeanStatistics) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "WeightedMean statistic slot test requires a GPU";

    constexpr uint32_t batchCapacity = 2;
    Network network("weighted_mean_slot_statistics");
    NetworkInput values = NetworkInput::Builder()
                              .network(network)
                              .name("values")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();
    NetworkInput weights = NetworkInput::Builder()
                               .network(network)
                               .name("weights")
                               .dimensions({1})
                               .dataType(DataType::FP32)
                               .build();
    WeightedMean metric = WeightedMean::Builder()
                              .network(network)
                              .values(values.getFeatureOutput().value())
                              .weights(weights.getFeatureOutput().value())
                              .build();
    NetworkOutput::Builder()
        .network(network)
        .name("weighted_mean")
        .inputTensor(metric.getMetric())
        // Exercise metric discovery through the transparent output converter.
        .dataType(DataType::FP16)
        .build();

    vector<Event> initializationDone;
    shared_ptr<PlacedNetwork> placed = network.place(
        batchCapacity,
        initializationDone,
        /*inferenceOnly=*/false,
        vector<int32_t>{0},
        /*forcedNumStampsPerGpu=*/1);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initializationDone)
        event.synchronize();
    placed->preallocateInputSlots(2);
    placed->preallocateOutputSlots(2);
    placed->synchronize();

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    auto makeBatch = [&](const array<float, 2>& valueData, const array<float, 2>& weightData) {
        ThorImplementation::Tensor valuesCpu(
            cpuPlacement, ThorImplementation::TensorDescriptor(DataType::FP32, {batchCapacity, 1}));
        ThorImplementation::Tensor weightsCpu(
            cpuPlacement, ThorImplementation::TensorDescriptor(DataType::FP32, {batchCapacity, 1}));
        std::copy(valueData.begin(), valueData.end(), valuesCpu.getMemPtr<float>());
        std::copy(weightData.begin(), weightData.end(), weightsCpu.getMemPtr<float>());
        Batch batch;
        batch.insert("values", valuesCpu);
        batch.insert("weights", weightsCpu);
        return batch;
    };

    Batch firstBatch = makeBatch({2.0f, 4.0f}, {1.0f, 2.0f});
    Batch secondBatch = makeBatch({10.0f, 20.0f}, {3.0f, 4.0f});
    map<string, ThorImplementation::Tensor> firstOutputs;
    map<string, Event> firstOutputEvents;
    map<string, ThorImplementation::Tensor> secondOutputs;
    map<string, Event> secondOutputEvents;

    Event firstDone = placed->submitBatch(
        0,
        firstBatch,
        firstOutputs,
        firstOutputEvents,
        /*isInferenceOnly=*/true,
        /*reusableProcessingFinishedEvent=*/nullptr,
        /*waitForOutputsOnProcessingStream=*/false,
        /*submitTiming=*/nullptr,
        /*outputSlotIndex=*/0);
    Event secondDone = placed->submitBatch(
        0,
        secondBatch,
        secondOutputs,
        secondOutputEvents,
        /*isInferenceOnly=*/true,
        /*reusableProcessingFinishedEvent=*/nullptr,
        /*waitForOutputsOnProcessingStream=*/false,
        /*submitTiming=*/nullptr,
        /*outputSlotIndex=*/1);

    ASSERT_EQ(firstOutputs.size(), 1u);
    ASSERT_EQ(secondOutputs.size(), 1u);
    EXPECT_TRUE(firstOutputs.count("weighted_mean"));
    EXPECT_TRUE(secondOutputs.count("weighted_mean"));
    EXPECT_FALSE(firstOutputs.count(METRIC_AGGREGATION_NUMERATOR_NAME));
    EXPECT_FALSE(firstOutputs.count(METRIC_AGGREGATION_DENOMINATOR_NAME));

    map<string, ThorImplementation::MetricBatchStatisticTensors> firstStatistics =
        placed->getMetricBatchStatisticTensorsForSlot(0, 0);
    map<string, ThorImplementation::MetricBatchStatisticTensors> secondStatistics =
        placed->getMetricBatchStatisticTensorsForSlot(0, 1);
    ASSERT_EQ(firstStatistics.size(), 1u);
    ASSERT_EQ(secondStatistics.size(), 1u);
    ASSERT_TRUE(firstStatistics.count("weighted_mean"));
    ASSERT_TRUE(secondStatistics.count("weighted_mean"));

    auto expectStatistics = [](ThorImplementation::MetricBatchStatisticTensors& statistics,
                               float numerator,
                               float denominator) {
        EXPECT_EQ(statistics.aggregation, MetricAggregation::RATIO);
        ASSERT_TRUE(statistics.numerator.has_value());
        ASSERT_TRUE(statistics.denominator.has_value());
        ASSERT_TRUE(statistics.readyEvent.isInitialized());
        statistics.readyEvent.synchronize();
        EXPECT_NEAR(*statistics.numerator->getMemPtr<float>(), numerator, 1e-6f);
        EXPECT_NEAR(*statistics.denominator->getMemPtr<float>(), denominator, 1e-6f);
    };
    expectStatistics(firstStatistics.at("weighted_mean"), 10.0f, 3.0f);
    expectStatistics(secondStatistics.at("weighted_mean"), 110.0f, 7.0f);

    firstDone.synchronize();
    secondDone.synchronize();
    placed->synchronize();
}
