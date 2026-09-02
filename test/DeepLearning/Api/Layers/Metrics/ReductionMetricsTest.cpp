#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Metrics/ReductionMetrics.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
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

void writeR10LOffsets(ThorImplementation::Tensor& offsetsTensor, DataType dtype, const vector<uint64_t>& offsets) {
    if (dtype == DataType::UINT32) {
        uint32_t* values = offsetsTensor.getMemPtr<uint32_t>();
        for (size_t i = 0; i < offsets.size(); ++i) values[i] = static_cast<uint32_t>(offsets[i]);
        return;
    }
    THOR_THROW_IF_FALSE(dtype == DataType::UINT64);
    uint64_t* values = offsetsTensor.getMemPtr<uint64_t>();
    copy(offsets.begin(), offsets.end(), values);
}

vector<float> copyR10LFp32ToHost(const ThorImplementation::Tensor& tensor) {
    THOR_THROW_IF_FALSE(tensor.getDataType() == DataType::FP32);
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor host = tensor.clone(cpuPlacement);
    Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    const float* values = host.getMemPtr<float>();
    return vector<float>(values, values + host.getTotalNumElements());
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


TEST(ReductionMetricApi, R10LRaggedSumAndMeanBuildWithDistinctAggregationContracts) {
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("r10l_ragged_reductions_" + ThorImplementation::TensorDescriptor::getElementTypeName(offsetsDType));
        RaggedTensor values = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("values")
                                  .valuesDataType(DataType::FP16)
                                  .offsetsDataType(offsetsDType)
                                  .trailingDimensions({2})
                                  .batchSize(4)
                                  .maxTotalValues(9)
                                  .maxValuesPerRow(4)
                                  .build();

        Sum sum = Sum::Builder().network(network).values(values).build();
        Mean mean = Mean::Builder().network(network).values(values).build();

        ASSERT_TRUE(sum.getUseRagged());
        ASSERT_TRUE(mean.getUseRagged());
        EXPECT_FALSE(sum.requiresLabels());
        EXPECT_FALSE(mean.requiresLabels());
        ASSERT_EQ(sum.getAllInputTensors().size(), 2U);
        ASSERT_EQ(mean.getAllInputTensors().size(), 2U);
        EXPECT_EQ(sum.getConnectionType(values.getOffsets()),
                  static_cast<int>(ThorImplementation::Metric::ConnectionType::STRUCTURAL));
        EXPECT_EQ(mean.getConnectionType(values.getOffsets()),
                  static_cast<int>(ThorImplementation::Metric::ConnectionType::STRUCTURAL));
        ASSERT_TRUE(sum.getRaggedValues().has_value());
        ASSERT_TRUE(mean.getRaggedValues().has_value());
        EXPECT_EQ(sum.getRaggedValues().value(), values);
        EXPECT_EQ(mean.getRaggedValues().value(), values);
        EXPECT_EQ(sum.getAggregation(), MetricAggregation::SUM);
        EXPECT_EQ(mean.getAggregation(), MetricAggregation::RATIO);
        EXPECT_EQ(sum.getMetric().getDataType(), DataType::FP32);
        EXPECT_EQ(mean.getMetric().getDataType(), DataType::FP32);

        const json sumJson = sum.architectureJson();
        const json meanJson = mean.architectureJson();
        EXPECT_TRUE(sumJson.contains("ragged_values"));
        EXPECT_FALSE(sumJson.contains("values"));
        EXPECT_TRUE(meanJson.contains("ragged_values"));
        EXPECT_FALSE(meanJson.contains("values"));
        EXPECT_EQ(sumJson.at("aggregation").get<MetricAggregation>(), MetricAggregation::SUM);
        EXPECT_EQ(meanJson.at("aggregation").get<MetricAggregation>(), MetricAggregation::RATIO);
    }
}

TEST(ReductionMetricApi, R10LExtremaRemainOutsideRaggedContract) {
    Network network("r10l_extrema_outside_contract");
    RaggedTensor values = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("values")
                              .valuesDataType(DataType::FP32)
                              .trailingDimensions({2})
                              .batchSize(3)
                              .maxTotalValues(7)
                              .build();
    EXPECT_THROW((void)Min::Builder().network(network).values(values), std::invalid_argument);
    EXPECT_THROW((void)Max::Builder().network(network).values(values), std::invalid_argument);
}

TEST(ReductionMetricApi, R10LPlacedRaggedSumAndMeanExposeActiveScalarStatisticsForBothOffsetWidths) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "R10L ragged metric execution requires a GPU";

    constexpr uint32_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 9;
    constexpr uint64_t trailingWidth = 2;
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Network network("r10l_metric_runtime_" + ThorImplementation::TensorDescriptor::getElementTypeName(offsetsDType));
        RaggedTensor values = RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("values")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(offsetsDType)
                                  .trailingDimensions({trailingWidth})
                                  .batchSize(batchSize)
                                  .maxTotalValues(maxTotalValues)
                                  .maxValuesPerRow(4)
                                  .build();
        Sum sum = Sum::Builder().network(network).values(values).build();
        Mean mean = Mean::Builder().network(network).values(values).build();
        NetworkOutput::Builder().network(network).name("sum").inputTensor(sum.getMetric()).dataType(DataType::FP32).build();
        NetworkOutput::Builder().network(network).name("mean").inputTensor(mean.getMetric()).dataType(DataType::FP32).build();

        vector<Event> initializationDone;
        shared_ptr<PlacedNetwork> placed = network.place(
            batchSize, initializationDone, /*inferenceOnly=*/true, vector<int32_t>{0}, /*forcedNumStampsPerGpu=*/1);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initializationDone) event.synchronize();
        placed->preallocateInputSlots(1);
        placed->preallocateOutputSlots(1);
        placed->synchronize();

        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::Tensor packedValues(
            cpuPlacement, ThorImplementation::TensorDescriptor(DataType::FP32, {maxTotalValues, trailingWidth}));
        float* packed = packedValues.getMemPtr<float>();
        fill(packed, packed + maxTotalValues * trailingWidth, numeric_limits<float>::quiet_NaN());
        const array<float, 10> active{1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f, 7.0f, 8.0f, 9.0f, -10.0f};
        copy(active.begin(), active.end(), packed);
        ThorImplementation::Tensor offsets(
            cpuPlacement, ThorImplementation::TensorDescriptor(offsetsDType, {batchSize + 1}));
        writeR10LOffsets(offsets, offsetsDType, {0, 2, 2, 5, 5});

        Batch batch;
        batch.insert("values", ThorImplementation::RaggedTensor(packedValues, offsets, /*maxValuesPerRow=*/4));
        map<string, ThorImplementation::Tensor> outputs;
        map<string, Event> outputReadyEvents;
        Event done = placed->submitBatch(0, batch, outputs, outputReadyEvents, /*isInferenceOnly=*/true);

        done.synchronize();
        outputReadyEvents.at("sum").synchronize();
        outputReadyEvents.at("mean").synchronize();
        const vector<float> sumValue = copyR10LFp32ToHost(outputs.at("sum"));
        const vector<float> meanValue = copyR10LFp32ToHost(outputs.at("mean"));
        ASSERT_EQ(sumValue.size(), 1U);
        ASSERT_EQ(meanValue.size(), 1U);
        const float expectedSum = accumulate(active.begin(), active.end(), 0.0f);
        EXPECT_NEAR(sumValue.front(), expectedSum, 1.0e-5f);
        EXPECT_NEAR(meanValue.front(), expectedSum / static_cast<float>(active.size()), 1.0e-5f);

        map<string, ThorImplementation::MetricBatchStatisticTensors> statistics =
            placed->getMetricBatchStatisticTensorsForSlot(0, 0);
        ASSERT_TRUE(statistics.count("sum"));
        ASSERT_TRUE(statistics.count("mean"));
        EXPECT_EQ(statistics.at("sum").aggregation, MetricAggregation::SUM);
        EXPECT_FALSE(statistics.at("sum").numerator.has_value());
        EXPECT_FALSE(statistics.at("sum").denominator.has_value());
        auto& meanStatistics = statistics.at("mean");
        EXPECT_EQ(meanStatistics.aggregation, MetricAggregation::RATIO);
        ASSERT_TRUE(meanStatistics.numerator.has_value());
        ASSERT_TRUE(meanStatistics.denominator.has_value());
        ASSERT_TRUE(meanStatistics.readyEvent.isInitialized());
        meanStatistics.readyEvent.synchronize();
        EXPECT_NEAR(*meanStatistics.numerator->getMemPtr<float>(), expectedSum, 1.0e-5f);
        EXPECT_FLOAT_EQ(*meanStatistics.denominator->getMemPtr<float>(), static_cast<float>(active.size()));
        placed->synchronize();

        // Tail-batch cardinality is row-based, not packed-capacity based. Keep
        // later rows populated in the canonical partition and mark only the
        // first two logical rows valid; the reduction must clamp the structural
        // partition at offsets[2] and ignore rows 2 and 3 completely.
        Batch partialBatch;
        partialBatch.insert("values", ThorImplementation::RaggedTensor(packedValues, offsets, /*maxValuesPerRow=*/4));
        partialBatch.setValidExampleCount(2);
        outputs.clear();
        outputReadyEvents.clear();
        done = placed->submitBatch(0, partialBatch, outputs, outputReadyEvents, /*isInferenceOnly=*/true);
        done.synchronize();
        outputReadyEvents.at("sum").synchronize();
        outputReadyEvents.at("mean").synchronize();
        const float partialExpectedSum = active[0] + active[1] + active[2] + active[3];
        EXPECT_NEAR(copyR10LFp32ToHost(outputs.at("sum")).front(), partialExpectedSum, 1.0e-5f);
        EXPECT_NEAR(copyR10LFp32ToHost(outputs.at("mean")).front(), partialExpectedSum / 4.0f, 1.0e-5f);
        statistics = placed->getMetricBatchStatisticTensorsForSlot(0, 0);
        auto& partialMeanStatistics = statistics.at("mean");
        ASSERT_TRUE(partialMeanStatistics.numerator.has_value());
        ASSERT_TRUE(partialMeanStatistics.denominator.has_value());
        partialMeanStatistics.readyEvent.synchronize();
        EXPECT_NEAR(*partialMeanStatistics.numerator->getMemPtr<float>(), partialExpectedSum, 1.0e-5f);
        EXPECT_FLOAT_EQ(*partialMeanStatistics.denominator->getMemPtr<float>(), 4.0f);
        placed->synchronize();

        // Reuse the same stamp/slot with an all-empty population. Packed values
        // remain deliberately nonzero/NaN-capable storage; offsets alone define
        // that there is no active contribution.
        ThorImplementation::Tensor emptyOffsets(
            cpuPlacement, ThorImplementation::TensorDescriptor(offsetsDType, {batchSize + 1}));
        writeR10LOffsets(emptyOffsets, offsetsDType, {0, 0, 0, 0, 0});
        Batch emptyBatch;
        emptyBatch.insert(
            "values", ThorImplementation::RaggedTensor(packedValues, emptyOffsets, /*maxValuesPerRow=*/4));
        outputs.clear();
        outputReadyEvents.clear();
        done = placed->submitBatch(0, emptyBatch, outputs, outputReadyEvents, /*isInferenceOnly=*/true);
        done.synchronize();
        outputReadyEvents.at("sum").synchronize();
        outputReadyEvents.at("mean").synchronize();
        EXPECT_FLOAT_EQ(copyR10LFp32ToHost(outputs.at("sum")).front(), 0.0f);
        EXPECT_FLOAT_EQ(copyR10LFp32ToHost(outputs.at("mean")).front(), 0.0f);

        statistics = placed->getMetricBatchStatisticTensorsForSlot(0, 0);
        ASSERT_TRUE(statistics.count("mean"));
        auto& emptyMeanStatistics = statistics.at("mean");
        ASSERT_TRUE(emptyMeanStatistics.numerator.has_value());
        ASSERT_TRUE(emptyMeanStatistics.denominator.has_value());
        emptyMeanStatistics.readyEvent.synchronize();
        EXPECT_FLOAT_EQ(*emptyMeanStatistics.numerator->getMemPtr<float>(), 0.0f);
        EXPECT_FLOAT_EQ(*emptyMeanStatistics.denominator->getMemPtr<float>(), 0.0f);
        placed->synchronize();
    }
}

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
