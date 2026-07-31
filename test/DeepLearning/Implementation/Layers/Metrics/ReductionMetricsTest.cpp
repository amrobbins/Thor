#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/ReductionMetrics.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include "gtest/gtest.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

using namespace std;
using namespace ThorImplementation;

namespace {

template <typename MetricT, typename ExpectedFn>
void expectUnaryReductionMetricComputes(ExpectedFn expectedFn) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const vector<uint64_t> dimensions{7, 5};
    TensorDescriptor descriptor(DataType::FP32, dimensions);

    Tensor valuesCpu(cpuPlacement, descriptor);
    Tensor valuesGpu(gpuPlacement, descriptor);

    float* values = static_cast<float*>(valuesCpu.getMemPtr());
    for (uint32_t i = 0; i < valuesCpu.getTotalNumElements(); ++i)
        values[i] = static_cast<float>(int(i) - 13) * 0.25f;

    vector<shared_ptr<Layer>> layers;
    shared_ptr<NetworkInput> valuesInput = make_shared<NetworkInput>(valuesGpu);
    layers.push_back(valuesInput);
    shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
    layers.push_back(noOpLayer);
    shared_ptr<MetricT> metric = make_shared<MetricT>();
    layers.push_back(metric);
    shared_ptr<NetworkOutput> metricOutput = make_shared<NetworkOutput>(gpuPlacement);
    layers.push_back(metricOutput);

    LayerTestHelper::connectTwoLayers(valuesInput, noOpLayer);
    LayerTestHelper::connectTwoLayers(noOpLayer, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(metric, metricOutput, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    ASSERT_TRUE(!metric->getErrorOutput().has_value());
    ASSERT_TRUE(!metric->getErrorInput().has_value());
    ASSERT_TRUE(metric->getFeatureOutput().has_value());

    valuesInput->forward(valuesCpu, false);

    Tensor metricCpu = metricOutput->getFeatureOutput().value().clone(cpuPlacement);
    metricCpu.copyFromAsync(metricOutput->getFeatureOutput().value(), valuesInput->getStream());
    valuesInput->getStream().synchronize();

    const float expected = expectedFn(valuesCpu);
    ASSERT_LT(std::abs(expected - *static_cast<float*>(metricCpu.getMemPtr())), 0.0001f);

    LayerTestHelper::tearDownNetwork(layers);
}

void expectBf16MeanComputesExpectedValue() {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const vector<uint64_t> dimensions{7, 5};
    TensorDescriptor descriptor(DataType::BF16, dimensions);

    Tensor valuesCpu(cpuPlacement, descriptor);
    Tensor valuesGpu(gpuPlacement, descriptor);

    __nv_bfloat16* values = static_cast<__nv_bfloat16*>(valuesCpu.getMemPtr());
    double expectedTotal = 0.0;
    for (uint32_t i = 0; i < valuesCpu.getTotalNumElements(); ++i) {
        values[i] = __nv_bfloat16(static_cast<float>(int(i) - 13) * 0.25f);
        expectedTotal += static_cast<float>(values[i]);
    }

    vector<shared_ptr<Layer>> layers;
    shared_ptr<NetworkInput> valuesInput = make_shared<NetworkInput>(valuesGpu);
    layers.push_back(valuesInput);
    shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
    layers.push_back(noOpLayer);
    shared_ptr<Mean> metric = make_shared<Mean>();
    layers.push_back(metric);
    shared_ptr<NetworkOutput> metricOutput = make_shared<NetworkOutput>(gpuPlacement);
    layers.push_back(metricOutput);

    LayerTestHelper::connectTwoLayers(valuesInput, noOpLayer);
    LayerTestHelper::connectTwoLayers(noOpLayer, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(metric, metricOutput, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    ASSERT_TRUE(metric->getFeatureOutput().has_value());
    ASSERT_EQ(metric->getFeatureOutput()->getDescriptor().getDataType(), DataType::FP32);

    valuesInput->forward(valuesCpu, false);

    Tensor metricCpu = metricOutput->getFeatureOutput().value().clone(cpuPlacement);
    metricCpu.copyFromAsync(metricOutput->getFeatureOutput().value(), valuesInput->getStream());
    valuesInput->getStream().synchronize();

    const float expected = static_cast<float>(expectedTotal / static_cast<double>(valuesCpu.getTotalNumElements()));
    ASSERT_LT(std::abs(expected - *static_cast<float*>(metricCpu.getMemPtr())), 0.0001f);

    LayerTestHelper::tearDownNetwork(layers);
}

float expectedMean(const Tensor& valuesCpu) {
    const float* values = static_cast<const float*>(valuesCpu.getMemPtr());
    double total = 0.0;
    for (uint32_t i = 0; i < valuesCpu.getTotalNumElements(); ++i)
        total += values[i];
    return static_cast<float>(total / static_cast<double>(valuesCpu.getTotalNumElements()));
}

float expectedSum(const Tensor& valuesCpu) {
    const float* values = static_cast<const float*>(valuesCpu.getMemPtr());
    double total = 0.0;
    for (uint32_t i = 0; i < valuesCpu.getTotalNumElements(); ++i)
        total += values[i];
    return static_cast<float>(total);
}

float expectedMin(const Tensor& valuesCpu) {
    const float* values = static_cast<const float*>(valuesCpu.getMemPtr());
    float value = std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < valuesCpu.getTotalNumElements(); ++i)
        value = std::min(value, values[i]);
    return value;
}

float expectedMax(const Tensor& valuesCpu) {
    const float* values = static_cast<const float*>(valuesCpu.getMemPtr());
    float value = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < valuesCpu.getTotalNumElements(); ++i)
        value = std::max(value, values[i]);
    return value;
}

void expectRatioStatistics(const shared_ptr<Metric>& metric,
                           uint32_t slotIndex,
                           float expectedNumerator,
                           float expectedDenominator,
                           float tolerance = 1e-6f) {
    optional<MetricBatchStatisticTensors> statistics =
        metric->getMetricBatchStatisticTensorsForSlot(slotIndex);
    ASSERT_TRUE(statistics.has_value());
    EXPECT_EQ(statistics->aggregation, Thor::MetricAggregation::RATIO);
    ASSERT_TRUE(statistics->numerator.has_value());
    ASSERT_TRUE(statistics->denominator.has_value());
    ASSERT_TRUE(statistics->readyEvent.isInitialized());
    statistics->readyEvent.synchronize();
    EXPECT_NEAR(*statistics->numerator->getMemPtr<float>(), expectedNumerator, tolerance);
    EXPECT_NEAR(*statistics->denominator->getMemPtr<float>(), expectedDenominator, tolerance);
}

}  // namespace

TEST(ReductionMetrics, MeanComputesExpectedValue) { expectUnaryReductionMetricComputes<Mean>(expectedMean); }

TEST(ReductionMetrics, MeanComputesExpectedValueForBf16Input) { expectBf16MeanComputesExpectedValue(); }

TEST(ReductionMetrics, SumComputesExpectedValue) { expectUnaryReductionMetricComputes<Sum>(expectedSum); }

TEST(ReductionMetrics, MinComputesExpectedValue) { expectUnaryReductionMetricComputes<Min>(expectedMin); }

TEST(ReductionMetrics, MaxComputesExpectedValue) { expectUnaryReductionMetricComputes<Max>(expectedMax); }

TEST(ReductionMetrics, WeightedMeanComputesExpectedValue) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const vector<uint64_t> dimensions{4, 5};
    TensorDescriptor descriptor(DataType::FP32, dimensions);

    Tensor valuesCpu(cpuPlacement, descriptor);
    Tensor weightsCpu(cpuPlacement, descriptor);
    Tensor valuesGpu(gpuPlacement, descriptor);
    Tensor weightsGpu(gpuPlacement, descriptor);

    float* values = static_cast<float*>(valuesCpu.getMemPtr());
    float* weights = static_cast<float*>(weightsCpu.getMemPtr());
    double weightedTotal = 0.0;
    double weightTotal = 0.0;
    for (uint32_t i = 0; i < valuesCpu.getTotalNumElements(); ++i) {
        values[i] = static_cast<float>(i + 1);
        weights[i] = static_cast<float>((i % 4) + 1);
        weightedTotal += static_cast<double>(values[i]) * static_cast<double>(weights[i]);
        weightTotal += weights[i];
    }

    vector<shared_ptr<Layer>> layers;
    shared_ptr<NetworkInput> valuesInput = make_shared<NetworkInput>(valuesGpu);
    layers.push_back(valuesInput);
    shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
    layers.push_back(noOpLayer);
    shared_ptr<NetworkInput> weightsInput = make_shared<NetworkInput>(weightsGpu);
    layers.push_back(weightsInput);
    shared_ptr<WeightedMean> metric = make_shared<WeightedMean>();
    layers.push_back(metric);
    shared_ptr<NetworkOutput> metricOutput = make_shared<NetworkOutput>(gpuPlacement);
    layers.push_back(metricOutput);

    LayerTestHelper::connectTwoLayers(valuesInput, noOpLayer);
    LayerTestHelper::connectTwoLayers(noOpLayer, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(weightsInput, metric, 0, static_cast<int>(Metric::ConnectionType::LABELS));
    LayerTestHelper::connectTwoLayers(metric, metricOutput, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    valuesInput->forward(valuesCpu, false);
    weightsInput->forward(weightsCpu, false);

    Tensor metricCpu = metricOutput->getFeatureOutput().value().clone(cpuPlacement);
    metricCpu.copyFromAsync(metricOutput->getFeatureOutput().value(), valuesInput->getStream());
    valuesInput->getStream().synchronize();

    const float expected = static_cast<float>(weightedTotal / weightTotal);
    ASSERT_LT(std::abs(expected - *static_cast<float*>(metricCpu.getMemPtr())), 0.0001f);
    expectRatioStatistics(metric, 0, static_cast<float>(weightedTotal), static_cast<float>(weightTotal), 1e-4f);

    LayerTestHelper::tearDownNetwork(layers);
}

namespace {

template <typename MetricT>
float runPartialUnaryReductionMetric(const vector<float>& values, uint32_t validExampleCount) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4, 2});
    Tensor valuesCpu(cpuPlacement, descriptor);
    Tensor valuesGpu(gpuPlacement, descriptor);
    std::copy(values.begin(), values.end(), valuesCpu.getMemPtr<float>());

    vector<shared_ptr<Layer>> layers;
    auto input = make_shared<NetworkInput>(valuesGpu);
    auto metric = make_shared<MetricT>();
    auto output = make_shared<NetworkOutput>(gpuPlacement);
    layers = {input, metric, output};
    LayerTestHelper::connectTwoLayers(input, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(metric, output, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    input->forward(valuesCpu, false, validExampleCount);
    Tensor resultCpu = output->getFeatureOutput().value().clone(cpuPlacement);
    resultCpu.copyFromAsync(output->getFeatureOutput().value(), input->getStream());
    input->getStream().synchronize();
    const float result = *resultCpu.getMemPtr<float>();
    LayerTestHelper::tearDownNetwork(layers);
    return result;
}

}  // namespace

TEST(ReductionMetrics, PartialBatchReductionsIgnoreInvalidTailRows) {
    const vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, -1000.0f, -2000.0f, 1000.0f, 2000.0f};
    EXPECT_NEAR(runPartialUnaryReductionMetric<Mean>(values, 2), 2.5f, 1e-6f);
    EXPECT_NEAR(runPartialUnaryReductionMetric<Sum>(values, 2), 10.0f, 1e-6f);
    EXPECT_NEAR(runPartialUnaryReductionMetric<Min>(values, 2), 1.0f, 1e-6f);
    EXPECT_NEAR(runPartialUnaryReductionMetric<Max>(values, 2), 4.0f, 1e-6f);
}

TEST(ReductionMetrics, PartialBatchWeightedMeanIgnoresInvalidTailRows) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4, 2});
    Tensor valuesCpu(cpuPlacement, descriptor);
    Tensor weightsCpu(cpuPlacement, descriptor);
    Tensor valuesGpu(gpuPlacement, descriptor);
    Tensor weightsGpu(gpuPlacement, descriptor);

    const vector<float> values = {1.0f, 3.0f, 5.0f, 7.0f, 1000.0f, 1000.0f, -1000.0f, -1000.0f};
    const vector<float> weights = {1.0f, 1.0f, 2.0f, 2.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    std::copy(values.begin(), values.end(), valuesCpu.getMemPtr<float>());
    std::copy(weights.begin(), weights.end(), weightsCpu.getMemPtr<float>());

    vector<shared_ptr<Layer>> layers;
    auto valuesInput = make_shared<NetworkInput>(valuesGpu);
    auto weightsInput = make_shared<NetworkInput>(weightsGpu);
    auto metric = make_shared<WeightedMean>();
    auto output = make_shared<NetworkOutput>(gpuPlacement);
    layers = {valuesInput, weightsInput, metric, output};
    LayerTestHelper::connectTwoLayers(valuesInput, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(weightsInput, metric, 0, static_cast<int>(Metric::ConnectionType::LABELS));
    LayerTestHelper::connectTwoLayers(metric, output, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    valuesInput->forward(valuesCpu, false, 2);
    weightsInput->forward(weightsCpu, false, 2);
    Tensor resultCpu = output->getFeatureOutput().value().clone(cpuPlacement);
    resultCpu.copyFromAsync(output->getFeatureOutput().value(), valuesInput->getStream());
    valuesInput->getStream().synchronize();
    EXPECT_NEAR(*resultCpu.getMemPtr<float>(), 28.0f / 6.0f, 1e-6f);
    expectRatioStatistics(metric, 0, 28.0f, 6.0f);

    LayerTestHelper::tearDownNetwork(layers);
}


TEST(ReductionMetrics, WeightedMeanRetainsNumeratorWhenDenominatorIsZero) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {2, 1});
    Tensor valuesCpu(cpuPlacement, descriptor);
    Tensor weightsCpu(cpuPlacement, descriptor);
    Tensor valuesGpu(gpuPlacement, descriptor);
    Tensor weightsGpu(gpuPlacement, descriptor);

    const vector<float> values = {1.0f, 3.0f};
    const vector<float> weights = {1.0f, -1.0f};
    std::copy(values.begin(), values.end(), valuesCpu.getMemPtr<float>());
    std::copy(weights.begin(), weights.end(), weightsCpu.getMemPtr<float>());

    auto valuesInput = make_shared<NetworkInput>(valuesGpu);
    auto weightsInput = make_shared<NetworkInput>(weightsGpu);
    auto metric = make_shared<WeightedMean>();
    auto output = make_shared<NetworkOutput>(gpuPlacement);
    vector<shared_ptr<Layer>> layers{valuesInput, weightsInput, metric, output};
    LayerTestHelper::connectTwoLayers(valuesInput, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(weightsInput, metric, 0, static_cast<int>(Metric::ConnectionType::LABELS));
    LayerTestHelper::connectTwoLayers(metric, output, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    valuesInput->forward(valuesCpu, false);
    weightsInput->forward(weightsCpu, false);

    Tensor resultCpu = output->getFeatureOutput().value().clone(cpuPlacement);
    resultCpu.copyFromAsync(output->getFeatureOutput().value(), valuesInput->getStream());
    valuesInput->getStream().synchronize();
    EXPECT_FLOAT_EQ(*resultCpu.getMemPtr<float>(), 0.0f);
    expectRatioStatistics(metric, 0, -2.0f, 0.0f);

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(ReductionMetrics, WeightedMeanStatisticSlotsDoNotOverwriteInFlightBatches) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {2, 1});
    Tensor firstValuesCpu(cpuPlacement, descriptor);
    Tensor firstWeightsCpu(cpuPlacement, descriptor);
    Tensor secondValuesCpu(cpuPlacement, descriptor);
    Tensor secondWeightsCpu(cpuPlacement, descriptor);
    Tensor valuesGpu(gpuPlacement, descriptor);
    Tensor weightsGpu(gpuPlacement, descriptor);

    auto valuesInput = make_shared<NetworkInput>(valuesGpu);
    auto weightsInput = make_shared<NetworkInput>(weightsGpu);
    auto metric = make_shared<WeightedMean>();
    auto output = make_shared<NetworkOutput>(gpuPlacement);
    vector<shared_ptr<Layer>> layers{valuesInput, weightsInput, metric, output};
    LayerTestHelper::connectTwoLayers(valuesInput, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(weightsInput, metric, 0, static_cast<int>(Metric::ConnectionType::LABELS));
    LayerTestHelper::connectTwoLayers(metric, output, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);
    metric->preallocateMetricStatisticSlots(2);

    const vector<float> firstValues = {2.0f, 4.0f};
    const vector<float> firstWeights = {1.0f, 2.0f};
    std::copy(firstValues.begin(), firstValues.end(), firstValuesCpu.getMemPtr<float>());
    std::copy(firstWeights.begin(), firstWeights.end(), firstWeightsCpu.getMemPtr<float>());
    metric->setActiveMetricStatisticSlot(0);
    valuesInput->forward(firstValuesCpu, false);
    weightsInput->forward(firstWeightsCpu, false);

    // NetworkInput exposes one statically connected feature tensor per input. Before
    // reusing those tensors for batch 2, wait until batch 1 has consumed both inputs
    // and copied its ratio statistics into slot 0's device buffers. Deliberately do
    // not synchronize the metric download stream: slot 0's host download may remain
    // in flight while batch 2 is submitted, which is the behavior under test.
    valuesInput->getStream().synchronize();

    const vector<float> secondValues = {10.0f, 20.0f};
    const vector<float> secondWeights = {3.0f, 4.0f};
    std::copy(secondValues.begin(), secondValues.end(), secondValuesCpu.getMemPtr<float>());
    std::copy(secondWeights.begin(), secondWeights.end(), secondWeightsCpu.getMemPtr<float>());
    metric->setActiveMetricStatisticSlot(1);
    valuesInput->forward(secondValuesCpu, false);
    weightsInput->forward(secondWeightsCpu, false);

    expectRatioStatistics(metric, 0, 10.0f, 3.0f);
    expectRatioStatistics(metric, 1, 110.0f, 7.0f);

    LayerTestHelper::tearDownNetwork(layers);
}
