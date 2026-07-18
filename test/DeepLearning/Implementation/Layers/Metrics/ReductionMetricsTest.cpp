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

    LayerTestHelper::tearDownNetwork(layers);
}
