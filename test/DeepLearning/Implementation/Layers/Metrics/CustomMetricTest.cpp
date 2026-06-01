#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <vector>

using namespace std;
using namespace ThorImplementation;

namespace {

DynamicExpression buildMeanSquaredErrorMetricExpression() {
    return DynamicExpression({"predictions", "labels"},
                             {"metric"},
                             [](const DynamicExpression::TensorMap& inputs,
                                const DynamicExpression::TensorMap& outputs,
                                Stream& stream) -> DynamicExpressionBuild {
                                 auto predictions = Expression::input("predictions");
                                 auto labels = Expression::input("labels");
                                 auto diff = predictions - labels;
                                 auto metric = (diff * diff).reduce_mean({0, 1}, {0}, DataType::FP32);
                                 auto expressionOutputs = Expression::outputs({{"metric", metric}});
                                 return DynamicExpressionBuild{
                                     std::make_shared<FusedEquation>(
                                         FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum())),
                                     inputs,
                                     {},
                                     outputs,
                                     {}};
                             });
}

}  // namespace

TEST(CustomMetric, ComputesExpressionBackedMeanSquaredError) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const vector<uint64_t> dimensions = {4, 3};
    TensorDescriptor descriptor(DataType::FP32, dimensions);

    Tensor predictionsCpu(cpuPlacement, descriptor);
    Tensor labelsCpu(cpuPlacement, descriptor);
    Tensor predictionsGpu(gpuPlacement, descriptor);
    Tensor labelsGpu(gpuPlacement, descriptor);

    const vector<float> predictionsValues = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.5f, -2.0f, 0.25f, 8.0f, 0.0f, -1.0f};
    const vector<float> labelsValues = {0.0f, 2.5f, 1.0f, 4.0f, 7.0f, 5.0f, 2.5f, -3.0f, 0.25f, 6.0f, 1.0f, -1.5f};
    ASSERT_EQ(predictionsValues.size(), predictionsCpu.getTotalNumElements());
    ASSERT_EQ(labelsValues.size(), labelsCpu.getTotalNumElements());

    auto* predictions = static_cast<float*>(predictionsCpu.getMemPtr());
    auto* labels = static_cast<float*>(labelsCpu.getMemPtr());
    for (uint64_t i = 0; i < predictionsValues.size(); ++i) {
        predictions[i] = predictionsValues[i];
        labels[i] = labelsValues[i];
    }

    vector<shared_ptr<Layer>> layers;
    shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
    layers.push_back(predictionsInput);
    shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
    layers.push_back(noOpLayer);
    shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
    layers.push_back(labelsInput);
    shared_ptr<CustomMetric> customMetric = make_shared<CustomMetric>(buildMeanSquaredErrorMetricExpression(),
                                                                      "predictions",
                                                                      "labels",
                                                                      "metric",
                                                                      "MSE");
    layers.push_back(customMetric);
    shared_ptr<NetworkOutput> metricOutput = make_shared<NetworkOutput>(gpuPlacement);
    layers.push_back(metricOutput);

    LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
    LayerTestHelper::connectTwoLayers(noOpLayer, customMetric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(labelsInput, customMetric, 0, static_cast<int>(Metric::ConnectionType::LABELS));
    LayerTestHelper::connectTwoLayers(customMetric, metricOutput, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    ASSERT_FALSE(customMetric->getErrorOutput().has_value());
    ASSERT_FALSE(customMetric->getErrorInput().has_value());
    ASSERT_TRUE(customMetric->getFeatureOutput().has_value());
    ASSERT_EQ(customMetric->getFeatureOutput().value().getDescriptor().getDimensions(), vector<uint64_t>({1}));
    ASSERT_EQ(customMetric->getFeatureOutput().value().getDescriptor().getDataType(), DataType::FP32);

    predictionsInput->forward(predictionsCpu, false);
    labelsInput->forward(labelsCpu, false);

    Stream stream = predictionsInput->getStream();
    Tensor metricHost = metricOutput->getFeatureOutput().value().clone(cpuPlacement);
    metricHost.copyFromAsync(metricOutput->getFeatureOutput().value(), stream);
    stream.synchronize();

    float expected = 0.0f;
    for (uint64_t i = 0; i < predictionsValues.size(); ++i) {
        const float diff = predictionsValues[i] - labelsValues[i];
        expected += diff * diff;
    }
    expected /= static_cast<float>(predictionsValues.size());

    const float actual = *static_cast<float*>(metricHost.getMemPtr());
    ASSERT_LT(std::fabs(actual - expected), 1e-5f) << "actual=" << actual << " expected=" << expected;

    LayerTestHelper::tearDownNetwork(layers);
}
