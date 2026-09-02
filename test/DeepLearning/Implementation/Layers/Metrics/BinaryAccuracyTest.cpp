#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/BinaryAccuracy.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

using namespace std;

using namespace ThorImplementation;

TEST(BinaryAccuracy, R1CorrectnessReductionUsesFp16WorkspaceBeforeFp32CubAccumulation) {
    const DynamicExpression expression = BinaryAccuracyDetail::makeExpression();
    const std::shared_ptr<const ExpressionDefinition> definition = expression.getSerializedDefinition();
    ASSERT_NE(definition, nullptr);

    PhysicalOutputs outputs = definition->outputs;
    std::vector<DataType> inputDTypes(outputs.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.name == "predictions" || input.name == "labels")
            inputDTypes.at(input.slot) = DataType::FP16;
    }
    ASSERT_NO_THROW(resolveOutputsDTypesInPlace(outputs, inputDTypes));

    bool foundCorrectnessReduction = false;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op != ExprOp::REDUCE_SUM || node.lhs == UINT32_MAX || node.lhs >= outputs.expr->nodes.size())
            continue;
        const ExprNode& parent = outputs.expr->nodes.at(node.lhs);
        if (parent.op != ExprOp::MUL)
            continue;
        const std::optional<DataType> parentStorage =
            materializedValueStorageDType(*outputs.expr, node.lhs);
        ASSERT_TRUE(parentStorage.has_value());
        EXPECT_EQ(parentStorage.value(), DataType::FP16);
        ASSERT_TRUE(node.output_dtype.has_value());
        EXPECT_EQ(node.output_dtype.value(), DataType::FP32);
        foundCorrectnessReduction = true;
    }
    EXPECT_TRUE(foundCorrectnessReduction);
}

TEST(BinaryAccuracy, ComputesCorrectElementWiseResult) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 20; ++test) {
        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 300) + 1);
        dimensions.push_back(1);
        uint32_t batchSize = dimensions[0];

        bool inferenceOnly = (rand() % 5) == 0;

        TensorDescriptor elementwiseDescriptorFP32(DataType::FP32, dimensions);

        Tensor labelsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor predictionsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor labelsGpu(gpuPlacement, elementwiseDescriptorFP32);
        Tensor predictionsGpu(gpuPlacement, elementwiseDescriptorFP32);

        float *labels = (float *)labelsCpu.getMemPtr();
        float *predictions = (float *)predictionsCpu.getMemPtr();
        unordered_set<float> predictionsUsed;
        for (uint32_t i = 0; i < batchSize; ++i) {
            labels[i] = rand() % 2;
            predictions[i] = ((rand() % 1000000) / 1000000.0f);
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<BinaryAccuracy> binaryAccuracy = make_shared<BinaryAccuracy>();
        if (inferenceOnly)
            binaryAccuracy->setConstructForInferenceOnly(true);
        layers.push_back(binaryAccuracy);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, binaryAccuracy, 0, (int)Metric::ConnectionType::FORWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, binaryAccuracy, 0, (int)Metric::ConnectionType::LABELS);
        shared_ptr<NetworkOutput> accuracyOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(accuracyOutput);
        LayerTestHelper::connectTwoLayers(binaryAccuracy, accuracyOutput, (int)Metric::ConnectionType::METRIC);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(!binaryAccuracy->getErrorOutput().has_value());
        ASSERT_TRUE(!binaryAccuracy->getErrorInput().has_value());

        ASSERT_TRUE(binaryAccuracy->getFeatureOutput().has_value());

        // Network is runnable here
        predictionsInput->forward(predictionsCpu, false);
        labelsInput->forward(labelsCpu, false);

        Tensor accuracyGpu_h = accuracyOutput->getFeatureOutput().value().clone(cpuPlacement);
        accuracyGpu_h.copyFromAsync(accuracyOutput->getFeatureOutput().value(), stream);

        stream.synchronize();

        // Compute the expected accuracy
        uint32_t correctCount = 0;
        for (uint32_t i = 0; i < batchSize; ++i) {
            float expected;
            if (predictions[i] >= 0.5)
                expected = 1;
            else
                expected = 0;
            if (labels[i] == expected)
                correctCount += 1;
        }
        float accuracy_h = correctCount / (float)batchSize;

        float delta = 0.0001;
        if (abs(accuracy_h - *((float *)accuracyGpu_h.getMemPtr())) >= delta) {
            printf("actual %f expected %f\n", *((float *)accuracyGpu_h.getMemPtr()), accuracy_h);
        }
        ASSERT_LT(abs(accuracy_h - *((float *)accuracyGpu_h.getMemPtr())), delta);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(BinaryAccuracy, PartialBatchIgnoresInvalidTailRows) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4, 1});

    Tensor predictionsCpu(cpuPlacement, descriptor);
    Tensor labelsCpu(cpuPlacement, descriptor);
    Tensor predictionsGpu(gpuPlacement, descriptor);
    Tensor labelsGpu(gpuPlacement, descriptor);

    float* predictions = predictionsCpu.getMemPtr<float>();
    float* labels = labelsCpu.getMemPtr<float>();
    predictions[0] = 0.9f;
    labels[0] = 1.0f;
    predictions[1] = 0.1f;
    labels[1] = 0.0f;
    predictions[2] = 0.9f;
    labels[2] = 0.0f;
    predictions[3] = 0.1f;
    labels[3] = 1.0f;

    vector<shared_ptr<Layer>> layers;
    auto predictionsInput = make_shared<NetworkInput>(predictionsGpu);
    auto labelsInput = make_shared<NetworkInput>(labelsGpu);
    auto metric = make_shared<BinaryAccuracy>();
    auto output = make_shared<NetworkOutput>(gpuPlacement);
    layers = {predictionsInput, labelsInput, metric, output};

    LayerTestHelper::connectTwoLayers(predictionsInput, metric, 0, static_cast<int>(Metric::ConnectionType::FORWARD));
    LayerTestHelper::connectTwoLayers(labelsInput, metric, 0, static_cast<int>(Metric::ConnectionType::LABELS));
    LayerTestHelper::connectTwoLayers(metric, output, static_cast<int>(Metric::ConnectionType::METRIC));
    LayerTestHelper::initializeNetwork(layers);

    ASSERT_TRUE(metric->supportsPartialBatches());
    predictionsInput->forward(predictionsCpu, false, 2);
    labelsInput->forward(labelsCpu, false, 2);

    Tensor resultCpu = output->getFeatureOutput().value().clone(cpuPlacement);
    resultCpu.copyFromAsync(output->getFeatureOutput().value(), predictionsInput->getStream());
    predictionsInput->getStream().synchronize();
    EXPECT_NEAR(*resultCpu.getMemPtr<float>(), 1.0f, 1e-6f);

    LayerTestHelper::tearDownNetwork(layers);
}
