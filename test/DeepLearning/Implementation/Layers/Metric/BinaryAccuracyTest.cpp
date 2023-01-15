#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "Thor.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <unordered_set>
#include <vector>

using std::set;
using std::unordered_set;
using std::vector;

using namespace ThorImplementation;

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

        TensorDescriptor elementwiseDescriptorFP32(TensorDescriptor::DataType::FP32, dimensions);

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

        vector<Layer *> layers;
        NetworkInput *predictionsInput = new NetworkInput(predictionsGpu);
        layers.push_back(predictionsInput);
        NoOpLayer *noOpLayer = new NoOpLayer();
        layers.push_back(noOpLayer);
        NetworkInput *labelsInput = new NetworkInput(labelsGpu);
        layers.push_back(labelsInput);
        BinaryAccuracy *binaryAccuracy = new BinaryAccuracy();
        if (inferenceOnly)
            binaryAccuracy->setConstructForInferenceOnly(true);
        layers.push_back(binaryAccuracy);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, binaryAccuracy, 0, (int)Metric::ConnectionType::FORWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, binaryAccuracy, 0, (int)Metric::ConnectionType::LABELS);
        NetworkOutput *accuracyOutput = nullptr;
        if (!inferenceOnly) {
            accuracyOutput = new NetworkOutput(gpuPlacement);
            layers.push_back(accuracyOutput);
            LayerTestHelper::connectTwoLayers(binaryAccuracy, accuracyOutput, (int)Metric::ConnectionType::METRIC);
        }
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(binaryAccuracy->getErrorOutput().isEmpty());
        ASSERT_TRUE(binaryAccuracy->getErrorInput().isEmpty());

        if (inferenceOnly) {
            assert(binaryAccuracy->getFeatureOutput().isEmpty());
            continue;
        }

        // Network is runnable here
        predictionsInput->forward(predictionsCpu, false);
        labelsInput->forward(labelsCpu, false);

        Tensor accuracyGpu_h = accuracyOutput->getFeatureOutput().get().clone(cpuPlacement);
        accuracyGpu_h.copyFromAsync(accuracyOutput->getFeatureOutput(), stream);

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
