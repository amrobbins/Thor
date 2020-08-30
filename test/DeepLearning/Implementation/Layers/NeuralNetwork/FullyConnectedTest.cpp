#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#include "MLDev.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <vector>

using std::vector;

TEST(FullyConnected, FullyConnectedWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);

    for (int test = 0; test < 5; ++test) {
        uint64_t numInputFeatures = (rand() % 300) + 1;
        uint64_t numOutputFeatures = (rand() % 300) + 1;
        uint64_t batchSize = (rand() % 300) + 1;
        bool inferenceOnly = true;  // FIXME
        bool hasBiases = false;     // FIXME
        Optional<float> learningRate;
        if (!inferenceOnly)
            learningRate = (1 + (rand() % 20000)) / 10000.0f;

        Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));
        Tensor weights = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numInputFeatures, numOutputFeatures}));
        Tensor featureOut = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numOutputFeatures}));
        Tensor featureOutGpu_h;
        Tensor biases;

        const int featureInSize = batchSize * numInputFeatures;
        half *featureInMem = (half *)featureIn.getMemPtr();
        for (int i = 0; i < featureInSize; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        const int weightsSize = numInputFeatures * numOutputFeatures;
        half *weightsMem = (half *)weights.getMemPtr();
        for (int i = 0; i < weightsSize; ++i) {
            weightsMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        if (hasBiases) {
            // FIXME: create and fill biases
        }

        vector<Layer *> layers;

        layers.push_back(
            new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions(), stream));
        layers.push_back(new NoOpLayer());
        FullyConnected *fullyConnectedLayer =
            new FullyConnected(numInputFeatures, numOutputFeatures, batchSize, inferenceOnly, hasBiases, learningRate);

        layers.push_back(fullyConnectedLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        featureOutGpu_h = layers.back()->getFeatureOutput();

        fullyConnectedLayer->getWeights().copyFromAsync(weights, stream);
        if (hasBiases)
            fullyConnectedLayer->getBiases().get().copyFromAsync(biases, stream);

        // Network is runnable here
        layers[0]->forward(featureIn);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());

        matrixMultiplyCpuHalf((half *)featureIn.getMemPtr(),
                              (half *)weights.getMemPtr(),
                              (half *)featureOut.getMemPtr(),
                              batchSize,
                              numInputFeatures,
                              numInputFeatures,
                              numOutputFeatures,
                              numInputFeatures,
                              numOutputFeatures,
                              numOutputFeatures,
                              false,
                              false,
                              false);

        stream.synchronize();

        half *cpuFeatureOut = (half *)featureOut.getMemPtr();
        half *gpuFeatureOut = (half *)featureOutGpu_h.getMemPtr();
        int numOutputElements = batchSize * numOutputFeatures;
        for (int i = 0; i < numOutputElements; ++i) {
            float thresh = std::max(abs((float)cpuFeatureOut[i]) / 500, 0.01f);
            EXPECT_LT(abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])), thresh);
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= thresh)
                printf("%f %f\n", (float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }
    }
}

// FIXME: make a test for multiple connections

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
