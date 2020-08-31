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

// FIXME: make a test with accumulate once cublasLt bug is fixed.
// FIXME: make a test for multiple connections

void backwardPass(FullyConnected *fullyConnectedLayer, bool hasBiases);

TEST(FullyConnected, FullyConnectedWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);

    for (int test = 0; test < 5; ++test) {
        uint64_t numInputFeatures = (rand() % 300) + 1;
        uint64_t numOutputFeatures = (rand() % 300) + 1;
        uint64_t batchSize = (rand() % 300) + 1;
        bool inferenceOnly = (rand() % 4) == 0;
        bool hasBiases = false;  // FIXME
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
        float maxDiff = numInputFeatures * 0.01;
        for (int i = 0; i < numOutputElements; ++i) {
            EXPECT_LT(abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])), maxDiff);
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= maxDiff)
                printf("%f %f\n", (float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        backwardPass(fullyConnectedLayer, hasBiases);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

void backwardPass(FullyConnected *fullyConnectedLayer, bool hasBiases) {
    assert(hasBiases == false);  // FIXME

    Stream stream = fullyConnectedLayer->getStreams()[0];

    Tensor featureInput = fullyConnectedLayer->getFeatureInputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor errorInput = fullyConnectedLayer->getErrorInputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor errorOutput = fullyConnectedLayer->getErrorOutputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor weights = fullyConnectedLayer->getWeights().clone(TensorPlacement::MemDevices::CPU);
    Tensor weightsGradient = fullyConnectedLayer->getWeightsGradient().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor biases;
    Tensor biasesGradient;
    if (hasBiases) {
        biases = fullyConnectedLayer->getBiases().get().clone(TensorPlacement::MemDevices::CPU);
        biasesGradient = fullyConnectedLayer->getBiasesGradient().get().clone(TensorPlacement::MemDevices::CPU);
    }

    int batchSize = featureInput.getDescriptor().getDimensions()[0];
    int numInputFeatures = featureInput.getDescriptor().getDimensions()[1];
    int numOutputFeatures = errorInput.getDescriptor().getDimensions()[1];

    // featureIn, weights and biases are populated still by the forward pass
    // just need to populate errorIn
    featureInput.copyFromAsync(fullyConnectedLayer->getFeatureInputs().front(), stream);
    weights.copyFromAsync(fullyConnectedLayer->getWeights(), stream);
    if (hasBiases) {
        biases.copyFromAsync(fullyConnectedLayer->getBiases(), stream);
        biasesGradient.copyFromAsync(fullyConnectedLayer->getBiasesGradient(), stream);
    }
    stream.synchronize();

    const int errorInputSize = batchSize * numOutputFeatures;
    half *errorInputMem = (half *)errorInput.getMemPtr();
    for (int i = 0; i < errorInputSize; ++i) {
        errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }
    fullyConnectedLayer->getErrorInputs().front().get().copyFromAsync(errorInput, stream);

    fullyConnectedLayer->backward(fullyConnectedLayer->getErrorInputs().front());

    matrixMultiplyCpuHalf((half *)errorInput.getMemPtr(),
                          (half *)weights.getMemPtr(),
                          (half *)errorOutput.getMemPtr(),
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numInputFeatures,
                          false,
                          true,
                          false);

    matrixMultiplyCpuHalf((half *)featureInput.getMemPtr(),
                          (half *)errorInput.getMemPtr(),
                          (half *)weightsGradient.getMemPtr(),
                          batchSize,
                          numInputFeatures,
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          true,
                          false,
                          false);

    Tensor errorOutputGpu_h = fullyConnectedLayer->getErrorOutputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor weightsGradientGpu_h = fullyConnectedLayer->getWeightsGradient().get().clone(TensorPlacement::MemDevices::CPU);

    errorOutputGpu_h.copyFromAsync(fullyConnectedLayer->getErrorOutputs().front(), stream);
    weightsGradientGpu_h.copyFromAsync(fullyConnectedLayer->getWeightsGradient(), stream);

    stream.synchronize();

    half *cpuErrorOutput = (half *)errorOutput.getMemPtr();
    half *gpuErrorOutput = (half *)errorOutputGpu_h.getMemPtr();
    int numOutputElements = batchSize * numInputFeatures;
    float maxDiff = numOutputFeatures * 0.01;
    for (int i = 0; i < numOutputElements; ++i) {
        EXPECT_LT(abs((float)(cpuErrorOutput[i]) - (float)(gpuErrorOutput[i])), maxDiff);
        if (abs((float)(cpuErrorOutput[i]) - (float)(gpuErrorOutput[i])) >= maxDiff)
            printf("%f %f\n", (float)(cpuErrorOutput[i]), (float)(gpuErrorOutput[i]));
    }

    half *cpuWeightsGradient = (half *)weightsGradient.getMemPtr();
    half *gpuWeightsGradient = (half *)weightsGradientGpu_h.getMemPtr();
    int numWeights = numInputFeatures * numOutputFeatures;
    maxDiff = batchSize * 0.01;
    for (int i = 0; i < numWeights; ++i) {
        EXPECT_LT(abs((float)(cpuWeightsGradient[i]) - (float)(gpuWeightsGradient[i])), maxDiff);
        if (abs((float)(cpuWeightsGradient[i]) - (float)(gpuWeightsGradient[i])) >= maxDiff)
            printf("%f %f\n", (float)(cpuWeightsGradient[i]), (float)(gpuWeightsGradient[i]));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
