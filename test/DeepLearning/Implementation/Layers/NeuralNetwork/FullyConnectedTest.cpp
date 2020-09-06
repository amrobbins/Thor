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

void backwardPass(FullyConnected *fullyConnectedLayer, bool hasBiases, bool accumulate);

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
        bool hasBiases = (rand() % 4) != 0;
        Optional<float> learningRate;
        if (!inferenceOnly)
            learningRate = (1 + (rand() % 20000)) / 10000.0f;

        Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));
        Tensor weights = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numInputFeatures, numOutputFeatures}));
        Tensor featureOut = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numOutputFeatures}));
        Tensor featureOutGpu_h;
        Tensor biases = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numOutputFeatures}));

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

        half *biasesMem = (half *)biases.getMemPtr();
        if (hasBiases) {
            for (unsigned int i = 0; i < numOutputFeatures; ++i) {
                biasesMem[i] = ((rand() % 100) / 2.0f) - 5.0f;
            }
        }

        vector<Layer *> layers;

        layers.push_back(
            new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions(), stream));
        layers.push_back(new NoOpLayer());
        FullyConnected *fullyConnectedLayer =
            new FullyConnected(numInputFeatures, numOutputFeatures, batchSize, inferenceOnly, hasBiases, learningRate);
        fullyConnectedLayer->setInferenceOnly(inferenceOnly);

        layers.push_back(fullyConnectedLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);

        FullyConnected::UniformRandomInitialization initializer(0.1, -0.1);
        fullyConnectedLayer->initializeWeights(&initializer);

        Tensor weightsInitial = fullyConnectedLayer->getWeights().clone(cpuPlacement);
        weightsInitial.copyFromAsync(fullyConnectedLayer->getWeights(), stream);
        stream.synchronize();
        int totalNumWeights = fullyConnectedLayer->getWeights().getDescriptor().getTotalNumElements();
        half *weightsInitialMem = (half *)weightsInitial.getMemPtr();
        for (int i = 0; i < totalNumWeights; ++i) {
            ASSERT_LT(abs((float)weightsInitialMem[i]), 0.1);
        }

        // Backward tensors must not be created, since they would be unused and would waist memory.
        if (inferenceOnly) {
            ASSERT_TRUE(fullyConnectedLayer->getErrorOutputs()[0].isEmpty());
            ASSERT_TRUE(fullyConnectedLayer->getWeightsGradient().isEmpty());
            ASSERT_TRUE(fullyConnectedLayer->getBiasesGradient().isEmpty());
            ASSERT_TRUE(fullyConnectedLayer->getGradientUpdateStream().isEmpty());
        }

        if (!hasBiases) {
            ASSERT_TRUE(fullyConnectedLayer->getBiases().isEmpty());
            ASSERT_TRUE(fullyConnectedLayer->getBiasesGradient().isEmpty());
        }

        featureOutGpu_h = layers.back()->getFeatureOutput();

        Event weightsUpdatedEvent = fullyConnectedLayer->updateWeightsAndBiases(weights, biases, stream.putEvent());
        stream.waitEvent(weightsUpdatedEvent);
        if (!inferenceOnly)
            fullyConnectedLayer->getGradientUpdateStream().get().waitEvent(weightsUpdatedEvent);

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

        if (hasBiases) {
            half *featureOutMem = (half *)featureOut.getMemPtr();
            for (unsigned int batch = 0; batch < batchSize; ++batch) {
                for (unsigned int outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                    featureOutMem[batch * numOutputFeatures + outputFeature] =
                        featureOutMem[batch * numOutputFeatures + outputFeature] + biasesMem[outputFeature];
                }
            }
        }

        stream.synchronize();

        half *cpuFeatureOut = (half *)featureOut.getMemPtr();
        half *gpuFeatureOut = (half *)featureOutGpu_h.getMemPtr();
        int numOutputElements = batchSize * numOutputFeatures;
        float maxDiff = numInputFeatures * 0.01;
        for (int i = 0; i < numOutputElements; ++i) {
            EXPECT_LT(abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])), maxDiff);
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= maxDiff) {
                int batch = i / numOutputFeatures;
                int outputFeature = i - (batch * numOutputFeatures);
                printf("%f %f   bias %f\n",
                       (float)(cpuFeatureOut[i]),
                       (float)(gpuFeatureOut[i]),
                       hasBiases ? (float)biasesMem[outputFeature] : 0.0f);
            }
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        bool accumulate = false;  // FIXME

        backwardPass(fullyConnectedLayer, hasBiases, accumulate);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

void backwardPass(FullyConnected *fullyConnectedLayer, bool hasBiases, bool accumulate) {
    assert(accumulate == false);  // FIXME

    Stream dataStream = fullyConnectedLayer->getStreams()[0];
    Stream gradientUpdateStream = fullyConnectedLayer->getGradientUpdateStream().get();

    Tensor errorInput = fullyConnectedLayer->getErrorInputs().front().get().clone(TensorPlacement::MemDevices::CPU);

    Tensor featureInput = fullyConnectedLayer->getFeatureInputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor errorOutput = fullyConnectedLayer->getErrorOutputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor weights = fullyConnectedLayer->getWeights().clone(TensorPlacement::MemDevices::CPU);
    Tensor weightsGradient = fullyConnectedLayer->getWeightsGradient().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor biases;
    Tensor biasesGradient;
    Tensor biasesGradientFloat;
    if (hasBiases) {
        biases = fullyConnectedLayer->getBiases().get().clone(TensorPlacement::MemDevices::CPU);
        biasesGradient = fullyConnectedLayer->getBiasesGradient().get().clone(TensorPlacement::MemDevices::CPU);
        biasesGradientFloat = biasesGradient.clone(TensorDescriptor::DataType::FP32);
    }

    int batchSize = featureInput.getDescriptor().getDimensions()[0];
    int numInputFeatures = featureInput.getDescriptor().getDimensions()[1];
    int numOutputFeatures = errorInput.getDescriptor().getDimensions()[1];

    // featureIn, weights and biases are populated still by the forward pass
    // just need to populate errorIn
    featureInput.copyFromAsync(fullyConnectedLayer->getFeatureInputs().front(), dataStream);
    weights.copyFromAsync(fullyConnectedLayer->getWeights(), dataStream);
    if (hasBiases) {
        biases.copyFromAsync(fullyConnectedLayer->getBiases(), dataStream);
        biasesGradient.copyFromAsync(fullyConnectedLayer->getBiasesGradient(), dataStream);
        biasesGradientFloat.copyFromAsync(biasesGradient, dataStream);
    }
    dataStream.synchronize();

    const int errorInputSize = batchSize * numOutputFeatures;
    half *errorInputMem = (half *)errorInput.getMemPtr();
    for (int i = 0; i < errorInputSize; ++i) {
        errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }
    fullyConnectedLayer->getErrorInputs().front().get().copyFromAsync(errorInput, dataStream);

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

    if (hasBiases) {
        // Reduce errorIn_batchSizexnumOutputFeatures to biasesGradient_numOutputFeatures by summing the elements of each batch item.
        float *biasesGradientMem = (float *)biasesGradientFloat.getMemPtr();
        half *errorInMem = (half *)errorInput.getMemPtr();
        for (int batch = 0; batch < batchSize; ++batch) {
            for (int outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                if (batch == 0 && !accumulate)
                    biasesGradientMem[outputFeature] = 0.0f;
                biasesGradientMem[outputFeature] =
                    biasesGradientMem[outputFeature] + (float)errorInMem[batch * numOutputFeatures + outputFeature];
            }
        }
        biasesGradient.copyFromAsync(biasesGradientFloat, gradientUpdateStream);
    }

    Tensor errorOutputGpu_h = fullyConnectedLayer->getErrorOutputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor weightsGradientGpu_h = fullyConnectedLayer->getWeightsGradient().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor biasesGradientGpu_h;
    if (hasBiases)
        biasesGradientGpu_h = fullyConnectedLayer->getBiasesGradient().get().clone(TensorPlacement::MemDevices::CPU);

    errorOutputGpu_h.copyFromAsync(fullyConnectedLayer->getErrorOutputs().front(), dataStream);
    weightsGradientGpu_h.copyFromAsync(fullyConnectedLayer->getWeightsGradient(), gradientUpdateStream);
    if (hasBiases)
        biasesGradientGpu_h.copyFromAsync(fullyConnectedLayer->getBiasesGradient(), gradientUpdateStream);

    dataStream.synchronize();
    gradientUpdateStream.synchronize();

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

    if (hasBiases) {
        half *cpuBiasesGradient = (half *)biasesGradient.getMemPtr();
        half *gpuBiasesGradient = (half *)biasesGradientGpu_h.getMemPtr();

        for (int i = 0; i < numOutputFeatures; ++i) {
            EXPECT_LT(abs((float)(cpuBiasesGradient[i]) - (float)(gpuBiasesGradient[i])), maxDiff);
            if (abs((float)(cpuBiasesGradient[i]) - (float)(gpuBiasesGradient[i])) < maxDiff) {
            } else {
                printf("%d of %d   %f %f\n", i + 1, numOutputFeatures, (float)(cpuBiasesGradient[i]), (float)(gpuBiasesGradient[i]));
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}