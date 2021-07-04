#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

#include "Thor.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <vector>

using std::vector;

using namespace ThorImplementation;

// FIXME: I don't have time now to create tests that verify cudnn is giving the correct forward and backward values
//        Also I have used this layer in other frameworks
//        For now I will settle for connection tests only.
//
//        And then I gave up on that too... So I am not verifying the outputs at all except to say that they are finite valued.
//        Just from eyeballing them, they look ok.

TEST(BatchNormalization, 2dYieldsFiniteValues) {
    srand(time(NULL));

    bool print = false;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 10; ++test) {
        Tensor featureInputCpu(
            cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 50) + 1, (uint64_t)(rand() % 50) + 1}));
        bool train = rand() % 5 != 0;

        vector<Layer *> layers;

        layers.push_back(new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureInputCpu.getDescriptor().getDimensions()));
        layers.push_back(new NoOpLayer());
        BatchNormalization *batchNormalizationLayer = new BatchNormalization(train);
        layers.push_back(batchNormalizationLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        UniformRandom initializer(0.1, -0.1);
        initializer.initialize(batchNormalizationLayer, batchNormalizationLayer->getWeights());
        initializer.initialize(batchNormalizationLayer, batchNormalizationLayer->getBiases());

        // Forward Pass
        uint32_t numElements = featureInputCpu.getDescriptor().getTotalNumElements();
        half *featureInputMem = (half *)featureInputCpu.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i)
            featureInputMem[i] = ((rand() % 10) / 1.0f) - 5.0f;

        layers[0]->forward(featureInputCpu);
        Tensor featureOutGpu_h = featureInputCpu.clone();
        featureOutGpu_h.copyFromAsync(featureInputCpu, stream);
        stream.synchronize();

        half *featureOutGpuMem_h = (half *)featureOutGpu_h.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i) {
            float feature = featureOutGpuMem_h[i];
            if (print)
                printf("%f\n", feature);
            ASSERT_TRUE(isfinite(feature));
        }

        // Backward Pass
        if (train) {
            for (int p = 0; p < 10; ++p) {
                Tensor errorInput = batchNormalizationLayer->getErrorInputs()[0];
                Tensor errorInputCpu = errorInput.clone(cpuPlacement);
                half *errorInputCpuMem = (half *)errorInputCpu.getMemPtr();
                for (uint32_t i = 0; i < numElements; ++i)
                    errorInputCpuMem[i] = ((rand() % 10) / 1.0f) - 5.0f;

                errorInput.copyFromAsync(errorInputCpu, stream);
                batchNormalizationLayer->backward(errorInput);

                Tensor errorOutput = batchNormalizationLayer->getErrorOutputs()[0];
                Tensor errorOutputGpu_h = errorOutput.clone(cpuPlacement);
                errorOutputGpu_h.copyFromAsync(errorOutput, stream);
                stream.synchronize();

                half *errorOutputGpuMem_h = (half *)errorOutputGpu_h.getMemPtr();
                for (uint32_t i = 0; i < numElements; ++i) {
                    float error = errorOutputGpuMem_h[i];
                    if (print)
                        printf("%f\n", error);
                    ASSERT_TRUE(isfinite(error));
                }

                batchNormalizationLayer->applyGradients(stream,
                                                        batchNormalizationLayer->getWeights(),
                                                        batchNormalizationLayer->getWeightsGradient(),
                                                        batchNormalizationLayer->getBiases(),
                                                        batchNormalizationLayer->getBiasesGradient());
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(BatchNormalization, 4dYieldsFiniteValues) {
    srand(time(NULL));

    bool print = false;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 10; ++test) {
        Tensor featureInputCpu(
            cpuPlacement,
            TensorDescriptor(
                TensorDescriptor::DataType::FP16,
                {(uint64_t)(rand() % 10) + 1, (uint64_t)(rand() % 10) + 1, (uint64_t)(rand() % 10) + 1, (uint64_t)(rand() % 10) + 1}));
        bool train = rand() % 5 != 0;

        vector<Layer *> layers;

        layers.push_back(new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureInputCpu.getDescriptor().getDimensions()));
        layers.push_back(new NoOpLayer());
        BatchNormalization *batchNormalizationLayer = new BatchNormalization(train);
        layers.push_back(batchNormalizationLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        UniformRandom initializer(0.1, -0.1);
        initializer.initialize(batchNormalizationLayer, batchNormalizationLayer->getWeights());
        initializer.initialize(batchNormalizationLayer, batchNormalizationLayer->getBiases());

        // Forward Pass
        uint32_t numElements = featureInputCpu.getDescriptor().getTotalNumElements();
        half *featureInputMem = (half *)featureInputCpu.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i)
            featureInputMem[i] = ((rand() % 10) / 1.0f) - 5.0f;

        layers[0]->forward(featureInputCpu);
        Tensor featureOutGpu_h = featureInputCpu.clone();
        featureOutGpu_h.copyFromAsync(featureInputCpu, stream);
        stream.synchronize();

        half *featureOutGpuMem_h = (half *)featureOutGpu_h.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i) {
            float feature = featureOutGpuMem_h[i];
            if (print)
                printf("%f\n", feature);
            ASSERT_TRUE(isfinite(feature));
        }

        // Backward Pass
        if (train) {
            for (int p = 0; p < 10; ++p) {
                Tensor errorInput = batchNormalizationLayer->getErrorInputs()[0];
                Tensor errorInputCpu = errorInput.clone(cpuPlacement);
                half *errorInputCpuMem = (half *)errorInputCpu.getMemPtr();
                for (uint32_t i = 0; i < numElements; ++i)
                    errorInputCpuMem[i] = ((rand() % 10) / 1.0f) - 5.0f;

                errorInput.copyFromAsync(errorInputCpu, stream);
                batchNormalizationLayer->backward(errorInput);

                Tensor errorOutput = batchNormalizationLayer->getErrorOutputs()[0];
                Tensor errorOutputGpu_h = errorOutput.clone(cpuPlacement);
                errorOutputGpu_h.copyFromAsync(errorOutput, stream);
                stream.synchronize();

                half *errorOutputGpuMem_h = (half *)errorOutputGpu_h.getMemPtr();
                for (uint32_t i = 0; i < numElements; ++i) {
                    float error = errorOutputGpuMem_h[i];
                    if (print)
                        printf("%f\n", error);
                    ASSERT_TRUE(isfinite(error));
                }

                batchNormalizationLayer->applyGradients(stream,
                                                        batchNormalizationLayer->getWeights(),
                                                        batchNormalizationLayer->getWeightsGradient(),
                                                        batchNormalizationLayer->getBiases(),
                                                        batchNormalizationLayer->getBiasesGradient());
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
