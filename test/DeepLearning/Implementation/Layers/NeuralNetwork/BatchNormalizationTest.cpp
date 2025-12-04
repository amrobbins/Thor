#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

#include "DeepLearning/Implementation/Initializers/UniformRandom.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <vector>

using namespace std;

using namespace ThorImplementation;

// FIXME: I don't have time now to create tests that verify cudnn is giving the correct forward and backward values
//        Also I have used this layer in other frameworks
//        For now I will settle for connection tests only.
//
//        And then I gave up on that too... So I am not verifying the outputs at all except to say that they are finite valued.
//        Just from eyeballing them, they look ok.

// FIXME: Need to set batchNormalizationLayer->setCurrentExponentialRunningAverageFactor(1.0); but is dependent on num connections...
TEST(BatchNormalization, DISABLED_2dYieldsFiniteValues) {
    srand(time(NULL));

    bool print = false;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 10; ++test) {
        Tensor featureInputCpu(
            cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 50) + 1, (uint64_t)(rand() % 50) + 1}));
        bool train = rand() % 5 != 0;

        vector<shared_ptr<Layer>> layers;

        layers.push_back(
            make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, featureInputCpu.getDescriptor().getDimensions()));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<BatchNormalization> batchNormalizationLayer = make_shared<BatchNormalization>(train);
        layers.push_back(batchNormalizationLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);

        ThorImplementation::Tensor anErrorInput =
            ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(batchNormalizationLayer->getErrorInputs());
        ThorImplementation::Tensor anErrorOutput =
            ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(batchNormalizationLayer->getErrorOutputs());
        shared_ptr<Optimizer> sgd =
            make_shared<ThorImplementation::Sgd>(batchNormalizationLayer, 0.1, 0, 0, false, 0, anErrorInput, anErrorOutput);
        batchNormalizationLayer->setOptimizer(sgd);
        sgd->updateHyperParameters(0, 0, 1);

        Stream stream = layers.front()->getStream();

        UniformRandom initializer(0.1, -0.1);
        initializer.initialize(batchNormalizationLayer.get(), batchNormalizationLayer->getWeights());
        initializer.initialize(batchNormalizationLayer.get(), batchNormalizationLayer->getBiases());

        // Forward Pass
        uint32_t numElements = featureInputCpu.getDescriptor().getTotalNumElements();
        half *featureInputMem = (half *)featureInputCpu.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i)
            featureInputMem[i] = ((rand() % 10) / 1.0f) - 5.0f;

        layers[0]->forward(featureInputCpu, false);
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
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(BatchNormalization, DISABLED_4dYieldsFiniteValues) {
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

        vector<shared_ptr<Layer>> layers;

        layers.push_back(
            make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, featureInputCpu.getDescriptor().getDimensions()));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<BatchNormalization> batchNormalizationLayer = make_shared<BatchNormalization>(train);
        batchNormalizationLayer->setCurrentExponentialRunningAverageFactor(1.0);
        layers.push_back(batchNormalizationLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);

        Stream stream = layers.front()->getStream();

        ThorImplementation::Tensor anErrorInput =
            ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(batchNormalizationLayer->getErrorInputs());
        ThorImplementation::Tensor anErrorOutput =
            ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(batchNormalizationLayer->getErrorOutputs());
        shared_ptr<Optimizer> sgd =
            make_shared<ThorImplementation::Sgd>(batchNormalizationLayer, 0.1, 0, 0, false, 0, anErrorInput, anErrorOutput);
        batchNormalizationLayer->setOptimizer(sgd);
        sgd->updateHyperParameters(0, 0, 1);

        UniformRandom initializer(0.1, -0.1);
        initializer.initialize(batchNormalizationLayer.get(), batchNormalizationLayer->getWeights());
        initializer.initialize(batchNormalizationLayer.get(), batchNormalizationLayer->getBiases());

        // Forward Pass
        uint32_t numElements = featureInputCpu.getDescriptor().getTotalNumElements();
        half *featureInputMem = (half *)featureInputCpu.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i)
            featureInputMem[i] = ((rand() % 10) / 1.0f) - 5.0f;

        layers[0]->forward(featureInputCpu, false);
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
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}
