#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <vector>

using namespace std;

using namespace ThorImplementation;

TEST(DropOut, InferenceWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 5) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 5) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(DataType::FP16, dimensions);
        Tensor sourceCpu(cpuPlacement, descriptor);
        Tensor sourceGpu(gpuPlacement, descriptor);
        Tensor destCpu(cpuPlacement, descriptor);

        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<DropOut> dropOutLayer = make_shared<DropOut>(0.25, true);
        ASSERT_TRUE(dropOutLayer->isTrainingMode());
        dropOutLayer->setTrainingMode(false);
        ASSERT_FALSE(dropOutLayer->isTrainingMode());
        layers.push_back(dropOutLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)destMem[i], (float)sourceMem[i]);
        }
    }
}

TEST(DropOut, RaggedInferenceIdentityAliasesValuesAndDoesNotRequireRuntimeExtent) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint64_t fullRows = 8;
    constexpr uint64_t elementsPerValue = 3;
    TensorDescriptor descriptor(DataType::FP32, {fullRows, elementsPerValue});

    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    for (uint64_t i = 0; i < sourceCpu.getTotalNumElements(); ++i) {
        sourceCpu.getMemPtr<float>()[i] = static_cast<float>(i + 1);
    }

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    auto dropOutLayer = make_shared<DropOut>(
        0.5f, /*training=*/false, /*trainingDropoutEnabled=*/true,
        DropOut::RaggedConfiguration{fullRows, elementsPerValue});
    layers.push_back(dropOutLayer);
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    Stream stream = layers.front()->getStream();
    LayerTestHelper::connectNetwork(layers);

    // The structural offsets port is part of the physical graph, but inference
    // identity must neither wait for it at runtime nor require its host cache.
    Tensor rowPartitionGpu(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    dropOutLayer->connectToPreviousLayer(
        nullptr, rowPartitionGpu, stream, /*backPropagateError=*/false, /*connectionType=*/1);
    LayerTestHelper::initializeNetwork(layers);

    ASSERT_TRUE(dropOutLayer->getFeatureInput().has_value());
    ASSERT_TRUE(dropOutLayer->getFeatureOutput().has_value());
    EXPECT_EQ(dropOutLayer->getFeatureInput().value(), dropOutLayer->getFeatureOutput().value());

    // Drive only the values edge. No offsets contents or RowPartitionRuntime cache
    // are supplied because a true inference identity does not inspect packed rows.
    layers.front()->forward(sourceCpu, false);
    auto networkOutput = dynamic_pointer_cast<NetworkOutput>(layers.back());
    stream.waitEvent(networkOutput->getOutputReadyEvent());
    Tensor resultCpu(cpuPlacement, descriptor);
    resultCpu.copyFromAsync(networkOutput->getFeatureOutput().value(), stream);
    stream.synchronize();

    for (uint64_t i = 0; i < sourceCpu.getTotalNumElements(); ++i) {
        EXPECT_FLOAT_EQ(resultCpu.getMemPtr<float>()[i], sourceCpu.getMemPtr<float>()[i]);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(DropOut, TrainingNoDropOut) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 6) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 5) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(DataType::FP16, dimensions);
        Tensor sourceCpu(cpuPlacement, descriptor);
        Tensor sourceGpu(gpuPlacement, descriptor);
        Tensor destCpu(cpuPlacement, descriptor);

        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<DropOut> dropOutLayer = make_shared<DropOut>(0.0f, false);
        ASSERT_FALSE(dropOutLayer->isTrainingMode());
        dropOutLayer->setTrainingMode(true);
        ASSERT_TRUE(dropOutLayer->isTrainingMode());
        layers.push_back(dropOutLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);
        ASSERT_EQ(dropOutLayer->getFeatureOutput().value(), dropOutLayer->getFeatureInput().value());
        ASSERT_EQ(dropOutLayer->getErrorOutput().value(), dropOutLayer->getErrorInput().value());
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            EXPECT_EQ((float)destMem[i], (float)sourceMem[i]);
        }

        // Backward pass
        Tensor errorInput = dropOutLayer->getErrorInput().value();
        Tensor errorOutput = dropOutLayer->getErrorOutput().value();
        Tensor errorInputCpu = Tensor(cpuPlacement, errorInput.getDescriptor());
        Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutput.getDescriptor());
        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();

        for (int i = 0; i < numElements; ++i) {
            errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        errorInput.copyFromAsync(errorInputCpu, stream);
        dropOutLayer->backward(errorInput);
        errorOutputCpu.copyFromAsync(errorOutput, stream);
        stream.synchronize();

        for (int i = 0; i < numElements; ++i) {
            EXPECT_EQ((float)errorOutputMem[i], (float)errorInputMem[i]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(DropOut, ValidationBypassesDropOut) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP16, {2, 3});
    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    Tensor destCpu(cpuPlacement, descriptor);

    half *sourceMem = static_cast<half *>(sourceCpu.getMemPtr());
    for (int i = 0; i < 6; ++i)
        sourceMem[i] = static_cast<float>(i + 1);

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    auto dropOutLayer = make_shared<DropOut>(0.5f, true);
    layers.push_back(dropOutLayer);
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    Stream stream = layers.front()->getStream();
    LayerTestHelper::connectAndInitializeNetwork(layers);
    layers[0]->forward(sourceCpu, true);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    destCpu.copyFromAsync(dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value(), stream);
    stream.synchronize();

    half *destMem = static_cast<half *>(destCpu.getMemPtr());
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(static_cast<float>(destMem[i]), static_cast<float>(sourceMem[i]));

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(DropOut, TrainingAllDropOut) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 6) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 5) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(DataType::FP16, dimensions);
        Tensor sourceCpu(cpuPlacement, descriptor);
        Tensor sourceGpu(gpuPlacement, descriptor);
        Tensor destCpu(cpuPlacement, descriptor);

        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<DropOut> dropOutLayer = make_shared<DropOut>(1.0f, false);
        dropOutLayer->setTrainingMode(true);
        ASSERT_TRUE(dropOutLayer->isTrainingMode());
        layers.push_back(dropOutLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            EXPECT_EQ((float)destMem[i], 0.0f);
        }

        // Backward pass
        Tensor errorInput = dropOutLayer->getErrorInput().value();
        Tensor errorOutput = dropOutLayer->getErrorOutput().value();
        Tensor errorInputCpu = Tensor(cpuPlacement, errorInput.getDescriptor());
        Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutput.getDescriptor());
        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();

        for (int i = 0; i < numElements; ++i) {
            errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        errorInput.copyFromAsync(errorInputCpu, stream);
        dropOutLayer->backward(errorInput);
        errorOutputCpu.copyFromAsync(errorOutput, stream);
        stream.synchronize();

        for (int i = 0; i < numElements; ++i) {
            EXPECT_EQ((float)errorOutputMem[i], 0.0f);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(DropOut, TrainingSomeDropOut) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 6) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 5) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(DataType::FP16, dimensions);
        Tensor sourceCpu(cpuPlacement, descriptor);
        Tensor sourceGpu(gpuPlacement, descriptor);
        Tensor destCpu(cpuPlacement, descriptor);

        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
            if (sourceMem[i] < (half)0.1 && sourceMem[i] > (half)-0.1)
                sourceMem[i] = 1.2f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));

        Stream stream = layers.front()->getStream();

        float dropOutRate = ((rand() % 60) / 100.0f) + 0.2;
        float scalingFactor = 0.0f;
        if (dropOutRate < 1.0f)
            scalingFactor = 1 / (1.0f - dropOutRate);

        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<DropOut> dropOutLayer = make_shared<DropOut>(dropOutRate, true);
        layers.push_back(dropOutLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        int numDropped = 0;
        for (int i = 0; i < numElements; ++i) {
            if ((float)destMem[i] == 0.0f)
                numDropped += 1;
        }

        for (int i = 0; i < numElements; ++i) {
            if (destMem[i] != (half)0.0f) {
                EXPECT_LT(abs((float)destMem[i] - (float)sourceMem[i] * scalingFactor), 0.2);
            }
        }

        // Backward pass
        Tensor errorInput = dropOutLayer->getErrorInput().value();
        Tensor errorOutput = dropOutLayer->getErrorOutput().value();
        Tensor errorInputCpu = Tensor(cpuPlacement, errorInput.getDescriptor());
        Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutput.getDescriptor());
        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();

        for (int i = 0; i < numElements; ++i) {
            errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        errorInput.copyFromAsync(errorInputCpu, stream);
        dropOutLayer->backward(errorInput);
        errorOutputCpu.copyFromAsync(errorOutput, stream);
        stream.synchronize();

        for (int i = 0; i < numElements; ++i) {
            if (destMem[i] == (half)0.0f) {
                EXPECT_EQ((float)errorOutputMem[i], 0.0f);
            } else {
                ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorInputMem[i] * scalingFactor), 0.2);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}


TEST(DropOut, Bfloat16TrainingForwardBackward) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::BF16, {4096});
    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    Tensor destCpu(cpuPlacement, descriptor);

    auto *sourceMem = static_cast<__nv_bfloat16 *>(sourceCpu.getMemPtr());
    for (uint64_t i = 0; i < sourceCpu.getTotalNumElements(); ++i)
        sourceMem[i] = __float2bfloat16_rn(1.0f);

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    // Preserve the test-only backward tensors. NetworkOutput intentionally
    // terminates and prunes the backward path when connected directly.
    layers.push_back(make_shared<NoOpLayer>());
    auto dropOutLayer = make_shared<DropOut>(0.5f, true);
    dropOutLayer->seed(12345);
    layers.push_back(dropOutLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    Stream stream = layers.front()->getStream();
    LayerTestHelper::connectAndInitializeNetwork(layers);

    layers.front()->forward(sourceCpu, false);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    destCpu.copyFromAsync(dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value(), stream);
    stream.synchronize();

    auto *destMem = static_cast<__nv_bfloat16 *>(destCpu.getMemPtr());
    uint64_t dropped = 0;
    uint64_t kept = 0;
    for (uint64_t i = 0; i < destCpu.getTotalNumElements(); ++i) {
        const float value = __bfloat162float(destMem[i]);
        if (value == 0.0f) {
            ++dropped;
        } else {
            ++kept;
            EXPECT_FLOAT_EQ(value, 2.0f);
        }
    }
    EXPECT_GT(dropped, 0);
    EXPECT_GT(kept, 0);

    Tensor errorInput = dropOutLayer->getErrorInput().value();
    Tensor errorOutput = dropOutLayer->getErrorOutput().value();
    Tensor errorInputCpu(cpuPlacement, errorInput.getDescriptor());
    Tensor errorOutputCpu(cpuPlacement, errorOutput.getDescriptor());
    auto *errorInputMem = static_cast<__nv_bfloat16 *>(errorInputCpu.getMemPtr());
    for (uint64_t i = 0; i < errorInputCpu.getTotalNumElements(); ++i)
        errorInputMem[i] = __float2bfloat16_rn(1.0f);

    errorInput.copyFromAsync(errorInputCpu, stream);
    dropOutLayer->backward(errorInput);
    errorOutputCpu.copyFromAsync(errorOutput, stream);
    stream.synchronize();

    auto *errorOutputMem = static_cast<__nv_bfloat16 *>(errorOutputCpu.getMemPtr());
    for (uint64_t i = 0; i < errorOutputCpu.getTotalNumElements(); ++i) {
        const float expected = __bfloat162float(destMem[i]) == 0.0f ? 0.0f : 2.0f;
        EXPECT_FLOAT_EQ(__bfloat162float(errorOutputMem[i]), expected);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(DropOut, TrainingDropoutControlUsesIdentityForwardAndBackwardWithoutChangingConfiguredRate) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP16, {2, 3});
    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    Tensor destCpu(cpuPlacement, descriptor);

    half* sourceMem = static_cast<half*>(sourceCpu.getMemPtr());
    for (int i = 0; i < 6; ++i) {
        sourceMem[i] = static_cast<float>(i + 1);
    }

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    layers.push_back(make_shared<NoOpLayer>());
    auto dropOutLayer = make_shared<DropOut>(0.5f, true);
    layers.push_back(dropOutLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    Stream stream = layers.front()->getStream();
    LayerTestHelper::connectAndInitializeNetwork(layers);

    ASSERT_FLOAT_EQ(dropOutLayer->getDropOutRate(), 0.5f);
    ASSERT_TRUE(dropOutLayer->isTrainingDropoutEnabled());
    ASSERT_NE(dropOutLayer->getFeatureOutput().value(), dropOutLayer->getFeatureInput().value());
    ASSERT_NE(dropOutLayer->getErrorOutput().value(), dropOutLayer->getErrorInput().value());

    dropOutLayer->setTrainingDropoutEnabled(false);
    ASSERT_FALSE(dropOutLayer->isTrainingDropoutEnabled());

    layers.front()->forward(sourceCpu, false);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    destCpu.copyFromAsync(dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value(), stream);
    stream.synchronize();

    half* destMem = static_cast<half*>(destCpu.getMemPtr());
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(static_cast<float>(destMem[i]), static_cast<float>(sourceMem[i]));
    }

    Tensor errorInput = dropOutLayer->getErrorInput().value();
    Tensor errorOutput = dropOutLayer->getErrorOutput().value();
    Tensor errorInputCpu(cpuPlacement, errorInput.getDescriptor());
    Tensor errorOutputCpu(cpuPlacement, errorOutput.getDescriptor());
    half* errorInputMem = static_cast<half*>(errorInputCpu.getMemPtr());
    for (int i = 0; i < 6; ++i) {
        errorInputMem[i] = static_cast<float>(10 + i);
    }

    errorInput.copyFromAsync(errorInputCpu, stream);
    dropOutLayer->backward(errorInput);
    errorOutputCpu.copyFromAsync(errorOutput, stream);
    stream.synchronize();

    half* errorOutputMem = static_cast<half*>(errorOutputCpu.getMemPtr());
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(static_cast<float>(errorOutputMem[i]), static_cast<float>(errorInputMem[i]));
    }

    dropOutLayer->setTrainingDropoutEnabled(true);
    ASSERT_TRUE(dropOutLayer->isTrainingDropoutEnabled());
    ASSERT_FLOAT_EQ(dropOutLayer->getDropOutRate(), 0.5f);

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(DropOut, RaggedTrainingUsesOnlyActivePrefixAndBackwardReusesForwardMask) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint64_t fullRows = 64;
    constexpr uint64_t activeRows = 37;
    constexpr uint64_t elementsPerValue = 4;
    constexpr uint64_t totalElements = fullRows * elementsPerValue;
    constexpr uint64_t activeElements = activeRows * elementsPerValue;
    TensorDescriptor descriptor(DataType::FP32, {fullRows, elementsPerValue});

    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    float* source = sourceCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < activeElements; ++i) source[i] = 1.0f;
    for (uint64_t i = activeElements; i < totalElements; ++i) source[i] = 1234.0f;

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    // External NetworkInput intentionally terminates backpropagation. Keep a
    // test-only upstream gradient edge alive so this test can exercise DropOut
    // backward, matching the established dense DropOut backward tests above.
    layers.push_back(make_shared<NoOpLayer>());
    auto dropOutLayer = make_shared<DropOut>(
        0.5f, true, true, DropOut::RaggedConfiguration{fullRows, elementsPerValue});
    dropOutLayer->seed(0x12345678ULL);
    layers.push_back(dropOutLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    Stream stream = layers.front()->getStream();
    LayerTestHelper::connectNetwork(layers);
    Tensor rowPartitionGpu(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor rowPartitionCpu(cpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    rowPartitionCpu.getMemPtr<uint32_t>()[0] = 0;
    rowPartitionCpu.getMemPtr<uint32_t>()[1] = static_cast<uint32_t>(activeRows / 2);
    rowPartitionCpu.getMemPtr<uint32_t>()[2] = static_cast<uint32_t>(activeRows);
    rowPartitionGpu.copyFromAsync(rowPartitionCpu, stream);
    RowPartitionRuntime rowPartition(
        rowPartitionGpu, RowPartitionDescriptor(/*batchSize=*/2, fullRows, DataType::UINT32));
    rowPartition.setHostActiveValueCount(activeRows);
    dropOutLayer->connectToPreviousLayer(nullptr, rowPartitionGpu, stream, /*backPropagateError=*/false, /*connectionType=*/1);
    LayerTestHelper::initializeNetwork(layers);

    // Drive DropOut directly with poisoned inactive capacity; the layer must
    // use the row partition and avoid reading rows outside the active prefix.
    Tensor packedInput = dropOutLayer->getFeatureInput().value();
    packedInput.copyFromAsync(sourceCpu, stream);
    dropOutLayer->forward(packedInput, false);
    dropOutLayer->forward(rowPartitionGpu, false);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    Tensor outputCpu(cpuPlacement, descriptor);
    outputCpu.copyFromAsync(dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value(), stream);
    stream.synchronize();

    const float* output = outputCpu.getMemPtr<float>();
    uint64_t dropped = 0;
    uint64_t kept = 0;
    for (uint64_t i = 0; i < activeElements; ++i) {
        if (output[i] == 0.0f) {
            ++dropped;
        } else {
            ++kept;
            EXPECT_FLOAT_EQ(output[i], 2.0f);
        }
    }
    EXPECT_GT(dropped, 0U);
    EXPECT_GT(kept, 0U);
    for (uint64_t i = activeElements; i < totalElements; ++i) EXPECT_FLOAT_EQ(output[i], 0.0f);

    Tensor errorInput = dropOutLayer->getErrorInput().value();
    Tensor errorOutput = dropOutLayer->getErrorOutput().value();
    Tensor errorInputCpu(cpuPlacement, descriptor);
    float* errorInputValues = errorInputCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < activeElements; ++i) errorInputValues[i] = 1.0f;
    for (uint64_t i = activeElements; i < totalElements; ++i) errorInputValues[i] = -9999.0f;
    errorInput.copyFromAsync(errorInputCpu, stream);
    dropOutLayer->backward(errorInput);

    Tensor errorOutputCpu(cpuPlacement, descriptor);
    errorOutputCpu.copyFromAsync(errorOutput, stream);
    stream.synchronize();
    const float* errorOutputValues = errorOutputCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < activeElements; ++i) {
        EXPECT_FLOAT_EQ(errorOutputValues[i], output[i] == 0.0f ? 0.0f : 2.0f);
    }
    for (uint64_t i = activeElements; i < totalElements; ++i) EXPECT_FLOAT_EQ(errorOutputValues[i], 0.0f);

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(DropOut, RaggedValidationIsIdentityOverActivePrefixAndCanonicalizesInactiveCapacity) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint64_t fullRows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t elementsPerValue = 3;
    constexpr uint64_t totalElements = fullRows * elementsPerValue;
    constexpr uint64_t activeElements = activeRows * elementsPerValue;
    TensorDescriptor descriptor(DataType::FP32, {fullRows, elementsPerValue});

    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    float* source = sourceCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < activeElements; ++i) source[i] = static_cast<float>(i + 1);
    for (uint64_t i = activeElements; i < totalElements; ++i) source[i] = 7777.0f;

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    auto dropOutLayer = make_shared<DropOut>(
        0.5f, true, true, DropOut::RaggedConfiguration{fullRows, elementsPerValue});
    layers.push_back(dropOutLayer);
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));
    Stream stream = layers.front()->getStream();
    LayerTestHelper::connectNetwork(layers);
    Tensor rowPartitionGpu(gpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    Tensor rowPartitionCpu(cpuPlacement, TensorDescriptor(DataType::UINT32, {3}));
    rowPartitionCpu.getMemPtr<uint32_t>()[0] = 0;
    rowPartitionCpu.getMemPtr<uint32_t>()[1] = static_cast<uint32_t>(activeRows / 2);
    rowPartitionCpu.getMemPtr<uint32_t>()[2] = static_cast<uint32_t>(activeRows);
    rowPartitionGpu.copyFromAsync(rowPartitionCpu, stream);
    RowPartitionRuntime rowPartition(
        rowPartitionGpu, RowPartitionDescriptor(/*batchSize=*/2, fullRows, DataType::UINT32));
    rowPartition.setHostActiveValueCount(activeRows);
    dropOutLayer->connectToPreviousLayer(nullptr, rowPartitionGpu, stream, /*backPropagateError=*/false, /*connectionType=*/1);
    LayerTestHelper::initializeNetwork(layers);

    Tensor packedInput = dropOutLayer->getFeatureInput().value();
    packedInput.copyFromAsync(sourceCpu, stream);
    dropOutLayer->forward(rowPartitionGpu, true);
    dropOutLayer->forward(packedInput, true);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    Tensor outputCpu(cpuPlacement, descriptor);
    outputCpu.copyFromAsync(dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput().value(), stream);
    stream.synchronize();

    const float* output = outputCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < activeElements; ++i) EXPECT_FLOAT_EQ(output[i], source[i]);
    for (uint64_t i = activeElements; i < totalElements; ++i) EXPECT_FLOAT_EQ(output[i], 0.0f);

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(DropOut, NativeRaggedKernelSeedReproducesTheSameActiveMask) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint64_t fullElements = 256;
    constexpr uint64_t activeElements = 157;
    TensorDescriptor descriptor(DataType::FP32, {fullElements});
    TensorDescriptor maskDescriptor(DataType::UINT8, {fullElements});

    Tensor inputCpu(cpuPlacement, descriptor);
    for (uint64_t i = 0; i < fullElements; ++i) inputCpu.getMemPtr<float>()[i] = 1.0f;
    Tensor inputGpu(gpuPlacement, descriptor);
    Tensor outputA(gpuPlacement, descriptor);
    Tensor outputB(gpuPlacement, descriptor);
    Tensor maskA(gpuPlacement, maskDescriptor);
    Tensor maskB(gpuPlacement, maskDescriptor);
    Stream stream(0);
    inputGpu.copyFromAsync(inputCpu, stream);

    constexpr uint64_t seed = 0xabcdef0123456789ULL;
    constexpr uint64_t sequence = 7;
    launchDropOutForward(inputGpu.getMemPtr(),
                         outputA.getMemPtr(),
                         maskA.getMemPtr<uint8_t>(),
                         DataType::FP32,
                         activeElements,
                         0.5f,
                         seed,
                         sequence,
                         stream);
    launchDropOutForward(inputGpu.getMemPtr(),
                         outputB.getMemPtr(),
                         maskB.getMemPtr<uint8_t>(),
                         DataType::FP32,
                         activeElements,
                         0.5f,
                         seed,
                         sequence,
                         stream);

    Tensor outputACpu(cpuPlacement, descriptor);
    Tensor outputBCpu(cpuPlacement, descriptor);
    Tensor maskACpu(cpuPlacement, maskDescriptor);
    Tensor maskBCpu(cpuPlacement, maskDescriptor);
    outputACpu.copyFromAsync(outputA, stream);
    outputBCpu.copyFromAsync(outputB, stream);
    maskACpu.copyFromAsync(maskA, stream);
    maskBCpu.copyFromAsync(maskB, stream);
    stream.synchronize();

    for (uint64_t i = 0; i < activeElements; ++i) {
        EXPECT_FLOAT_EQ(outputACpu.getMemPtr<float>()[i], outputBCpu.getMemPtr<float>()[i]);
        EXPECT_EQ(maskACpu.getMemPtr<uint8_t>()[i], maskBCpu.getMemPtr<uint8_t>()[i]);
    }
}
