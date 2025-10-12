#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "Thor.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
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

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
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
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput();

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

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
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
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput();

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
        Tensor errorInput = dropOutLayer->getErrorInput();
        Tensor errorOutput = dropOutLayer->getErrorOutput();
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

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
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
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput();

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
        Tensor errorInput = dropOutLayer->getErrorInput();
        Tensor errorOutput = dropOutLayer->getErrorOutput();
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

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
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
        Tensor outputGpu = dynamic_pointer_cast<NetworkOutput>(layers.back())->getFeatureOutput();

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
        Tensor errorInput = dropOutLayer->getErrorInput();
        Tensor errorOutput = dropOutLayer->getErrorOutput();
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
