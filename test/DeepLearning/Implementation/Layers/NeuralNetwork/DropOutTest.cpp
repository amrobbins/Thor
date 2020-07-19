#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "MLDev.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <vector>

using std::set;
using std::vector;

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

        Stream stream(0);
        sourceGpu.copyFromAsync(sourceCpu, stream);

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(sourceGpu, stream));
        layers.push_back(new NoOpLayer());
        DropOut *dropOutLayer = new DropOut(0.25, false);
        layers.push_back(dropOutLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = ((NetworkOutput *)layers.back())->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(sourceGpu);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)destMem[i], (float)sourceMem[i]);
        }

        // Backward pass
        Tensor errorInput = dropOutLayer->getErrorInput();
        Tensor errorOutput = dropOutLayer->getErrorOutput();
        Tensor errorInputCpu = Tensor(cpuPlacement, errorInput.getDescriptor());
        Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutput.getDescriptor());

        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        errorInput.copyFromAsync(errorInputCpu, stream);
        dropOutLayer->backProp(sourceGpu, errorInput, errorOutput, stream);
        errorOutputCpu.copyFromAsync(errorOutput, stream);
        stream.synchronize();

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)errorOutputMem[i], (float)errorInputMem[i]);
        }

        LayerTestHelper::tearDownNetwork(layers);
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

        Stream stream(0);
        sourceGpu.copyFromAsync(sourceCpu, stream);

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(sourceGpu, stream));
        layers.push_back(new NoOpLayer());
        DropOut *dropOutLayer = new DropOut(0.0f, true);
        layers.push_back(dropOutLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = ((NetworkOutput *)layers.back())->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(sourceGpu);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
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
        dropOutLayer->backProp(sourceGpu, errorInput, errorOutput, stream);
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

        Stream stream(0);
        sourceGpu.copyFromAsync(sourceCpu, stream);

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(sourceGpu, stream));
        layers.push_back(new NoOpLayer());
        DropOut *dropOutLayer = new DropOut(1.0f, true);
        layers.push_back(dropOutLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = ((NetworkOutput *)layers.back())->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(sourceGpu);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
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
        dropOutLayer->backProp(sourceGpu, errorInput, errorOutput, stream);
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
            if (sourceMem[i] < 0.1 && sourceMem[i] > -0.1)
                sourceMem[i] = 1.2f;
        }

        Stream stream(0);
        sourceGpu.copyFromAsync(sourceCpu, stream);

        float dropOutRate = ((rand() % 60) / 100.0f) + 0.2;
        float scalingFactor = 0.0f;
        if (dropOutRate < 1.0f)
            scalingFactor = 1 / (1.0f - dropOutRate);

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(sourceGpu, stream));
        layers.push_back(new NoOpLayer());
        DropOut *dropOutLayer = new DropOut(dropOutRate, true);
        layers.push_back(dropOutLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = ((NetworkOutput *)layers.back())->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(sourceGpu);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
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
            if (destMem[i] != 0.0f) {
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
        dropOutLayer->backProp(sourceGpu, errorInput, errorOutput, stream);
        errorOutputCpu.copyFromAsync(errorOutput, stream);
        stream.synchronize();

        for (int i = 0; i < numElements; ++i) {
            if (destMem[i] == 0.0f) {
                EXPECT_EQ((float)errorOutputMem[i], 0.0f);
            } else {
                ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorInputMem[i] * scalingFactor), 0.2);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
