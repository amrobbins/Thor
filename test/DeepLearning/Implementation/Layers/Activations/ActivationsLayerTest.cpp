#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <vector>

using std::set;
using std::vector;

using namespace ThorImplementation;

TEST(Relu, Works) {
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
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu(gpuPlacement, descriptor);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Relu *relu = new Relu();
        layers.push_back(relu);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        Stream stream = layers.front()->getStream();
        featureInGpu.copyFromAsync(featureInCpu, stream);

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInGpu, false);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            half rectified = featureInMem[i];
            if (rectified < 0.0f)
                rectified = 0.0f;
            ASSERT_EQ((float)destMem[i], (float)rectified);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = relu->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        relu->backProp(featureInGpu, errorInGpu, errorOutGpu, stream);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half zero = 0.0f;
        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            half expectedValue = featureInMem[i] > zero ? errorInMem[i] : zero;
            half actualValue = errorOutMem[i];
            ASSERT_EQ((float)expectedValue, (float)actualValue);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Tanh, Works) {
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
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu(gpuPlacement, descriptor);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Tanh *tanhLayer = new Tanh();
        layers.push_back(tanhLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        Stream stream = layers.front()->getStream();
        featureInGpu.copyFromAsync(featureInCpu, stream);

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInGpu, false);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)destMem[i], (float)(half)tanh((float)featureInMem[i]));
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = tanhLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        tanhLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        float thresh = 0.0001;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = errorInMem[i] * (1.0f - tanh(featureInMem[i]) * tanh(featureInMem[i]));
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Linear, Works) {
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
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu(gpuPlacement, descriptor);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Linear *linear = new Linear();
        layers.push_back(linear);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        Stream stream = layers.front()->getStream();
        featureInGpu.copyFromAsync(featureInCpu, stream);

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInGpu, false);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)destMem[i], featureInMem[i]);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = linear->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        linear->backProp(featureInGpu, errorInGpu, errorOutGpu, stream);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)errorInMem[i], (float)errorOutMem[i]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
