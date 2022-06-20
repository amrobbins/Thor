#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <set>
#include <vector>

using std::set;
using std::vector;

using namespace ThorImplementation;

TEST(Relu, Works) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 100; ++test) {
        vector<unsigned long> dimensions;
        int numDimensions = (rand() % 5) + 1;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 8) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu = featureInCpu.clone(gpuPlacement);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Relu *relu = new Relu();
        layers.push_back(relu);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInCpu, false);
        Stream stream = layers.front()->getStream();
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        stream.synchronize();

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            half rectified = featureInMem[i];
            if (rectified < 0.0f)
                rectified = 0.0f;
            if ((float)destMem[i] != (float)rectified) {
                printf("%d of %d\n", i, numElements);
                fflush(stdout);
            }
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

        featureInGpu.copyFromAsync(featureInCpu, stream);
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

    for (int test = 0; test < 100; ++test) {
        int numDimensions = (rand() % 5) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 8) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu = featureInCpu.clone(gpuPlacement);
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

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInCpu, false);
        Stream stream = layers.front()->getStream();
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        stream.synchronize();

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

/**
 * x when x >= 0
 * alpha * (exp(x) - 1) when x < 0
 * where alpha is a scalar parameter that defaults to 1.0 and must be >= 0.0
 */
float elu(float featureIn, float alpha) {
    if (featureIn >= 0.0f)
        return featureIn;
    else
        return alpha * (exp(featureIn) - 1);
}

/**
 * d/dx(x)elu(x) = 1 when x >= 0
 * d/dx(alpha * (exp(x) - 1)) = alpha * exp(x) when x < 0
 * where alpha is a scalar parameter that defaults to 1.0 and must be >= 0.0
 */
float eluBackward(float featureIn, float errorIn, float alpha) {
    if (featureIn >= 0.0f)
        return errorIn;
    else
        return errorIn * (alpha * exp(featureIn));
}

TEST(Elu, Works) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 100; ++test) {
        int numDimensions = (rand() % 5) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 8) + 1);
            numElements *= dimensions.back();
        }
        float alpha = rand() % 10000;
        alpha /= 2000.0f;

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu = featureInCpu.clone(gpuPlacement);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Elu *eluLayer = new Elu(alpha);
        layers.push_back(eluLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInCpu, false);
        Stream stream = layers.front()->getStream();
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        stream.synchronize();

        half *destMem = (half *)destCpu.getMemPtr();
        float thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = (float)(half)elu((float)featureInMem[i], alpha);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = eluLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        eluLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = eluBackward(featureInMem[i], errorInMem[i], alpha);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/**
 * softSign(x) === x * sigmoid(x) === x / (1 + exp(-x))
 */
float swish(float featureIn) { return featureIn / (1 + exp(-featureIn)); }

/**
 * d/dx(x/(1 + exp(-x))) = (e^x * (x + e^x + 1))/(e^x + 1)^2
 */
float swishBackward(float featureIn, float errorIn) {
    float expX = exp(featureIn);
    return errorIn * (expX * (featureIn + expX + 1)) / ((expX + 1) * (expX + 1));
}

TEST(Swish, Works) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 100; ++test) {
        int numDimensions = (rand() % 5) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 8) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu = featureInCpu.clone(gpuPlacement);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Swish *swishLayer = new Swish();
        layers.push_back(swishLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInCpu, false);
        Stream stream = layers.front()->getStream();
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        stream.synchronize();

        half *destMem = (half *)destCpu.getMemPtr();
        float thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = (float)(half)swish((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = swishLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        swishLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = swishBackward(featureInMem[i], errorInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Exponential, Works) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 100; ++test) {
        int numDimensions = (rand() % 5) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 8) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu = featureInCpu.clone(gpuPlacement);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            featureInMem[i] = ((rand() % 40) / 10.0f) - 2.0f;
        }

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Exponential *exponentialLayer = new Exponential();
        layers.push_back(exponentialLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInCpu, false);
        Stream stream = layers.front()->getStream();
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        stream.synchronize();

        half *destMem = (half *)destCpu.getMemPtr();
        float thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = (float)(half)exp((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = exponentialLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 40) / 10.0f) - 2.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        exponentialLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = (float)errorInMem[i] * exp((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            EXPECT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/**
 * softSign(x) === x / (abs(x) + 1)
 */
float softSign(float featureIn) { return featureIn / (abs(featureIn) + 1); }

/**
 * d/dx(x/(abs(x) + 1)) = 1/(abs(x) + 1)^2
 */
float softSignBackward(float featureIn, float errorIn) {
    float root = 1.0f / (abs(featureIn) + 1);
    return errorIn * (root * root);
}

TEST(SoftSign, Works) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 100; ++test) {
        int numDimensions = (rand() % 5) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 8) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, dimensions);
        Tensor featureInCpu(cpuPlacement, descriptor);
        Tensor featureInGpu = featureInCpu.clone(gpuPlacement);
        Tensor destCpu(cpuPlacement, descriptor);

        half *featureInMem = (half *)featureInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        SoftSign *softSignLayer = new SoftSign();
        layers.push_back(softSignLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput();

        // Network is runnable here
        layers[0]->forward(featureInCpu, false);
        Stream stream = layers.front()->getStream();
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        stream.synchronize();

        half *destMem = (half *)destCpu.getMemPtr();
        float thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = (float)(half)softSign((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = softSignLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        softSignLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = softSignBackward(featureInMem[i], errorInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
