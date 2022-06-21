#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <math.h>
#include <cmath>
#include <set>
#include <vector>

using namespace std;

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

/**
 * max( min(0.2 * x + 0.5, 1.0), 0.0)
 */
float hardSigmoid(float featureIn) { return max(min(0.2 * featureIn + 0.5, 1.0), 0.0); }

/**
 * d/dx(x) = 0.2 when 1 < x > 0; else 0
 */
float hardSigmoidBackward(float featureIn, float errorIn) {
    if (featureIn < 1.0f && featureIn > 0.0f)
        return errorIn * 0.2f;
    else
        return 0.0f;
}

TEST(HardSigmoid, Works) {
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
        HardSigmoid *hardSigmoidLayer = new HardSigmoid();
        layers.push_back(hardSigmoidLayer);
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
            float expectedValueFloat = (float)(half)hardSigmoid((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = hardSigmoidLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        hardSigmoidLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = hardSigmoidBackward(featureInMem[i], errorInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/**
 * ln(e(x) + 1)
 */
float softPlus(float featureIn) { return log(exp(featureIn) + 1); }

/**
 * d/dx(ln(exp(x) + 1)) = e^x/(e^x + 1)
 */
float softPlusBackward(float featureIn, float errorIn) {
    float expX = exp(featureIn);
    return errorIn * expX / (expX + 1.0f);
}

TEST(SoftPlus, Works) {
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
        SoftPlus *softPlusLayer = new SoftPlus();
        layers.push_back(softPlusLayer);
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
            float expectedValueFloat = (float)(half)softPlus((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = softPlusLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        softPlusLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = softPlusBackward(featureInMem[i], errorInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/**
 * sigmoid(x) = 1 / (1 + exp(-x))
 */
float sigmoid(float featureIn) { return 1.0f / (1.0f + exp(-featureIn)); }

/**
 * d/dx(1/(1 + exp(-x))) = e^x/(e^x + 1)^2
 */
float sigmoidBackward(float featureIn, float errorIn) {
    float expX = exp(featureIn);
    float expX_1 = expX + 1.0f;
    return errorIn * expX / (expX_1 * expX_1);
}

TEST(Sigmoid, Works) {
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
        Sigmoid *sigmoidLayer = new Sigmoid();
        layers.push_back(sigmoidLayer);
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
            float expectedValueFloat = (float)(half)sigmoid((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = sigmoidLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        sigmoidLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = sigmoidBackward(featureInMem[i], errorInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/**
 * scale * x when x >= 0
 * scale * alpha * (exp(x) - 1) when x < 0
 * where scale = 1.05070098 and alpha = 1.67326324 are pre-set values
 */
float selu(float featureIn) {
    constexpr float scale = 1.05070098;
    constexpr float alpha = 1.67326324;
    if (featureIn >= 0)
        return scale * featureIn;
    else
        return scale * alpha * (exp(featureIn) - 1.0f);
}

/**
 * d/dx(x) = scale when x >= 0
 * d/dx(alpha * (exp(x) - 1)) = scale * alpha * exp(x) when x < 0
 * where scale = 1.05070098 and alpha = 1.67326324 are pre-set values
 */
float seluBackward(float featureIn, float errorIn) {
    constexpr float scale = 1.05070098;
    constexpr float alpha = 1.67326324;
    if (featureIn >= 0)
        return errorIn * scale;
    else
        return errorIn * scale * alpha * exp(featureIn);
}

TEST(Selu, Works) {
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
        Selu *seluLayer = new Selu();
        layers.push_back(seluLayer);
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
            float expectedValueFloat = (float)(half)selu((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = seluLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        seluLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = seluBackward(featureInMem[i], errorInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/**
 * 0.5 * x * (1 + tanh((2/π)^0.5 * (x + 0.044715 * x^3)))
 */
float gelu(float featureIn) {
    return 0.5f * featureIn * (1.0f + tanh(sqrt(2.0f / (float)M_PI) * (featureIn + 0.044715f * (featureIn * featureIn * featureIn))));
}

/**
 * d/dx(0.5 * x * (1 + tanh((2/π)^0.5 * (x + 0.044715 * x^3))))
 *   = 0.5 * tanh(0.797885 * (x + 0.044715 * x^3)) + (0.0535161 * x^3 + 0.398942 * x) * sech^2(0.797885 * (x + 0.044715 * x^3)) + 0.5
 */
float geluBackward(float featureIn, float errorIn) {
    float xCubed = featureIn * featureIn * featureIn;
    float sechStuff = 1.0f / cosh(0.797885f * (featureIn + 0.044715f * xCubed));
    float derivative = 0.5f * tanh(0.797885f * (featureIn + 0.044715f * xCubed)) +
                       (0.0535161f * xCubed + 0.398942f * featureIn) * sechStuff * sechStuff + 0.5f;
    return errorIn * derivative;
}

TEST(Gelu, Works) {
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
        Gelu *geluLayer = new Gelu();
        layers.push_back(geluLayer);
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
            float expectedValueFloat = (float)(half)gelu((float)featureInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = geluLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);

        geluLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float expectedValueFloat = geluBackward(featureInMem[i], errorInMem[i]);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = errorOutMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/**
 *   e^xi / sum(e^xj, j = 1 to K)
 */
float softmax(float featureIn, float sumxj, float maxX) { return exp(featureIn - maxX) / sumxj; }

/**
 * The derivative of softmax is:
 * let sumExj = sum(exp(x_j), j = 1 to N)
 * exp(x_j)/sumExj * (1(i == k) - exp(x_i)/sumExj)
 *
 * Where 1(i == k) equals 1 when i == k and 0 otherwise.
 * N is the number of classes of a training example
 *
 */
float softmaxBackward(float featureIn, float errorIn) {
    float xCubed = featureIn * featureIn * featureIn;
    float sechStuff = 1.0f / cosh(0.797885f * (featureIn + 0.044715f * xCubed));
    float derivative = 0.5f * tanh(0.797885f * (featureIn + 0.044715f * xCubed)) +
                       (0.0535161f * xCubed + 0.398942f * featureIn) * sechStuff * sechStuff + 0.5f;
    return errorIn * derivative;
}

TEST(Softmax, Works) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 10; ++test) {
        int numDimensions = 2;
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
        vector<vector<float>> featureInVector;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            featureInVector.emplace_back();
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                featureInVector.back().push_back((float)featureInMem[i * dimensions[1] + j]);
            }
        }

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(featureInGpu));
        layers.push_back(new NoOpLayer());
        Softmax *softmaxLayer = new Softmax();
        layers.push_back(softmaxLayer);
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
        float sumxj = 0.0f;
        vector<vector<float>> softmaxForward;
        float maxX = 0.0f;
        for (int i = 0; i < numElements; ++i) {
            uint32_t batchItem = i / dimensions[1];
            uint32_t element = i % dimensions[1];
            if (element == 0) {
                maxX = featureInVector[batchItem][0];
                for (uint j = 1; j < featureInVector[batchItem].size(); ++j) {
                    if (featureInVector[batchItem][j] > maxX)
                        maxX = featureInVector[batchItem][j];
                }
                sumxj = 0.0f;
                for (uint j = 0; j < featureInVector[batchItem].size(); ++j)
                    sumxj += exp(featureInVector[batchItem][j] - maxX);
                softmaxForward.emplace_back();
            }
            float expectedValueFloat = (float)(half)softmax((float)featureInMem[i], sumxj, maxX);
            softmaxForward.back().push_back(expectedValueFloat);
            half expectedValue = (half)expectedValueFloat;
            half actualValue = destMem[i];
            ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
        }

        // Backward pass
        Tensor errorInCpu(cpuPlacement, descriptor);
        Tensor errorInCpuFloat(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = softmaxLayer->getErrorOutput();

        half *errorInMem = (half *)errorInCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            errorInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        errorInGpu.copyFromAsync(errorInCpu, stream);
        errorInCpuFloat.copyFromAsync(errorInCpu, stream);

        softmaxLayer->backward(errorInGpu);
        errorOutCpu.copyFromAsync(errorOutGpu, stream);
        stream.synchronize();

        half *errorOutMem = (half *)errorOutCpu.getMemPtr();
        Tensor I(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor smI(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor smTsm(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor smJacobian(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor expectedEOut(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dimensions[1]}));
        thresh = 0.01;
        for (uint32_t batchItem = 0; batchItem < dimensions[0]; ++batchItem) {
            diagonalize((float *)softmaxForward[batchItem].data(), (float *)smI.getMemPtr(), dimensions[1]);
            matrixMultiplyCpu(softmaxForward[batchItem].data(),
                              softmaxForward[batchItem].data(),
                              (float *)smTsm.getMemPtr(),
                              dimensions[1],
                              1,
                              1,
                              dimensions[1],
                              1,
                              dimensions[1],
                              dimensions[1],
                              false,
                              false,
                              false);
            matrixSubtractCpu(
                (float *)smI.getMemPtr(), (float *)smTsm.getMemPtr(), (float *)smJacobian.getMemPtr(), dimensions[1], dimensions[1]);
            matrixMultiplyCpu((float *)errorInCpuFloat.getMemPtr() + batchItem * dimensions[1],
                              (float *)smJacobian.getMemPtr(),
                              (float *)expectedEOut.getMemPtr(),
                              1,
                              dimensions[1],
                              dimensions[1],
                              dimensions[1],
                              dimensions[1],
                              dimensions[1],
                              dimensions[1],
                              false,
                              false,
                              false);
            for (uint32_t classType = 0; classType < dimensions[1]; ++classType) {
                half expectedValue = (half)(((float *)expectedEOut.getMemPtr())[classType]);
                half actualValue = errorOutMem[batchItem * dimensions[1] + classType];
                ASSERT_LT(abs((float)expectedValue - (float)actualValue), thresh);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
