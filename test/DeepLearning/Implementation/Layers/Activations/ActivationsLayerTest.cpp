#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include <cuda_fp16.h>
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

using namespace std;

using namespace ThorImplementation;

/**
 *   e^xi / sum(e^xj, j = 1 to K)
 */
static float softmax(float featureIn, float sumxj, float maxX) { return exp(featureIn - maxX) / sumxj; }

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

        TensorDescriptor descriptor(DataType::FP16, dimensions);
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

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(featureInGpu));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<Softmax> softmaxLayer = make_shared<Softmax>();
        layers.push_back(softmaxLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(featureInCpu, false);
        Stream stream = layers.front()->getStream();
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
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
                for (uint32_t j = 1; j < featureInVector[batchItem].size(); ++j) {
                    if (featureInVector[batchItem][j] > maxX)
                        maxX = featureInVector[batchItem][j];
                }
                sumxj = 0.0f;
                for (uint32_t j = 0; j < featureInVector[batchItem].size(); ++j)
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
        Tensor errorInCpuFloat(cpuPlacement, TensorDescriptor(DataType::FP32, dimensions));
        Tensor errorOutCpu(cpuPlacement, descriptor);
        Tensor errorInGpu(gpuPlacement, descriptor);
        Tensor errorOutGpu = softmaxLayer->getErrorOutput().value();

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
        Tensor I(cpuPlacement, TensorDescriptor(DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor smI(cpuPlacement, TensorDescriptor(DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor smTsm(cpuPlacement, TensorDescriptor(DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor smJacobian(cpuPlacement, TensorDescriptor(DataType::FP32, {dimensions[1], dimensions[1]}));
        Tensor expectedEOut(cpuPlacement, TensorDescriptor(DataType::FP32, {dimensions[1]}));
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
