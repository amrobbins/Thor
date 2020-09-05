#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

#include "MLDev.h"

#include <stdio.h>
#include <unistd.h>
#include <cmath>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <vector>

using std::vector;

Tensor maxPooling(Tensor featureIn,
                  uint32_t windowHeight,
                  uint32_t windowWidth,
                  uint32_t verticalStride,
                  uint32_t horizontalStride,
                  uint32_t batchSize,
                  uint32_t numFeatures,
                  uint32_t inputHeight,
                  uint32_t inputWidth,
                  uint32_t verticalPadding,
                  uint32_t horizontalPadding) {
    vector<unsigned long> featureInDimensions = featureIn.getDescriptor().getDimensions();
    assert(featureInDimensions[0] == batchSize);
    assert(featureInDimensions[1] == numFeatures);
    assert(featureInDimensions[2] == inputHeight);
    assert(featureInDimensions[3] == inputWidth);

    uint32_t outputHeight = ConvolutionTestHelper::computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
    uint32_t outputWidth = ConvolutionTestHelper::computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

    Tensor featureOut(TensorPlacement::MemDevices::CPU,
                      TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numFeatures, outputHeight, outputWidth}));

    uint32_t paddedInputHeight = inputHeight + 2 * verticalPadding;
    uint32_t paddedInputWidth = inputWidth + 2 * horizontalPadding;

    if (omp_get_num_procs() > 1)
        omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(static, 1)
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
        for (uint32_t feature = 0; feature < numFeatures; ++feature) {
            for (uint32_t row = 0; row + windowHeight <= paddedInputHeight; row += verticalStride) {
                for (uint32_t col = 0; col + windowWidth <= paddedInputWidth; col += horizontalStride) {
                    float maxValue = -INFINITY;
                    for (uint32_t windowRow = 0; windowRow < windowHeight; ++windowRow) {
                        for (uint32_t windowCol = 0; windowCol < windowWidth; ++windowCol) {
                            uint32_t imageCol = col + windowCol;
                            uint32_t imageRow = row + windowRow;
                            bool inHorizontalPadding = imageCol < horizontalPadding || imageCol >= horizontalPadding + inputWidth;
                            bool inVerticalPadding = imageRow < verticalPadding || imageRow >= verticalPadding + inputHeight;
                            if (!inHorizontalPadding && !inVerticalPadding) {
                                half value = *(half *)featureIn.getElement(
                                    {batch, feature, imageRow - verticalPadding, imageCol - horizontalPadding});
                                if (value > maxValue)
                                    maxValue = value;
                            }
                        }
                    }
                    uint32_t outputRow = row / verticalStride;
                    uint32_t outputCol = col / horizontalStride;
                    *(half *)featureOut.getElement({batch, feature, outputRow, outputCol}) = maxValue;
                }
            }
        }
    }

    return featureOut;
}

Tensor averagePooling(Tensor featureIn,
                      uint32_t windowHeight,
                      uint32_t windowWidth,
                      uint32_t verticalStride,
                      uint32_t horizontalStride,
                      uint32_t batchSize,
                      uint32_t numFeatures,
                      uint32_t inputHeight,
                      uint32_t inputWidth,
                      uint32_t verticalPadding,
                      uint32_t horizontalPadding) {
    vector<unsigned long> featureInDimensions = featureIn.getDescriptor().getDimensions();
    assert(featureInDimensions[0] == batchSize);
    assert(featureInDimensions[1] == numFeatures);
    assert(featureInDimensions[2] == inputHeight);
    assert(featureInDimensions[3] == inputWidth);

    uint32_t outputHeight = ConvolutionTestHelper::computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
    uint32_t outputWidth = ConvolutionTestHelper::computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

    Tensor featureOut(TensorPlacement::MemDevices::CPU,
                      TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numFeatures, outputHeight, outputWidth}));

    uint32_t paddedInputHeight = inputHeight + 2 * verticalPadding;
    uint32_t paddedInputWidth = inputWidth + 2 * horizontalPadding;

    if (omp_get_num_procs() > 1)
        omp_set_num_threads(omp_get_num_procs() - 1);

#pragma omp parallel for schedule(static, 1)
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
        for (uint32_t feature = 0; feature < numFeatures; ++feature) {
            for (uint32_t row = 0; row + windowHeight <= paddedInputHeight; row += verticalStride) {
                for (uint32_t col = 0; col + windowWidth <= paddedInputWidth; col += horizontalStride) {
                    float averageValue = 0;
                    int numValues = 0;
                    for (uint32_t windowRow = 0; windowRow < windowHeight; ++windowRow) {
                        for (uint32_t windowCol = 0; windowCol < windowWidth; ++windowCol) {
                            uint32_t imageCol = col + windowCol;
                            uint32_t imageRow = row + windowRow;
                            bool inHorizontalPadding = imageCol < horizontalPadding || imageCol >= horizontalPadding + inputWidth;
                            bool inVerticalPadding = imageRow < verticalPadding || imageRow >= verticalPadding + inputHeight;
                            if (!inHorizontalPadding && !inVerticalPadding) {
                                half value = *(half *)featureIn.getElement(
                                    {batch, feature, imageRow - verticalPadding, imageCol - horizontalPadding});
                                numValues += 1;
                                averageValue += value;
                            } else {
                                numValues += 1;
                            }
                        }
                    }
                    averageValue /= numValues;
                    uint32_t outputRow = row / verticalStride;
                    uint32_t outputCol = col / horizontalStride;
                    *(half *)featureOut.getElement({batch, feature, outputRow, outputCol}) = averageValue;
                }
            }
        }
    }

    return featureOut;
}

TEST(Pooling, MaxPoolingWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);

    for (int test = 0; test < 10; ++test) {
        uint32_t batchSize = (rand() % 10) + 1;
        uint32_t numFeatures = (rand() % 10) + 1;
        uint32_t inputHeight = (rand() % 50) + 1;
        uint32_t inputWidth = (rand() % 50) + 1;
        uint32_t windowHeight = (rand() % inputHeight) + 1;
        uint32_t windowWidth = (rand() % inputWidth) + 1;
        uint32_t verticalPadding = rand() % windowHeight;
        uint32_t horizontalPadding = rand() % windowWidth;
        uint32_t verticalStride = (rand() % 5) + 1;
        uint32_t horizontalStride = (rand() % 5) + 1;
        uint32_t outputHeight =
            ConvolutionTestHelper::computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
        uint32_t outputWidth =
            ConvolutionTestHelper::computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

        bool inferenceOnly = (rand() % 4) == 0;

        Tensor featureIn =
            Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numFeatures, inputHeight, inputWidth}));
        Tensor featureOut;
        Tensor featureOutGpu_h;

        const int featureInSize = featureIn.getDescriptor().getTotalNumElements();
        half *featureInMem = (half *)featureIn.getMemPtr();
        for (int i = 0; i < featureInSize; ++i) {
            featureInMem[i] = ((rand() % 1000) / 5.0f) - 100.0f;
        }

        vector<Layer *> layers;

        layers.push_back(
            new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions(), stream));
        layers.push_back(new NoOpLayer());
        Pooling *poolingLayer = new Pooling(Pooling::Type::MAX,
                                            windowHeight,
                                            windowWidth,
                                            verticalStride,
                                            horizontalStride,
                                            batchSize,
                                            numFeatures,
                                            inputHeight,
                                            inputWidth,
                                            verticalPadding,
                                            horizontalPadding);
        poolingLayer->setInferenceOnly(inferenceOnly);

        layers.push_back(poolingLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);

        // Backward tensors must not be created, since they would be unused and would waist memory.
        if (inferenceOnly) {
            ASSERT_TRUE(poolingLayer->getErrorOutput().isEmpty());
        }

        featureOutGpu_h = layers.back()->getFeatureOutput();

        // Forward pass

        // Network is runnable here
        layers[0]->forward(featureIn);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());

        featureOut = maxPooling(featureIn,
                                windowHeight,
                                windowWidth,
                                verticalStride,
                                horizontalStride,
                                batchSize,
                                numFeatures,
                                inputHeight,
                                inputWidth,
                                verticalPadding,
                                horizontalPadding);
        vector<unsigned long> featureOutDimensions = featureOut.getDescriptor().getDimensions();
        assert(featureOutDimensions[0] == batchSize);
        assert(featureOutDimensions[1] == numFeatures);
        assert(featureOutDimensions[2] == outputHeight);
        assert(featureOutDimensions[3] == outputWidth);

        stream.synchronize();

        half *cpuFeatureOut = (half *)featureOut.getMemPtr();
        half *gpuFeatureOut = (half *)featureOutGpu_h.getMemPtr();
        int numOutputElements = featureOut.getDescriptor().getTotalNumElements();
        for (int i = 0; i < numOutputElements; ++i) {
            ASSERT_EQ((float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
            if (cpuFeatureOut[i] != gpuFeatureOut[i]) {
                printf("cpu: %f gpu: %f\n", (float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
            }
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Pooling, AveragePoolingWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);

    for (int test = 0; test < 10; ++test) {
        uint32_t batchSize = (rand() % 10) + 1;
        uint32_t numFeatures = (rand() % 10) + 1;
        uint32_t inputHeight = (rand() % 50) + 1;
        uint32_t inputWidth = (rand() % 50) + 1;
        uint32_t windowHeight = (rand() % inputHeight) + 1;
        uint32_t windowWidth = (rand() % inputWidth) + 1;
        uint32_t verticalPadding = rand() % windowHeight;
        uint32_t horizontalPadding = rand() % windowWidth;
        uint32_t verticalStride = (rand() % 5) + 1;
        uint32_t horizontalStride = (rand() % 5) + 1;
        uint32_t outputHeight =
            ConvolutionTestHelper::computeOutputDimensionSize(inputHeight, verticalPadding, windowHeight, verticalStride);
        uint32_t outputWidth =
            ConvolutionTestHelper::computeOutputDimensionSize(inputWidth, horizontalPadding, windowWidth, horizontalStride);

        bool inferenceOnly = (rand() % 4) == 0;

        Tensor featureIn =
            Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numFeatures, inputHeight, inputWidth}));
        Tensor featureOut;
        Tensor featureOutGpu_h;

        const int featureInSize = featureIn.getDescriptor().getTotalNumElements();
        half *featureInMem = (half *)featureIn.getMemPtr();
        for (int i = 0; i < featureInSize; ++i) {
            featureInMem[i] = ((rand() % 1000) / 5.0f) - 100.0f;
        }

        vector<Layer *> layers;

        layers.push_back(
            new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions(), stream));
        layers.push_back(new NoOpLayer());
        Pooling *poolingLayer = new Pooling(Pooling::Type::AVERAGE,
                                            windowHeight,
                                            windowWidth,
                                            verticalStride,
                                            horizontalStride,
                                            batchSize,
                                            numFeatures,
                                            inputHeight,
                                            inputWidth,
                                            verticalPadding,
                                            horizontalPadding);
        poolingLayer->setInferenceOnly(inferenceOnly);

        layers.push_back(poolingLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);

        // Backward tensors must not be created, since they would be unused and would waist memory.
        if (inferenceOnly) {
            ASSERT_TRUE(poolingLayer->getErrorOutput().isEmpty());
        }

        featureOutGpu_h = layers.back()->getFeatureOutput();

        // Forward pass

        // Network is runnable here
        layers[0]->forward(featureIn);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());

        featureOut = averagePooling(featureIn,
                                    windowHeight,
                                    windowWidth,
                                    verticalStride,
                                    horizontalStride,
                                    batchSize,
                                    numFeatures,
                                    inputHeight,
                                    inputWidth,
                                    verticalPadding,
                                    horizontalPadding);
        vector<unsigned long> featureOutDimensions = featureOut.getDescriptor().getDimensions();
        assert(featureOutDimensions[0] == batchSize);
        assert(featureOutDimensions[1] == numFeatures);
        assert(featureOutDimensions[2] == outputHeight);
        assert(featureOutDimensions[3] == outputWidth);

        stream.synchronize();

        half *cpuFeatureOut = (half *)featureOut.getMemPtr();
        half *gpuFeatureOut = (half *)featureOutGpu_h.getMemPtr();
        int numOutputElements = featureOut.getDescriptor().getTotalNumElements();
        float maxDiff = 0.1;
        for (int i = 0; i < numOutputElements; ++i) {
            ASSERT_LT(abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])), maxDiff);
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= maxDiff) {
                printf("cpu: %f gpu: %f\n", (float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
            }
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
