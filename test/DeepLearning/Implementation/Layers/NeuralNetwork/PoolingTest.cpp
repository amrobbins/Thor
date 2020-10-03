#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

#include "Thor.h"

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

Tensor maxPoolingForward(Tensor featureIn,
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
            for (uint32_t row = 0; row + (windowHeight - 1) < paddedInputHeight; row += verticalStride) {
                for (uint32_t col = 0; col + (windowWidth - 1) < paddedInputWidth; col += horizontalStride) {
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

Tensor maxPoolingBackward(Tensor featureIn,
                          Tensor featureOut,
                          Tensor errorIn,
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
    Tensor errorOut(TensorPlacement::MemDevices::CPU,
                    TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numFeatures, inputHeight, inputWidth}));
    half *errorOutMem = (half *)errorOut.getMemPtr();
    for (uint32_t i = 0; i < errorOut.getDescriptor().getTotalNumElements(); ++i)
        errorOutMem[i] = 0.0f;

    uint32_t paddedInputHeight = inputHeight + 2 * verticalPadding;
    uint32_t paddedInputWidth = inputWidth + 2 * horizontalPadding;

#pragma omp parallel for schedule(static, 1)
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
        for (uint32_t feature = 0; feature < numFeatures; ++feature) {
            for (uint32_t verticalWindow = 0; verticalWindow * verticalStride + (windowHeight - 1) < paddedInputHeight; ++verticalWindow) {
                for (uint32_t horizontalWindow = 0; horizontalWindow * horizontalStride + (windowWidth - 1) < paddedInputWidth;
                     ++horizontalWindow) {
                    for (uint32_t windowRow = 0; windowRow < windowHeight; ++windowRow) {
                        for (uint32_t windowCol = 0; windowCol < windowWidth; ++windowCol) {
                            uint32_t imageRow = verticalWindow * verticalStride + windowRow;
                            uint32_t imageCol = horizontalWindow * horizontalStride + windowCol;
                            uint32_t inputRow = imageRow - verticalPadding;
                            uint32_t inputCol = imageCol - horizontalPadding;

                            // printf("horizontalWindow %d imageCol %d\n", horizontalWindow, imageCol);

                            if (inputRow < 0 || inputCol < 0 || inputRow >= inputHeight || inputCol >= inputWidth)
                                continue;

                            half featureInValue = *(half *)featureIn.getElement({batch, feature, inputRow, inputCol});

                            uint32_t outputRow = verticalWindow;
                            uint32_t outputCol = horizontalWindow;

                            // vector<uint64_t> featureOutDimensions = featureOut.getDescriptor().getDimensions();
                            // printf("inputRows %d inputCols %d windowHeight %d windowWidth %d verticalPadding %d horizontalPadding %d
                            // verticalStride %d horizontalStride %d imageRow %d imageCol %d outputRow %d outputCol %d\n", inputHeight,
                            // inputWidth, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride,
                            // imageRow, imageCol, outputRow, outputCol); printf("[%ld][%ld][%ld][%ld] vs [%d][%d][%d][%d]\n",
                            // featureOutDimensions[0], featureOutDimensions[1], featureOutDimensions[2], featureOutDimensions[3], batch,
                            // feature, outputRow, outputCol); fflush(stdout);

                            half featureOutValue = *(half *)featureOut.getElement({batch, feature, outputRow, outputCol});

                            // If the input is the max in the window
                            if (featureOutValue == featureInValue) {
                                half errorValue = *(half *)errorIn.getElement({batch, feature, outputRow, outputCol});
                                // if(*(half *)errorOut.getElement({batch, feature, inputRow, inputCol}) != 0)
                                printf("Summing..... %f = %f + %f    at %d  (windowRow %d windowCol %d)\n",
                                       (float)*(half *)errorOut.getElement({batch, feature, inputRow, inputCol}) + (float)errorValue,
                                       (float)*(half *)errorOut.getElement({batch, feature, inputRow, inputCol}),
                                       (float)errorValue,
                                       batch * numFeatures * inputHeight * inputWidth + feature * inputHeight * inputWidth +
                                           inputRow * inputWidth + inputCol,
                                       windowRow,
                                       windowCol);
                                *(half *)errorOut.getElement({batch, feature, inputRow, inputCol}) =
                                    *(half *)errorOut.getElement({batch, feature, inputRow, inputCol}) + errorValue;

                                // Looks like cudnn considers just the first max element in the window as the winner.
                                windowRow = windowHeight;
                                windowCol = windowWidth;
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }

    return errorOut;
}

Tensor averagePoolingForward(Tensor featureIn,
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
            for (uint32_t row = 0; row + (windowHeight - 1) < paddedInputHeight; row += verticalStride) {
                for (uint32_t col = 0; col + (windowWidth - 1) < paddedInputWidth; col += horizontalStride) {
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
                            }
                        }
                    }
                    if (numValues != 0)
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

    for (int test = 0; test < 100; ++test) {  // FIXME
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
            featureInMem[i] = ((rand() % 100000) / 500.0f) - 100.0f;
        }

        vector<Layer *> layers;

        layers.push_back(new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
        layers.push_back(new NoOpLayer());
        Pooling *poolingLayer = new Pooling(
            Pooling::Type::MAX, windowHeight, windowWidth, verticalStride, horizontalStride, verticalPadding, horizontalPadding);
        poolingLayer->setConstructForInferenceOnly(inferenceOnly);

        layers.push_back(poolingLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        // Backward tensors must not be created, since they would be unused and would waist memory.
        if (inferenceOnly) {
            ASSERT_TRUE(poolingLayer->getErrorOutput().isEmpty());
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        featureOutGpu_h = layers.back()->getFeatureOutput();

        // Forward pass

        // Network is runnable here
        layers[0]->forward(featureIn);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());

        featureOut = maxPoolingForward(featureIn,
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
        const int featureOutSize = featureOut.getDescriptor().getTotalNumElements();

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
        // For max pooling, if(featureOut[i] = maxInWindow) then back prop error as is. Else back prop 0.
        Tensor errorInput = poolingLayer->getErrorInput();
        Tensor errorOutput = poolingLayer->getErrorOutput();
        Tensor errorInputCpu = Tensor(cpuPlacement, errorInput.getDescriptor());
        Tensor errorOutputGpu_h = Tensor(cpuPlacement, errorOutput.getDescriptor());
        half *errorInputCpuMem = (half *)errorInputCpu.getMemPtr();

        for (int i = 0; i < featureOutSize; ++i) {
            errorInputCpuMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        Tensor errorOutputCpu = maxPoolingBackward(featureIn,
                                                   featureOut,
                                                   errorInputCpu,
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

        errorInput.copyFromAsync(errorInputCpu, stream);
        poolingLayer->backward(errorInput);
        errorOutputGpu_h.copyFromAsync(errorOutput, stream);
        stream.synchronize();
        half *errorOutputCpuMem = (half *)errorOutputCpu.getMemPtr();
        half *errorOutputGpuMem_h = (half *)errorOutputGpu_h.getMemPtr();
        float maxDiff = (windowWidth * windowHeight) * 0.005;
        for (int i = 0; i < featureInSize; ++i) {
            float diff = abs((float)errorOutputCpuMem[i] - (float)errorOutputGpuMem_h[i]);
            if (true || diff >= maxDiff)
                printf("%d CPU %f GPU %f\n", i, (float)errorOutputCpuMem[i], (float)errorOutputGpuMem_h[i]);
            if (diff >= 0.05) {
                fflush(stdout);
                ASSERT_LT(diff, maxDiff);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Pooling, AveragePoolingWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

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

        layers.push_back(new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
        layers.push_back(new NoOpLayer());
        Pooling *poolingLayer = new Pooling(
            Pooling::Type::AVERAGE, windowHeight, windowWidth, verticalStride, horizontalStride, verticalPadding, horizontalPadding);
        poolingLayer->setConstructForInferenceOnly(inferenceOnly);

        layers.push_back(poolingLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        Stream stream = layers.front()->getStream();

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

        featureOut = averagePoolingForward(featureIn,
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
