#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

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

TEST(Convolution2d, Convolution2dWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);

    for (int test = 0; test < 20; ++test) {
        const int numInputColumns = 1 + (rand() % 50);
        const int numInputRows = 1 + (rand() % 50);
        const int filterHorizontalStride = numInputColumns == 1 ? 1 : 1 + (rand() % (numInputColumns - 1));
        const int filterVerticalStride = numInputRows == 1 ? 1 : 1 + (rand() % (numInputRows - 1));
        const int filterWidth = numInputColumns == 1 ? 1 : 1 + (rand() % (numInputColumns - 1));
        const int filterHeight = numInputRows == 1 ? 1 : 1 + (rand() % (numInputRows - 1));
        const int leftAndRightPadWidth = rand() % 10;
        const int topAndBottomPadHeight = rand() % 10;
        const int numInputChannels = 1 + (rand() % 10);
        const int numOutputChannels = 1 + (rand() % 10);
        const int batchSize = 1 + (rand() % 10);
        const bool inferenceOnly = (rand() % 4) == 0;
        const bool hasBias = (rand() % 4) != 0;
        Optional<float> learningRate;
        if (!inferenceOnly)
            learningRate = (1 + (rand() % 20000)) / 10000.0f;

        int numOutputRows =
            ConvolutionTestHelper::computeOutputDimensionSize(numInputRows, topAndBottomPadHeight, filterHeight, filterVerticalStride);
        int numOutputColumns =
            ConvolutionTestHelper::computeOutputDimensionSize(numInputColumns, leftAndRightPadWidth, filterWidth, filterHorizontalStride);

        vector<Layer *> layers;

        vector<unsigned long> inputDimensions;
        inputDimensions.push_back(batchSize);
        inputDimensions.push_back(numInputChannels);
        inputDimensions.push_back(numInputRows);
        inputDimensions.push_back(numInputColumns);
        TensorDescriptor inputDescriptor(TensorDescriptor::DataType::FP16, inputDimensions);
        int numInputElements = inputDescriptor.getTotalNumElements();

        TensorDescriptor errorOutputDescriptor = inputDescriptor;

        vector<unsigned long> outputDimensions;
        outputDimensions.push_back(batchSize);
        outputDimensions.push_back(numOutputChannels);
        outputDimensions.push_back(numOutputRows);
        outputDimensions.push_back(numOutputColumns);
        TensorDescriptor outputDescriptor(TensorDescriptor::DataType::FP16, outputDimensions);
        int numOutputElements = outputDescriptor.getTotalNumElements();

        vector<unsigned long> weightsDimensions;
        weightsDimensions.push_back(numOutputChannels);
        weightsDimensions.push_back(numInputChannels);
        weightsDimensions.push_back(filterHeight);
        weightsDimensions.push_back(filterWidth);
        TensorDescriptor weightsDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);

        Tensor featureInputCpu(cpuPlacement, inputDescriptor);
        half *featureInputMem = (half *)featureInputCpu.getMemPtr();
        for (int i = 0; i < numInputElements; ++i) {
            featureInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        Tensor weightsCpu(cpuPlacement, weightsDescriptor);
        int numWeights = weightsCpu.getDescriptor().getTotalNumElements();
        half *weightsMem = (half *)weightsCpu.getMemPtr();
        for (int i = 0; i < numWeights; ++i) {
            weightsMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        Optional<Tensor> biasesCpu;
        if (hasBias) {
            TensorDescriptor biasesDescriptor(TensorDescriptor::DataType::FP16, numOutputChannels);
            biasesCpu = Tensor(cpuPlacement, biasesDescriptor);
            half *biasesMem = (half *)biasesCpu.get().getMemPtr();
            for (int i = 0; i < numOutputChannels; ++i) {
                biasesMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
            }
        }

        Tensor featureOutputCpu(cpuPlacement, outputDescriptor);
        Tensor featureOutputGpu_h(cpuPlacement, outputDescriptor);

        layers.push_back(
            new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureInputCpu.getDescriptor().getDimensions(), stream));
        layers.push_back(new NoOpLayer());
        Convolution2d *convolution2dLayer = new Convolution2d(filterWidth,
                                                              filterHeight,
                                                              filterHorizontalStride,
                                                              filterVerticalStride,
                                                              leftAndRightPadWidth,
                                                              topAndBottomPadHeight,
                                                              numInputChannels,
                                                              numOutputChannels,
                                                              batchSize,
                                                              numInputColumns,
                                                              numInputRows,
                                                              inferenceOnly,
                                                              hasBias,
                                                              learningRate);
        layers.push_back(convolution2dLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        featureOutputGpu_h = layers.back()->getFeatureOutput();

        // convolution2dLayer->setCallBackWhenGradientsReady(weightUpdateCallback);

        convolution2dLayer->getWeights().copyFromAsync(weightsCpu, stream);
        if (hasBias)
            convolution2dLayer->getBiases().get().copyFromAsync(biasesCpu, stream);

        // Network is runnable here
        layers[0]->forward(featureInputCpu);
        stream.waitEvent(((NetworkOutput *)layers.back())->getOutputReadyEvent());

        ConvolutionKernelRequirement convolutionKernelRequirement(MachineEvaluator::instance().getGpuType(0),
                                                                  filterWidth,
                                                                  filterHeight,
                                                                  filterHorizontalStride,
                                                                  filterVerticalStride,
                                                                  leftAndRightPadWidth,
                                                                  topAndBottomPadHeight,
                                                                  numInputChannels,
                                                                  numOutputChannels,
                                                                  batchSize,
                                                                  numInputColumns,
                                                                  numInputRows);

        ConvolutionTestHelper::cpuConvolutionForward(
            featureInputCpu, weightsCpu, biasesCpu, featureOutputCpu, convolutionKernelRequirement);

        stream.synchronize();

        half *cpuFeatureOut = (half *)featureOutputCpu.getMemPtr();
        half *gpuFeatureOut = (half *)featureOutputGpu_h.getMemPtr();
        for (int i = 0; i < numOutputElements; ++i) {
            float thresh = std::max(abs((float)cpuFeatureOut[i]) / 500, 0.01f);
            EXPECT_LT(abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])), thresh);
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= thresh)
                printf("%f %f\n", (float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass
        Tensor errorInputGpu = convolution2dLayer->getErrorInputs().front();
        Tensor errorOutputGpu = convolution2dLayer->getErrorOutputs().front();
        Tensor errorOutputGpu_h = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
        Tensor errorInputCpu = Tensor(cpuPlacement, errorInputGpu.getDescriptor());
        Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());

        half *errorInputMem = (half *)errorInputCpu.getMemPtr();
        for (int i = 0; i < numOutputElements; ++i) {
            errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        errorInputGpu.copyFromAsync(errorInputCpu, stream);
        convolution2dLayer->backward(errorInputGpu);
        errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);
        stream.synchronize();
        convolution2dLayer->getGradientUpdateStream().get().synchronize();

        // Backward Data
        ConvolutionTestHelper::cpuConvolutionBackwardData(errorInputCpu, weightsCpu, errorOutputCpu, convolutionKernelRequirement);

        // Verify CPU and GPU results match
        for (unsigned int n = 0; n < errorOutputDescriptor.getDimensions()[0]; ++n) {
            for (unsigned int c = 0; c < errorOutputDescriptor.getDimensions()[1]; ++c) {
                for (unsigned int h = 0; h < errorOutputDescriptor.getDimensions()[2]; ++h) {
                    for (unsigned int w = 0; w < errorOutputDescriptor.getDimensions()[3]; ++w) {
                        float cpuVal = *(half *)errorOutputCpu.getElement({(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                        float gpuVal = *(half *)errorOutputGpu_h.getElement({(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                        float thresh = batchSize * 0.1 + abs(cpuVal * 0.05);
                        EXPECT_LT(abs(cpuVal - gpuVal), thresh);
                        if (abs(cpuVal - gpuVal) >= thresh)
                            printf("%f %f   at [%d, %d, %d, %d]\n", cpuVal, gpuVal, n, c, h, w);
                    }
                }
            }
        }

        // Backward Filter
        Tensor weightsGpu = convolution2dLayer->getWeights();
        Tensor weightsGpu_h = weightsGpu.clone(cpuPlacement);
        Tensor weightsGradientCpu = weightsGpu_h.clone();
        weightsGpu_h.copyFromAsync(weightsGpu, stream);
        stream.synchronize();

        ConvolutionTestHelper::cpuConvolutionBackwardFilter(
            featureInputCpu, errorInputCpu, weightsGradientCpu, convolutionKernelRequirement, false);

        for (int o = 0; o < numOutputChannels; ++o) {
            for (int i = 0; i < numInputChannels; ++i) {
                for (int h = 0; h < filterHeight; ++h) {
                    for (int w = 0; w < filterWidth; ++w) {
                        float cpuGradient = *(half *)weightsGradientCpu.getElement({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                        float cpuWeight = *(half *)weightsCpu.getElement({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                        float cpuVal = cpuWeight - cpuGradient * learningRate.get();
                        float gpuVal = *(half *)weightsGpu_h.getElement({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                        float thresh = batchSize * 0.1 + abs(cpuVal * 0.05);
                        EXPECT_LT(abs(cpuVal - gpuVal), thresh);
                        if (abs(cpuVal - gpuVal) >= thresh)
                            printf("%f %f   at [%d, %d, %d, %d]\n", cpuVal, gpuVal, o, i, h, w);
                    }
                }
            }
        }

        // Backward Bias
        if (hasBias) {
            Tensor biasesGpu = convolution2dLayer->getBiases();
            Tensor biasesGpu_h = biasesGpu.clone(cpuPlacement);
            Tensor biasesGradientCpu = biasesGpu_h.clone();
            biasesGpu_h.copyFromAsync(biasesGpu, stream);
            stream.synchronize();

            ConvolutionTestHelper::cpuConvolutionBackwardBias(errorInputCpu, biasesGradientCpu);

            for (int i = 0; i < numOutputChannels; ++i) {
                float cpuGradient = *(half *)biasesGradientCpu.getElement({(uint64_t)i});
                float cpuBias = *(half *)biasesCpu.get().getElement({(uint64_t)i});
                float cpuVal = cpuBias - cpuGradient * learningRate.get();
                float gpuVal = *(half *)biasesGpu_h.getElement({(uint64_t)i});
                float thresh = batchSize * 0.001 + abs(cpuVal * 0.005);
                EXPECT_LT(abs(cpuVal - gpuVal), thresh);
                if (abs(cpuVal - gpuVal) >= thresh)
                    printf("%f %f   at [%d] batchSize %d thresh %f\n", cpuVal, gpuVal, i, batchSize, thresh);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

// FIXME: make a test for multiple connections

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
