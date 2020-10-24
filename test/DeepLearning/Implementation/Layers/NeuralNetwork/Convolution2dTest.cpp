#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

#include "Thor.h"

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

using namespace ThorImplementation;

// FIXME: make a test for multiple connections

void backwardPass(Convolution2d *convolution2dLayer,
                  int numOutputElements,
                  int numWeights,
                  int numInputChannels,
                  int numOutputChannels,
                  int batchSize,
                  Tensor featureInputCpu,
                  Tensor weightsCpu,
                  Optional<Tensor> biasesCpu,
                  Stream stream,
                  ConvolutionKernelRequirement convolutionKernelRequirement,
                  int filterHeight,
                  int filterWidth,
                  bool accumulate);

TEST(Convolution2d, Convolution2dWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 5; ++test) {
        const int numInputColumns = 1 + (rand() % 50);
        const int numInputRows = 1 + (rand() % 50);
        const int filterHorizontalStride = numInputColumns == 1 ? 1 : 1 + (rand() % (numInputColumns - 1));
        const int filterVerticalStride = numInputRows == 1 ? 1 : 1 + (rand() % (numInputRows - 1));
        const int filterWidth = numInputColumns == 1 ? 1 : 1 + (rand() % (numInputColumns - 1));
        const int filterHeight = numInputRows == 1 ? 1 : 1 + (rand() % (numInputRows - 1));
        const int leftAndRightPadWidth = filterWidth < 10 ? rand() % filterWidth : rand() % 10;
        const int topAndBottomPadHeight = filterHeight < 10 ? rand() % filterHeight : rand() % 10;
        const int numInputChannels = 1 + (rand() % 10);
        const int numOutputChannels = 1 + (rand() % 10);
        const int batchSize = 1 + (rand() % 10);
        const bool inferenceOnly = (rand() % 4) == 0;
        const bool hasBias = (rand() % 4) != 0;

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
        TensorDescriptor biasesDescriptor;
        half *biasesMem = nullptr;
        if (hasBias) {
            biasesDescriptor = TensorDescriptor(TensorDescriptor::DataType::FP16, numOutputChannels);
            biasesCpu = Tensor(cpuPlacement, biasesDescriptor);
            biasesMem = (half *)biasesCpu.get().getMemPtr();
            for (int i = 0; i < numOutputChannels; ++i) {
                biasesMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
            }
        }

        Tensor featureOutputCpu(cpuPlacement, outputDescriptor);
        Tensor featureOutputGpu_h;

        layers.push_back(new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureInputCpu.getDescriptor().getDimensions()));
        layers.push_back(new NoOpLayer());
        Convolution2d *convolution2dLayer = new Convolution2d(filterWidth,
                                                              filterHeight,
                                                              filterHorizontalStride,
                                                              filterVerticalStride,
                                                              leftAndRightPadWidth,
                                                              topAndBottomPadHeight,
                                                              numOutputChannels,
                                                              hasBias);
        convolution2dLayer->setConstructForInferenceOnly(inferenceOnly);

        layers.push_back(convolution2dLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        // Backward tensors must not be created, since they would be unused and would waist memory.
        if (inferenceOnly) {
            ASSERT_TRUE(convolution2dLayer->getErrorOutputs()[0].isEmpty());
            ASSERT_TRUE(convolution2dLayer->getWeightsGradient().isEmpty());
            ASSERT_TRUE(convolution2dLayer->getBiasesGradient().isEmpty());
            ASSERT_FALSE(convolution2dLayer->getGradientUpdateStream().isInitialized());
        }

        if (!hasBias) {
            ASSERT_TRUE(convolution2dLayer->getBiases().isEmpty());
            ASSERT_TRUE(convolution2dLayer->getBiasesGradient().isEmpty());
        }

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
        const float thresh = std::max(batchSize * (filterWidth * 0.02 + filterHeight * 0.02), 1.01);
        for (int i = 0; i < numOutputElements; ++i) {
            int threshAdjust = abs(cpuFeatureOut[i]) > 300.0f ? 3 : 0;
            EXPECT_LT(abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])), thresh + threshAdjust);
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= thresh)
                printf("%f %f\n", (float)(cpuFeatureOut[i]), (float)(gpuFeatureOut[i]));
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass # 1 no accumulate
        backwardPass(convolution2dLayer,
                     numOutputElements,
                     numWeights,
                     numInputChannels,
                     numOutputChannels,
                     batchSize,
                     featureInputCpu,
                     weightsCpu,
                     biasesCpu,
                     stream,
                     convolutionKernelRequirement,
                     filterHeight,
                     filterWidth,
                     false);

        backwardPass(convolution2dLayer,
                     numOutputElements,
                     numWeights,
                     numInputChannels,
                     numOutputChannels,
                     batchSize,
                     featureInputCpu,
                     weightsCpu,
                     biasesCpu,
                     stream,
                     convolutionKernelRequirement,
                     filterHeight,
                     filterWidth,
                     true);

        for (int i = 0; i < numWeights; ++i) {
            weightsMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        if (hasBias) {
            for (int i = 0; i < numOutputChannels; ++i) {
                biasesMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
            }
        }
        Event weightsUpdateEvent = convolution2dLayer->updateWeightsAndBiases(weightsCpu, biasesCpu, stream.putEvent());
        stream.waitEvent(weightsUpdateEvent);
        convolution2dLayer->getGradientUpdateStream().waitEvent(weightsUpdateEvent);

        backwardPass(convolution2dLayer,
                     numOutputElements,
                     numWeights,
                     numInputChannels,
                     numOutputChannels,
                     batchSize,
                     featureInputCpu,
                     weightsCpu,
                     biasesCpu,
                     stream,
                     convolutionKernelRequirement,
                     filterHeight,
                     filterWidth,
                     false);

        // Test apply gradients
        float learningRate = 1.0f / ((rand() % 100) + 1);
        convolution2dLayer->setLearningRate(learningRate);
        Tensor weights = convolution2dLayer->getWeights().clone(TensorPlacement::MemDevices::CPU);
        Tensor weightsGpu_h = convolution2dLayer->getWeights().clone(TensorPlacement::MemDevices::CPU);
        Tensor weightsGradientGpu_h = convolution2dLayer->getWeightsGradient().get().clone(TensorPlacement::MemDevices::CPU);
        Tensor biases;
        Tensor biasesGpu_h;
        Tensor biasesGradientGpu_h;
        if (hasBias) {
            biases = convolution2dLayer->getBiases().get().clone(TensorPlacement::MemDevices::CPU);
            biasesGpu_h = convolution2dLayer->getBiases().get().clone(TensorPlacement::MemDevices::CPU);
            biasesGradientGpu_h = convolution2dLayer->getBiasesGradient().get().clone(TensorPlacement::MemDevices::CPU);
        }

        weights.copyFromAsync(convolution2dLayer->getWeights(), stream);
        weightsGradientGpu_h.copyFromAsync(convolution2dLayer->getWeightsGradient(), stream);
        if (hasBias) {
            biases.copyFromAsync(convolution2dLayer->getBiases(), stream);
            biasesGradientGpu_h.copyFromAsync(convolution2dLayer->getBiasesGradient(), stream);
        }
        stream.synchronize();

        Event gradientsApplied = convolution2dLayer->updateWeightsAndBiasesWithScaledGradient();
        stream.waitEvent(gradientsApplied);
        weightsGpu_h.copyFromAsync(convolution2dLayer->getWeights(), stream);
        if (hasBias)
            biasesGpu_h.copyFromAsync(convolution2dLayer->getBiases(), stream);

        stream.synchronize();

        float maxDiff = 0.03;

        weightsMem = (half *)weights.getMemPtr();
        half *weightsGpuMem = (half *)weightsGpu_h.getMemPtr();
        half *weightsGradientMem = (half *)weightsGradientGpu_h.getMemPtr();
        for (int i = 0; i < numWeights; ++i) {
            half expected = weightsMem[i] - learningRate * weightsGradientMem[i];
            EXPECT_LT(abs((float)expected - (float)weightsGpuMem[i]), maxDiff);
            if (abs((float)expected - (float)weightsGpuMem[i]) >= maxDiff) {
                printf("%d  cpu %f  gpu %f     weight %f   gradient %f  learningRate %f\n",
                       i,
                       (float)expected,
                       (float)weightsGpuMem[i],
                       (float)weightsMem[i],
                       (float)weightsGradientMem[i],
                       learningRate);
            }
        }

        if (hasBias) {
            int numBiases = convolution2dLayer->getBiases().get().getDescriptor().getTotalNumElements();
            half *biasesMem = (half *)biases.getMemPtr();
            half *biasesGpuMem = (half *)biasesGpu_h.getMemPtr();
            half *biasesGradientMem = (half *)biasesGradientGpu_h.getMemPtr();
            for (int i = 0; i < numBiases; ++i) {
                half expected = biasesMem[i] - learningRate * biasesGradientMem[i];
                EXPECT_LT(abs((float)expected - (float)biasesGpuMem[i]), maxDiff);
                if (abs((float)expected - (float)biasesGpuMem[i]) >= maxDiff) {
                    printf("%d  cpu %f  gpu %f     bias %f   gradient %f  learningRate %f\n",
                           i,
                           (float)expected,
                           (float)biasesGpuMem[i],
                           (float)biasesMem[i],
                           (float)biasesGradientMem[i],
                           learningRate);
                }
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

void backwardPass(Convolution2d *convolution2dLayer,
                  int numOutputElements,
                  int numWeights,
                  int numInputChannels,
                  int numOutputChannels,
                  int batchSize,
                  Tensor featureInputCpu,
                  Tensor weightsCpu,
                  Optional<Tensor> biasesCpu,
                  Stream stream,
                  ConvolutionKernelRequirement convolutionKernelRequirement,
                  int filterHeight,
                  int filterWidth,
                  bool accumulate) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    Tensor errorInputGpu = convolution2dLayer->getErrorInputs().front();
    Tensor errorOutputGpu = convolution2dLayer->getErrorOutputs().front();
    Tensor errorOutputGpu_h = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
    Tensor errorInputCpu = Tensor(cpuPlacement, errorInputGpu.getDescriptor());
    Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());

    half *errorInputMem = (half *)errorInputCpu.getMemPtr();
    for (int i = 0; i < numOutputElements; ++i) {
        errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }

    Tensor weightsGradientCpu = weightsCpu.clone();
    Tensor weightsGradientGpu = convolution2dLayer->getWeightsGradient();

    if (accumulate) {
        weightsGradientCpu.copyFromAsync(weightsGradientGpu, stream);
    } else {
        memset(weightsGradientCpu.getMemPtr(), 0, weightsGradientCpu.getDescriptor().getArraySizeInBytes());
    }

    // printf("cpu before %f\n", (float)*(half*)weightsGradientCpu.getElement({0, 0, 0, 0}));

    Optional<Tensor> biasesGradientCpu = biasesCpu.isPresent() ? Optional<Tensor>(biasesCpu.get().clone()) : Optional<Tensor>::empty();
    Optional<Tensor> biasesGradientGpu = convolution2dLayer->getBiasesGradient();

    if (biasesCpu.isPresent() && accumulate) {
        biasesGradientCpu.get().copyFromAsync(biasesGradientGpu, stream);
    }

    // Launch backward pass
    errorInputGpu.copyFromAsync(errorInputCpu, stream);
    convolution2dLayer->backward(errorInputGpu);
    errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);
    stream.synchronize();

    // Backward Data
    ConvolutionTestHelper::cpuConvolutionBackwardData(errorInputCpu, weightsCpu, errorOutputCpu, convolutionKernelRequirement);

    const float thresh = std::max(batchSize * (filterWidth * 0.02 + filterHeight * 0.02), 1.01);

    // Verify CPU and GPU results match
    vector<unsigned long> errorOutputDimensions = errorOutputGpu.getDescriptor().getDimensions();
    for (unsigned int n = 0; n < errorOutputDimensions[0]; ++n) {
        for (unsigned int c = 0; c < errorOutputDimensions[1]; ++c) {
            for (unsigned int h = 0; h < errorOutputDimensions[2]; ++h) {
                for (unsigned int w = 0; w < errorOutputDimensions[3]; ++w) {
                    float cpuVal = *(half *)errorOutputCpu.getElement({(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                    float gpuVal = *(half *)errorOutputGpu_h.getElement({(uint64_t)n, (uint64_t)c, (uint64_t)h, (uint64_t)w});
                    int threshAdjust = abs(cpuVal) > 300.0f ? 3 : 0;
                    EXPECT_LT(abs(cpuVal - gpuVal), thresh + threshAdjust);
                    if (abs(cpuVal - gpuVal) >= thresh + threshAdjust)
                        printf("%f %f   at [%d, %d, %d, %d]\n", cpuVal, gpuVal, n, c, h, w);
                }
            }
        }
    }

    // Backward Filter
    ConvolutionTestHelper::cpuConvolutionBackwardFilter(
        featureInputCpu, errorInputCpu, weightsGradientCpu, convolutionKernelRequirement, accumulate);
    Tensor weightsGradientGpu_h = Tensor(cpuPlacement, weightsGradientGpu.getDescriptor());
    weightsGradientGpu_h.copyFromAsync(weightsGradientGpu, convolution2dLayer->getGradientUpdateStream());
    convolution2dLayer->getGradientUpdateStream().synchronize();

    // printf("cpu after %f gpu after %f\n", (float)*(half*)weightsGradientCpu.getElement({0, 0, 0, 0}),
    // (float)*(half*)weightsGradientGpu_h.getElement({0, 0, 0, 0}));

    for (int o = 0; o < numOutputChannels; ++o) {
        for (int i = 0; i < numInputChannels; ++i) {
            for (int h = 0; h < filterHeight; ++h) {
                for (int w = 0; w < filterWidth; ++w) {
                    float cpuGradient = *(half *)weightsGradientCpu.getElement({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                    // float cpuWeight = *(half *)weightsCpu.getElement({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                    float cpuVal = cpuGradient;
                    float gpuVal = *(half *)weightsGradientGpu_h.getElement({(uint64_t)o, (uint64_t)i, (uint64_t)h, (uint64_t)w});
                    int threshAdjust = abs(cpuVal) > 300.0f ? 3 : 0;
                    if (abs(cpuVal - gpuVal) >= thresh + threshAdjust) {
                        printf("%f %f   at [%d, %d, %d, %d]\n", cpuVal, gpuVal, o, i, h, w);
                        int inputImageHeight = featureInputCpu.getDescriptor().getDimensions()[2];
                        int inputImageWidth = featureInputCpu.getDescriptor().getDimensions()[3];
                        Tensor featureOutput = convolution2dLayer->getFeatureOutputs()[0];
                        int outputImageHeight = featureOutput.getDescriptor().getDimensions()[2];
                        int outputImageWidth = featureOutput.getDescriptor().getDimensions()[3];
                        printf(
                            "accumulate %d inputImageHeight %d inputImageWidth %d outputImageHeight %d outputImageWidth %d filterHeight %d "
                            "filterWidth %d verticalStride %d horizontalStride %d verticalPadding %d horizontalPadding %d numWeights %d "
                            "numInputChannels %d numOutputChannels %d batchSize %d\n",
                            accumulate,
                            inputImageHeight,
                            inputImageWidth,
                            outputImageHeight,
                            outputImageWidth,
                            filterHeight,
                            filterWidth,
                            convolutionKernelRequirement.getFilterVerticalStride(),
                            convolutionKernelRequirement.getFilterHorizontalStride(),
                            convolutionKernelRequirement.getTopAndBottomPadHeight(),
                            convolutionKernelRequirement.getLeftAndRightPadWidth(),
                            numWeights,
                            numInputChannels,
                            numOutputChannels,
                            batchSize);

                        printf("\n");
                        convolution2dLayer->printBackwardFilterKernelInfo();
                        printf("\n");
                    }
                    EXPECT_LT(abs(cpuVal - gpuVal), thresh + threshAdjust);
                }
            }
        }
    }

    // Backward Bias
    if (biasesCpu.isPresent()) {
        ConvolutionTestHelper::cpuConvolutionBackwardBias(errorInputCpu, biasesGradientCpu, accumulate);

        Tensor biasesGradientGpu_h = Tensor(cpuPlacement, biasesGradientGpu.get().getDescriptor());
        biasesGradientGpu_h.copyFromAsync(biasesGradientGpu, convolution2dLayer->getGradientUpdateStream());
        convolution2dLayer->getGradientUpdateStream().synchronize();

        for (int i = 0; i < numOutputChannels; ++i) {
            float cpuGradient = *(half *)biasesGradientCpu.get().getElement({(uint64_t)i});
            // float cpuBias = *(half *)biasesCpu.get().getElement({(uint64_t)i});
            float cpuVal = cpuGradient;
            float gpuVal = *(half *)biasesGradientGpu_h.getElement({(uint64_t)i});
            int threshAdjust = abs(cpuVal) > 300.0f ? 3 : 0;
            EXPECT_LT(abs(cpuVal - gpuVal), thresh + threshAdjust);
            if (abs(cpuVal - gpuVal) >= thresh + threshAdjust)
                printf("%f %f   at [%d] batchSize %d thresh %f\n", cpuVal, gpuVal, i, batchSize, thresh + threshAdjust);
        }
    }
}

TEST(Convolution2dInitializers, UniformRandomInitializerWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 5; ++test) {
        const int numInputColumns = 1 + (rand() % 50);
        const int numInputRows = 1 + (rand() % 50);
        const int filterHorizontalStride = numInputColumns == 1 ? 1 : 1 + (rand() % (numInputColumns - 1));
        const int filterVerticalStride = numInputRows == 1 ? 1 : 1 + (rand() % (numInputRows - 1));
        const int filterWidth = numInputColumns == 1 ? 1 : 1 + (rand() % (numInputColumns - 1));
        const int filterHeight = numInputRows == 1 ? 1 : 1 + (rand() % (numInputRows - 1));
        const int leftAndRightPadWidth = filterWidth < 10 ? rand() % filterWidth : rand() % 10;
        const int topAndBottomPadHeight = filterHeight < 10 ? rand() % filterHeight : rand() % 10;
        const int numInputChannels = 1 + (rand() % 10);
        const int numOutputChannels = 1 + (rand() % 10);
        const int batchSize = 1 + (rand() % 10);
        const bool hasBias = (rand() % 4) != 0;

        vector<unsigned long> inputDimensions;
        inputDimensions.push_back(batchSize);
        inputDimensions.push_back(numInputChannels);
        inputDimensions.push_back(numInputRows);
        inputDimensions.push_back(numInputColumns);
        TensorDescriptor inputDescriptor(TensorDescriptor::DataType::FP16, inputDimensions);

        Tensor featureIn = Tensor(cpuPlacement, inputDescriptor);

        vector<Layer *> layers;
        layers.push_back(new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
        layers.push_back(new NoOpLayer());
        Convolution2d *convolution2dLayer = new Convolution2d(filterWidth,
                                                              filterHeight,
                                                              filterHorizontalStride,
                                                              filterVerticalStride,
                                                              leftAndRightPadWidth,
                                                              topAndBottomPadHeight,
                                                              numOutputChannels,
                                                              hasBias);
        layers.push_back(convolution2dLayer);
        layers.push_back(new NoOpLayer());
        layers.push_back(new NetworkOutput(cpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        UniformRandomInitializer initializer(0.1, -0.1);
        initializer.initialize(convolution2dLayer, convolution2dLayer->getWeights());
        if (hasBias)
            initializer.initialize(convolution2dLayer, convolution2dLayer->getBiases());

        Tensor weights = convolution2dLayer->getWeights().clone(cpuPlacement);
        weights.copyFromAsync(convolution2dLayer->getWeights(), stream);
        Tensor biases;
        if (hasBias) {
            biases = convolution2dLayer->getBiases().get().clone(cpuPlacement);
            biases.copyFromAsync(convolution2dLayer->getBiases(), stream);
        }

        stream.synchronize();

        int totalNumWeights = convolution2dLayer->getWeights().getDescriptor().getTotalNumElements();
        half *weightsMem = (half *)weights.getMemPtr();
        for (int i = 0; i < totalNumWeights; ++i) {
            ASSERT_LT(abs((float)weightsMem[i]), 0.1);
        }

        if (hasBias) {
            int totalNumBiases = convolution2dLayer->getBiases().get().getDescriptor().getTotalNumElements();
            half *biasesMem = (half *)biases.getMemPtr();
            for (int i = 0; i < totalNumBiases; ++i) {
                ASSERT_LT(abs((float)biasesMem[i]), 0.1);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
