#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "Thor.h"

#include <math.h>
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

TEST(CategoricalCrossEntropyLoss, ComputesCorrectElementWiseResult_perClassLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 10; ++test) {
        vector<unsigned long> dimensions;
        int numElements = 1;
        int batchSize = (rand() % 30) + 1;
        int numClasses = (rand() % 300) + 1;
        numElements = batchSize * numClasses;
        dimensions.push_back(batchSize);
        dimensions.push_back(numClasses);

        bool inferenceOnly = (rand() % 5) == 0;

        TensorDescriptor predictionsDescriptor(TensorDescriptor::DataType::FP16, dimensions);
        TensorDescriptor labelsDescriptor(TensorDescriptor::DataType::FP32, dimensions);
        TensorDescriptor lossDescriptor(TensorDescriptor::DataType::FP32, dimensions);

        Tensor labelsCpu(cpuPlacement, labelsDescriptor);
        Tensor activationsCpu(cpuPlacement, predictionsDescriptor);
        Tensor labelsGpu(gpuPlacement, labelsDescriptor);
        Tensor activationsGpu(gpuPlacement, predictionsDescriptor);

        vector<unsigned long> batchDimensions;
        batchDimensions.push_back(batchSize);
        Tensor lossCpu(cpuPlacement, lossDescriptor);
        Tensor lossGpu_h(cpuPlacement, lossDescriptor);

        float *labels = (float *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (int b = 0; b < batchSize; ++b) {
                for (int c = 0; c < numClasses; ++c) {
                    int i = b * numClasses + c;
                    activations[i] = ((rand() % 1500) / 999.0f);
                    totalActivations += activations[i];
                }
            }
        }

        for (int b = 0; b < batchSize; ++b) {
            for (int c = 0; c < numClasses; ++c) {
                labels[b * numClasses + c] = 0.0f;
            }
            uint32_t trueClass = rand() % numClasses;
            labels[b * numClasses + trueClass] = 1.0f;
        }

        vector<Layer *> layers;
        NetworkInput *activationsInput = new NetworkInput(activationsGpu);
        activationsInput->setName("activations");
        layers.push_back(activationsInput);
        NoOpLayer *noOpLayer = new NoOpLayer();
        layers.push_back(noOpLayer);
        NetworkInput *labelsInput = new NetworkInput(labelsGpu);
        labelsInput->setName("Labels");
        layers.push_back(labelsInput);
        Softmax *softmax = new Softmax(true);
        layers.push_back(softmax);
        TensorFanout *tensorFanout = new TensorFanout();
        layers.push_back(tensorFanout);
        CrossEntropy *crossEntropy = new CrossEntropy(false);
        if (inferenceOnly)
            crossEntropy->setConstructForInferenceOnly(true);
        layers.push_back(crossEntropy);
        NetworkOutput *predictionsOutput = new NetworkOutput(gpuPlacement);
        predictionsOutput->setName("predictions");
        layers.push_back(predictionsOutput);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        lossOutput->setName("loss");
        layers.push_back(lossOutput);

        Stream stream = activationsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        // noOpLayer is needed so that back prop path is not pruned
        LayerTestHelper::connectTwoLayers(noOpLayer, softmax);
        // Since losses have multiple input connection types, the type needs to be specified when connecting to CrossEntropy
        LayerTestHelper::connectTwoLayers(softmax, tensorFanout);
        LayerTestHelper::connectTwoLayers(tensorFanout, predictionsOutput);
        LayerTestHelper::connectTwoLayers(tensorFanout, crossEntropy, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, crossEntropy, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(crossEntropy, lossOutput, (int)Loss::ConnectionType::LOSS);
        LayerTestHelper::initializeNetwork(layers);

        if (inferenceOnly) {
            ASSERT_TRUE(crossEntropy->getErrorOutput().isEmpty());
        }
        ASSERT_TRUE(crossEntropy->getErrorInput().isEmpty());

        Tensor outputGpu = lossOutput->getFeatureOutput();
        assert(outputGpu.getDataType() == TensorDescriptor::DataType::FP16);

        // Network is runnable here
        activationsInput->forward(activationsCpu, false);
        labelsInput->forward(labelsCpu, false);

        labelsStream.waitEvent(lossOutput->getOutputReadyEvent());
        assert(lossGpu_h.getDataType() == TensorDescriptor::DataType::FP32);
        lossGpu_h.copyFromAsync(outputGpu, labelsStream);
        Tensor predictionsGpu_h = predictionsOutput->getFeatureOutput().get().clone(cpuPlacement);
        labelsStream.waitEvent(predictionsOutput->getOutputReadyEvent());
        predictionsGpu_h.copyFromAsync(predictionsOutput->getFeatureOutput(), labelsStream);

        labelsStream.synchronize();

        Tensor errorOutputGpu;
        Tensor errorOutputCpu;
        Tensor errorOutputGpu_h;
        if (!inferenceOnly) {
            errorOutputGpu = softmax->getErrorOutput();
            errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
            errorOutputGpu_h = errorOutputCpu.clone();
            labelsStream.waitEvent(stream.putEvent());
            errorOutputGpu_h.copyFromAsync(errorOutputGpu, labelsStream);
            labelsStream.synchronize();
        }

        // Compute the expected loss
        float *labelsMem = (float *)labelsCpu.getMemPtr();
        half *activationsMem = (half *)activationsCpu.getMemPtr();
        float *lossMem = (float *)lossCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, lossDescriptor);
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        TensorDescriptor exponentialsDescriptor(TensorDescriptor::DataType::FP32, dimensions);
        Tensor exponentials(cpuPlacement, exponentialsDescriptor);
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (int i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] = exp((float)activationsMem[b * numClasses + i]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numClasses + i];
            }
        }
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numClasses + i] < 0.001f || !isfinite(exponentialsMem[b * numClasses + i]))
                    exponentialsMem[b * numClasses + i] = 0.001f;
            }
        }

        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numClasses; ++i) {
                lossMem[b * numClasses + i] = labelsMem[b * numClasses + i] * log(exponentialsMem[b * numClasses + i]) * -1.0f;
            }
        }

        // Verify the softmax output (predictions)
        float thresh = 0.001f;
        half *predictionsGpuMem = (half *)predictionsGpu_h.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            if (abs(exponentialsMem[i] - predictionsGpuMem[i]) >= thresh)
                printf("%d   cpu %f gpu %f [%d, %d] of [%d, %d]\n",
                       i,
                       exponentialsMem[i],
                       (float)predictionsGpuMem[i],
                       i / batchSize,
                       i % batchSize,
                       batchSize,
                       numClasses);
            ASSERT_LT(abs(exponentialsMem[i] - (float)predictionsGpuMem[i]), thresh);
        }

        // Verify the loss output
        ASSERT_EQ(lossGpu_h.getDataType(), TensorDescriptor::DataType::FP32);
        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        thresh = 0.01f;
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numClasses; ++i) {
                if (abs(lossMem[b * numClasses + i] - lossMemFromGpu[b * numClasses + i]) >= thresh)
                    printf("cpuLoss %f gpuLoss %f    %d\n", lossMem[b * numClasses + i], lossMemFromGpu[b * numClasses + i], b);
                ASSERT_LT(abs(lossMem[b * numClasses + i] - lossMemFromGpu[b * numClasses + i]), thresh);
            }
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass
        assert(Loss::getLossScalingFactor() > 0.0f);
        assert(Loss::getLossScalingFactor() < 1000000.0f);

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numClasses; ++i) {
                errorOutputMem[b * numClasses + i] =
                    (Loss::getLossScalingFactor() * (exponentialsMem[b * numClasses + i] - labelsMem[b * numClasses + i]));
            }
        }

        // Verify the loss gradient
        assert(errorOutputGpu.getDataType() == TensorDescriptor::DataType::FP16);
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        thresh = 0.1f;
        for (int i = 0; i < numElements; ++i) {
            if (abs((float)errorOutputMem[i] - (float)(errorOutputFromGpu[i])) >= thresh)
                printf("[%d,%d] cpuGradient %f gpuGradient %f   batchSize %d numClasses %d\n",
                       i / numClasses,
                       i % numClasses,
                       (float)errorOutputMem[i],
                       (float)errorOutputFromGpu[i],
                       batchSize,
                       numClasses);
            ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
        }
        fflush(stdout);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

/*
TEST(CategoricalCrossEntropyLoss, ComputesCorrectBatchResult_perClassLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 20; ++test) {
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < 2; ++i) {
            dimensions.push_back((rand() % 300) + 1);
            if (i == 1)
                dimensions.back() += 1;
            numElements *= dimensions.back();
        }
        int batchSize = dimensions[0];
        int numClasses = dimensions[1];

        bool inferenceOnly = (rand() % 5) == 0;

        TensorDescriptor elementwiseDescriptorFP32(TensorDescriptor::DataType::FP32, dimensions);
        TensorDescriptor elementwiseDescriptorFP16(TensorDescriptor::DataType::FP16, dimensions);

        Tensor labelsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor activationsCpu(cpuPlacement, elementwiseDescriptorFP16);
        Tensor labelsGpu(gpuPlacement, elementwiseDescriptorFP32);
        Tensor activationsGpu(gpuPlacement, elementwiseDescriptorFP16);

        vector<unsigned long> batchDimensions;
        TensorDescriptor batchwiseDescriptor(TensorDescriptor::DataType::FP32, {1});
        Tensor lossGpu_h(cpuPlacement, batchwiseDescriptor);

        float *labels = (float *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (int i = 0; i < numElements; ++i) {
                activations[i] = ((rand() % 1500) / 999.0f);
                totalActivations += activations[i];
                labels[i] = ((rand() % 1500) / 999.0f);
            }
        }

        vector<Layer *> layers;
        NetworkInput *activationsInput = new NetworkInput(activationsGpu);
        layers.push_back(activationsInput);
        NoOpLayer *noOpLayer = new NoOpLayer();
        layers.push_back(noOpLayer);
        NetworkInput *labelsInput = new NetworkInput(labelsGpu);
        layers.push_back(labelsInput);
        CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss();
        if (inferenceOnly)
            categoricalCrossEntropyLoss->setConstructForInferenceOnly(true);
        layers.push_back(categoricalCrossEntropyLoss);
        NetworkOutput *predictionsOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(predictionsOutput);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = activationsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, predictionsOutput, (int)Loss::ConnectionType::PREDICTIONS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, lossOutput, (int)Loss::ConnectionType::BATCH_LOSS);
        LayerTestHelper::initializeNetwork(layers);

        if (inferenceOnly) {
            ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorOutput().isEmpty());
        }
        ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorInput().isEmpty());

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsCpu, false);
        labelsInput->forward(labelsCpu, false);

        labelsStream.waitEvent(lossOutput->getOutputReadyEvent());
        lossGpu_h.copyFromAsync(outputGpu, labelsStream);
        Tensor predictionsGpu_h = predictionsOutput->getFeatureOutput().get().clone(cpuPlacement);
        labelsStream.waitEvent(predictionsOutput->getOutputReadyEvent());
        predictionsGpu_h.copyFromAsync(predictionsOutput->getFeatureOutput(), labelsStream);

        Tensor errorOutputGpu;
        Tensor errorOutputCpu;
        Tensor errorOutputGpu_h;
        if (!inferenceOnly) {
            errorOutputGpu = categoricalCrossEntropyLoss->getErrorOutput();
            errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
            errorOutputGpu_h = errorOutputCpu.clone();
            errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);
        }

        labelsStream.synchronize();

        // Compute the expected loss
        float *labelsMem = (float *)labelsCpu.getMemPtr();
        half *activationsMem = (half *)activationsCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {(uint64_t)batchSize}));
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        Tensor exponentials(cpuPlacement, elementwiseDescriptorFP32);
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (int i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] = exp((float)activationsMem[b * numClasses + i]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numClasses + i];
            }
        }
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numClasses + i] < 0.001f || !isfinite(exponentialsMem[b * numClasses + i]))
                    exponentialsMem[b * numClasses + i] = 0.001f;
            }
        }

        double batchLoss = 0.0;
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numClasses; ++i) {
                batchLoss -= labelsMem[b * numClasses + i] * log(exponentialsMem[b * numClasses + i]);
            }
        }

        // Verify the softmax output (predictions)
        float thresh = 0.001f;
        float *predictionsGpuMem = (float *)predictionsGpu_h.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            if (abs(exponentialsMem[i] - predictionsGpuMem[i]) > thresh)
                printf("%d   cpu %f gpu %f\n", i, exponentialsMem[i], predictionsGpuMem[i]);
            ASSERT_LT(abs(exponentialsMem[i] - predictionsGpuMem[i]), thresh);
        }

        // Verify the loss output
        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        thresh = std::max((float)batchLoss / 320000.0f, 0.001f);
        ASSERT_LT(abs(batchLoss - lossMemFromGpu[0]), thresh);
        if (abs(batchLoss - *lossMemFromGpu) >= thresh)
            printf("cpuF %f gpuF %f   batch\n", batchLoss, lossMemFromGpu[0]);

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        stream.synchronize();

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numClasses; ++i) {
                errorOutputMem[b * numClasses + i] =
                    (Loss::getLossScalingFactor() *
                     (exponentialsMem[b * numClasses + i] - labelsMem[b * numClasses + i]));
            }
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        thresh = 0.1f;
        for (int i = 0; i < numElements; ++i) {
            EXPECT_LT(abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
            if (abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]) >= thresh) {
                printf("cpu %f gpu %f\n", (float)errorOutputMem[i], (float)errorOutputFromGpu[i]);
                fflush(stdout);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectElementWiseResult_classIndexLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        vector<unsigned long> dimensions;
        uint32_t numElements = 1;
        for (uint32_t i = 0; i < 2; ++i) {
            dimensions.push_back((rand() % 300) + 1);
            if (i == 1)
                dimensions.back() += 1;
            numElements *= dimensions.back();
        }
        uint32_t batchSize = dimensions[0];
        uint32_t numClasses = dimensions[1];

        bool inferenceOnly = (rand() % 5) == 0;

        Tensor labelsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT16, {batchSize, 1}));
        Tensor activationsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numClasses}));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor activationsGpu = activationsCpu.clone(gpuPlacement);
        Tensor lossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
        Tensor lossGpu_h = lossCpu.clone();

        uint16_t *labels = (uint16_t *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (uint32_t b = 0; b < batchSize; ++b) {
                labels[b] = rand() % numClasses;
                for (uint32_t c = 0; c < numClasses; ++c) {
                    activations[b * numClasses + c] = ((rand() % 1500) / 999.0f);
                    totalActivations += activations[b * numClasses + c];
                }
            }
        }

        vector<Layer *> layers;
        NetworkInput *activationsInput = new NetworkInput(activationsGpu);
        layers.push_back(activationsInput);
        NoOpLayer *noOpLayer = new NoOpLayer();
        layers.push_back(noOpLayer);
        NetworkInput *labelsInput = new NetworkInput(labelsGpu);
        layers.push_back(labelsInput);
        CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss();
        if (inferenceOnly)
            categoricalCrossEntropyLoss->setConstructForInferenceOnly(true);
        layers.push_back(categoricalCrossEntropyLoss);
        NetworkOutput *predictionsOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(predictionsOutput);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = activationsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, predictionsOutput, (int)Loss::ConnectionType::PREDICTIONS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, lossOutput, (int)Loss::ConnectionType::LOSS);
        LayerTestHelper::initializeNetwork(layers);

        if (inferenceOnly) {
            ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorOutput().isEmpty());
        }
        ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorInput().isEmpty());

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsCpu, false);
        labelsInput->forward(labelsCpu, false);

        labelsStream.waitEvent(lossOutput->getOutputReadyEvent());
        lossGpu_h.copyFromAsync(outputGpu, labelsStream);
        Tensor predictionsGpu_h = predictionsOutput->getFeatureOutput().get().clone(cpuPlacement);
        labelsStream.waitEvent(predictionsOutput->getOutputReadyEvent());
        predictionsGpu_h.copyFromAsync(predictionsOutput->getFeatureOutput(), labelsStream);

        Tensor errorOutputGpu;
        Tensor errorOutputCpu;
        Tensor errorOutputGpu_h;
        if (!inferenceOnly) {
            errorOutputGpu = categoricalCrossEntropyLoss->getErrorOutput();
            errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
            errorOutputGpu_h = errorOutputCpu.clone();
            errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);
        }

        labelsStream.synchronize();

        // Compute the expected loss
        half *activationsMem = (half *)activationsCpu.getMemPtr();
        float *lossMem = (float *)lossCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        Tensor exponentials(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (uint32_t i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] = exp((float)activationsMem[b * numClasses + i]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numClasses + i];
            }
        }
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numClasses + i] < 0.001f || !isfinite(exponentialsMem[b * numClasses + i]))
                    exponentialsMem[b * numClasses + i] = 0.001f;
            }
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            uint16_t label = labels[b];
            lossMem[b] = -log(exponentialsMem[b * numClasses + label]);
        }

        // Verify the softmax output (predictions)
        float thresh = 0.001f;
        float *predictionsGpuMem = (float *)predictionsGpu_h.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i) {
            if (abs(exponentialsMem[i] - predictionsGpuMem[i]) > thresh || !isfinite(exponentialsMem[i] - predictionsGpuMem[i]))
                printf("%d   cpu %f gpu %f\n", i, exponentialsMem[i], predictionsGpuMem[i]);
            ASSERT_LT(abs(exponentialsMem[i] - predictionsGpuMem[i]), thresh);
        }

        // Verify the loss output
        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            float thresh = std::max(lossMem[b] / 320000.0f, 0.001f);
            if (abs(lossMem[b] - lossMemFromGpu[b]) >= thresh)
                printf("cpuF %f gpuF %f    %d\n", lossMem[b], lossMemFromGpu[b], b);
            ASSERT_LT(abs(lossMem[b] - lossMemFromGpu[b]), thresh);
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        stream.synchronize();

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                errorOutputMem[b * numClasses + i] =
                    (Loss::getLossScalingFactor() * (exponentialsMem[b * numClasses + i] - (labels[b] == i ? 1.0f : 0.0f)));
            }
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        thresh = 0.1f;
        for (uint32_t i = 0; i < numElements; ++i) {
            if (abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]) >= thresh) {
                printf("cpu %f gpu %f   %d\n", (float)errorOutputMem[i], (float)errorOutputFromGpu[i], i);
                fflush(stdout);
            }
            ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectBatchResult_classIndexLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        vector<unsigned long> dimensions;
        uint32_t numElements = 1;
        for (uint32_t i = 0; i < 2; ++i) {
            dimensions.push_back((rand() % 300) + 1);
            if (i == 1)
                dimensions.back() += 1;
            numElements *= dimensions.back();
        }
        uint32_t batchSize = dimensions[0];
        uint32_t numClasses = dimensions[1];

        bool inferenceOnly = (rand() % 5) == 0;

        Tensor labelsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT16, {batchSize, 1}));
        Tensor activationsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numClasses}));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor activationsGpu = activationsCpu.clone(gpuPlacement);
        Tensor lossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor lossGpu_h = lossCpu.clone();

        uint16_t *labels = (uint16_t *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (uint32_t b = 0; b < batchSize; ++b) {
                labels[b] = rand() % numClasses;
                for (uint32_t c = 0; c < numClasses; ++c) {
                    activations[b * numClasses + c] = ((rand() % 1500) / 999.0f);
                    totalActivations += activations[b * numClasses + c];
                }
            }
        }

        vector<Layer *> layers;
        NetworkInput *activationsInput = new NetworkInput(activationsGpu);
        layers.push_back(activationsInput);
        NoOpLayer *noOpLayer = new NoOpLayer();
        layers.push_back(noOpLayer);
        NetworkInput *labelsInput = new NetworkInput(labelsGpu);
        layers.push_back(labelsInput);
        CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss();
        if (inferenceOnly)
            categoricalCrossEntropyLoss->setConstructForInferenceOnly(true);
        layers.push_back(categoricalCrossEntropyLoss);
        NetworkOutput *predictionsOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(predictionsOutput);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = activationsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, predictionsOutput, (int)Loss::ConnectionType::PREDICTIONS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, lossOutput, (int)Loss::ConnectionType::BATCH_LOSS);
        LayerTestHelper::initializeNetwork(layers);

        if (inferenceOnly) {
            ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorOutput().isEmpty());
        }
        ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorInput().isEmpty());

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsCpu, false);
        labelsInput->forward(labelsCpu, false);

        labelsStream.waitEvent(lossOutput->getOutputReadyEvent());
        lossGpu_h.copyFromAsync(outputGpu, labelsStream);
        Tensor predictionsGpu_h = predictionsOutput->getFeatureOutput().get().clone(cpuPlacement);
        labelsStream.waitEvent(predictionsOutput->getOutputReadyEvent());
        predictionsGpu_h.copyFromAsync(predictionsOutput->getFeatureOutput(), labelsStream);

        Tensor errorOutputGpu;
        Tensor errorOutputCpu;
        Tensor errorOutputGpu_h;
        if (!inferenceOnly) {
            errorOutputGpu = categoricalCrossEntropyLoss->getErrorOutput();
            errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
            errorOutputGpu_h = errorOutputCpu.clone();
            errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);
        }

        labelsStream.synchronize();

        // Compute the expected loss
        half *activationsMem = (half *)activationsCpu.getMemPtr();
        float *lossMem = (float *)lossCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        Tensor exponentials(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (uint32_t i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] = exp((float)activationsMem[b * numClasses + i]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numClasses + i];
            }
        }
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                exponentialsMem[b * numClasses + i] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numClasses + i] < 0.001f || !isfinite(exponentialsMem[b * numClasses + i]))
                    exponentialsMem[b * numClasses + i] = 0.001f;
            }
        }

        lossMem[0] = 0.0f;
        for (uint32_t b = 0; b < batchSize; ++b) {
            uint16_t label = labels[b];
            lossMem[0] -= log(exponentialsMem[b * numClasses + label]);
        }

        // Verify the softmax output (predictions)
        float thresh = 0.001f;
        float *predictionsGpuMem = (float *)predictionsGpu_h.getMemPtr();
        for (uint32_t i = 0; i < numElements; ++i) {
            if (abs(exponentialsMem[i] - predictionsGpuMem[i]) > thresh)
                printf("%d   cpu %f gpu %f\n", i, exponentialsMem[i], predictionsGpuMem[i]);
            ASSERT_LT(abs(exponentialsMem[i] - predictionsGpuMem[i]), thresh);
        }

        // Verify the loss output
        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        thresh = std::max(lossMem[0] / 32000.0f, 0.01f);
        if (abs(lossMem[0] - lossMemFromGpu[0]) >= thresh)
            printf("cpuF %f gpuF %f    %d\n", lossMem[0], lossMemFromGpu[0], 0);
        ASSERT_LT(abs(lossMem[0] - lossMemFromGpu[0]), thresh);

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        stream.synchronize();

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                errorOutputMem[b * numClasses + i] =
                    (Loss::getLossScalingFactor() * (exponentialsMem[b * numClasses + i] - (labels[b] == i ? 1.0f : 0.0f)));
            }
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        thresh = 0.1f;
        for (uint32_t i = 0; i < numElements; ++i) {
            if (abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]) >= thresh) {
                printf("cpu %f gpu %f   %d\n", (float)errorOutputMem[i], (float)errorOutputFromGpu[i], i);
                fflush(stdout);
            }
            ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
