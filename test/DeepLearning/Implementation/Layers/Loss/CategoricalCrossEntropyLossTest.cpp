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

using std::set;
using std::vector;

using namespace ThorImplementation;

TEST(CategoricalCrossEntropyLoss, ComputesCorrectElementWiseResult) {
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
        int numElementsPerBatch = dimensions[1];

        bool inferenceOnly = (rand() % 5) == 0;

        TensorDescriptor elementwiseDescriptorFP32(TensorDescriptor::DataType::FP32, dimensions);
        TensorDescriptor elementwiseDescriptorFP16(TensorDescriptor::DataType::FP16, dimensions);

        Tensor labelsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor activationsCpu(cpuPlacement, elementwiseDescriptorFP16);
        Tensor labelsGpu(gpuPlacement, elementwiseDescriptorFP32);
        Tensor activationsGpu(gpuPlacement, elementwiseDescriptorFP16);

        vector<unsigned long> batchDimensions;
        batchDimensions.push_back(batchSize);
        TensorDescriptor batchwiseDescriptor(TensorDescriptor::DataType::FP32, batchDimensions);
        Tensor lossCpu(cpuPlacement, batchwiseDescriptor);
        Tensor lossGpu_h(cpuPlacement, batchwiseDescriptor);

        float *labels = (float *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (int i = 0; i < numElements; ++i) {
                activations[i] = ((rand() % 10000) / 999.0f);
                totalActivations += activations[i];
                labels[i] = ((rand() % 1000) / 999.0f);
            }
        }

        uint32_t lossScalingFactor = 1;
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

        labelsGpu.copyFromAsync(labelsCpu, labelsStream);
        activationsGpu.copyFromAsync(activationsCpu, stream);

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, predictionsOutput, (int)Loss::ConnectionType::PREDICTIONS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, lossOutput, (int)Loss::ConnectionType::ELEMENTWISE_LOSS);
        LayerTestHelper::initializeNetwork(layers);

        if (inferenceOnly) {
            ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorOutput().isEmpty());
        }
        ASSERT_TRUE(categoricalCrossEntropyLoss->getErrorInput().isEmpty());

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsGpu, false);
        labelsInput->forward(labelsGpu, false);

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
        float *lossMem = (float *)lossCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, batchwiseDescriptor);
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        Tensor exponentials(cpuPlacement, elementwiseDescriptorFP32);
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (int i = 0; i < numElementsPerBatch; ++i) {
                exponentialsMem[b * numElementsPerBatch + i] = exp((float)activationsMem[b * numElementsPerBatch + i]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numElementsPerBatch + i];
            }
        }
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i) {
                exponentialsMem[b * numElementsPerBatch + i] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numElementsPerBatch + i] < 1.0e-15f || !isfinite(exponentialsMem[b * numElementsPerBatch + i]))
                    exponentialsMem[b * numElementsPerBatch + i] = 1.0e-15f;
            }
        }

        for (int b = 0; b < batchSize; ++b) {
            lossMem[b] = 0.0f;
            for (int i = 0; i < numElementsPerBatch; ++i) {
                lossMem[b] -= labelsMem[b * numElementsPerBatch + i] * log(exponentialsMem[b * numElementsPerBatch + i]);
            }
        }

        // Verify the softmax output (predictions)
        float thresh = 0.01f;
        float *predictionsGpuMem = (float *)predictionsGpu_h.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            EXPECT_LT(abs(exponentialsMem[i] - predictionsGpuMem[i]), thresh);
            if (abs(exponentialsMem[i] - predictionsGpuMem[i]) > thresh)
                printf("%d   cpu %f gpu %f\n", i, exponentialsMem[i], predictionsGpuMem[i]);
        }

        // Verify the loss output
        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            float thresh = std::max(lossMem[b] / 320000.0f, 0.001f);
            EXPECT_LT(abs(lossMem[b] - lossMemFromGpu[b]), thresh);
            if (abs(lossMem[b] - lossMemFromGpu[b]) >= thresh)
                printf("cpuF %f gpuF %f    %d\n", lossMem[b], lossMemFromGpu[b], b);
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        stream.synchronize();

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i) {
                errorOutputMem[b * numElementsPerBatch + i] =
                    (lossScalingFactor * (exponentialsMem[b * numElementsPerBatch + i] - labelsMem[b * numElementsPerBatch + i]));
            }
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        thresh = 0.01f;
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

TEST(CategoricalCrossEntropyLoss, ComputesCorrectBatchResult) {
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
        int numElementsPerBatch = dimensions[1];

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
                activations[i] = ((rand() % 10000) / 999.0f);
                totalActivations += activations[i];
                labels[i] = ((rand() % 1000) / 999.0f);
            }
        }

        uint32_t lossScalingFactor = 1;
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

        labelsGpu.copyFromAsync(labelsCpu, labelsStream);
        activationsGpu.copyFromAsync(activationsCpu, stream);

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
        activationsInput->forward(activationsGpu, false);
        labelsInput->forward(labelsGpu, false);

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
            for (int i = 0; i < numElementsPerBatch; ++i) {
                exponentialsMem[b * numElementsPerBatch + i] = exp((float)activationsMem[b * numElementsPerBatch + i]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numElementsPerBatch + i];
            }
        }
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i) {
                exponentialsMem[b * numElementsPerBatch + i] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numElementsPerBatch + i] < 1.0e-15f || !isfinite(exponentialsMem[b * numElementsPerBatch + i]))
                    exponentialsMem[b * numElementsPerBatch + i] = 1.0e-15f;
            }
        }

        double batchLoss = 0.0;
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i) {
                batchLoss -= labelsMem[b * numElementsPerBatch + i] * log(exponentialsMem[b * numElementsPerBatch + i]);
            }
        }

        // Verify the softmax output (predictions)
        float thresh = 0.01f;
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
            for (int i = 0; i < numElementsPerBatch; ++i) {
                errorOutputMem[b * numElementsPerBatch + i] =
                    (lossScalingFactor * (exponentialsMem[b * numElementsPerBatch + i] - labelsMem[b * numElementsPerBatch + i]));
            }
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        thresh = 0.01f;
        for (int i = 0; i < numElements; ++i) {
            ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
            if (abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]) >= thresh) {
                printf("cpu %f gpu %f\n", (float)errorOutputMem[i], (float)errorOutputFromGpu[i]);
                fflush(stdout);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
