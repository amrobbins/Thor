#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "MLDev.h"

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

TEST(CategoricalCrossEntropyLoss, ComputesCorrectResult) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 50; ++test) {
        int numDimensions = (rand() % 5) + 2;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 5) + 1);
            numElements *= dimensions.back();
        }
        int batchSize = dimensions.front();
        int numElementsPerBatch = numElements / batchSize;

        bool inferenceOnly = false;  // FIXME: (rand() % 5) == 0;

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

        Stream stream(0);
        Stream labelsStream(0);

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
        labelsGpu.copyFromAsync(labelsCpu, stream);
        activationsGpu.copyFromAsync(activationsCpu, stream);

        float lossScalingFactor = (rand() % 1000) / 100;
        if (rand() % 5)
            lossScalingFactor = 1.0f;

        vector<Layer *> layers;
        NetworkInput *activationsInput = new NetworkInput(activationsGpu, stream);
        layers.push_back(activationsInput);
        NoOpLayer *noOpLayer = new NoOpLayer();
        layers.push_back(noOpLayer);
        NetworkInput *labelsInput = new NetworkInput(labelsGpu, labelsStream);
        layers.push_back(labelsInput);
        CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss(lossScalingFactor);
        if (inferenceOnly)
            categoricalCrossEntropyLoss->setInferenceOnly(true);
        layers.push_back(categoricalCrossEntropyLoss);
        NetworkOutput *predictionsOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(predictionsOutput);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalCrossEntropyLoss, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalCrossEntropyLoss, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, predictionsOutput, (int)Loss::ConnectionType::PREDICTIONS);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, lossOutput, (int)Loss::ConnectionType::LOSS);
        LayerTestHelper::initializeNetwork(layers);

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsGpu);
        labelsInput->forward(labelsGpu);

        labelsStream.waitEvent(lossOutput->getOutputReadyEvent());
        lossGpu_h.copyFromAsync(outputGpu, labelsStream);
        Tensor predictionsGpu_h = predictionsOutput->getFeatureOutput().get().clone(cpuPlacement);
        labelsStream.waitEvent(predictionsOutput->getOutputReadyEvent());
        predictionsGpu_h.copyFromAsync(predictionsOutput->getFeatureOutput(), labelsStream);

        Tensor errorOutputGpu = categoricalCrossEntropyLoss->getErrorOutput();
        Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
        Tensor errorOutputGpu_h = errorOutputCpu.clone();
        categoricalCrossEntropyLoss->backward(Optional<Tensor>::empty());
        errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
