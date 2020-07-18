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

TEST(CategoricalCrossEntropyLoss, ComputesCorrectResultFp32) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 25; ++test) {
        int numDimensions = (rand() % 5) + 2;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 5) + 1);
            numElements *= dimensions.back();
        }
        int batchSize = dimensions.front();
        int numElementsPerBatch = numElements / batchSize;

        TensorDescriptor elementwiseDescriptor(TensorDescriptor::DataType::FP32, dimensions);
        Tensor labelsCpu(cpuPlacement, elementwiseDescriptor);
        Tensor activationsCpu(cpuPlacement, elementwiseDescriptor);
        Tensor labelsGpu(gpuPlacement, elementwiseDescriptor);
        Tensor activationsGpu(gpuPlacement, elementwiseDescriptor);

        vector<unsigned long> batchDimensions;
        batchDimensions.push_back(batchSize);
        TensorDescriptor batchwiseDescriptor(TensorDescriptor::DataType::FP32, batchDimensions);
        Tensor lossCpu(cpuPlacement, batchwiseDescriptor);
        Tensor lossGpu_h(cpuPlacement, batchwiseDescriptor);

        Stream stream(0);

        float *labels = (float *)labelsCpu.getMemPtr();
        float *activations = (float *)activationsCpu.getMemPtr();
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
        NetworkInput *labelsInput = new NetworkInput(labelsGpu, stream);
        layers.push_back(labelsInput);
        CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss(lossScalingFactor);
        layers.push_back(categoricalCrossEntropyLoss);
        NetworkOutput *lossOutput = new NetworkOutput();
        layers.push_back(lossOutput);

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalCrossEntropyLoss);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalCrossEntropyLoss);
        LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, lossOutput);
        LayerTestHelper::initializeNetwork(layers);

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsGpu);
        labelsInput->forward(labelsGpu);
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        lossGpu_h.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        // Compute the expected loss
        float *labelsMem = (float *)labelsCpu.getMemPtr();
        float *activationsMem = (float *)activationsCpu.getMemPtr();
        float *lossMem = (float *)lossCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, batchwiseDescriptor);
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        Tensor exponentials(cpuPlacement, elementwiseDescriptor);
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (int i = 0; i < numElementsPerBatch; ++i) {
                exponentialsMem[b * numElementsPerBatch + i] = exp(activationsMem[b * numElementsPerBatch + i]);
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

        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            float thresh = std::max(lossMem[b] / 320000.0f, 0.001f);
            EXPECT_LT(abs(lossMem[b] - lossMemFromGpu[b]), thresh);
            if (abs(lossMem[b] - lossMemFromGpu[b]) >= thresh)
                printf("cpuF %f gpuF %f    %d\n", lossMem[b], lossMemFromGpu[b], b);
        }

        // Backward pass
        Tensor errorOutputGpu = categoricalCrossEntropyLoss->getErrorOutput();
        Tensor errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
        Tensor errorOutputGpu_h = errorOutputCpu.clone();

        categoricalCrossEntropyLoss->backward(Optional<Tensor>::empty());
        errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);

        float *errorOutputMem = (float *)errorOutputCpu.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i) {
                errorOutputMem[b * numElementsPerBatch + i] = (float)(half)(
                    lossScalingFactor * (exponentialsMem[b * numElementsPerBatch + i] - labelsMem[b * numElementsPerBatch + i]));
            }
        }

        stream.synchronize();

        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        float thresh = 0.01f;
        for (int i = 0; i < numElements; ++i) {
            EXPECT_LT(abs(errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
            if (abs(errorOutputMem[i] - (float)errorOutputFromGpu[i]) >= thresh)
                printf("cpu %f gpu %f\n", errorOutputMem[i], (float)errorOutputFromGpu[i]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
