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

TEST(MeanSquaredError, ComputesCorrectResult_BatchLoss) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        vector<unsigned long> dimensions;
        uint32_t numElements = 1;
        for (uint32_t i = 0; i < 2; ++i) {
            dimensions.push_back((rand() % 500) + 1);
            numElements *= dimensions.back();
        }
        Tensor labelsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor predictionsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor predictionsGpu = predictionsCpu.clone(gpuPlacement);
        Tensor elementLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor elementLossGpu = elementLossCpu.clone(gpuPlacement);
        Tensor elementLossGradientCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor elementLossGradientGpu = elementLossGradientCpu.clone(gpuPlacement);

        half *labels = (half *)labelsCpu.getMemPtr();
        half *predictions = (half *)predictionsCpu.getMemPtr();
        half *elementLoss = (half *)elementLossCpu.getMemPtr();
        half *elementLossGradient = (half *)elementLossGradientCpu.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); i++) {
            predictions[i] = ((rand() % 1500) / 999.0f);
            labels[i] = ((rand() % 1500) / 999.0f);
            half val = labels[i] - predictions[i];
            elementLoss[i] = val * val;
            elementLossGradient[i] = (half)2.0f * val * 100;
        }

        vector<Layer *> layers;
        NetworkInput *predictionsInput = new NetworkInput(predictionsGpu);
        layers.push_back(predictionsInput);
        NoOpLayer *noOpLayer = new NoOpLayer();
        layers.push_back(noOpLayer);
        NetworkInput *labelsInput = new NetworkInput(labelsGpu);
        layers.push_back(labelsInput);
        MeanSquaredError *meanSquaredError = new MeanSquaredError();
        layers.push_back(meanSquaredError);
        NetworkOutput *elementLossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanSquaredError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanSquaredError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanSquaredError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanSquaredError, elementLossOutput, (int)Loss::ConnectionType::LOSS);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(meanSquaredError->getErrorInput().isEmpty());
        if (inferenceOnly) {
            ASSERT_TRUE(meanSquaredError->getErrorOutput().isEmpty());
        }

        // Network is runnable here
        predictionsInput->forward(predictionsCpu, false);
        labelsInput->forward(labelsCpu, false);

        labelsStream.waitEvent(stream.putEvent());
        labelsStream.waitEvent(elementLossOutput->getOutputReadyEvent());

        Tensor predictionsOutputCpu = predictionsInput->getFeatureOutput().get().clone(cpuPlacement);
        predictionsOutputCpu.copyFromAsync(predictionsInput->getFeatureOutput().get(), labelsStream);
        labelsStream.synchronize();
        half *predictionsGpuMem = (half *)predictionsOutputCpu.getMemPtr();
        for (uint32_t i = 0; i < predictionsOutputCpu.getTotalNumElements(); ++i) {
            if (predictionsGpuMem[i] != predictions[i])
                printf("%d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
            ASSERT_EQ(predictionsGpuMem[i], predictions[i]);
        }

        Tensor labelsOutputCpu = labelsInput->getFeatureOutput().get().clone(cpuPlacement);
        labelsOutputCpu.copyFromAsync(labelsInput->getFeatureOutput().get(), labelsStream);
        labelsStream.synchronize();
        half *labelsGpuMem = (half *)labelsOutputCpu.getMemPtr();
        for (uint32_t i = 0; i < labelsOutputCpu.getTotalNumElements(); ++i) {
            if (labelsGpuMem[i] != labels[i])
                printf("%d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
            ASSERT_EQ(labelsGpuMem[i], labels[i]);
        }

        // Verify the loss
        Tensor elementLossGpu_h = elementLossCpu.clone(cpuPlacement);
        elementLossGpu_h.copyFromAsync(elementLossOutput->getFeatureOutput(), labelsStream);
        labelsStream.synchronize();
        half *elementLossGpu_h_mem = (half *)elementLossGpu_h.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
            if (elementLoss[i] != elementLossGpu_h_mem[i])
                printf("%d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
            ASSERT_EQ(elementLoss[i], elementLossGpu_h_mem[i]);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanSquaredError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            half *elementLossGradientGpu_h_mem = (half *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                if (elementLoss[i] != elementLossGpu_h_mem[i])
                    printf("gradient %d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
                ASSERT_EQ(elementLossGradient[i], elementLossGradientGpu_h_mem[i]);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
