#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "Thor.h"

#include <stdio.h>
#include "gtest/gtest.h"

using namespace std;
using namespace ThorImplementation;

TEST(LossShaper, NumericalBatchFp16) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 1);
        dimensions.push_back((rand() % 400) + 1);
        vector<uint64_t> reducedDimensions;
        reducedDimensions.push_back(dimensions[1]);

        const TensorDescriptor::DataType dataType = TensorDescriptor::DataType::FP16;

        Tensor elementwiseLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        // Accumulators are FP32:
        Tensor batchLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, reducedDimensions));

        half *elementWiseLossCpuMem = (half *)elementwiseLossCpu.getMemPtr();
        float *batchLossCpuMem = (float *)batchLossCpu.getMemPtr();
        for (uint32_t b = 0; b < dimensions[0]; ++b) {
            for (uint32_t e = 0; e < dimensions[1]; ++e) {
                half val = (rand() % 1000) / 250.0f;
                elementWiseLossCpuMem[b * dimensions[1] + e] = val;
                batchLossCpuMem[e] = batchLossCpuMem[e] + val;
            }
        }
        for (uint32_t e = 0; e < dimensions[1]; ++e) {
            batchLossCpuMem[e] = batchLossCpuMem[e] / dimensions[0];
        }

        vector<Layer *> layers;
        NetworkInput *lossInput = new NetworkInput(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        NoOpLayer *noOpLayer1 = new NoOpLayer();
        layers.push_back(noOpLayer1);
        LossShaper *lossShaper = new LossShaper(LossShaper::InputLossType::NUMERICAL_LOSS, LossShaper::OutputLossType::BATCH_LOSS);
        layers.push_back(lossShaper);
        NoOpLayer *noOpLayer2 = new NoOpLayer();
        layers.push_back(noOpLayer2);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = lossInput->getStream();

        LayerTestHelper::connectTwoLayers(lossInput, noOpLayer1);
        LayerTestHelper::connectTwoLayers(noOpLayer1, lossShaper);
        LayerTestHelper::connectTwoLayers(lossShaper, noOpLayer2);
        LayerTestHelper::connectTwoLayers(noOpLayer2, lossOutput);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(lossShaper->getErrorInput().isEmpty());
        ASSERT_TRUE(lossShaper->getErrorOutput().isEmpty());
        ASSERT_EQ(lossOutput->getFeatureOutput().get().getDimensions(), reducedDimensions);

        // Network is runnable here
        lossInput->forward(elementwiseLossCpu, false);
        Tensor batchLossGpu_h = batchLossCpu.clone(dataType);
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        batchLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().get(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        half *batchLossGpuMem_h = (half *)batchLossGpu_h.getMemPtr();
        for (uint32_t e = 0; e < dimensions[1]; ++e) {
            float diff = ((half)batchLossCpuMem[e]) - batchLossGpuMem_h[e];
            ASSERT_LT(abs(diff), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, NumericalBatchFp32) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 1);
        dimensions.push_back((rand() % 400) + 1);
        vector<uint64_t> reducedDimensions;
        reducedDimensions.push_back(dimensions[1]);

        const TensorDescriptor::DataType dataType = TensorDescriptor::DataType::FP32;

        Tensor elementwiseLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        // Accumulators are FP32:
        Tensor batchLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, reducedDimensions));

        float *elementWiseLossCpuMem = (float *)elementwiseLossCpu.getMemPtr();
        float *batchLossCpuMem = (float *)batchLossCpu.getMemPtr();
        for (uint32_t b = 0; b < dimensions[0]; ++b) {
            for (uint32_t e = 0; e < dimensions[1]; ++e) {
                float val = (rand() % 1000) / 250.0f;
                elementWiseLossCpuMem[b * dimensions[1] + e] = val;
                batchLossCpuMem[e] = batchLossCpuMem[e] + val;
            }
        }
        for (uint32_t e = 0; e < dimensions[1]; ++e) {
            batchLossCpuMem[e] = batchLossCpuMem[e] / dimensions[0];
        }

        vector<Layer *> layers;
        NetworkInput *lossInput = new NetworkInput(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        NoOpLayer *noOpLayer1 = new NoOpLayer();
        layers.push_back(noOpLayer1);
        LossShaper *lossShaper = new LossShaper(LossShaper::InputLossType::NUMERICAL_LOSS, LossShaper::OutputLossType::BATCH_LOSS);
        layers.push_back(lossShaper);
        NoOpLayer *noOpLayer2 = new NoOpLayer();
        layers.push_back(noOpLayer2);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = lossInput->getStream();

        LayerTestHelper::connectTwoLayers(lossInput, noOpLayer1);
        LayerTestHelper::connectTwoLayers(noOpLayer1, lossShaper);
        LayerTestHelper::connectTwoLayers(lossShaper, noOpLayer2);
        LayerTestHelper::connectTwoLayers(noOpLayer2, lossOutput);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(lossShaper->getErrorInput().isEmpty());
        ASSERT_TRUE(lossShaper->getErrorOutput().isEmpty());
        ASSERT_EQ(lossOutput->getFeatureOutput().get().getDimensions(), reducedDimensions);

        // Network is runnable here
        lossInput->forward(elementwiseLossCpu, false);
        Tensor batchLossGpu_h = batchLossCpu.clone(dataType);
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        batchLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().get(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        float *batchLossGpuMem_h = (float *)batchLossGpu_h.getMemPtr();
        for (uint32_t e = 0; e < dimensions[1]; ++e) {
            float diff = batchLossCpuMem[e] - batchLossGpuMem_h[e];
            ASSERT_LT(abs(diff), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, CategoricalClasswise) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 1);
        dimensions.push_back((rand() % 400) + 1);
        vector<uint64_t> reducedDimensions;
        reducedDimensions.push_back(dimensions[1]);

        const TensorDescriptor::DataType dataType = TensorDescriptor::DataType::FP32;

        Tensor perExampleClasswiseLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        // Accumulators are FP32:
        Tensor classwiseLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, reducedDimensions));

        float *perExampleClasswiseLossCpuMem = (float *)perExampleClasswiseLossCpu.getMemPtr();
        float *classwiseLossCpuMem = (float *)classwiseLossCpu.getMemPtr();
        for (uint32_t b = 0; b < dimensions[0]; ++b) {
            for (uint32_t e = 0; e < dimensions[1]; ++e) {
                float val = (rand() % 1000) / 250.0f;
                perExampleClasswiseLossCpuMem[b * dimensions[1] + e] = val;
                classwiseLossCpuMem[e] = classwiseLossCpuMem[e] + val;
            }
        }
        for (uint32_t e = 0; e < dimensions[1]; ++e) {
            classwiseLossCpuMem[e] = classwiseLossCpuMem[e] / dimensions[0];
        }

        vector<Layer *> layers;
        NetworkInput *lossInput = new NetworkInput(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        NoOpLayer *noOpLayer1 = new NoOpLayer();
        layers.push_back(noOpLayer1);
        LossShaper *lossShaper = new LossShaper(LossShaper::InputLossType::CATEGORICAL_LOSS, LossShaper::OutputLossType::CLASSWISE_LOSS);
        layers.push_back(lossShaper);
        NoOpLayer *noOpLayer2 = new NoOpLayer();
        layers.push_back(noOpLayer2);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = lossInput->getStream();

        LayerTestHelper::connectTwoLayers(lossInput, noOpLayer1);
        LayerTestHelper::connectTwoLayers(noOpLayer1, lossShaper);
        LayerTestHelper::connectTwoLayers(lossShaper, noOpLayer2);
        LayerTestHelper::connectTwoLayers(noOpLayer2, lossOutput);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(lossShaper->getErrorInput().isEmpty());
        ASSERT_TRUE(lossShaper->getErrorOutput().isEmpty());
        ASSERT_EQ(lossOutput->getFeatureOutput().get().getDimensions(), reducedDimensions);

        // Network is runnable here
        lossInput->forward(perExampleClasswiseLossCpu, false);
        Tensor classwiseLossGpu_h = classwiseLossCpu.clone(dataType);
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        classwiseLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().get(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        float *classwiseLossGpuMem_h = (float *)classwiseLossGpu_h.getMemPtr();
        for (uint32_t e = 0; e < dimensions[1]; ++e) {
            float diff = classwiseLossCpuMem[e] - classwiseLossGpuMem_h[e];
            ASSERT_LT(abs(diff), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, CategoricalBatch) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 1);
        dimensions.push_back((rand() % 400) + 1);
        vector<uint64_t> reducedDimensions;
        reducedDimensions.push_back(1);

        const TensorDescriptor::DataType dataType = TensorDescriptor::DataType::FP32;

        Tensor perExampleClasswiseLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));

        float *perExampleClasswiseLossCpuMem = (float *)perExampleClasswiseLossCpu.getMemPtr();
        float batchLoss = 0.0f;
        for (uint32_t b = 0; b < dimensions[0]; ++b) {
            for (uint32_t e = 0; e < dimensions[1]; ++e) {
                float val = (rand() % 1000) / 250.0f;
                perExampleClasswiseLossCpuMem[b * dimensions[1] + e] = val;
                batchLoss = batchLoss + val;
            }
        }
        batchLoss = batchLoss / dimensions[0];

        vector<Layer *> layers;
        NetworkInput *lossInput = new NetworkInput(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        NoOpLayer *noOpLayer1 = new NoOpLayer();
        layers.push_back(noOpLayer1);
        LossShaper *lossShaper = new LossShaper(LossShaper::InputLossType::CATEGORICAL_LOSS, LossShaper::OutputLossType::BATCH_LOSS);
        layers.push_back(lossShaper);
        NoOpLayer *noOpLayer2 = new NoOpLayer();
        layers.push_back(noOpLayer2);
        NetworkOutput *lossOutput = new NetworkOutput(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = lossInput->getStream();

        LayerTestHelper::connectTwoLayers(lossInput, noOpLayer1);
        LayerTestHelper::connectTwoLayers(noOpLayer1, lossShaper);
        LayerTestHelper::connectTwoLayers(lossShaper, noOpLayer2);
        LayerTestHelper::connectTwoLayers(noOpLayer2, lossOutput);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(lossShaper->getErrorInput().isEmpty());
        ASSERT_TRUE(lossShaper->getErrorOutput().isEmpty());
        ASSERT_EQ(lossOutput->getFeatureOutput().get().getDimensions(), reducedDimensions);

        // Network is runnable here
        lossInput->forward(perExampleClasswiseLossCpu, false);
        Tensor classwiseLossGpu_h(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        classwiseLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().get(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        float *classwiseLossGpuMem_h = (float *)classwiseLossGpu_h.getMemPtr();
        float diff = batchLoss - classwiseLossGpuMem_h[0];
        ASSERT_LT(abs(diff), thresh);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
