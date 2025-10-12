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
        dimensions.push_back((rand() % 400) + 2);
        dimensions.push_back((rand() % 400) + 2);
        vector<uint64_t> reducedDimensions = {1, 1};

        const TensorDescriptor::DataType dataType = TensorDescriptor::DataType::FP16;

        Tensor rawLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        // Accumulators are FP32:
        float batchLossCpu = 0.0f;

        half *rawLossCpuMem = (half *)rawLossCpu.getMemPtr();
        for (uint32_t b = 0; b < dimensions[0]; ++b) {
            for (uint32_t c = 0; c < dimensions[1]; ++c) {
                half val = (rand() % 1000) / 250.0f;
                rawLossCpuMem[b * dimensions[1] + c] = val;
                batchLossCpu += (float)val;
            }
        }
        batchLossCpu /= dimensions[0];

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> lossInput = make_shared<NetworkInput>(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        shared_ptr<NoOpLayer> noOpLayer1 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer1);
        shared_ptr<LossShaper> lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::BATCH);
        layers.push_back(lossShaper);
        shared_ptr<NoOpLayer> noOpLayer2 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer2);
        shared_ptr<NetworkOutput> lossOutput = make_shared<NetworkOutput>(gpuPlacement);
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
        lossInput->forward(rawLossCpu, false);
        Tensor batchLossGpu_h(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        batchLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().get(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 1.0f;
        half batchLossGpuMem_h = *((half *)batchLossGpu_h.getMemPtr());
        float diff = batchLossCpu - (float)batchLossGpuMem_h;
        ASSERT_LT(abs(diff), thresh);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, NumericalElementWiseFp32) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 2);
        dimensions.push_back((rand() % 400) + 2);
        vector<uint64_t> reducedDimensions = {dimensions[0], 1};

        const TensorDescriptor::DataType dataType = TensorDescriptor::DataType::FP32;

        Tensor rawLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        Tensor elementwiseLossCpu(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));

        float *rawLossCpuMem = (float *)rawLossCpu.getMemPtr();
        float *elementwiseLossCpuMem = (float *)elementwiseLossCpu.getMemPtr();
        for (uint32_t b = 0; b < dimensions[0]; ++b) {
            elementwiseLossCpuMem[b] = 0.0f;
            for (uint32_t c = 0; c < dimensions[1]; ++c) {
                float val = (rand() % 1000) / 250.0f;
                rawLossCpuMem[b * dimensions[1] + c] = val;
                elementwiseLossCpuMem[b] += val;
            }
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> lossInput = make_shared<NetworkInput>(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        shared_ptr<NoOpLayer> noOpLayer1 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer1);
        shared_ptr<LossShaper> lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::ELEMENTWISE);
        layers.push_back(lossShaper);
        shared_ptr<NoOpLayer> noOpLayer2 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer2);
        shared_ptr<NetworkOutput> lossOutput = make_shared<NetworkOutput>(gpuPlacement);
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
        lossInput->forward(rawLossCpu, false);
        Tensor elementwiseLossGpu_h(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        elementwiseLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().get(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        float *elementwiseLossGpuMem_h = ((float *)elementwiseLossGpu_h.getMemPtr());
        for (uint32_t e = 0; e < dimensions[0]; ++e) {
            float diff = elementwiseLossCpuMem[e] - elementwiseLossGpuMem_h[e];
            ASSERT_LT(abs(diff), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, NumericalClassWiseFp32) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 2);
        dimensions.push_back((rand() % 400) + 2);
        vector<uint64_t> reducedDimensions = {1, dimensions[1]};

        const TensorDescriptor::DataType dataType = TensorDescriptor::DataType::FP32;

        Tensor rawLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        Tensor classwiseLossCpu(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));

        float *rawLossCpuMem = (float *)rawLossCpu.getMemPtr();
        float *classwiseLossCpuMem = (float *)classwiseLossCpu.getMemPtr();
        for (uint32_t c = 0; c < dimensions[1]; ++c) {
            classwiseLossCpuMem[c] = 0.0f;
            for (uint32_t b = 0; b < dimensions[0]; ++b) {
                float val = (rand() % 1000) / 250.0f;
                rawLossCpuMem[b * dimensions[1] + c] = val;
                classwiseLossCpuMem[c] += val;
            }
            classwiseLossCpuMem[c] /= dimensions[0];
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> lossInput = make_shared<NetworkInput>(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        shared_ptr<NoOpLayer> noOpLayer1 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer1);
        shared_ptr<LossShaper> lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::CLASSWISE);
        layers.push_back(lossShaper);
        shared_ptr<NoOpLayer> noOpLayer2 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer2);
        shared_ptr<NetworkOutput> lossOutput = make_shared<NetworkOutput>(gpuPlacement);
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
        lossInput->forward(rawLossCpu, false);
        Tensor classwiseLossGpu_h(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        classwiseLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().get(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        float *classwiseLossGpuMem_h = ((float *)classwiseLossGpu_h.getMemPtr());
        for (uint32_t c = 0; c < dimensions[1]; ++c) {
            float diff = classwiseLossCpuMem[c] - classwiseLossGpuMem_h[c];
            ASSERT_LT(abs(diff), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}
