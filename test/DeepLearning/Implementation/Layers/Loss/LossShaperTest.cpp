#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

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

        const DataType dataType = DataType::FP16;

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

        ASSERT_TRUE(!lossShaper->getErrorInput().has_value());
        ASSERT_TRUE(!lossShaper->getErrorOutput().has_value());
        ASSERT_EQ(lossOutput->getFeatureOutput().value().getDimensions(), reducedDimensions);

        // Network is runnable here
        lossInput->forward(rawLossCpu, false);
        Tensor batchLossGpu_h(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        batchLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().value(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 1.0f;
        half batchLossGpuMem_h = *((half *)batchLossGpu_h.getMemPtr());
        float diff = batchLossCpu - (float)batchLossGpuMem_h;
        ASSERT_LT(abs(diff), thresh);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, NumericalPerExampleFp32) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 2);
        dimensions.push_back((rand() % 400) + 2);
        vector<uint64_t> reducedDimensions = {dimensions[0], 1};

        const DataType dataType = DataType::FP32;

        Tensor rawLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        Tensor perExampleLossCpu(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));

        float *rawLossCpuMem = (float *)rawLossCpu.getMemPtr();
        float *perExampleLossCpuMem = (float *)perExampleLossCpu.getMemPtr();
        for (uint32_t b = 0; b < dimensions[0]; ++b) {
            perExampleLossCpuMem[b] = 0.0f;
            for (uint32_t c = 0; c < dimensions[1]; ++c) {
                float val = (rand() % 1000) / 250.0f;
                rawLossCpuMem[b * dimensions[1] + c] = val;
                perExampleLossCpuMem[b] += val;
            }
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> lossInput = make_shared<NetworkInput>(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        shared_ptr<NoOpLayer> noOpLayer1 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer1);
        shared_ptr<LossShaper> lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::PER_EXAMPLE);
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

        ASSERT_TRUE(!lossShaper->getErrorInput().has_value());
        ASSERT_TRUE(!lossShaper->getErrorOutput().has_value());
        ASSERT_EQ(lossOutput->getFeatureOutput().value().getDimensions(), reducedDimensions);

        // Network is runnable here
        lossInput->forward(rawLossCpu, false);
        Tensor perExampleLossGpu_h(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        perExampleLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().value(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        float *perExampleLossGpuMem_h = ((float *)perExampleLossGpu_h.getMemPtr());
        for (uint32_t e = 0; e < dimensions[0]; ++e) {
            float diff = perExampleLossCpuMem[e] - perExampleLossGpuMem_h[e];
            ASSERT_LT(abs(diff), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, NumericalPerOutputFp32) {
    srand(time(NULL));

    for (uint32_t i = 0; i < 10; ++i) {
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

        vector<uint64_t> dimensions;
        dimensions.push_back((rand() % 400) + 2);
        dimensions.push_back((rand() % 400) + 2);
        vector<uint64_t> reducedDimensions = {1, dimensions[1]};

        const DataType dataType = DataType::FP32;

        Tensor rawLossCpu(cpuPlacement, TensorDescriptor(dataType, dimensions));
        Tensor perOutputLossCpu(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));

        float *rawLossCpuMem = (float *)rawLossCpu.getMemPtr();
        float *perOutputLossCpuMem = (float *)perOutputLossCpu.getMemPtr();
        for (uint32_t c = 0; c < dimensions[1]; ++c) {
            perOutputLossCpuMem[c] = 0.0f;
            for (uint32_t b = 0; b < dimensions[0]; ++b) {
                float val = (rand() % 1000) / 250.0f;
                rawLossCpuMem[b * dimensions[1] + c] = val;
                perOutputLossCpuMem[c] += val;
            }
            perOutputLossCpuMem[c] /= dimensions[0];
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> lossInput = make_shared<NetworkInput>(gpuPlacement, dataType, dimensions);
        layers.push_back(lossInput);
        shared_ptr<NoOpLayer> noOpLayer1 = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer1);
        shared_ptr<LossShaper> lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::PER_OUTPUT);
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

        ASSERT_TRUE(!lossShaper->getErrorInput().has_value());
        ASSERT_TRUE(!lossShaper->getErrorOutput().has_value());
        ASSERT_EQ(lossOutput->getFeatureOutput().value().getDimensions(), reducedDimensions);

        // Network is runnable here
        lossInput->forward(rawLossCpu, false);
        Tensor perOutputLossGpu_h(cpuPlacement, TensorDescriptor(dataType, reducedDimensions));
        stream.waitEvent(lossOutput->getOutputReadyEvent());
        perOutputLossGpu_h.copyFromAsync(lossOutput->getFeatureOutput().value(), stream);

        stream.waitEvent(lossOutput->getOutputReadyEvent());
        stream.synchronize();

        float thresh = 0.01f;
        float *perOutputLossGpuMem_h = ((float *)perOutputLossGpu_h.getMemPtr());
        for (uint32_t c = 0; c < dimensions[1]; ++c) {
            float diff = perOutputLossCpuMem[c] - perOutputLossGpuMem_h[c];
            ASSERT_LT(abs(diff), thresh);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(LossShaper, NumericalPerOutputRankThreePreservesNonBatchLayout) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    const vector<uint64_t> inputDimensions = {3, 2, 2};
    const vector<uint64_t> outputDimensions = {1, 2, 2};

    Tensor inputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, inputDimensions));
    float* input = inputCpu.getMemPtr<float>();
    for (uint32_t i = 0; i < inputCpu.getTotalNumElements(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    vector<shared_ptr<Layer>> layers;
    auto lossInput = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, inputDimensions);
    layers.push_back(lossInput);
    auto lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::PER_OUTPUT);
    layers.push_back(lossShaper);
    auto lossOutput = make_shared<NetworkOutput>(gpuPlacement);
    layers.push_back(lossOutput);

    LayerTestHelper::connectTwoLayers(lossInput, lossShaper);
    LayerTestHelper::connectTwoLayers(lossShaper, lossOutput);
    LayerTestHelper::initializeNetwork(layers);

    ASSERT_EQ(lossOutput->getFeatureOutput().value().getDimensions(), outputDimensions);

    lossInput->forward(inputCpu, false);
    Stream stream = lossInput->getStream();
    stream.waitEvent(lossOutput->getOutputReadyEvent());
    Tensor outputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, outputDimensions));
    outputCpu.copyFromAsync(lossOutput->getFeatureOutput().value(), stream);
    stream.synchronize();

    const float* output = outputCpu.getMemPtr<float>();
    const float expected[] = {5.0f, 6.0f, 7.0f, 8.0f};
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(output[i], expected[i]);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(LossShaper, PartialBatchUsesValidExampleCountForBatchMean) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    const vector<uint64_t> inputDimensions = {4, 2};

    Tensor inputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, inputDimensions));
    float* input = inputCpu.getMemPtr<float>();
    const float values[] = {1.0f, 2.0f,
                            3.0f, 4.0f,
                            0.0f, 0.0f,
                            0.0f, 0.0f};
    for (uint32_t i = 0; i < inputCpu.getTotalNumElements(); ++i)
        input[i] = values[i];

    vector<shared_ptr<Layer>> layers;
    auto lossInput = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, inputDimensions);
    layers.push_back(lossInput);
    auto lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::BATCH);
    layers.push_back(lossShaper);
    auto lossOutput = make_shared<NetworkOutput>(gpuPlacement);
    layers.push_back(lossOutput);

    LayerTestHelper::connectTwoLayers(lossInput, lossShaper);
    LayerTestHelper::connectTwoLayers(lossShaper, lossOutput);
    LayerTestHelper::initializeNetwork(layers);

    lossInput->forward(inputCpu, false, 2);
    Stream stream = lossInput->getStream();
    stream.waitEvent(lossOutput->getOutputReadyEvent());
    Tensor outputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {1, 1}));
    outputCpu.copyFromAsync(lossOutput->getFeatureOutput().value(), stream);
    stream.synchronize();

    EXPECT_FLOAT_EQ(outputCpu.getMemPtr<float>()[0], 5.0f);
    LayerTestHelper::tearDownNetwork(layers);
}

TEST(LossShaper, PartialBatchUsesValidExampleCountForPerOutputMean) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    const vector<uint64_t> inputDimensions = {4, 2};

    Tensor inputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, inputDimensions));
    float* input = inputCpu.getMemPtr<float>();
    const float values[] = {1.0f, 2.0f,
                            3.0f, 4.0f,
                            0.0f, 0.0f,
                            0.0f, 0.0f};
    for (uint32_t i = 0; i < inputCpu.getTotalNumElements(); ++i)
        input[i] = values[i];

    vector<shared_ptr<Layer>> layers;
    auto lossInput = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, inputDimensions);
    layers.push_back(lossInput);
    auto lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::PER_OUTPUT);
    layers.push_back(lossShaper);
    auto lossOutput = make_shared<NetworkOutput>(gpuPlacement);
    layers.push_back(lossOutput);

    LayerTestHelper::connectTwoLayers(lossInput, lossShaper);
    LayerTestHelper::connectTwoLayers(lossShaper, lossOutput);
    LayerTestHelper::initializeNetwork(layers);

    lossInput->forward(inputCpu, false, 2);
    Stream stream = lossInput->getStream();
    stream.waitEvent(lossOutput->getOutputReadyEvent());
    Tensor outputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {1, 2}));
    outputCpu.copyFromAsync(lossOutput->getFeatureOutput().value(), stream);
    stream.synchronize();

    EXPECT_FLOAT_EQ(outputCpu.getMemPtr<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(outputCpu.getMemPtr<float>()[1], 3.0f);
    LayerTestHelper::tearDownNetwork(layers);
}
