#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

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
            elementLossGradient[i] = (half)2.0f * val * (half)Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanSquaredError> meanSquaredError = make_shared<MeanSquaredError>(TensorDescriptor::DataType::FP16);
        layers.push_back(meanSquaredError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanSquaredError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanSquaredError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanSquaredError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanSquaredError, elementLossOutput, 0);
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

TEST(MeanSquaredError, ComputesCorrectResult_BatchLoss_FP16_FP32Labels) {
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
        Tensor labelsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor predictionsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor predictionsGpu = predictionsCpu.clone(gpuPlacement);
        Tensor elementLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor elementLossGpu = elementLossCpu.clone(gpuPlacement);
        Tensor elementLossGradientCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor elementLossGradientGpu = elementLossGradientCpu.clone(gpuPlacement);

        float *labels = (float *)labelsCpu.getMemPtr();
        half *predictions = (half *)predictionsCpu.getMemPtr();
        half *elementLoss = (half *)elementLossCpu.getMemPtr();
        half *elementLossGradient = (half *)elementLossGradientCpu.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); i++) {
            predictions[i] = ((rand() % 1500) / 999.0f);
            labels[i] = ((rand() % 1500) / 999.0f);
            half val = (half)labels[i] - predictions[i];
            elementLoss[i] = val * val;
            elementLossGradient[i] = (half)2.0f * val * (half)Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanSquaredError> meanSquaredError = make_shared<MeanSquaredError>(TensorDescriptor::DataType::FP16);
        layers.push_back(meanSquaredError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanSquaredError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanSquaredError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanSquaredError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanSquaredError, elementLossOutput, 0);
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
        float *labelsGpuMem = (float *)labelsOutputCpu.getMemPtr();
        for (uint32_t i = 0; i < labelsOutputCpu.getTotalNumElements(); ++i) {
            if (labelsGpuMem[i] != labels[i])
                printf("%d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
            ASSERT_EQ(labelsGpuMem[i], labels[i]);
        }

        // Verify the loss
        float thresh = 0.002;
        Tensor elementLossGpu_h = elementLossCpu.clone(cpuPlacement);
        elementLossGpu_h.copyFromAsync(elementLossOutput->getFeatureOutput(), labelsStream);
        labelsStream.synchronize();
        half *elementLossGpu_h_mem = (half *)elementLossGpu_h.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
            if (abs((float)elementLoss[i] - (float)elementLossGpu_h_mem[i]) >= thresh)
                printf("%d (%ld, %ld)  %f vs %f\n", i, dimensions[0], dimensions[1], (float)elementLoss[i], (float)elementLossGpu_h_mem[i]);
            ASSERT_LT(abs((float)elementLoss[i] - (float)elementLossGpu_h_mem[i]), thresh);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanSquaredError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            half *elementLossGradientGpu_h_mem = (half *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                if (abs((float)elementLossGradient[i] - (float)elementLossGradientGpu_h_mem[i]) >= thresh)
                    printf("gradient %d (%ld, %ld)  %f vs %f   %f  %f  %i\n",
                           i,
                           dimensions[0],
                           dimensions[1],
                           (float)elementLossGradient[i],
                           (float)elementLossGradientGpu_h_mem[i],
                           (float)predictions[i],
                           (float)labels[i],
                           Loss::getLossScalingFactor());
                ASSERT_LT(abs((float)elementLossGradient[i] - (float)elementLossGradientGpu_h_mem[i]), thresh);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(MeanSquaredError, ComputesCorrectResult_BatchLoss_FP16PredictionsGradient_FP32LabelsLoss) {
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
        Tensor labelsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor predictionsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor predictionsGpu = predictionsCpu.clone(gpuPlacement);
        Tensor elementLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor elementLossGpu = elementLossCpu.clone(gpuPlacement);
        Tensor elementLossGradientCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, dimensions));
        Tensor elementLossGradientGpu = elementLossGradientCpu.clone(gpuPlacement);

        float *labels = (float *)labelsCpu.getMemPtr();
        half *predictions = (half *)predictionsCpu.getMemPtr();
        float *elementLoss = (float *)elementLossCpu.getMemPtr();
        half *elementLossGradient = (half *)elementLossGradientCpu.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); i++) {
            predictions[i] = ((rand() % 1500) / 999.0f);
            labels[i] = ((rand() % 1500) / 999.0f);
            half val = (half)labels[i] - predictions[i];
            elementLoss[i] = val * val;
            elementLossGradient[i] = (half)2.0f * val * (half)Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanSquaredError> meanSquaredError = make_shared<MeanSquaredError>(TensorDescriptor::DataType::FP32);
        layers.push_back(meanSquaredError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanSquaredError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanSquaredError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanSquaredError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanSquaredError, elementLossOutput, 0);
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
        float *labelsGpuMem = (float *)labelsOutputCpu.getMemPtr();
        for (uint32_t i = 0; i < labelsOutputCpu.getTotalNumElements(); ++i) {
            if (labelsGpuMem[i] != labels[i])
                printf("%d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
            ASSERT_EQ(labelsGpuMem[i], labels[i]);
        }

        // Verify the loss
        float thresh = 0.005;
        Tensor elementLossGpu_h = elementLossCpu.clone(cpuPlacement);
        elementLossGpu_h.copyFromAsync(elementLossOutput->getFeatureOutput(), labelsStream);
        labelsStream.synchronize();
        float *elementLossGpu_h_mem = (float *)elementLossGpu_h.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
            if (abs(elementLoss[i] - elementLossGpu_h_mem[i]) >= thresh)
                printf("%d (%ld, %ld)  %f vs %f\n", i, dimensions[0], dimensions[1], (float)elementLoss[i], (float)elementLossGpu_h_mem[i]);
            ASSERT_LT(abs(elementLoss[i] - elementLossGpu_h_mem[i]), thresh);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            thresh = 0.1;
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanSquaredError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            half *elementLossGradientGpu_h_mem = (half *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                if (abs((float)elementLossGradient[i] - (float)elementLossGradientGpu_h_mem[i]) >= thresh)
                    printf("gradient %d (%ld, %ld)  %f vs %f   %f  %f  %i\n",
                           i,
                           dimensions[0],
                           dimensions[1],
                           (float)elementLossGradient[i],
                           (float)elementLossGradientGpu_h_mem[i],
                           (float)predictions[i],
                           (float)labels[i],
                           Loss::getLossScalingFactor());
                ASSERT_LT(abs((float)elementLossGradient[i] - (float)elementLossGradientGpu_h_mem[i]), thresh);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(MeanSquaredError, ComputesCorrectResult_BatchLoss_FP32) {
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
        Tensor labelsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor predictionsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor predictionsGpu = predictionsCpu.clone(gpuPlacement);
        Tensor elementLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor elementLossGpu = elementLossCpu.clone(gpuPlacement);
        Tensor elementLossGradientCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor elementLossGradientGpu = elementLossGradientCpu.clone(gpuPlacement);

        float *labels = (float *)labelsCpu.getMemPtr();
        float *predictions = (float *)predictionsCpu.getMemPtr();
        float *elementLoss = (float *)elementLossCpu.getMemPtr();
        float *elementLossGradient = (float *)elementLossGradientCpu.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); i++) {
            predictions[i] = ((rand() % 1500) / 999.0f);
            labels[i] = ((rand() % 1500) / 999.0f);
            float val = labels[i] - predictions[i];
            elementLoss[i] = val * val;
            elementLossGradient[i] = 2.0f * val * Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanSquaredError> meanSquaredError = make_shared<MeanSquaredError>(TensorDescriptor::DataType::FP32);
        layers.push_back(meanSquaredError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanSquaredError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanSquaredError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanSquaredError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanSquaredError, elementLossOutput, 0);
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
        float *predictionsGpuMem = (float *)predictionsOutputCpu.getMemPtr();
        for (uint32_t i = 0; i < predictionsOutputCpu.getTotalNumElements(); ++i) {
            if (predictionsGpuMem[i] != predictions[i])
                printf("%d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
            ASSERT_EQ(predictionsGpuMem[i], predictions[i]);
        }

        Tensor labelsOutputCpu = labelsInput->getFeatureOutput().get().clone(cpuPlacement);
        labelsOutputCpu.copyFromAsync(labelsInput->getFeatureOutput().get(), labelsStream);
        labelsStream.synchronize();
        float *labelsGpuMem = (float *)labelsOutputCpu.getMemPtr();
        for (uint32_t i = 0; i < labelsOutputCpu.getTotalNumElements(); ++i) {
            if (labelsGpuMem[i] != labels[i])
                printf("%d (%ld, %ld)\n", i, dimensions[0], dimensions[1]);
            ASSERT_EQ(labelsGpuMem[i], labels[i]);
        }

        // Verify the loss
        Tensor elementLossGpu_h = elementLossCpu.clone(cpuPlacement);
        elementLossGpu_h.copyFromAsync(elementLossOutput->getFeatureOutput(), labelsStream);
        labelsStream.synchronize();
        float *elementLossGpu_h_mem = (float *)elementLossGpu_h.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
            if (elementLoss[i] != elementLossGpu_h_mem[i])
                printf("%d (%ld, %ld)  %f vs %f\n", i, dimensions[0], dimensions[1], (float)elementLoss[i], (float)elementLossGpu_h_mem[i]);
            ASSERT_EQ(elementLoss[i], elementLossGpu_h_mem[i]);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanSquaredError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            float *elementLossGradientGpu_h_mem = (float *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                if (elementLoss[i] != elementLossGpu_h_mem[i])
                    printf("gradient %d (%ld, %ld)  %f vs %f   %f  %f  %i\n",
                           i,
                           dimensions[0],
                           dimensions[1],
                           (float)elementLossGradient[i],
                           (float)elementLossGradientGpu_h_mem[i],
                           (float)predictions[i],
                           (float)labels[i],
                           Loss::getLossScalingFactor());
                ASSERT_EQ(elementLossGradient[i], elementLossGradientGpu_h_mem[i]);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(MeanSquaredError, ComputesCorrectResult_BatchLoss_FP32_FP16Labels) {
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
        Tensor predictionsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor predictionsGpu = predictionsCpu.clone(gpuPlacement);
        Tensor elementLossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor elementLossGpu = elementLossCpu.clone(gpuPlacement);
        Tensor elementLossGradientCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, dimensions));
        Tensor elementLossGradientGpu = elementLossGradientCpu.clone(gpuPlacement);

        half *labels = (half *)labelsCpu.getMemPtr();
        float *predictions = (float *)predictionsCpu.getMemPtr();
        float *elementLoss = (float *)elementLossCpu.getMemPtr();
        float *elementLossGradient = (float *)elementLossGradientCpu.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); i++) {
            predictions[i] = ((rand() % 1500) / 999.0f);
            labels[i] = ((rand() % 1500) / 999.0f);
            float val = (float)labels[i] - (float)predictions[i];
            elementLoss[i] = val * val;
            elementLossGradient[i] = 2.0f * val * Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanSquaredError> meanSquaredError = make_shared<MeanSquaredError>(TensorDescriptor::DataType::FP32);
        layers.push_back(meanSquaredError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanSquaredError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanSquaredError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanSquaredError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanSquaredError, elementLossOutput, 0);
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
        float *predictionsGpuMem = (float *)predictionsOutputCpu.getMemPtr();
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
        float *elementLossGpu_h_mem = (float *)elementLossGpu_h.getMemPtr();
        for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
            if (elementLoss[i] != elementLossGpu_h_mem[i])
                printf("%d (%ld, %ld)  %f vs %f\n", i, dimensions[0], dimensions[1], (float)elementLoss[i], (float)elementLossGpu_h_mem[i]);
            ASSERT_EQ(elementLoss[i], elementLossGpu_h_mem[i]);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanSquaredError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            float *elementLossGradientGpu_h_mem = (float *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                if (elementLoss[i] != elementLossGpu_h_mem[i])
                    printf("gradient %d (%ld, %ld)  %f vs %f   %f  %f  %i\n",
                           i,
                           dimensions[0],
                           dimensions[1],
                           (float)elementLossGradient[i],
                           (float)elementLossGradientGpu_h_mem[i],
                           (float)predictions[i],
                           (float)labels[i],
                           Loss::getLossScalingFactor());
                ASSERT_EQ(elementLossGradient[i], elementLossGradientGpu_h_mem[i]);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}
