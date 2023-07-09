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

using namespace ThorImplementation;
using namespace std;

float sgn(float x) {
    if (x > 0)
        return 1.0f;
    else if (x < 0)
        return -1.0f;
    return 0.0f;
}

float clampLabel(float label, float epsilon) {
    if (label >= 0.0f && label < epsilon)
        label = epsilon;
    else if (label < 0.0f && label > -epsilon)
        label = -epsilon;
    return label;
}

half clampLabel(half label, float epsilonF) {
    half epsilon = epsilonF;
    if (label >= half(0.0f) && label < epsilon)
        label = epsilon;
    else if (label < half(0.0f) && label > -epsilon)
        label = -epsilon;
    return label;
}

TEST(MeanAbsolutePercentageError, ComputesCorrectResult_FP16) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        float epsilon = ((rand() % 5) + 1) / 10000.0f;
        float maxMagnitude = 500.0f + (rand() % 500);

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
            if (rand() % 2)
                predictions[i] = predictions[i] * -1.0f;
            labels[i] = ((rand() % 1500) / 999.0f);
            if (predictions[i] == 0.0f && (rand() % 5))
                labels[i] = 0.0f;
            if (rand() % 2)
                labels[i] = labels[i] * -1.0f;

            if (labels[i] == predictions[i])
                elementLoss[i] = 0.0f;
            else
                elementLoss[i] = 100.0f * fabsf((clampLabel(labels[i], epsilon) - predictions[i]) / clampLabel(labels[i], epsilon));
            if (elementLoss[i] < maxMagnitude) {
                if (!(elementLoss[i] > -maxMagnitude))
                    elementLoss[i] = -maxMagnitude;
            } else {
                elementLoss[i] = maxMagnitude;
            }

            // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
            half x = predictions[i];
            half y = clampLabel(labels[i], epsilon);
            if (labels[i] == predictions[i])
                elementLossGradient[i] = 0.0f;
            else
                elementLossGradient[i] = 100.0f * (x - y) * (sqrt(((y - x) * (y - x)) / (y * y)) / ((y - x) * (y - x)));
            if (elementLossGradient[i] < maxMagnitude) {
                if (!(elementLossGradient[i] > -maxMagnitude))
                    elementLossGradient[i] = -maxMagnitude;
            } else {
                elementLossGradient[i] = maxMagnitude;
            }
            elementLossGradient[i] = elementLossGradient[i] * Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanAbsolutePercentageError> meanAbsolutePercentageError =
            make_shared<MeanAbsolutePercentageError>(TensorDescriptor::DataType::FP16, epsilon, maxMagnitude);
        layers.push_back(meanAbsolutePercentageError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanAbsolutePercentageError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanAbsolutePercentageError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanAbsolutePercentageError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanAbsolutePercentageError, elementLossOutput, 0);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(meanAbsolutePercentageError->getErrorInput().isEmpty());
        if (inferenceOnly) {
            ASSERT_TRUE(meanAbsolutePercentageError->getErrorOutput().isEmpty());
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
            float thresh = 0.001f;
            if (thresh < fabsf(elementLoss[i] * 0.002f))
                thresh = fabsf(elementLoss[i] * 0.002f);
            if (abs(elementLoss[i] - elementLossGpu_h_mem[i]) >= thresh) {
                printf("%d (%ld, %ld)  %f vs %f    %f %f\n",
                       i,
                       dimensions[0],
                       dimensions[1],
                       (float)elementLoss[i],
                       (float)elementLossGpu_h_mem[i],
                       (float)predictions[i],
                       (float)labels[i]);
                printf("100 * |(%f - %f) / %f| = %f\n",
                       (float)clampLabel(labels[i], epsilon),
                       (float)predictions[i],
                       (float)clampLabel(labels[i], epsilon),
                       (float)elementLoss[i]);
            }
            thresh = fmaxf(thresh, elementLoss[i] * 0.001f);
            ASSERT_LT(abs(elementLoss[i] - elementLossGpu_h_mem[i]), thresh);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanAbsolutePercentageError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            half *elementLossGradientGpu_h_mem = (half *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                float thresh = fmaxf(abs(elementLossGradient[i] * 0.06f), 0.0001f);
                if (fabsf(elementLossGradient[i] - elementLossGradientGpu_h_mem[i]) >= thresh)
                    printf("gradient %d (%ld, %ld)  %f vs %f   %f  %f  %d\n",
                           i,
                           dimensions[0],
                           dimensions[1],
                           (float)elementLossGradient[i],
                           (float)elementLossGradientGpu_h_mem[i],
                           (float)predictions[i],
                           (float)clampLabel(labels[i], epsilon),
                           Loss::getLossScalingFactor());
                ASSERT_LT(abs(elementLossGradient[i] - elementLossGradientGpu_h_mem[i]), thresh);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(MeanAbsolutePercentageError, ComputesCorrectResult_FP32) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        float epsilon = ((rand() % 5) + 1) / 10000.0f;
        float maxMagnitude = 500.0f + (rand() % 500);

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
            if (rand() % 2)
                predictions[i] = predictions[i] * -1.0f;
            labels[i] = ((rand() % 1500) / 999.0f);
            if (predictions[i] == 0.0f && (rand() % 5))
                labels[i] = 0.0f;
            if (rand() % 2)
                labels[i] = labels[i] * -1.0f;

            if (labels[i] == predictions[i])
                elementLoss[i] = 0.0f;
            else
                elementLoss[i] = 100.0f * fabsf((clampLabel(labels[i], epsilon) - predictions[i]) / clampLabel(labels[i], epsilon));
            if (elementLoss[i] < maxMagnitude) {
                if (!(elementLoss[i] > -maxMagnitude))
                    elementLoss[i] = -maxMagnitude;
            } else {
                elementLoss[i] = maxMagnitude;
            }

            // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
            float x = predictions[i];
            float y = clampLabel(labels[i], epsilon);
            if (labels[i] == predictions[i])
                elementLossGradient[i] = 0.0f;
            else
                elementLossGradient[i] = 100.0f * (x - y) * (sqrt(((y - x) * (y - x)) / (y * y)) / ((y - x) * (y - x)));
            if (elementLossGradient[i] < maxMagnitude) {
                if (!(elementLossGradient[i] > -maxMagnitude))
                    elementLossGradient[i] = -maxMagnitude;
            } else {
                elementLossGradient[i] = maxMagnitude;
            }
            elementLossGradient[i] = elementLossGradient[i] * Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanAbsolutePercentageError> meanAbsolutePercentageError =
            make_shared<MeanAbsolutePercentageError>(TensorDescriptor::DataType::FP32, epsilon, maxMagnitude);
        layers.push_back(meanAbsolutePercentageError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanAbsolutePercentageError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanAbsolutePercentageError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanAbsolutePercentageError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanAbsolutePercentageError, elementLossOutput, 0);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(meanAbsolutePercentageError->getErrorInput().isEmpty());
        if (inferenceOnly) {
            ASSERT_TRUE(meanAbsolutePercentageError->getErrorOutput().isEmpty());
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
            float thresh = 0.001f;
            if (thresh < fabsf(elementLoss[i] * 0.002f))
                thresh = fabsf(elementLoss[i] * 0.002f);
            if (abs(elementLoss[i] - elementLossGpu_h_mem[i]) >= thresh) {
                printf("%d (%ld, %ld)  %f vs %f    %f %f\n",
                       i,
                       dimensions[0],
                       dimensions[1],
                       (float)elementLoss[i],
                       (float)elementLossGpu_h_mem[i],
                       (float)predictions[i],
                       (float)labels[i]);
                printf("100 * |(%f - %f) / %f| = %f\n",
                       (float)clampLabel(labels[i], epsilon),
                       (float)predictions[i],
                       (float)clampLabel(labels[i], epsilon),
                       (float)elementLoss[i]);
            }
            thresh = fmaxf(thresh, elementLoss[i] * 0.001f);
            ASSERT_LT(abs(elementLoss[i] - elementLossGpu_h_mem[i]), thresh);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanAbsolutePercentageError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            float *elementLossGradientGpu_h_mem = (float *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                float thresh = fmaxf(abs(elementLossGradient[i] * 0.06f), 0.0001f);
                if (fabsf(elementLossGradient[i] - elementLossGradientGpu_h_mem[i]) >= thresh)
                    printf("gradient %d (%ld, %ld)  %f vs %f   %f  %f  %d\n",
                           i,
                           dimensions[0],
                           dimensions[1],
                           (float)elementLossGradient[i],
                           (float)elementLossGradientGpu_h_mem[i],
                           (float)predictions[i],
                           (float)clampLabel(labels[i], epsilon),
                           Loss::getLossScalingFactor());
                ASSERT_LT(abs(elementLossGradient[i] - elementLossGradientGpu_h_mem[i]), thresh);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(MeanAbsolutePercentageError, ComputesCorrectResult_FP32_FP16Labels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        float epsilon = ((rand() % 5) + 1) / 10000.0f;
        float maxMagnitude = 500.0f + (rand() % 500);

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
            if (rand() % 2)
                predictions[i] = predictions[i] * -1.0f;
            labels[i] = ((rand() % 1500) / 999.0f);
            if (predictions[i] == 0.0f && (rand() % 5))
                labels[i] = 0.0f;
            if (rand() % 2)
                labels[i] = labels[i] * -1.0f;

            if (labels[i] == predictions[i])
                elementLoss[i] = 0.0f;
            else
                elementLoss[i] = 100.0f * fabsf((clampLabel(labels[i], epsilon) - predictions[i]) / clampLabel(labels[i], epsilon));
            if (elementLoss[i] < maxMagnitude) {
                if (!(elementLoss[i] > -maxMagnitude))
                    elementLoss[i] = -maxMagnitude;
            } else {
                elementLoss[i] = maxMagnitude;
            }

            // d/dx(100 abs((y - x)/y)) = 100 * (x - y) * (sqrt( (y-x)^2 / y^2 ) / (y - x)^2)
            float x = predictions[i];
            float y = clampLabel(labels[i], epsilon);
            if (labels[i] == predictions[i])
                elementLossGradient[i] = 0.0f;
            else
                elementLossGradient[i] = 100.0f * (x - y) * (sqrt(((y - x) * (y - x)) / (y * y)) / ((y - x) * (y - x)));
            if (elementLossGradient[i] < maxMagnitude) {
                if (!(elementLossGradient[i] > -maxMagnitude))
                    elementLossGradient[i] = -maxMagnitude;
            } else {
                elementLossGradient[i] = maxMagnitude;
            }
            elementLossGradient[i] = elementLossGradient[i] * Loss::getLossScalingFactor();
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<MeanAbsolutePercentageError> meanAbsolutePercentageError =
            make_shared<MeanAbsolutePercentageError>(TensorDescriptor::DataType::FP32, epsilon, maxMagnitude);
        layers.push_back(meanAbsolutePercentageError);
        shared_ptr<NetworkOutput> elementLossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(elementLossOutput);

        bool inferenceOnly = (rand() % 5) == 0;
        meanAbsolutePercentageError->setConstructForInferenceOnly(inferenceOnly);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, meanAbsolutePercentageError, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, meanAbsolutePercentageError, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(meanAbsolutePercentageError, elementLossOutput, 0);
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(meanAbsolutePercentageError->getErrorInput().isEmpty());
        if (inferenceOnly) {
            ASSERT_TRUE(meanAbsolutePercentageError->getErrorOutput().isEmpty());
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
            float thresh = 0.001f;
            if (thresh < fabsf(elementLoss[i] * 0.002f))
                thresh = fabsf(elementLoss[i] * 0.002f);
            if (abs(elementLoss[i] - elementLossGpu_h_mem[i]) >= thresh) {
                printf("%d (%ld, %ld)  %f vs %f    %f %f\n",
                       i,
                       dimensions[0],
                       dimensions[1],
                       (float)elementLoss[i],
                       (float)elementLossGpu_h_mem[i],
                       (float)predictions[i],
                       (float)labels[i]);
                printf("100 * |(%f - %f) / %f| = %f\n",
                       (float)clampLabel(labels[i], epsilon),
                       (float)predictions[i],
                       (float)clampLabel(labels[i], epsilon),
                       (float)elementLoss[i]);
            }
            thresh = fmaxf(thresh, elementLoss[i] * 0.001f);
            ASSERT_LT(abs(elementLoss[i] - elementLossGpu_h_mem[i]), thresh);
        }

        if (!inferenceOnly) {
            // Verify the loss gradient
            Tensor elementLossGradientGpu_h = elementLossGradientCpu.clone(cpuPlacement);
            elementLossGradientGpu_h.copyFromAsync(meanAbsolutePercentageError->getErrorOutput(), labelsStream);
            labelsStream.synchronize();
            float *elementLossGradientGpu_h_mem = (float *)elementLossGradientGpu_h.getMemPtr();
            for (uint32_t i = 0; i < elementLossCpu.getTotalNumElements(); ++i) {
                float thresh = fmaxf(abs(elementLossGradient[i] * 0.06f), 0.0001f);
                if (fabsf(elementLossGradient[i] - elementLossGradientGpu_h_mem[i]) >= thresh)
                    printf("gradient %d (%ld, %ld)  %f vs %f   %f  %f  %d\n",
                           i,
                           dimensions[0],
                           dimensions[1],
                           (float)elementLossGradient[i],
                           (float)elementLossGradientGpu_h_mem[i],
                           (float)predictions[i],
                           (float)clampLabel(labels[i], epsilon),
                           Loss::getLossScalingFactor());
                ASSERT_LT(abs(elementLossGradient[i] - elementLossGradientGpu_h_mem[i]), thresh);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
