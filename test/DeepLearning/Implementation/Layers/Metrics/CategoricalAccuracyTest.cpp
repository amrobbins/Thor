#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/CategoricalAccuracy.h"
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
#include <unordered_set>
#include <vector>

using namespace std;

using namespace ThorImplementation;

TEST(CategoricalAccuracy, ComputesCorrectElementWiseResult_indicatorPerClassLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 20; ++test) {
        vector<uint64_t> dimensions;
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

        TensorDescriptor elementwiseDescriptorFP32(TensorDescriptor::DataType::FP32, dimensions);

        Tensor labelsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor predictionsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor labelsGpu(gpuPlacement, elementwiseDescriptorFP32);
        Tensor predictionsGpu(gpuPlacement, elementwiseDescriptorFP32);

        float *labels = (float *)labelsCpu.getMemPtr();
        float *predictions = (float *)predictionsCpu.getMemPtr();
        unordered_set<float> predictionsUsed;
        for (uint32_t i = 0; i < batchSize; ++i) {
            uint32_t trueClass = rand() % numClasses;
            for (uint32_t j = 0; j < numClasses; ++j) {
                labels[i * numClasses + j] = j == trueClass ? 1.0f : 0.0f;
                float prediction;
                // avoid ties
                do {
                    prediction = ((rand() % 1000000) / 1000000.0f);
                } while (predictionsUsed.count(prediction) == 1);
                predictionsUsed.insert(prediction);
                predictions[i * numClasses + j] = prediction;
            }
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<CategoricalAccuracy> categoricalAccuracy = make_shared<CategoricalAccuracy>();
        if (inferenceOnly)
            categoricalAccuracy->setConstructForInferenceOnly(true);
        layers.push_back(categoricalAccuracy);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalAccuracy, 0, (int)Metric::ConnectionType::FORWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalAccuracy, 0, (int)Metric::ConnectionType::LABELS);
        shared_ptr<NetworkOutput> accuracyOutput = nullptr;
        if (!inferenceOnly) {
            accuracyOutput = make_shared<NetworkOutput>(gpuPlacement);
            layers.push_back(accuracyOutput);
            LayerTestHelper::connectTwoLayers(categoricalAccuracy, accuracyOutput, (int)Metric::ConnectionType::METRIC);
        }
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(categoricalAccuracy->getErrorOutput().isEmpty());
        ASSERT_TRUE(categoricalAccuracy->getErrorInput().isEmpty());

        if (inferenceOnly) {
            assert(categoricalAccuracy->getFeatureOutput().isEmpty());
            continue;
        }

        // Network is runnable here
        predictionsInput->forward(predictionsCpu, false);
        labelsInput->forward(labelsCpu, false);

        Tensor accuracyGpu_h = accuracyOutput->getFeatureOutput().get().clone(cpuPlacement);
        accuracyGpu_h.copyFromAsync(accuracyOutput->getFeatureOutput(), stream);

        stream.synchronize();

        // Compute the expected accuracy
        float *labelsMem = (float *)labelsCpu.getMemPtr();
        float *predictionsMem = (float *)predictionsCpu.getMemPtr();
        uint32_t correctCount = 0;
        for (uint32_t i = 0; i < batchSize; ++i) {
            float highestProbability = predictions[i * numClasses + 0];
            uint32_t indexOfHighestProbability = 0;
            for (uint32_t j = 1; j < numClasses; ++j) {
                if (predictions[i * numClasses + j] > highestProbability) {
                    highestProbability = predictionsMem[i * numClasses + j];
                    indexOfHighestProbability = j;
                }
            }
            if (labelsMem[i * numClasses + indexOfHighestProbability] == 1.0f)
                correctCount += 1;
        }
        float accuracy_h = correctCount / (float)batchSize;

        float delta = 0.0001;
        ASSERT_LT(abs(accuracy_h - *((float *)accuracyGpu_h.getMemPtr())), delta);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(CategoricalAccuracy, ComputesCorrectElementWiseResult_classIndexLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 20; ++test) {
        vector<uint64_t> dimensions;
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

        TensorDescriptor elementwiseDescriptorFP32(TensorDescriptor::DataType::FP32, dimensions);
        TensorDescriptor classIndexDescriptorFP32(TensorDescriptor::DataType::UINT32, {batchSize, 1});

        Tensor labelsCpu(cpuPlacement, classIndexDescriptorFP32);
        Tensor predictionsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor labelsGpu(gpuPlacement, classIndexDescriptorFP32);
        Tensor predictionsGpu(gpuPlacement, elementwiseDescriptorFP32);

        uint32_t *labels = (uint32_t *)labelsCpu.getMemPtr();
        float *predictions = (float *)predictionsCpu.getMemPtr();
        unordered_set<float> predictionsUsed;
        for (uint32_t i = 0; i < batchSize; ++i) {
            labels[i] = rand() % numClasses;
            for (uint32_t j = 0; j < numClasses; ++j) {
                float prediction;
                // avoid ties
                do {
                    prediction = ((rand() % 1000000) / 1000000.0f);
                } while (predictionsUsed.count(prediction) == 1);
                predictionsUsed.insert(prediction);
                predictions[i * numClasses + j] = prediction;
            }
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> predictionsInput = make_shared<NetworkInput>(predictionsGpu);
        layers.push_back(predictionsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<CategoricalAccuracy> categoricalAccuracy = make_shared<CategoricalAccuracy>();
        if (inferenceOnly)
            categoricalAccuracy->setConstructForInferenceOnly(true);
        layers.push_back(categoricalAccuracy);

        Stream stream = predictionsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(predictionsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, categoricalAccuracy, 0, (int)Metric::ConnectionType::FORWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, categoricalAccuracy, 0, (int)Metric::ConnectionType::LABELS);
        shared_ptr<NetworkOutput> accuracyOutput = nullptr;
        if (!inferenceOnly) {
            accuracyOutput = make_shared<NetworkOutput>(gpuPlacement);
            layers.push_back(accuracyOutput);
            LayerTestHelper::connectTwoLayers(categoricalAccuracy, accuracyOutput, (int)Metric::ConnectionType::METRIC);
        }
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(categoricalAccuracy->getErrorOutput().isEmpty());
        ASSERT_TRUE(categoricalAccuracy->getErrorInput().isEmpty());

        if (inferenceOnly) {
            assert(categoricalAccuracy->getFeatureOutput().isEmpty());
            continue;
        }

        // Network is runnable here
        predictionsInput->forward(predictionsCpu, false);
        labelsInput->forward(labelsCpu, false);

        Tensor accuracyGpu_h = accuracyOutput->getFeatureOutput().get().clone(cpuPlacement);
        accuracyGpu_h.copyFromAsync(accuracyOutput->getFeatureOutput(), stream);

        stream.synchronize();

        // Compute the expected accuracy
        uint32_t *labelsMem = (uint32_t *)labelsCpu.getMemPtr();
        float *predictionsMem = (float *)predictionsCpu.getMemPtr();
        uint32_t correctCount = 0;
        for (uint32_t i = 0; i < batchSize; ++i) {
            float highestProbability = predictions[i * numClasses + 0];
            float indexOfHighestProbability = 0.0f;
            for (uint32_t j = 1; j < numClasses; ++j) {
                if (predictions[i * numClasses + j] > highestProbability) {
                    highestProbability = predictionsMem[i * numClasses + j];
                    indexOfHighestProbability = j;
                }
            }
            if (labelsMem[i] == indexOfHighestProbability)
                correctCount += 1;
        }
        float accuracy_h = correctCount / (float)batchSize;

        float delta = 0.0001;
        ASSERT_LT(abs(accuracy_h - *((float *)accuracyGpu_h.getMemPtr())), delta);

        LayerTestHelper::tearDownNetwork(layers);
    }
}
