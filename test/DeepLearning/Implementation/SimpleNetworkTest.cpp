#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

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

int popRandomElement(set<int> &elements) {
    int numElements = elements.size();
    int chosenElement = rand() % numElements;
    auto it = elements.begin();
    for (int i = 0; i < chosenElement; ++i)
        ++it;
    int element = *it;
    elements.erase(it);
    return element;
}

void createLabeledFeatures(Tensor featureIn, Tensor labelsIn, vector<set<int>> &featuresPerClass) {
    int numClasses = featuresPerClass.size();
    int batchSize = featureIn.getDescriptor().getDimensions()[0];
    int numFeatures = featureIn.getDescriptor().getDimensions()[1];

    half *featureInMem = (half *)featureIn.getMemPtr();
    float *labelsMem = (float *)labelsIn.getMemPtr();

    for (int batchItem = 0; batchItem < batchSize; ++batchItem) {
        int classNum = rand() % numClasses;
        set<int> classFeatures = featuresPerClass[classNum];

        for (int feature = 0; feature < numFeatures; ++feature) {
            featureInMem[batchItem * numFeatures + feature] = (rand() % 100) / 400.0f;
            if (classFeatures.count(feature) == 1)
                featureInMem[batchItem * numFeatures + feature] = featureInMem[batchItem * numFeatures + feature] + half(0.5f);
        }

        for (int classLabel = 0; classLabel < numClasses; ++classLabel)
            labelsMem[batchItem * numClasses + classLabel] = classLabel == classNum ? 1.0f : 0.0f;
    }
}

int getClassNum(Tensor labels, int batchItem, float &confidence) {
    int numClasses = labels.getDescriptor().getDimensions()[1];

    float maxLabel = -1.0f;
    int label = -1;
    float *labelsMem = (float *)labels.getMemPtr();

    for (int classNum = 0; classNum < numClasses; ++classNum) {
        if (labelsMem[numClasses * batchItem + classNum] > maxLabel) {
            maxLabel = labelsMem[numClasses * batchItem + classNum];
            label = classNum;
        }
    }

    confidence = maxLabel;
    return label;
}

TEST(SimpleFullyConnectedNetwork, Learns) {
    srand(time(NULL));

    constexpr int NUM_CLASSES = 8;
    constexpr int FEATURES_PER_CLASS = 8;
    constexpr int BATCH_SIZE = 32;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Tensor featureIn =
        Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {BATCH_SIZE, NUM_CLASSES * FEATURES_PER_CLASS}));
    Tensor labelsIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {BATCH_SIZE, NUM_CLASSES}));

    NetworkInput *featureInput =
        new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions());
    NetworkInput *labelsInput = new NetworkInput(gpuPlacement, TensorDescriptor::DataType::FP32, labelsIn.getDescriptor().getDimensions());
    FullyConnected *fullyConnectedLayer = new FullyConnected(NUM_CLASSES * FEATURES_PER_CLASS, true);
    Relu *relu = new Relu();
    FullyConnected *logitsLayer = new FullyConnected(NUM_CLASSES, true);
    CategoricalCrossEntropyLoss *categoricalCrossEntropyLoss = new CategoricalCrossEntropyLoss();
    NetworkOutput *predictionsOutput = new NetworkOutput(cpuPlacement);
    NetworkOutput *lossOutput = new NetworkOutput(cpuPlacement);

    Stream stream = featureInput->getStream();
    Stream labelsStream = labelsInput->getStream();

    LayerTestHelper::connectTwoLayers(featureInput, fullyConnectedLayer);
    LayerTestHelper::connectTwoLayers(fullyConnectedLayer, relu);
    LayerTestHelper::connectTwoLayers(relu, logitsLayer);
    LayerTestHelper::connectTwoLayers(logitsLayer, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
    LayerTestHelper::connectTwoLayers(labelsInput, categoricalCrossEntropyLoss, 0, (int)Loss::ConnectionType::LABELS);
    LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, predictionsOutput, (int)Loss::ConnectionType::PREDICTIONS);
    LayerTestHelper::connectTwoLayers(categoricalCrossEntropyLoss, lossOutput, (int)Loss::ConnectionType::BATCH_LOSS);

    vector<Layer *> layers;
    layers.push_back(featureInput);
    layers.push_back(labelsInput);
    layers.push_back(fullyConnectedLayer);
    layers.push_back(relu);
    layers.push_back(logitsLayer);
    layers.push_back(categoricalCrossEntropyLoss);
    layers.push_back(predictionsOutput);
    layers.push_back(lossOutput);

    LayerTestHelper::initializeNetwork(layers);

    // Initialize weights
    UniformRandom initializer(0.1, -0.1);
    initializer.initialize(fullyConnectedLayer, fullyConnectedLayer->getWeights());
    initializer.initialize(fullyConnectedLayer, fullyConnectedLayer->getBiases());
    initializer.initialize(logitsLayer, logitsLayer->getWeights());
    initializer.initialize(logitsLayer, logitsLayer->getBiases());

    // Network is initialized and can be run
    set<int> featureIndices;
    for (int i = 0; i < NUM_CLASSES * FEATURES_PER_CLASS; ++i) {
        featureIndices.insert(i);
    }

    vector<set<int>> featuresPerClass;

    for (int i = 0; i < NUM_CLASSES; ++i) {
        featuresPerClass.emplace_back();
        for (int j = 0; j < FEATURES_PER_CLASS; ++j) {
            featuresPerClass.back().insert(popRandomElement(featureIndices));
        }
    }

    constexpr bool PRINT = true;

    fullyConnectedLayer->setLearningRate(0.32);
    logitsLayer->setLearningRate(0.32);
    for (int i = 0; i < 405; ++i) {
        if (i == 200) {
            fullyConnectedLayer->setLearningRate(0.032);
            logitsLayer->setLearningRate(0.032);
        }
        if (i == 300) {
            fullyConnectedLayer->setLearningRate(0.0096);
            logitsLayer->setLearningRate(0.0096);
        }
        createLabeledFeatures(featureIn, labelsIn, featuresPerClass);
        featureInput->forward(featureIn, false);
        labelsInput->forward(labelsIn, false);
        Event batchDone0 = fullyConnectedLayer->updateWeightsAndBiasesWithScaledGradient();
        Event batchDone1 = logitsLayer->updateWeightsAndBiasesWithScaledGradient();

        if (PRINT && (i < 5 || i >= 400)) {
            predictionsOutput->getOutputReadyEvent().synchronize();
            Tensor predictions = predictionsOutput->getFeatureOutput();

            lossOutput->getOutputReadyEvent().synchronize();
            Tensor loss = lossOutput->getFeatureOutput();

            printf("batch %d,  batch loss %f\n", i, ((float *)loss.getMemPtr())[0]);
            for (int i = 0; i < 10; ++i) {
                int label;
                float confidence;
                label = getClassNum(predictions, i, confidence);
                float unused;
                printf("actual class %d    predicted class %d  confidence %d%%\n",
                       getClassNum(labelsIn, i, unused),
                       label,
                       (int)(confidence * 100));
            }
        }

        labelsStream.waitEvent(batchDone0);
        labelsStream.waitEvent(batchDone1);
        stream.waitEvent(batchDone0);
        stream.waitEvent(batchDone1);
    }

    lossOutput->getOutputReadyEvent().synchronize();
    Tensor loss = lossOutput->getFeatureOutput();

    ASSERT_LT(((float *)loss.getMemPtr())[0], 10.0f);

    LayerTestHelper::tearDownNetwork(layers);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
