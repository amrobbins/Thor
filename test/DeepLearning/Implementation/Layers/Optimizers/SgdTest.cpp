#include "Thor.h"
#include "gtest/gtest.h"

#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

using namespace ThorImplementation;
using namespace std;

float computeCurrentLearningRate(float initialLearningRate, float decay, float epoch) {
    float currentLearningRate = initialLearningRate * pow(1.0 - (double)decay, (double)epoch);
    return currentLearningRate;
}

TEST(SgdTest, TestConstrutorSettersGetters) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    uint32_t batchSize = (rand() % 200) + 1;
    uint32_t numInputFeatures = (rand() % 200) + 1;
    uint32_t numOutputFeatures = (rand() % 200) + 1;
    bool hasBias = rand() % 2;

    Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

    float initialLearningRate = 10.0 / (1 + rand() % 100);
    float decay = 1.0f / (1.0f + (rand() % 10));
    float momentum = rand() % 2 ? 0.0f : 1.0f / (1.0f + (rand() % 10));
    ;
    bool useNesterovMomentum = rand() % 2;

    vector<shared_ptr<Layer>> layers;

    layers.push_back(make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
    layers.push_back(make_shared<NoOpLayer>());
    shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBias);
    layers.push_back(fullyConnectedLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

    Stream dataStream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);

    ThorImplementation::Tensor anErrorInput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
    ThorImplementation::Tensor anErrorOutput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
    shared_ptr<Sgd> sgd = make_shared<ThorImplementation::Sgd>(
        fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, anErrorInput, anErrorOutput);
    fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));

    ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
    ASSERT_EQ(sgd->getDecay(), decay);
    ASSERT_EQ(sgd->getMomentum(), momentum);
    ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

    initialLearningRate = 10.0 / (1 + rand() % 100);
    decay = 1.0f / (1.0f + (rand() % 10));
    momentum = rand() % 2 ? 0.0f : 1.0f / (1.0f + (rand() % 10));
    ;
    useNesterovMomentum = rand() % 2;

    sgd->setInitialLearningRate(initialLearningRate);
    sgd->setDecay(decay);
    sgd->setMomentum(momentum);
    sgd->setUseNesterovMomentum(useNesterovMomentum);

    ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
    ASSERT_EQ(sgd->getDecay(), decay);
    ASSERT_EQ(sgd->getMomentum(), momentum);
    ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

    unordered_map<std::string, float> hyperParameters;

    uint64_t epoch = rand() % 10;
    uint64_t batchesPerEpoch = 1 + (rand() % 3000);
    uint64_t batch = rand() % batchesPerEpoch;
    uint8_t flags = 0b00000000;
    const uint8_t CLR = 0b00000001;
    const uint8_t ILR = 0b00000010;
    const uint8_t D = 0b00000100;
    const uint8_t M = 0b00001000;
    const uint8_t UNM = 0b00010000;
    hyperParameters = sgd->getAllHyperParameters(epoch, batch, batchesPerEpoch);
    for (auto it = hyperParameters.begin(); it != hyperParameters.end(); ++it) {
        string parameter = it->first;
        float value = it->second;

        if (parameter == "currentLearningRate") {
            ASSERT_EQ(value, computeCurrentLearningRate(initialLearningRate, decay, epoch));
            ASSERT_EQ(flags & CLR, 0);
            flags |= CLR;
        } else if (parameter == "initialLearningRate") {
            ASSERT_EQ(value, initialLearningRate);
            ASSERT_EQ(flags & ILR, 0);
            flags |= ILR;
        } else if (parameter == "decay") {
            ASSERT_EQ(value, decay);
            ASSERT_EQ(flags & D, 0);
            flags |= D;
        } else if (parameter == "momentum") {
            ASSERT_EQ(value, momentum);
            ASSERT_EQ(flags & M, 0);
            flags |= M;
        } else if (parameter == "useNesterovMomentum") {
            ASSERT_EQ(value, useNesterovMomentum);
            ASSERT_EQ(flags & UNM, 0);
            flags |= UNM;
        } else {
            ASSERT_EQ(true, false);
        }
    }

    epoch = rand() % 10;
    batchesPerEpoch = 1 + (rand() % 3000);
    batch = rand() % batchesPerEpoch;
    flags = 0b00000000;
    hyperParameters = sgd->getAllHyperParameters(epoch, batch, batchesPerEpoch);
    for (auto it = hyperParameters.begin(); it != hyperParameters.end(); ++it) {
        string parameter = it->first;
        float value = it->second;

        if (parameter == "currentLearningRate") {
            ASSERT_EQ(value, computeCurrentLearningRate(initialLearningRate, decay, epoch));
            ASSERT_EQ(flags & CLR, 0);
            flags |= CLR;
        } else if (parameter == "initialLearningRate") {
            ASSERT_EQ(value, initialLearningRate);
            ASSERT_EQ(flags & ILR, 0);
            flags |= ILR;
        } else if (parameter == "decay") {
            ASSERT_EQ(value, decay);
            ASSERT_EQ(flags & D, 0);
            flags |= D;
        } else if (parameter == "momentum") {
            ASSERT_EQ(value, momentum);
            ASSERT_EQ(flags & M, 0);
            flags |= M;
        } else if (parameter == "useNesterovMomentum") {
            ASSERT_EQ(value, useNesterovMomentum);
            ASSERT_EQ(flags & UNM, 0);
            flags |= UNM;
        } else {
            ASSERT_EQ(true, false);
        }
    }
}

TEST(SgdTest, TestWeightsUpdate) {}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
