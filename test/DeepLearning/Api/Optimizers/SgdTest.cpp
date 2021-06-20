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

using namespace Thor;

/*
class Sgd : public Optimizer {
   public:
    class Builder;

    Sgd();
    virtual ~Sgd();

    virtual void setNetwork(Thor::Network *network);

    // returns a map of updated parameters
    std::unordered_map<std::string, float> updateParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    std::unordered_map<std::string, float> initializeStampedNetworkParameters(ThorImplementation::StampedNetwork &stampedNetwork,
                                                                              uint64_t epoch,
                                                                              uint64_t batch,
                                                                              uint64_t batchesPerEpoch);
    std::unordered_map<std::string, float> getAllParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);

   private:
    float initialLearningRate;
    float decay;
    float momentum;
    bool useNesterov;
    uint64_t currentEpoch;
    bool parametersInitialized;

    Thor::Network *network;
};
*/

Network buildNetwork(uint32_t numFCLayers) {
    Network network;
    Tensor latestOutputTensor;
    UniformRandomInitializer::Builder uniformRandomInitializerBuilder = UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1);

    NetworkInput networkInput =
        NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(Tensor::DataType::FP16).build();
    latestOutputTensor = networkInput.getFeatureOutput();

    for (uint32_t i = 0; i < numFCLayers; ++i) {
        FullyConnected fullyConnected = FullyConnected::Builder()
                                            .network(network)
                                            .featureInput(latestOutputTensor)
                                            .numOutputFeatures(500)
                                            .hasBias(true)
                                            .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                                            .biasInitializerBuilder(uniformRandomInitializerBuilder)
                                            .noActivation()
                                            .build();
        latestOutputTensor = fullyConnected.getFeatureOutput();
    }

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(network).name("output").inputTensor(latestOutputTensor).dataType(Tensor::DataType::FP16).build();

    return network;
}

TEST(Sgd, SgdBuilds) {
    std::shared_ptr<Sgd> sgd = Sgd::Builder().initialLearningRate(0.01).decay(0.9).momentum(0).useNesterov(true).build();
    ASSERT_NE(sgd, nullptr);
}

TEST(Sgd, SgdInitializesParametersWithOneStamp) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.72;
    float decay = 0.9;
    float momentum = 0.0;
    std::shared_ptr<Sgd> sgd =
        Sgd::Builder().initialLearningRate(initialLearningRate).decay(decay).momentum(momentum).useNesterov(true).build();
    sgd->setNetwork(&network);

    ThorImplementation::StampedNetwork stampedNetwork0;
    ThorImplementation::StampedNetwork stampedNetwork1;
    network.stampNetwork(0, 5, stampedNetwork0);

    uint32_t epoch = 0;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 10;
    std::unordered_map<std::string, float> params = sgd->updateParameters(epoch, batch, batchesPerEpoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 1U);
    ASSERT_LT(abs(params["currentLearningRate"] - initialLearningRate), 0.0001);

    // Check that the proper values are updated in the network
    ASSERT_LT(abs(stampedNetwork0.trainableLayers[0]->getLearningRate() - initialLearningRate), 0.0001);
    ASSERT_LT(abs(stampedNetwork0.trainableLayers[1]->getLearningRate() - initialLearningRate), 0.0001);
}

TEST(Sgd, SgdInitializesParametersWithTwoStamps) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.25;
    float decay = 0.8;
    float momentum = 0.0;
    std::shared_ptr<Sgd> sgd =
        Sgd::Builder().initialLearningRate(initialLearningRate).decay(decay).momentum(momentum).useNesterov(true).build();
    sgd->setNetwork(&network);

    ThorImplementation::StampedNetwork stampedNetwork0;
    ThorImplementation::StampedNetwork stampedNetwork1;
    network.stampNetwork(0, 5, stampedNetwork0);
    network.stampNetwork(0, 5, stampedNetwork1);

    uint32_t epoch = 0;
    uint32_t batch = 4;
    uint32_t batchesPerEpoch = 25;
    std::unordered_map<std::string, float> params = sgd->updateParameters(epoch, batch, batchesPerEpoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 1U);
    ASSERT_LT(abs(params["currentLearningRate"] - initialLearningRate), 0.0001);

    // Check that the proper values are updated in the network
    ASSERT_LT(abs(stampedNetwork0.trainableLayers[0]->getLearningRate() - initialLearningRate), 0.0001);
    ASSERT_LT(abs(stampedNetwork0.trainableLayers[1]->getLearningRate() - initialLearningRate), 0.0001);
    ASSERT_LT(abs(stampedNetwork1.trainableLayers[0]->getLearningRate() - initialLearningRate), 0.0001);
    ASSERT_LT(abs(stampedNetwork1.trainableLayers[1]->getLearningRate() - initialLearningRate), 0.0001);
}

TEST(Sgd, SgdUpdatesParameters) {
    Network network = buildNetwork(1);
    float initialLearningRate = 0.43;
    float decay = 0.8;
    float momentum = 0.0;
    std::shared_ptr<Sgd> sgd =
        Sgd::Builder().initialLearningRate(initialLearningRate).decay(decay).momentum(momentum).useNesterov(true).build();
    sgd->setNetwork(&network);

    ThorImplementation::StampedNetwork stampedNetwork0;
    ThorImplementation::StampedNetwork stampedNetwork1;
    ThorImplementation::StampedNetwork stampedNetwork2;
    network.stampNetwork(0, 5, stampedNetwork0);
    network.stampNetwork(0, 5, stampedNetwork1);
    network.stampNetwork(0, 5, stampedNetwork2);

    sgd->updateParameters(0, 0, 10);
    uint32_t epoch = 5;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 50;
    std::unordered_map<std::string, float> params = sgd->updateParameters(epoch, batch, batchesPerEpoch);
    float expected = initialLearningRate * pow(decay, epoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 1U);
    ASSERT_LT(abs(params["currentLearningRate"] - expected), 0.0001);

    // Check that the proper values are updated in the network
    ASSERT_LT(abs(stampedNetwork0.trainableLayers[0]->getLearningRate() - expected), 0.0001);
    ASSERT_LT(abs(stampedNetwork1.trainableLayers[0]->getLearningRate() - expected), 0.0001);
    ASSERT_LT(abs(stampedNetwork2.trainableLayers[0]->getLearningRate() - expected), 0.0001);
}

TEST(Sgd, SgdInitializesStampedNetworkParameters) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.5;
    float decay = 0.32;
    float momentum = 0.0;
    std::shared_ptr<Sgd> sgd =
        Sgd::Builder().initialLearningRate(initialLearningRate).decay(decay).momentum(momentum).useNesterov(true).build();
    sgd->setNetwork(&network);

    ThorImplementation::StampedNetwork stampedNetwork0;
    network.stampNetwork(0, 5, stampedNetwork0);

    sgd->updateParameters(0, 0, 10);
    uint32_t epoch = 1;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 50;
    std::unordered_map<std::string, float> params = sgd->initializeStampedNetworkParameters(stampedNetwork0, epoch, batch, batchesPerEpoch);
    float expected = initialLearningRate * pow(decay, epoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 1U);
    ASSERT_LT(abs(params["currentLearningRate"] - expected), 0.0001);

    // Check that the proper values are updated in the network
    ASSERT_LT(abs(stampedNetwork0.trainableLayers[0]->getLearningRate() - expected), 0.0001);
    ASSERT_LT(abs(stampedNetwork0.trainableLayers[1]->getLearningRate() - expected), 0.0001);
}

TEST(Sgd, SgdReportsParameters) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.5;
    float decay = 0.32;
    float momentum = 0.0;
    std::shared_ptr<Sgd> sgd =
        Sgd::Builder().initialLearningRate(initialLearningRate).decay(decay).momentum(momentum).useNesterov(true).build();
    sgd->setNetwork(&network);

    ThorImplementation::StampedNetwork stampedNetwork0;
    network.stampNetwork(0, 5, stampedNetwork0);

    std::unordered_map<std::string, float> params = sgd->getAllParameters(0, 0, 0);

    // Check that the proper values are reported
    ASSERT_EQ(params.size(), 5U);
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_LT(abs(params["currentLearningRate"] - initialLearningRate), 0.0001);
    ASSERT_EQ(params.count("initialLearningRate"), 1U);
    ASSERT_LT(abs(params["initialLearningRate"] - initialLearningRate), 0.0001);
    ASSERT_EQ(params.count("decay"), 1U);
    ASSERT_LT(abs(params["decay"] - decay), 0.0001);
    ASSERT_EQ(params.count("momentum"), 1U);
    ASSERT_LT(abs(params["momentum"] - momentum), 0.0001);
    ASSERT_EQ(params.count("useNesterov"), 1U);
    ASSERT_TRUE(params["useNesterov"]);

    sgd->updateParameters(0, 0, 10);
    uint32_t epoch = 2;
    uint32_t batch = 3;
    uint32_t batchesPerEpoch = 50;
    sgd->updateParameters(epoch, batch, batchesPerEpoch);
    float expected = initialLearningRate * pow(decay, epoch);
    params.clear();
    params = sgd->getAllParameters(epoch, batch, batchesPerEpoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.size(), 5U);
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_LT(abs(params["currentLearningRate"] - expected), 0.0001);
    ASSERT_EQ(params.count("initialLearningRate"), 1U);
    ASSERT_LT(abs(params["initialLearningRate"] - initialLearningRate), 0.0001);
    ASSERT_EQ(params.count("decay"), 1U);
    ASSERT_LT(abs(params["decay"] - decay), 0.0001);
    ASSERT_EQ(params.count("momentum"), 1U);
    ASSERT_LT(abs(params["momentum"] - momentum), 0.0001);
    ASSERT_EQ(params.count("useNesterov"), 1U);
    ASSERT_TRUE(params["useNesterov"]);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
