#include "Thor.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <vector>

using namespace std;

using namespace Thor;

static Network buildNetwork(uint32_t numFCLayers) {
    Network network;
    Tensor latestOutputTensor;
    UniformRandom::Builder uniformRandomInitializerBuilder = UniformRandom::Builder().minValue(-0.1).maxValue(0.1);

    NetworkInput networkInput =
        NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(Tensor::DataType::FP16).build();
    NetworkInput labels =
        NetworkInput::Builder().network(network).name("labels").dimensions({500}).dataType(Tensor::DataType::FP16).build();
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

    CategoricalCrossEntropy lossLayer = CategoricalCrossEntropy::Builder()
                                            .network(network)
                                            .predictions(latestOutputTensor)
                                            .labels(labels.getFeatureOutput())
                                            .reportsBatchLoss()
                                            .receivesOneHotLabels()
                                            .build();

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(network).name("output").inputTensor(lossLayer.getLoss()).dataType(Tensor::DataType::FP16).build();

    return network;
}

TEST(Adam, Builds) {
    Optional<Network *> optionalTest;
    ASSERT_FALSE(optionalTest.isPresent());

    Network network = buildNetwork(2);
    shared_ptr<Adam> adam = Adam::Builder().network(network).alpha(0.01).beta1(0.9).beta2(0).epsilon(0.0003).build();
    ASSERT_NE(adam, nullptr);
}

TEST(Adam, InitializesParametersWithOneStamp) {
    Network network = buildNetwork(2);
    float alpha = 0.72;
    float beta1 = 0.9;
    float beta2 = 0.0;
    float epsilon = 1e-5f;
    shared_ptr<Adam> adam = Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).network(network).build();

    Network::StatusCode statusCode = network.place(32, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    uint32_t epoch = 0;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 10;
    unordered_map<string, float> params = adam->updateHyperParameters(epoch, batch, batchesPerEpoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params.size(), 1U);
    ASSERT_EQ(params["t"], 0.0f);
}

/* FIXME: put this back in once multile stamps is supported
TEST(Adam, InitializesParametersWithTwoStamps) {
    Network network = buildNetwork(2);
    float alpha = 0.25;
    float beta1 = 0.8;
    float beta2 = 0.0;
    shared_ptr<Adam> adam =
        Adam::Builder().network(network).alpha(alpha).beta1(beta1).beta2(beta2).epsilon(true).build();

    Network::StatusCode statusCode = network.place(32, {0}, 2);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    uint32_t epoch = 0;
    uint32_t batch = 4;
    uint32_t batchesPerEpoch = 25;
    unordered_map<string, float> params = adam->updateHyperParameters(epoch, batch, batchesPerEpoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params.size(), 1U);
    ASSERT_EQ(params["t"], 1.0f);
}
 */

TEST(Adam, ReportsParameters) {
    Network network = buildNetwork(2);
    float alpha = 0.5;
    float beta1 = 0.32;
    float beta2 = 0.6;
    float epsilon = 1e-8f;
    shared_ptr<Adam> adam = Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).network(network).build();

    ThorImplementation::StampedNetwork stampedNetwork0;
    Network::StatusCode statusCode = network.place(32, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    unordered_map<string, float> params = adam->getAllHyperParameters(10, 3, 100);

    ASSERT_EQ(params.size(), 5U);
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params["t"], 0);
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params["alpha"], alpha);
    ASSERT_EQ(params.count("beta1"), 1U);
    ASSERT_EQ(params["beta1"], beta1);
    ASSERT_EQ(params.count("beta2"), 1U);
    ASSERT_EQ(params["beta2"], beta2);
    ASSERT_EQ(params.count("epsilon"), 1U);
    // Since fp16 is used, the minimum epsilon is around 5e-8f, check that this is enforced.
    ASSERT_GT(params["epsilon"], 1e-8f);

    // Ensure that optimizer is connected to each trainable layer and its paratmeters are initialized properly
    vector<shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer>> trainableLayers = stampedNetwork0.getTrainableLayers();
    for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
        shared_ptr<ThorImplementation::FullyConnected> fc = dynamic_pointer_cast<ThorImplementation::FullyConnected>(trainableLayers[i]);
        ASSERT_NE(fc, nullptr);
        Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = fc->getOptimizer();
        assert(maybeOptimizer.isPresent());
        shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
        shared_ptr<ThorImplementation::Adam> adam = dynamic_pointer_cast<ThorImplementation::Adam>(optimizer);
        ASSERT_NE(adam, nullptr);
        ASSERT_EQ(adam->getT(), 0);
        ASSERT_EQ(adam->getAlpha(), alpha);
        ASSERT_EQ(adam->getBeta1(), beta1);
        ASSERT_EQ(adam->getBeta2(), beta2);
        ASSERT_EQ(adam->getEpsilon(), epsilon);
    }
}

TEST(Adam, SettersAndGetters) {
    Network network = buildNetwork(2);
    float alpha = 0.5;
    float beta1 = 0.32;
    float beta2 = 0.6;
    float epsilon = 1e-6f;
    shared_ptr<Adam> adam = Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).network(network).build();

    ThorImplementation::StampedNetwork stampedNetwork0;
    Network::StatusCode statusCode = network.place(32, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    unordered_map<string, float> params = adam->getAllHyperParameters(10, 3, 100);

    ASSERT_EQ(params.size(), 5U);
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params["t"], 0);
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params["alpha"], alpha);
    ASSERT_EQ(params.count("beta1"), 1U);
    ASSERT_EQ(params["beta1"], beta1);
    ASSERT_EQ(params.count("beta2"), 1U);
    ASSERT_EQ(params["beta2"], beta2);
    ASSERT_EQ(params.count("epsilon"), 1U);
    ASSERT_EQ(params["epsilon"], epsilon);

    // Test the setters
    alpha = 0.75f;
    beta1 = 0.65f;
    beta2 = 0.77f;
    epsilon = 1e-4f;
    adam->setAlpha(alpha);
    EXPECT_EQ(adam->getAlpha(), alpha);
    adam->setBeta1(beta1);
    EXPECT_EQ(adam->getBeta1(), beta1);
    adam->setBeta2(beta2);
    EXPECT_EQ(adam->getBeta2(), beta2);
    adam->setEpsilon(epsilon);
    EXPECT_EQ(adam->getEpsilon(), epsilon);

    // Ensure that optimizer is connected to each trainable layer and its paratmeters are initialized properly
    vector<shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer>> trainableLayers = stampedNetwork0.getTrainableLayers();
    for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
        shared_ptr<ThorImplementation::FullyConnected> fc = dynamic_pointer_cast<ThorImplementation::FullyConnected>(trainableLayers[i]);
        ASSERT_NE(fc, nullptr);
        Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = fc->getOptimizer();
        assert(maybeOptimizer.isPresent());
        shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
        shared_ptr<ThorImplementation::Adam> adam = dynamic_pointer_cast<ThorImplementation::Adam>(optimizer);
        ASSERT_NE(adam, nullptr);
        ASSERT_EQ(adam->getT(), 0);
        ASSERT_EQ(adam->getAlpha(), alpha);
        ASSERT_EQ(adam->getBeta1(), beta1);
        ASSERT_EQ(adam->getBeta2(), beta2);
        ASSERT_EQ(adam->getEpsilon(), epsilon);
    }
}
