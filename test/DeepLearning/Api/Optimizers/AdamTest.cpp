#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Optimizers/Adam.h"

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
    shared_ptr<Initializer> uniformRandomInitializer = UniformRandom::Builder().minValue(-0.1).maxValue(0.1).build();

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
                                            .weightsInitializer(uniformRandomInitializer)
                                            .biasInitializer(uniformRandomInitializer)
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

    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, false, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    uint32_t epoch = 0;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 10;
    Optimizer::updateHyperParameters(&network, epoch, batch, batchesPerEpoch);
    unordered_map<string, float> params = adam->getAllHyperParameters();

    // Check that the proper values are reported
    ASSERT_EQ(params.size(), 5U);
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params["t"], 0.0f);
    ASSERT_EQ(params.count("alpha"), 1U);
    ASSERT_EQ(params["alpha"], alpha);
    ASSERT_EQ(params.count("t"), 1U);
    ASSERT_EQ(params["beta1"], beta1);
    ASSERT_EQ(params.count("beta2"), 1U);
    ASSERT_EQ(params["beta2"], beta2);
    ASSERT_EQ(params.count("epsilon"), 1U);
    ASSERT_EQ(params["epsilon"], epsilon);
}

/* FIXME: put this back in once multile stamps is supported
TEST(Adam, InitializesParametersWithTwoStamps) {
    Network network = buildNetwork(2);
    float alpha = 0.25;
    float beta1 = 0.8;
    float beta2 = 0.0;
    shared_ptr<Adam> adam =
        Adam::Builder().network(network).alpha(alpha).beta1(beta1).beta2(beta2).epsilon(true).build();

    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, {0}, 2);
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
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, false, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    unordered_map<string, float> params = adam->getAllHyperParameters();

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

    // Ensure that optimizer is connected to each trainable layer and its paratmeters are initialized properly
    for (uint32_t i = 0; i < stampedNetwork0.getNumTrainableLayers(); ++i) {
        shared_ptr<ThorImplementation::FullyConnected> fc =
            dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork0.getTrainableLayer(i));
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
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, false, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    unordered_map<string, float> params = adam->getAllHyperParameters();

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
    for (uint32_t i = 0; i < stampedNetwork0.getNumTrainableLayers(); ++i) {
        shared_ptr<ThorImplementation::FullyConnected> fc =
            dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork0.getTrainableLayer(i));
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
