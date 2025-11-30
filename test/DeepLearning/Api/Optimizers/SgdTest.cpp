#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

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

TEST(Sgd, SetAndGetInitialLearningRate) {
    Network network = buildNetwork(3);
    shared_ptr<Sgd> sgd = Sgd::Builder().initialLearningRate(0.2f).network(network).build();

    sgd->setInitialLearningRate(0.1f);
    ASSERT_FLOAT_EQ(0.1f, sgd->getInitialLearningRate());

    for (uint32_t i = 0; i < network.getNumStamps(); ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network.getStampedNetwork(i);
        for (uint32_t j = 0; j < stampedNetwork.getNumTrainableLayers(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            ASSERT_EQ(sgd->getInitialLearningRate(), 0.1f);
        }
    }
}

TEST(Sgd, SetAndGetDecay) {
    Network network = buildNetwork(4);
    shared_ptr<Sgd> sgd = Sgd::Builder().decay(0.5f).network(network).build();

    sgd->setDecay(0.2f);
    ASSERT_FLOAT_EQ(0.2f, sgd->getDecay());

    for (uint32_t i = 0; i < network.getNumStamps(); ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network.getStampedNetwork(i);
        for (uint32_t j = 0; j < stampedNetwork.getNumTrainableLayers(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            ASSERT_EQ(sgd->getDecay(), 0.2f);
        }
    }
}

TEST(Sgd, SetAndGetMomentum) {
    Network network = buildNetwork(5);
    shared_ptr<Sgd> sgd = Sgd::Builder().decay(0.5f).network(network).build();

    sgd->setMomentum(0.3f);
    ASSERT_FLOAT_EQ(0.3f, sgd->getMomentum());

    for (uint32_t i = 0; i < network.getNumStamps(); ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network.getStampedNetwork(i);
        for (uint32_t j = 0; j < stampedNetwork.getNumTrainableLayers(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            ASSERT_EQ(sgd->getMomentum(), 0.3f);
        }
    }
}

TEST(Sgd, SetAndGetUseNesterovMomentum) {
    Network network = buildNetwork(6);
    shared_ptr<Sgd> sgd = Sgd::Builder().useNesterovMomentum(false).network(network).build();

    sgd->setUseNesterovMomentum(true);
    ASSERT_TRUE(sgd->getUseNesterovMomentum());

    for (uint32_t i = 0; i < network.getNumStamps(); ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network.getStampedNetwork(i);
        for (uint32_t j = 0; j < stampedNetwork.getNumTrainableLayers(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            ASSERT_EQ(sgd->getUseNesterovMomentum(), true);
        }
    }

    sgd->setUseNesterovMomentum(false);
    ASSERT_FALSE(sgd->getUseNesterovMomentum());

    for (uint32_t i = 0; i < network.getNumStamps(); ++i) {
        ThorImplementation::StampedNetwork &stampedNetwork = network.getStampedNetwork(i);
        for (uint32_t j = 0; j < stampedNetwork.getNumTrainableLayers(); ++j) {
            shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer = stampedNetwork.getTrainableLayer(j);
            Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = trainableLayer->getOptimizer();
            assert(maybeOptimizer.isPresent());
            shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
            shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
            assert(sgd != NULL);
            ASSERT_EQ(sgd->getUseNesterovMomentum(), false);
        }
    }
}

TEST(Sgd, SgdBuilds) {
    Optional<Network *> optionalTest;
    ASSERT_FALSE(optionalTest.isPresent());

    Network network = buildNetwork(2);
    shared_ptr<Sgd> sgd =
        Sgd::Builder().network(network).initialLearningRate(0.01).decay(0.9).momentum(0).useNesterovMomentum(true).build();
    ASSERT_NE(sgd, nullptr);
}

TEST(Sgd, SgdInitializesParametersWithOneStamp) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.72;
    float decay = 0.9;
    float momentum = 0.0;
    shared_ptr<Sgd> sgd = Sgd::Builder()
                              .initialLearningRate(initialLearningRate)
                              .decay(decay)
                              .momentum(momentum)
                              .useNesterovMomentum(true)
                              .network(network)
                              .build();

    ThorImplementation::StampedNetwork stampedNetwork0;
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    uint32_t epoch = 0;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 10;
    Optimizer::updateHyperParameters(&network, epoch, batch, batchesPerEpoch);
    unordered_map<string, float> params = sgd->getAllHyperParameters();

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 5U);
    ASSERT_LT(abs(params["currentLearningRate"] - initialLearningRate), 0.0001);
    ASSERT_EQ(params["useNesterovMomentum"], float(true));
    ASSERT_EQ(params["momentum"], momentum);
    ASSERT_EQ(params["initialLearningRate"], initialLearningRate);
    ASSERT_EQ(params["decay"], decay);
}

/* FIXME: put this back in once multiple stamps is supported
TEST(Sgd, SgdInitializesParametersWithTwoStamps) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.25;
    float decay = 0.8;
    float momentum = 0.0;
    shared_ptr<Sgd> sgd =
        Sgd::Builder().network(network).initialLearningRate(initialLearningRate).decay(decay).momentum(momentum).useNesterovMomentum(true).build();

    ThorImplementation::StampedNetwork stampedNetwork0;
    ThorImplementation::StampedNetwork stampedNetwork1;
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, {0}, 2);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    uint32_t epoch = 0;
    uint32_t batch = 4;
    uint32_t batchesPerEpoch = 25;
    unordered_map<string, float> params = sgd->updateHyperParameters(epoch, batch, batchesPerEpoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 1U);
    ASSERT_LT(abs(params["currentLearningRate"] - initialLearningRate), 0.0001);
}
 */

TEST(Sgd, SgdUpdatesParameters) {
    Network network = buildNetwork(1);
    float initialLearningRate = 0.43;
    float decay = 0.8;
    float momentum = 0.0;
    shared_ptr<Sgd> sgd = Sgd::Builder()
                              .network(network)
                              .initialLearningRate(initialLearningRate)
                              .decay(decay)
                              .momentum(momentum)
                              .useNesterovMomentum(true)
                              .build();

    ThorImplementation::StampedNetwork stampedNetwork0;
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    Optimizer::updateHyperParameters(&network, 0, 0, 10);
    uint32_t epoch = 5;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 50;
    Optimizer::updateHyperParameters(&network, epoch, batch, batchesPerEpoch);
    unordered_map<string, float> params = sgd->getAllHyperParameters();
    float expected = initialLearningRate * pow(1.0f - decay, epoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 5U);
    ASSERT_LT(abs(params["currentLearningRate"] - expected), 0.0001);
    ASSERT_EQ(params["useNesterovMomentum"], float(true));
    ASSERT_EQ(params["momentum"], momentum);
    ASSERT_EQ(params["initialLearningRate"], initialLearningRate);
    ASSERT_EQ(params["decay"], decay);
}

TEST(Sgd, SgdInitializesStampedNetworkParameters) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.5;
    float decay = 0.32;
    float momentum = 0.0;
    shared_ptr<Sgd> sgd = Sgd::Builder()
                              .initialLearningRate(initialLearningRate)
                              .decay(decay)
                              .momentum(momentum)
                              .useNesterovMomentum(true)
                              .network(network)
                              .build();

    ThorImplementation::StampedNetwork stampedNetwork0;
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    Optimizer::updateHyperParameters(&network, 0, 0, 10);
    uint32_t epoch = 1;
    uint32_t batch = 0;
    uint32_t batchesPerEpoch = 50;
    Optimizer::updateHyperParameters(&network, epoch, batch, batchesPerEpoch);
    unordered_map<string, float> params = sgd->getAllHyperParameters();
    float expected = initialLearningRate * pow(1.0f - decay, epoch);

    // Check that the proper values are reported
    ASSERT_EQ(params.count("currentLearningRate"), 1U);
    ASSERT_EQ(params.size(), 5U);
    ASSERT_LT(abs(params["currentLearningRate"] - expected), 0.0001);
    ASSERT_EQ(params["useNesterovMomentum"], float(true));
    ASSERT_EQ(params["momentum"], momentum);
    ASSERT_EQ(params["initialLearningRate"], initialLearningRate);
    ASSERT_EQ(params["decay"], decay);
}

TEST(Sgd, SgdReportsParameters) {
    Network network = buildNetwork(2);
    float initialLearningRate = 0.5;
    float decay = 0.32;
    float momentum = 0.0;
    bool useNesteroveMomentum = true;
    shared_ptr<Sgd> sgd = Sgd::Builder()
                              .initialLearningRate(initialLearningRate)
                              .decay(decay)
                              .momentum(momentum)
                              .useNesterovMomentum(useNesteroveMomentum)
                              .network(network)
                              .build();

    ThorImplementation::StampedNetwork stampedNetwork0;
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode = network.place(32, initDoneEvents, {0}, 1);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);

    Optimizer::updateHyperParameters(&network, 0, 1, 10);
    unordered_map<string, float> params = sgd->getAllHyperParameters();

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
    ASSERT_EQ(params.count("useNesterovMomentum"), 1U);
    ASSERT_TRUE(params["useNesterovMomentum"]);

    // Ensure that optimizer is connected to each trainable layer and its paratmeters are initialized properly
    for (uint32_t i = 0; i < stampedNetwork0.getNumTrainableLayers(); ++i) {
        shared_ptr<ThorImplementation::FullyConnected> fc =
            dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork0.getTrainableLayer(i));
        ASSERT_NE(fc, nullptr);
        Optional<shared_ptr<ThorImplementation::Optimizer>> maybeOptimizer = fc->getOptimizer();
        assert(maybeOptimizer.isPresent());
        shared_ptr<ThorImplementation::Optimizer> optimizer = maybeOptimizer.get();
        shared_ptr<ThorImplementation::Sgd> sgd = dynamic_pointer_cast<ThorImplementation::Sgd>(optimizer);
        ASSERT_NE(sgd, nullptr);
        ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
        ASSERT_EQ(sgd->getDecay(), decay);
        ASSERT_EQ(sgd->getMomentum(), momentum);
        ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesteroveMomentum);
    }

    Optimizer::updateHyperParameters(&network, 0, 0, 10);
    uint32_t epoch = 2;
    uint32_t batch = 3;
    uint32_t batchesPerEpoch = 50;
    Optimizer::updateHyperParameters(&network, epoch, batch, batchesPerEpoch);
    float expected = initialLearningRate * pow(1.0f - decay, epoch);
    params.clear();
    params = sgd->getAllHyperParameters();

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
    ASSERT_EQ(params.count("useNesterovMomentum"), 1U);
    ASSERT_TRUE(params["useNesterovMomentum"]);
}

TEST(ApiSgd, Update) {}

// FIXME: Test stamp directly. Connect it to tensors and set its optimizer, set its weights and gradients and test backward to
// ensure the stamped layer's SGD is numerically correct.
