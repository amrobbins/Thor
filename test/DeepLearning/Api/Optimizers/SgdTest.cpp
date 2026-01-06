#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

#include <stdio.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <vector>

using namespace std;
using json = nlohmann::json;
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

    MeanAbsoluteError lossLayer = MeanAbsoluteError::Builder()
                                      .network(network)
                                      .predictions(latestOutputTensor)
                                      .labels(labels.getFeatureOutput())
                                      .reportsBatchLoss()
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
    Network::StatusCode statusCode = network.place(32, initDoneEvents, false, {0}, 1);
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
    Network::StatusCode statusCode = network.place(32, initDoneEvents, false, {0}, 1);
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
    Network::StatusCode statusCode = network.place(32, initDoneEvents, false, {0}, 1);
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
    Network::StatusCode statusCode = network.place(32, initDoneEvents, false, {0}, 1);
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

TEST(Sgd, SerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType dataType = Tensor::DataType::FP16;

    vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(initialNetwork)
                                        .featureInput(networkInput.getFeatureOutput())
                                        .numOutputFeatures(numOutputFeatures)
                                        .hasBias(hasBias)
                                        .noActivation()
                                        .build();

    Tensor logits = fullyConnected.getFeatureOutputs()[0];
    uint32_t numClasses = logits.getDimensions()[0];
    NetworkInput labelsInput =
        NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions({numClasses}).dataType(dataType).build();

    MeanAbsoluteError meanAbsoluteError = MeanAbsoluteError::Builder()
                                              .network(initialNetwork)
                                              .predictions(logits)
                                              .reportsRawLoss()
                                              .lossDataType(dataType)
                                              .labels(labelsInput.getFeatureOutput())
                                              .build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("lossOutput")
                                      .inputTensor(meanAbsoluteError.getLoss())
                                      .dataType(dataType)
                                      .build();

    float initialLearningRate = (1 + (rand() % 1000)) / 1000.0f;
    float decay = (rand() % 1000) / 1000.0f;
    float momentum = (1 + (rand() % 1000)) / 1000.0f;
    bool useNesterovMomentum = rand() % 2;

    // Looks like I am missing the step where I go over all the API TWB layers and the optimizer
    shared_ptr<Sgd> sgd = Sgd::Builder()
                              .network(initialNetwork)
                              .initialLearningRate(initialLearningRate)
                              .decay(decay)
                              .momentum(momentum)
                              .useNesterovMomentum(useNesterovMomentum)
                              .build();

    ASSERT_TRUE(fullyConnected.isInitialized());

    vector<uint64_t> outputDimensions = {numOutputFeatures};
    vector<Tensor> featureInputs = fullyConnected.getFeatureInputs();
    vector<Tensor> featureOutputs = fullyConnected.getFeatureOutputs();
    assert(featureInputs[0] == networkInput.getFeatureOutput());

    ASSERT_EQ(fullyConnected.getFeatureOutput(networkInput.getFeatureOutput()), featureOutputs[0]);

    assert(fullyConnected.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), outputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode;
    statusCode = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the fully connected layer from the network and write to its weights
    ASSERT_EQ(initialNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialNetwork.getStampedNetwork(0);
    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::FullyConnected> physicalFCLayer =
        dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(0));
    ASSERT_TRUE(physicalFCLayer != nullptr);
    ThorImplementation::Tensor weights = physicalFCLayer->getWeights();
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor weightsCpu = weights.clone(cpuPlacement);
    half *weightsCpuMem = (half *)weightsCpu.getMemPtr();
    for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
        weightsCpuMem[i] = i;
    }
    weights.copyFromAsync(weightsCpu, stream);

    ThorImplementation::Tensor biases;
    ThorImplementation::Tensor biasesCpu;
    if (hasBias) {
        biases = physicalFCLayer->getBiases();
        biasesCpu = biases.clone(cpuPlacement);
        half *biasesCpuMem = (half *)biasesCpu.getMemPtr();
        for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
            biasesCpuMem[i] = i * i + 6;
        }
        biases.copyFromAsync(biasesCpu, stream);
    }

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    // The network attached the optimizer to its copy of the FC layer
    json fullyConnectedJ;
    bool fcFound = false;
    shared_ptr<FullyConnected> initalNetworkFC;
    for (int32_t i = 0; i < initialNetwork.getNumTrainableLayers(); ++i) {
        shared_ptr<TrainableWeightsBiasesLayer> layer = initialNetwork.getTrainableLayer(i);
        initalNetworkFC = dynamic_pointer_cast<FullyConnected>(layer);
        if (initalNetworkFC) {
            fullyConnectedJ = initalNetworkFC->serialize(archiveWriter, stream);
            fcFound = true;
            break;
        }
    }
    ASSERT_TRUE(fcFound);

    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json labelsInputJ = labelsInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);
    json crossEntropyJ = meanAbsoluteError.serialize(archiveWriter, stream);

    ThorImplementation::StampedNetwork &initial_stamped_network = initialNetwork.getStampedNetwork(0);
    shared_ptr<ThorImplementation::Layer> initial_phys_layer = initial_stamped_network.getPhysicalLayerFromApiLayer(fullyConnected.getId());
    shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> initial_phys_twb =
        dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(initial_phys_layer);
    assert(initial_phys_twb != nullptr);
    json sgdJ = sgd->serialize(archiveWriter, stream, &fullyConnected, initial_phys_twb);

    ASSERT_EQ(fullyConnectedJ["version"], "1.0.0");
    ASSERT_EQ(fullyConnectedJ["layer_type"], "fully_connected");

    EXPECT_TRUE(fullyConnectedJ.contains("num_output_features"));
    EXPECT_TRUE(fullyConnectedJ.contains("has_bias"));
    EXPECT_FALSE(fullyConnectedJ.contains("activation"));
    EXPECT_FALSE(fullyConnectedJ.contains("drop_out"));
    EXPECT_FALSE(fullyConnectedJ.contains("batch_normalization"));
    EXPECT_FALSE(fullyConnectedJ.contains("activation"));
    EXPECT_EQ(fullyConnectedJ.contains("biases_tensor"), hasBias);
    EXPECT_TRUE(fullyConnectedJ.contains("weights_tensor"));
    EXPECT_TRUE(fullyConnectedJ.contains("inputs"));
    EXPECT_TRUE(fullyConnectedJ.contains("outputs"));

    ASSERT_TRUE(fullyConnectedJ.at("num_output_features").is_number_integer());
    ASSERT_TRUE(fullyConnectedJ.at("has_bias").is_boolean());
    ASSERT_TRUE(fullyConnectedJ.at("weights_tensor").is_string());
    ASSERT_TRUE(fullyConnectedJ.at("inputs").is_array());
    ASSERT_TRUE(fullyConnectedJ.at("outputs").is_array());

    EXPECT_EQ(fullyConnectedJ.at("num_output_features").get<uint32_t>(), numOutputFeatures);
    EXPECT_EQ(fullyConnectedJ.at("has_bias").get<bool>(), hasBias);

    const auto &inputs = fullyConnectedJ.at("inputs");
    ASSERT_EQ(inputs.size(), 1U) << "Expect exactly one input";
    const auto &in0 = inputs.at(0);
    ASSERT_TRUE(in0.is_object());
    ASSERT_TRUE(in0.at("data_type").is_string());
    EXPECT_EQ(in0.at("data_type").get<string>(), "fp16");

    ASSERT_TRUE(in0.at("dimensions").is_array());
    ASSERT_EQ(in0.at("dimensions").size(), 1U);
    EXPECT_TRUE(in0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(in0.at("dimensions").at(0).get<uint32_t>(), inputDimensions[0]);

    ASSERT_TRUE(in0.at("id").is_number_integer());

    const auto &outputs = fullyConnectedJ.at("outputs");
    ASSERT_EQ(outputs.size(), 1U) << "Expect exactly one output";
    const auto &out0 = outputs.at(0);
    ASSERT_TRUE(out0.is_object());
    ASSERT_TRUE(out0.at("data_type").is_string());
    EXPECT_EQ(out0.at("data_type").get<string>(), "fp16");

    ASSERT_TRUE(out0.at("dimensions").is_array());
    ASSERT_EQ(out0.at("dimensions").size(), 1U);
    EXPECT_TRUE(out0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(out0.at("dimensions").at(0).get<uint32_t>(), numOutputFeatures);

    ASSERT_TRUE(out0.at("id").is_number_integer());

    string file_prefix = "layer" + to_string(fullyConnected.getId());
    EXPECT_FALSE(fullyConnectedJ.at("weights_tensor").get<string>().empty());
    EXPECT_EQ(fullyConnectedJ.at("weights_tensor").get<string>(), file_prefix + "_weights.gds");
    if (hasBias) {
        EXPECT_FALSE(fullyConnectedJ.at("biases_tensor").get<string>().empty());
        EXPECT_EQ(fullyConnectedJ.at("biases_tensor").get<string>(), file_prefix + "_biases.gds");
    }

    const auto &optimizer = fullyConnectedJ.at("optimizer");
    ASSERT_TRUE(optimizer.is_object());
    ASSERT_EQ(optimizer.at("decay").get<float>(), decay);
    ASSERT_EQ(optimizer.at("epoch").get<uint32_t>(), 0);
    ASSERT_EQ(optimizer.at("initial_learning_rate").get<float>(), initialLearningRate);
    ASSERT_EQ(optimizer.at("momentum").get<float>(), momentum);
    ASSERT_EQ(optimizer.at("optimizer_type").get<string>(), "sgd");
    ASSERT_EQ(optimizer.at("use_nesterov").get<bool>(), useNesterovMomentum);
    ASSERT_EQ(optimizer.at("version").get<string>(), "1.0.0");

    // printf("%s\n", networkInputJ.dump(4).c_str());
    // printf("%s\n", labelsInputJ.dump(4).c_str());
    // printf("%s\n", fullyConnectedJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());
    // printf("%s\n", crossEntropyJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    archiveWriter.finishArchive();
    thor_file::TarReader archiveReader("testModel", "/tmp/");

    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, labelsInputJ, &newNetwork);
    Layer::deserialize(archiveReader, fullyConnectedJ, &newNetwork);
    Layer::deserialize(archiveReader, crossEntropyJ, &newNetwork);
    Layer::deserialize(archiveReader, networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    statusCode = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &newStampedNetwork = newNetwork.getStampedNetwork(0);

    // Find the FullyConnected layer and verify its optimizer's parameters
    shared_ptr<ThorImplementation::FullyConnected> physicalFCLayerDes =
        dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(0));
    ASSERT_TRUE(physicalFCLayerDes != nullptr);

    shared_ptr<ThorImplementation::Sgd> stampedSgd = dynamic_pointer_cast<ThorImplementation::Sgd>(physicalFCLayerDes->getOptimizer());
    ASSERT_TRUE(stampedSgd != nullptr);

    ASSERT_EQ(stampedSgd->getInitialLearningRate(), initialLearningRate);
    ASSERT_EQ(stampedSgd->getDecay(), decay);
    ASSERT_EQ(stampedSgd->getMomentum(), momentum);
    ASSERT_EQ(stampedSgd->getUseNesterovMomentum(), useNesterovMomentum);

    filesystem::remove("/tmp/testModel.thor");
}
