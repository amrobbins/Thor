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

#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"

using json = nlohmann::json;

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

inline float randFloat(float lo = 0.0f, float hi = 1.0f) {
    static uint32_t state = static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ 0x9E3779B9u;

    state = state * 1664525u + 1013904223u;
    const float t = (state >> 8) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * t;
}

TEST(Adam, SerializeDeserialize) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 4; ++t) {
        Network initialNetwork;

        Tensor::DataType dataType = Tensor::DataType::FP16;
        vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};
        uint32_t numOutputFeatures = 1 + (rand() % 1000);

        NetworkInput networkInput =
            NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

        bool hasBias = rand() % 2;
        FullyConnected::Builder fullyConnectedBuilder = FullyConnected::Builder()
                                                            .network(initialNetwork)
                                                            .featureInput(networkInput.getFeatureOutput())
                                                            .hasBias(hasBias)
                                                            .noActivation()
                                                            .numOutputFeatures(numOutputFeatures);

        FullyConnected fullyConnected = fullyConnectedBuilder.build();

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

        float alpha = randFloat(0.1, 0.9);
        float beta1 = randFloat(0.1, 0.9);
        float beta2 = randFloat(0.1, 0.9);
        float epsilon = randFloat(0.1, 0.9);
        shared_ptr<Adam> adam = Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).network(initialNetwork).build();

        NetworkOutput networkOutput = NetworkOutput::Builder()
                                          .network(initialNetwork)
                                          .name("testOutput")
                                          .inputTensor(meanAbsoluteError.getLoss())
                                          .dataType(dataType)
                                          .build();

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

        // Fetch the fully connected layer from the network
        ASSERT_EQ(initialNetwork.getNumStamps(), 1UL);
        ThorImplementation::StampedNetwork &stampedNetwork = initialNetwork.getStampedNetwork(0);
        ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
        shared_ptr<ThorImplementation::FullyConnected> physicalFCLayer =
            dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(0));
        ASSERT_TRUE(physicalFCLayer != nullptr);
        shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalFCLayer->getOptimizer();
        shared_ptr<ThorImplementation::Adam> physicalAdam = dynamic_pointer_cast<ThorImplementation::Adam>(physicalOptimizer);

        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

        // Check initialization and if saving state, write some state to check that it got saved and restored
        bool saveOptimizerState = rand() % 2;
        ThorImplementation::Tensor m = physicalAdam->getM().clone(cpuPlacement);
        ThorImplementation::Tensor v = physicalAdam->getV().clone(cpuPlacement);
        m.copyFromAsync(physicalAdam->getM(), stream);
        v.copyFromAsync(physicalAdam->getV(), stream);

        ASSERT_EQ(physicalAdam->getM().getDataType(), ThorImplementation::TensorDescriptor::DataType::FP16);
        ASSERT_EQ(physicalAdam->getV().getDataType(), ThorImplementation::TensorDescriptor::DataType::FP16);

        ThorImplementation::Tensor mBias;
        ThorImplementation::Tensor vBias;
        if (hasBias) {
            mBias = physicalAdam->getMBias().get().clone(cpuPlacement);
            vBias = physicalAdam->getVBias().get().clone(cpuPlacement);
            mBias.copyFromAsync(physicalAdam->getMBias(), stream);
            vBias.copyFromAsync(physicalAdam->getVBias(), stream);
        }
        stream.synchronize();

        half *mPtr = m.getMemPtr<half>();
        half *vPtr = v.getMemPtr<half>();
        half *mBiasPtr;
        half *vBiasPtr;
        if (hasBias) {
            mBiasPtr = mBias.getMemPtr<half>();
            vBiasPtr = vBias.getMemPtr<half>();
        }

        for (uint32_t in = 0; in < inputDimensions[0]; ++in) {
            for (uint32_t out = 0; out < numOutputFeatures; ++out) {
                uint32_t index = in * numOutputFeatures + out;
                ASSERT_EQ(float(mPtr[index]), 0.0f);
                ASSERT_EQ(float(vPtr[index]), 0.0f);
            }
        }
        if (hasBias) {
            for (uint32_t out = 0; out < numOutputFeatures; ++out) {
                ASSERT_EQ(float(mBiasPtr[out]), 0.0f);
                ASSERT_EQ(float(vBiasPtr[out]), 0.0f);
            }
        }

        // Let's write some numbers into optimizer state when it is being saved to ensure it is preserved
        if (saveOptimizerState) {
            for (uint32_t in = 0; in < inputDimensions[0]; ++in) {
                for (uint32_t out = 0; out < numOutputFeatures; ++out) {
                    uint32_t index = in * numOutputFeatures + out;
                    mPtr[index] = half(randFloat(-2.0f, 2.0f));
                    vPtr[index] = half(randFloat(-2.0f, 2.0f));
                }
            }
            physicalAdam->getM().copyFromAsync(m, stream);
            physicalAdam->getV().copyFromAsync(v, stream);
            if (hasBias) {
                for (uint32_t out = 0; out < numOutputFeatures; ++out) {
                    mBiasPtr[out] = half(randFloat(-2.0f, 2.0f));
                    vBiasPtr[out] = half(randFloat(-2.0f, 2.0f));
                }
                physicalAdam->getMBias().get().copyFromAsync(mBias, stream);
                physicalAdam->getVBias().get().copyFromAsync(vBias, stream);
            }
            stream.synchronize();
        }

        // The network attached the optimizer to its copy of the FC layer
        json fullyConnectedJ;
        bool fcFound = false;
        shared_ptr<FullyConnected> initalNetworkFC;
        for (int32_t i = 0; i < initialNetwork.getNumTrainableLayers(); ++i) {
            shared_ptr<TrainableWeightsBiasesLayer> layer = initialNetwork.getTrainableLayer(i);
            initalNetworkFC = dynamic_pointer_cast<FullyConnected>(layer);
            if (initalNetworkFC) {
                fullyConnectedJ = initalNetworkFC->serialize(archiveWriter, stream, saveOptimizerState);
                fcFound = true;
                break;
            }
        }
        ASSERT_TRUE(fcFound);

        json networkInputJ = networkInput.serialize(archiveWriter, stream);
        json labelsInputJ = labelsInput.serialize(archiveWriter, stream);
        json meanAbsoluteErrorJ = meanAbsoluteError.serialize(archiveWriter, stream);
        json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

        ASSERT_TRUE(fullyConnectedJ.contains("optimizer"));
        json adamJ = fullyConnectedJ["optimizer"];

        // printf("%s\n", networkInputJ.dump(4).c_str());
        // printf("%s\n", labelsInputJ.dump(4).c_str());
        // printf("%s\n", fullyConnectedJ.dump(4).c_str());
        // printf("%s\n", meanAbsoluteErrorJ.dump(4).c_str());
        // printf("%s\n", networkOutputJ.dump(4).c_str());
        // printf("--------------------\n");
        // printf("%s\n", adamJ.dump(4).c_str());

        ASSERT_EQ(adamJ.at("optimizer_type").get<string>(), "adam");
        ASSERT_EQ(adamJ.at("version").get<string>(), adam->getVersion());
        ASSERT_EQ(adamJ.at("t").get<float>(), 0.0f);
        ASSERT_EQ(adamJ.at("alpha").get<float>(), alpha);
        ASSERT_EQ(adamJ.at("beta1").get<float>(), beta1);
        ASSERT_EQ(adamJ.at("beta2").get<float>(), beta2);
        ASSERT_EQ(adamJ.at("epsilon").get<float>(), epsilon);

        ASSERT_EQ(saveOptimizerState, adamJ.contains("m_tensor"));
        ASSERT_EQ(saveOptimizerState, adamJ.contains("v_tensor"));
        ASSERT_EQ(saveOptimizerState && hasBias, adamJ.contains("m_bias_tensor"));
        ASSERT_EQ(saveOptimizerState && hasBias, adamJ.contains("v_bias_tensor"));

        archiveWriter.finishArchive();
        shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");
        Network newNetwork;

        // Deserialize everything
        // place
        // Ensure adam weights are correctly initial values

        Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
        Layer::deserialize(archiveReader, labelsInputJ, &newNetwork);
        Layer::deserialize(archiveReader, fullyConnectedJ, &newNetwork);
        Layer::deserialize(archiveReader, meanAbsoluteErrorJ, &newNetwork);
        Layer::deserialize(archiveReader, networkOutputJ, &newNetwork);

        batchSize = 1 + (rand() % 16);
        statusCode = newNetwork.place(batchSize, initDoneEvents);
        ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
        for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
            stream.waitEvent(initDoneEvents[i]);
        }
        initDoneEvents.clear();

        // Find the FC to find the Adam
        ThorImplementation::StampedNetwork &newStampedNetwork = newNetwork.getStampedNetwork(0);
        shared_ptr<ThorImplementation::FullyConnected> physicalFCLayerDes;
        for (uint32_t i = 0; i < newStampedNetwork.getNumTrainableLayers(); ++i) {
            physicalFCLayerDes = dynamic_pointer_cast<ThorImplementation::FullyConnected>(newStampedNetwork.getTrainableLayer(i));
            if (physicalFCLayerDes != nullptr)
                break;
        }
        ASSERT_TRUE(physicalFCLayerDes != nullptr);

        shared_ptr<ThorImplementation::Adam> physicalAdamDes =
            dynamic_pointer_cast<ThorImplementation::Adam>(physicalFCLayerDes->getOptimizer());

        ThorImplementation::Tensor mDeser = physicalAdamDes->getM().clone(cpuPlacement);
        ThorImplementation::Tensor vDeser = physicalAdamDes->getV().clone(cpuPlacement);
        mDeser.copyFromAsync(physicalAdamDes->getM(), stream);
        vDeser.copyFromAsync(physicalAdamDes->getV(), stream);

        ThorImplementation::Tensor mBiasDeser;
        ThorImplementation::Tensor vBiasDeser;
        if (hasBias) {
            mBiasDeser = physicalAdamDes->getMBias().get().clone(cpuPlacement);
            vBiasDeser = physicalAdamDes->getVBias().get().clone(cpuPlacement);
            mBiasDeser.copyFromAsync(physicalAdamDes->getMBias(), stream);
            vBiasDeser.copyFromAsync(physicalAdamDes->getVBias(), stream);
        }
        stream.synchronize();

        half *mDeserPtr = mDeser.getMemPtr<half>();
        half *vDeserPtr = vDeser.getMemPtr<half>();
        half *mBiasDeserPtr;
        half *vBiasDeserPtr;
        if (hasBias) {
            mBiasDeserPtr = mBiasDeser.getMemPtr<half>();
            vBiasDeserPtr = vBiasDeser.getMemPtr<half>();
        }

        for (uint32_t in = 0; in < inputDimensions[0]; ++in) {
            for (uint32_t out = 0; out < numOutputFeatures; ++out) {
                uint32_t index = in * numOutputFeatures + out;
                ASSERT_EQ(float(mPtr[index]), float(mDeserPtr[index]));
                ASSERT_EQ(float(vPtr[index]), float(vDeserPtr[index]));
            }
        }
        if (hasBias) {
            for (uint32_t out = 0; out < numOutputFeatures; ++out) {
                ASSERT_EQ(float(mBiasPtr[out]), float(mBiasDeserPtr[out]));
                ASSERT_EQ(float(vBiasPtr[out]), float(vBiasDeserPtr[out]));
            }
        }
    }

    filesystem::remove("/tmp/testModel.thor.tar");
}
