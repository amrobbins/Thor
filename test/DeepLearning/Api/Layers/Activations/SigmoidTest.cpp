#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Activations, SigmoidBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Sigmoid::Builder sigmoidBuilder;
    sigmoidBuilder.network(network);
    sigmoidBuilder.featureInput(featureInput);
    shared_ptr<Sigmoid> sigmoid = dynamic_pointer_cast<Sigmoid>(sigmoidBuilder.build());

    ASSERT_TRUE(sigmoid->isInitialized());

    Optional<Tensor> actualInput = sigmoid->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = sigmoid->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = sigmoid->clone();
    Sigmoid *clone = dynamic_cast<Sigmoid *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(sigmoid->getId(), clone->getId());
    ASSERT_GT(sigmoid->getId(), 1u);

    ASSERT_TRUE(*sigmoid == *clone);
    ASSERT_FALSE(*sigmoid != *clone);
    ASSERT_FALSE(*sigmoid > *clone);
    ASSERT_FALSE(*sigmoid < *clone);
}

TEST(Activations, SigmoidSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Sigmoid::Builder sigmoidBuilder = Sigmoid::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput());
    shared_ptr<Sigmoid> sigmoid = dynamic_pointer_cast<Sigmoid>(sigmoidBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(sigmoid->getFeatureOutput())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(sigmoid->isInitialized());

    Tensor featureInput = sigmoid->getFeatureInput();
    Tensor featureOutput = sigmoid->getFeatureOutput();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(sigmoid->getFeatureOutput().isPresent());
    ASSERT_EQ(sigmoid->getFeatureOutput().get(), featureOutput);

    ASSERT_TRUE(sigmoid->getFeatureInput().isPresent());
    assert(sigmoid->getFeatureInput().get() == featureInput);

    ASSERT_EQ(featureInput.getDataType(), dataType);
    ASSERT_EQ(featureInput.getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutput.getDataType(), dataType);
    ASSERT_EQ(featureOutput.getDimensions(), inputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode placementStatus;
    placementStatus = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the layer from the network
    vector<ThorImplementation::StampedNetwork> stampedNetworks = initialNetwork.getStampedNetworks();
    ASSERT_EQ(stampedNetworks.size(), 1UL);
    ThorImplementation::StampedNetwork stampedNetwork = stampedNetworks[0];

    json sigmoidJ = sigmoid->serialize("/tmp/", stream);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    ASSERT_EQ(sigmoidJ["factory"], "activation");
    ASSERT_EQ(sigmoidJ["version"], "1.0.0");
    ASSERT_EQ(sigmoidJ["layer_type"], "sigmoid");

    EXPECT_TRUE(sigmoidJ.contains("feature_input"));
    EXPECT_TRUE(sigmoidJ.contains("feature_output"));

    const auto &input = sigmoidJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    string dataTypeString = dataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = sigmoidJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    //     printf("%s\n", networkInputJ.dump(4).c_str());
    //     printf("%s\n", sigmoidJ.dump(4).c_str());
    //     printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Sigmoid::deserialize(sigmoidJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    placementStatus = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    stampedNetworks.clear();
    stampedNetworks = newNetwork.getStampedNetworks();
    ASSERT_EQ(stampedNetworks.size(), 1UL);
    stampedNetwork = stampedNetworks[0];

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Sigmoid> stampedSigmoid = dynamic_pointer_cast<ThorImplementation::Sigmoid>(otherLayers[0]);
    ASSERT_NE(stampedSigmoid, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
}

TEST(Activations, SigmoidRegistered) {
    srand(time(nullptr));

    Network initialNetwork;
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    float alpha = float(rand() % 200) / 100.0f;

    Sigmoid::Builder sigmoidBuilder = Sigmoid::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput());
    shared_ptr<Sigmoid> sigmoid = dynamic_pointer_cast<Sigmoid>(sigmoidBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(sigmoid->getFeatureOutput())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(sigmoid->isInitialized());

    Stream stream(0);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json sigmoidJ = sigmoid->serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    // Test that it is registered with Activation to deserialize
    Network newNetwork;
    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Activation::deserialize(sigmoidJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    vector<Event> initDoneEvents;
    uint32_t batchSize = 1 + (rand() % 16);
    Network::StatusCode placementStatus = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    vector<ThorImplementation::StampedNetwork> stampedNetworks = newNetwork.getStampedNetworks();
    ASSERT_EQ(stampedNetworks.size(), 1UL);
    ThorImplementation::StampedNetwork stampedNetwork = stampedNetworks[0];

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Sigmoid> stampedSigmoid = dynamic_pointer_cast<ThorImplementation::Sigmoid>(otherLayers[0]);
    ASSERT_NE(stampedSigmoid, nullptr);
}
