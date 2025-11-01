#include "DeepLearning/Api/Layers/Activations/Elu.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Activations, EluBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    bool setAlpha = rand() % 2;

    Tensor featureInput(dataType, dimensions);
    Elu::Builder eluBuilder;
    eluBuilder.network(network);
    eluBuilder.featureInput(featureInput);
    float alpha;
    if (setAlpha) {
        alpha = (rand() % 100) / (1 + (rand() % 100));
        eluBuilder.alpha(alpha);
    } else {
        alpha = 1.0f;
    }
    shared_ptr<Elu> elu = dynamic_pointer_cast<Elu>(eluBuilder.build());

    ASSERT_TRUE(elu->isInitialized());

    Optional<Tensor> actualInput = elu->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = elu->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = elu->clone();
    Elu *clone = dynamic_cast<Elu *>(cloneLayer.get());
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

    ASSERT_EQ(elu->getId(), clone->getId());
    ASSERT_GT(elu->getId(), 1u);

    ASSERT_TRUE(*elu == *clone);
    ASSERT_FALSE(*elu != *clone);
    ASSERT_FALSE(*elu > *clone);
    ASSERT_FALSE(*elu < *clone);
}

TEST(Activations, EluSerializeDeserialize) {
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

    Elu::Builder eluBuilder = Elu::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput()).alpha(alpha);
    shared_ptr<Elu> elu = dynamic_pointer_cast<Elu>(eluBuilder.build());

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(initialNetwork).name("testOutput").inputTensor(elu->getFeatureOutput()).dataType(dataType).build();

    ASSERT_TRUE(elu->isInitialized());

    Tensor featureInput = elu->getFeatureInput();
    Tensor featureOutput = elu->getFeatureOutput();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(elu->getFeatureOutput().isPresent());
    ASSERT_EQ(elu->getFeatureOutput().get(), featureOutput);

    ASSERT_TRUE(elu->getFeatureInput().isPresent());
    assert(elu->getFeatureInput().get() == featureInput);

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

    json eluJ = elu->serialize("/tmp/", stream);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    ASSERT_EQ(eluJ["factory"], "activation");
    ASSERT_EQ(eluJ["version"], "1.0.0");
    ASSERT_EQ(eluJ["layer_type"], "elu");

    ASSERT_EQ(eluJ["alpha"], alpha);
    EXPECT_TRUE(eluJ.contains("feature_input"));
    EXPECT_TRUE(eluJ.contains("feature_output"));

    const auto &input = eluJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    string dataTypeString = dataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = eluJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    //     printf("%s\n", networkInputJ.dump(4).c_str());
    //     printf("%s\n", eluJ.dump(4).c_str());
    //     printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Elu::deserialize(eluJ, &newNetwork);
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
    shared_ptr<ThorImplementation::Elu> stampedElu = dynamic_pointer_cast<ThorImplementation::Elu>(otherLayers[0]);
    ASSERT_NE(stampedElu, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
}

TEST(Activations, EluRegistered) {
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

    Elu::Builder eluBuilder = Elu::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput()).alpha(alpha);
    shared_ptr<Elu> elu = dynamic_pointer_cast<Elu>(eluBuilder.build());

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(initialNetwork).name("testOutput").inputTensor(elu->getFeatureOutput()).dataType(dataType).build();

    ASSERT_TRUE(elu->isInitialized());

    Stream stream(0);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json eluJ = elu->serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    // Test that it is registered with Activation to deserialize
    Network newNetwork;
    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Activation::deserialize(eluJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    // FIXME: Check the Elu layer
}