#include <optional>
#include "DeepLearning/Api/Layers/Activations/Exponential.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Activations, ExponentialBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Exponential::Builder exponentialBuilder;
    exponentialBuilder.network(network);
    exponentialBuilder.featureInput(featureInput);
    shared_ptr<Exponential> exponential = dynamic_pointer_cast<Exponential>(exponentialBuilder.build());

    ASSERT_TRUE(exponential->isInitialized());

    std::optional<Tensor> actualInput = exponential->getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), dataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = exponential->getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), dataType);
    ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = exponential->clone();
    Exponential *clone = dynamic_cast<Exponential *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    std::optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.has_value());
    ASSERT_EQ(cloneInput.value().getDataType(), dataType);
    ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);

    std::optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.has_value());
    ASSERT_EQ(cloneOutput.value().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.value().getDimensions(), dimensions);

    ASSERT_NE(exponential->getId(), clone->getId());
    ASSERT_GT(exponential->getId(), 1u);
}

TEST(Activations, ExponentialSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Exponential::Builder exponentialBuilder =
        Exponential::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Exponential> exponential = dynamic_pointer_cast<Exponential>(exponentialBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(exponential->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(exponential->isInitialized());

    Tensor featureInput = exponential->getFeatureInput().value();
    Tensor featureOutput = exponential->getFeatureOutput().value();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(exponential->getFeatureOutput().has_value());
    ASSERT_EQ(exponential->getFeatureOutput().value(), featureOutput);

    ASSERT_TRUE(exponential->getFeatureInput().has_value());
    assert(exponential->getFeatureInput().value() == featureInput);

    ASSERT_EQ(featureInput.getDataType(), dataType);
    ASSERT_EQ(featureInput.getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutput.getDataType(), dataType);
    ASSERT_EQ(featureOutput.getDimensions(), inputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> initialPlacedNetwork = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(initialPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the layer from the network
    ASSERT_EQ(initialPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialPlacedNetwork->getStampedNetwork(0);

    thor_file::TarWriter archiveWriter("testModel");

    json exponentialJ = exponential->serialize(archiveWriter, stream);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = exponential.get();
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(exponentialJ, fromLayerJ);

    ASSERT_EQ(exponentialJ["factory"], "activation");
    ASSERT_EQ(exponentialJ["version"], "1.0.0");
    ASSERT_EQ(exponentialJ["layer_type"], "exponential");

    EXPECT_TRUE(exponentialJ.contains("feature_input"));
    EXPECT_TRUE(exponentialJ.contains("feature_output"));

    const auto &input = exponentialJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    string dataTypeString = dataType == DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = exponentialJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    //     printf("%s\n", networkInputJ.dump(4).c_str());
    //     printf("%s\n", exponentialJ.dump(4).c_str());
    //     printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork("newNetwork");

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Exponential::deserialize(exponentialJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &newStamp = newPlacedNetwork->getStampedNetwork(0);

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = newStamp.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Exponential> stampedExponential = dynamic_pointer_cast<ThorImplementation::Exponential>(otherLayers[0]);
    ASSERT_NE(stampedExponential, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = newStamp.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = newStamp.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);

    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedExponential->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().has_value());
    ASSERT_EQ(stampedInput->getFeatureOutput().value(), stampedExponential->getFeatureInput().value());
    ASSERT_EQ(stampedExponential->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());

    filesystem::remove("/tmp/testModel.thor.tar");
}

TEST(Activations, ExponentialRegistered) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Exponential::Builder exponentialBuilder =
        Exponential::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Exponential> exponential = dynamic_pointer_cast<Exponential>(exponentialBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(exponential->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(exponential->isInitialized());

    thor_file::TarWriter archiveWriter("testModel");
    Stream stream(0);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json exponentialJ = exponential->serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Test that it is registered with Activation to deserialize
    Network newNetwork("newNetwork");
    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Activation::deserialize(exponentialJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    vector<Event> initDoneEvents;
    uint32_t batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = newPlacedNetwork->getStampedNetwork(0);

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Exponential> stampedExponential = dynamic_pointer_cast<ThorImplementation::Exponential>(otherLayers[0]);
    ASSERT_NE(stampedExponential, nullptr);

    filesystem::remove("/tmp/testModel.thor.tar");
}
