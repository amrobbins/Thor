#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Activations/Selu.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Activations, SeluBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Selu::Builder seluBuilder;
    seluBuilder.network(network);
    seluBuilder.featureInput(featureInput);
    shared_ptr<Selu> selu = dynamic_pointer_cast<Selu>(seluBuilder.build());

    ASSERT_TRUE(selu->isInitialized());

    Optional<Tensor> actualInput = selu->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = selu->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = selu->clone();
    Selu *clone = dynamic_cast<Selu *>(cloneLayer.get());
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

    ASSERT_EQ(selu->getId(), clone->getId());
    ASSERT_GT(selu->getId(), 1u);

    ASSERT_TRUE(*selu == *clone);
    ASSERT_FALSE(*selu != *clone);
    ASSERT_FALSE(*selu > *clone);
    ASSERT_FALSE(*selu < *clone);
}

TEST(Activations, SeluSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Selu::Builder seluBuilder = Selu::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput());
    shared_ptr<Selu> selu = dynamic_pointer_cast<Selu>(seluBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(selu->getFeatureOutput())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(selu->isInitialized());

    Tensor featureInput = selu->getFeatureInput();
    Tensor featureOutput = selu->getFeatureOutput();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(selu->getFeatureOutput().isPresent());
    ASSERT_EQ(selu->getFeatureOutput().get(), featureOutput);

    ASSERT_TRUE(selu->getFeatureInput().isPresent());
    assert(selu->getFeatureInput().get() == featureInput);

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
    ASSERT_EQ(initialNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialNetwork.getStampedNetwork(0);

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    json seluJ = selu->serialize(archiveWriter, stream);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    ASSERT_EQ(seluJ["factory"], "activation");
    ASSERT_EQ(seluJ["version"], "1.0.0");
    ASSERT_EQ(seluJ["layer_type"], "selu");

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = selu.get();
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(seluJ, fromLayerJ);

    EXPECT_TRUE(seluJ.contains("feature_input"));
    EXPECT_TRUE(seluJ.contains("feature_output"));

    const auto &input = seluJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    string dataTypeString = dataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = seluJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    //     printf("%s\n", networkInputJ.dump(4).c_str());
    //     printf("%s\n", seluJ.dump(4).c_str());
    //     printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Selu::deserialize(seluJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    placementStatus = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &newStamp = newNetwork.getStampedNetwork(0);

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = newStamp.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Selu> stampedSelu = dynamic_pointer_cast<ThorImplementation::Selu>(otherLayers[0]);
    ASSERT_NE(stampedSelu, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = newStamp.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = newStamp.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);

    ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
    ASSERT_TRUE(stampedSelu->getFeatureOutput().isPresent());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().isPresent());
    ASSERT_EQ(stampedInput->getFeatureOutput().get(), stampedSelu->getFeatureInput().get());
    ASSERT_EQ(stampedSelu->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    filesystem::remove("/tmp/testModel.thor");
}

TEST(Activations, SeluRegistered) {
    srand(time(nullptr));

    Network initialNetwork;
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1 + (rand() % 5);
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Selu::Builder seluBuilder = Selu::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput());
    shared_ptr<Selu> selu = dynamic_pointer_cast<Selu>(seluBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(selu->getFeatureOutput())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(selu->isInitialized());

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);
    Stream stream(0);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json seluJ = selu->serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Test that it is registered with Activation to deserialize
    Network newNetwork;
    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Activation::deserialize(seluJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    vector<Event> initDoneEvents;
    uint32_t batchSize = 1 + (rand() % 16);
    Network::StatusCode placementStatus = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = newNetwork.getStampedNetwork(0);

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Selu> stampedSelu = dynamic_pointer_cast<ThorImplementation::Selu>(otherLayers[0]);
    ASSERT_NE(stampedSelu, nullptr);

    filesystem::remove("/tmp/testModel.thor");
}
