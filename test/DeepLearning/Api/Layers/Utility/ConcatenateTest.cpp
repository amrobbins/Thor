#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;

TEST(UtilityApi, ConcatenateBuilds) {
    srand(time(nullptr));

    Network network;

    uint32_t numDimensions = 1 + (rand() % 4);
    uint32_t concatenationAxis = rand() % numDimensions;
    uint32_t concatenationAxisSize = 1 + (rand() % 50);
    vector<uint64_t> concatenatedDimensions(numDimensions, 0U);
    uint32_t numTensors = 1 + (rand() % 5);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    vector<uint64_t> fixedDimensionSize;
    for(uint32_t d = 0; d < numDimensions; ++d) {
        fixedDimensionSize.push_back(1 + (rand() % 5));
    }
    concatenatedDimensions = fixedDimensionSize;
    concatenatedDimensions[concatenationAxis] = 0;

    vector<vector<uint64_t>> tensorDimensions;
    for (uint32_t t = 0; t < numTensors; ++t) {
        tensorDimensions.emplace_back();
        for(uint32_t d = 0; d < numDimensions; ++d) {
            uint64_t size;
            if (d == concatenationAxis) {
                size = 1 + (rand() % 5);
                concatenatedDimensions[concatenationAxis] += size;
            } else {
                size = fixedDimensionSize[d];
            }
            tensorDimensions[t].push_back(size);
        }
    }

    vector<Tensor> tensors;
    for (uint32_t t = 0; t < numTensors; ++t) {
        tensors.push_back(Tensor(dataType, tensorDimensions[t]));
    }

    Concatenate::Builder concatenateBuilder = Concatenate::Builder().network(network).concatenationAxis(concatenationAxis);
    for (uint32_t t = 0; t < numTensors; ++t) {
        concatenateBuilder.featureInput(tensors[t]);
    }
    Concatenate concatenate = concatenateBuilder.build();

    ASSERT_TRUE(concatenate.isInitialized());

    vector<Tensor> actualInputs = concatenate.getFeatureInputs();
    ASSERT_EQ(actualInputs.size(), numTensors);
    for (uint32_t t = 0; t < numTensors; ++t) {
        ASSERT_EQ(actualInputs[t].getDataType(), dataType);
        ASSERT_EQ(actualInputs[t].getDimensions(), tensorDimensions[t]);
    }

    vector<Tensor> actualOutputs = concatenate.getFeatureOutputs();
    ASSERT_EQ(actualOutputs.size(), 1U);
    ASSERT_EQ(actualOutputs[0].getDataType(), dataType);
    vector<uint64_t> outputDimensions = actualOutputs[0].getDimensions();
    ASSERT_EQ(outputDimensions.size(), numDimensions);
    ASSERT_EQ(actualOutputs[0].getDimensions(), concatenatedDimensions);


    shared_ptr<Layer> cloneLayer = concatenate.clone();
    Concatenate *clone = dynamic_cast<Concatenate *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    vector<Tensor> cloneInputs = clone->getFeatureInputs();
    ASSERT_EQ(cloneInputs.size(), numTensors);
    for (uint32_t t = 0; t < numTensors; ++t) {
        ASSERT_EQ(cloneInputs[t].getDataType(), dataType);
        ASSERT_EQ(cloneInputs[t].getDimensions(), tensorDimensions[t]);
    }

    ASSERT_EQ(concatenate.getId(), clone->getId());
    ASSERT_GT(concatenate.getId(), 1u);

    vector<Tensor> cloneOutputs = clone->getFeatureOutputs();
    ASSERT_EQ(cloneOutputs.size(), 1U);
    ASSERT_EQ(cloneOutputs[0].getDataType(), dataType);
    outputDimensions.clear();
    outputDimensions = cloneOutputs[0].getDimensions();
    ASSERT_EQ(outputDimensions.size(), numDimensions);
    ASSERT_EQ(cloneOutputs[0].getDimensions(), concatenatedDimensions);

    ASSERT_TRUE(concatenate == *clone);
    ASSERT_FALSE(concatenate != *clone);
    ASSERT_FALSE(concatenate > *clone);
    ASSERT_FALSE(concatenate < *clone);
}

// FIXME: Modify for concatenate
//TEST(UtilityLayers, ConcatenateSerializeDeserialize) {
//    srand(time(nullptr));
//
//    Network initialNetwork;
//    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
//    vector<uint64_t> inputDimensions;
//    uint32_t numDimensions = 1 + (rand() % 5);
//    for (uint32_t i = 0; i < numDimensions; ++i)
//        inputDimensions.push_back(1 + (rand() % 5));
//
//    NetworkInput networkInput =
//        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();
//
//    uint32_t concatenationAxis = rand() % numDimensions;
//
//    Concatenate::Builder concatenateBuilder = Concatenate::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput()).alpha(concatenationAxis);
//    shared_ptr<Concatenate> concatenate = dynamic_pointer_cast<Concatenate>(concatenateBuilder.build());
//
//    NetworkOutput networkOutput =
//        NetworkOutput::Builder().network(initialNetwork).name("testOutput").inputTensor(concatenate->getFeatureOutput()).dataType(dataType).build();
//
//    ASSERT_TRUE(concatenate->isInitialized());
//
//    Tensor featureInput = concatenate->getFeatureInput();
//    Tensor featureOutput = concatenate->getFeatureOutput();
//    assert(featureInput == networkInput.getFeatureOutput());
//
//    ASSERT_TRUE(concatenate->getFeatureOutput().isPresent());
//    ASSERT_EQ(concatenate->getFeatureOutput().get(), featureOutput);
//
//    ASSERT_TRUE(concatenate->getFeatureInput().isPresent());
//    assert(concatenate->getFeatureInput().get() == featureInput);
//
//    ASSERT_EQ(featureInput.getDataType(), dataType);
//    ASSERT_EQ(featureInput.getDimensions(), inputDimensions);
//
//    ASSERT_EQ(featureOutput.getDataType(), dataType);
//    ASSERT_EQ(featureOutput.getDimensions(), inputDimensions);
//
//    // Now stamp the network and test serialization
//    Stream stream(0);
//    uint32_t batchSize = 1 + (rand() % 16);
//    vector<Event> initDoneEvents;
//    Network::StatusCode placementStatus;
//    placementStatus = initialNetwork.place(batchSize, initDoneEvents);
//    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
//    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
//        stream.waitEvent(initDoneEvents[i]);
//    }
//    initDoneEvents.clear();
//
//    // Fetch the layer from the network
//    vector<ThorImplementation::StampedNetwork> stampedNetworks = initialNetwork.getStampedNetworks();
//    ASSERT_EQ(stampedNetworks.size(), 1UL);
//    ThorImplementation::StampedNetwork stampedNetwork = stampedNetworks[0];
//
//    json concatenateJ = concatenate->serialize("/tmp/", stream);
//    json networkInputJ = networkInput.serialize("/tmp/", stream);
//    json networkOutputJ = networkOutput.serialize("/tmp/", stream);
//
//    ASSERT_EQ(concatenateJ["factory"], "activation");
//    ASSERT_EQ(concatenateJ["version"], "1.0.0");
//    ASSERT_EQ(concatenateJ["layer_type"], "concatenate");
//
//    ASSERT_EQ(concatenateJ["alpha"], concatenationAxis);
//    EXPECT_TRUE(concatenateJ.contains("feature_input"));
//    EXPECT_TRUE(concatenateJ.contains("feature_output"));
//
//    const auto &input = concatenateJ.at("feature_input");
//    ASSERT_TRUE(input.is_object());
//    ASSERT_TRUE(input.at("data_type").is_string());
//    string dataTypeString = dataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
//    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
//    ASSERT_TRUE(input.at("dimensions").is_array());
//    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
//    ASSERT_TRUE(input.at("id").is_number_integer());
//
//    const auto &output = concatenateJ.at("feature_output");
//    ASSERT_TRUE(output.is_object());
//    ASSERT_TRUE(output.at("data_type").is_string());
//    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
//    ASSERT_TRUE(output.at("dimensions").is_array());
//    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
//    ASSERT_TRUE(output.at("id").is_number_integer());
//
//    //     printf("%s\n", networkInputJ.dump(4).c_str());
//    //     printf("%s\n", concatenateJ.dump(4).c_str());
//    //     printf("%s\n", networkOutputJ.dump(4).c_str());
//
//    ////////////////////////////
//    // Deserialize
//    ////////////////////////////
//    // Verify that the layer gets added to the network and that its weights are set to the correct values
//    Network newNetwork;
//
//    NetworkInput::deserialize(networkInputJ, &newNetwork);
//    Layer::deserialize(concatenateJ, &newNetwork);
//    NetworkOutput::deserialize(networkOutputJ, &newNetwork);
//
//    batchSize = 1 + (rand() % 16);
//    placementStatus = newNetwork.place(batchSize, initDoneEvents);
//    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
//    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
//        stream.waitEvent(initDoneEvents[i]);
//    }
//    initDoneEvents.clear();
//
//    stampedNetworks.clear();
//    stampedNetworks = newNetwork.getStampedNetworks();
//    ASSERT_EQ(stampedNetworks.size(), 1UL);
//    stampedNetwork = stampedNetworks[0];
//
//    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
//    ASSERT_EQ(otherLayers.size(), 1U);
//    shared_ptr<ThorImplementation::Concatenate> stampedConcatenate = dynamic_pointer_cast<ThorImplementation::Concatenate>(otherLayers[0]);
//    ASSERT_NE(stampedConcatenate, nullptr);
//
//    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
//    ASSERT_EQ(inputLayers.size(), 1U);
//    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
//    ASSERT_NE(inputLayers[0], nullptr);
//
//    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
//    ASSERT_EQ(outputLayers.size(), 1U);
//    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
//    ASSERT_NE(outputLayers[0], nullptr);
//
//    ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
//    ASSERT_TRUE(stampedConcatenate->getFeatureOutput().isPresent());
//    ASSERT_TRUE(stampedOutput->getFeatureOutput().isPresent());
//    ASSERT_EQ(stampedInput->getFeatureOutput().get(), stampedConcatenate->getFeatureInput().get());
//    ASSERT_EQ(stampedConcatenate->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());
//}
