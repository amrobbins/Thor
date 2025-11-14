#include "DeepLearning/Api/Layers/Utility/Pooling.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, PoolingNoPaddingBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(10 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    uint32_t windowHeight = 1 + (rand() % dimensions[1]);
    uint32_t windowWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);

    Tensor featureInput(dataType, dimensions);
    Pooling pooling = Pooling::Builder()
                          .network(network)
                          .featureInput(featureInput)
                          .type(Pooling::Type::AVERAGE)
                          .windowHeight(windowHeight)
                          .windowWidth(windowWidth)
                          .verticalStride(verticalStride)
                          .horizontalStride(horizontalStride)
                          .noPadding()
                          .build();

    ASSERT_TRUE(pooling.isInitialized());

    Optional<Tensor> actualInput = pooling.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    vector<uint64_t> outputDimensions;
    uint32_t outputHeight = Pooling::Builder::computeOutputDimension(dimensions[1], verticalStride, windowHeight, 0);
    uint32_t outputWidth = Pooling::Builder::computeOutputDimension(dimensions[2], horizontalStride, windowWidth, 0);
    outputDimensions.push_back(dimensions[0]);
    outputDimensions.push_back(outputHeight);
    outputDimensions.push_back(outputWidth);

    Optional<Tensor> actualOutput = pooling.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
    ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
    ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
    ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(pooling.getVerticalPadding(), 0u);
    ASSERT_EQ(pooling.getHorizontalPadding(), 0u);

    shared_ptr<Layer> cloneLayer = pooling.clone();
    Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getWindowHeight(), windowHeight);
    ASSERT_EQ(clone->getWindowWidth(), windowWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), 0u);
    ASSERT_EQ(clone->getHorizontalPadding(), 0u);

    ASSERT_EQ(pooling.getId(), clone->getId());
    ASSERT_GT(pooling.getId(), 1u);

    ASSERT_TRUE(pooling == *clone);
    ASSERT_FALSE(pooling != *clone);
    ASSERT_FALSE(pooling > *clone);
    ASSERT_FALSE(pooling < *clone);
}

TEST(UtilityApiLayers, PoolingSamePaddingBuilds) {
    srand(time(nullptr));

    for (int test = 0; test < 50; ++test) {
        Network network;

        vector<uint64_t> dimensions;
        int numDimensions = 3;
        for (int i = 0; i < numDimensions; ++i)
            dimensions.push_back(10 + (rand() % 1000));
        Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        uint32_t windowHeight = 1 + (rand() % dimensions[1]);
        uint32_t windowWidth = 1 + (rand() % dimensions[2]);
        uint32_t verticalStride = 1;  // due to same padding
        uint32_t horizontalStride = 1;

        Tensor featureInput(dataType, dimensions);
        Pooling pooling = Pooling::Builder()
                              .network(network)
                              .featureInput(featureInput)
                              .type(Pooling::Type::MAX)
                              .windowHeight(windowHeight)
                              .windowWidth(windowWidth)
                              .verticalStride(verticalStride)
                              .horizontalStride(horizontalStride)
                              .samePadding()
                              .build();

        ASSERT_TRUE(pooling.isInitialized());

        Optional<Tensor> actualInput = pooling.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), dataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualOutput = pooling.getFeatureOutput();
        ASSERT_TRUE(actualOutput.isPresent());
        ASSERT_EQ(actualOutput.get().getDataType(), dataType);
        ASSERT_EQ(actualOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(actualOutput.get().getDimensions()[0], dimensions[0]);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = actualOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        uint32_t verticalPadding = Pooling::Builder::computeSamePadding(dimensions[1], verticalStride, windowHeight);
        uint32_t horizontalPadding = Pooling::Builder::computeSamePadding(dimensions[2], horizontalStride, windowWidth);
        ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
        ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
        ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
        ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
        ASSERT_EQ(pooling.getVerticalPadding(), verticalPadding);
        ASSERT_EQ(pooling.getHorizontalPadding(), horizontalPadding);

        shared_ptr<Layer> cloneLayer = pooling.clone();
        Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), dataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneOutput = clone->getFeatureOutput();
        ASSERT_TRUE(cloneOutput.isPresent());
        ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
        ASSERT_EQ(cloneOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(cloneOutput.get().getDimensions()[0], dimensions[0]);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = cloneOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(clone->getWindowHeight(), windowHeight);
        ASSERT_EQ(clone->getWindowWidth(), windowWidth);
        ASSERT_EQ(clone->getVerticalStride(), verticalStride);
        ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
        ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
        ASSERT_EQ(clone->getHorizontalPadding(), horizontalPadding);

        ASSERT_EQ(pooling.getId(), clone->getId());
        ASSERT_GT(pooling.getId(), 1u);

        ASSERT_TRUE(pooling == *clone);
        ASSERT_FALSE(pooling != *clone);
        ASSERT_FALSE(pooling > *clone);
        ASSERT_FALSE(pooling < *clone);
    }
}

TEST(UtilityApiLayers, PoolingDefaultPaddingBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(10 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    uint32_t windowHeight = 1 + (rand() % dimensions[1]);
    uint32_t windowWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);

    Tensor featureInput(dataType, dimensions);
    Pooling pooling = Pooling::Builder()
                          .network(network)
                          .featureInput(featureInput)
                          .type(Pooling::Type::AVERAGE)
                          .windowHeight(windowHeight)
                          .windowWidth(windowWidth)
                          .verticalStride(verticalStride)
                          .horizontalStride(horizontalStride)
                          .build();

    ASSERT_TRUE(pooling.isInitialized());

    Optional<Tensor> actualInput = pooling.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    vector<uint64_t> outputDimensions;
    uint32_t outputHeight = Pooling::Builder::computeOutputDimension(dimensions[1], verticalStride, windowHeight, 0);
    uint32_t outputWidth = Pooling::Builder::computeOutputDimension(dimensions[2], horizontalStride, windowWidth, 0);
    outputDimensions.push_back(dimensions[0]);
    outputDimensions.push_back(outputHeight);
    outputDimensions.push_back(outputWidth);

    Optional<Tensor> actualOutput = pooling.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
    ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
    ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
    ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(pooling.getVerticalPadding(), 0u);
    ASSERT_EQ(pooling.getHorizontalPadding(), 0u);

    shared_ptr<Layer> cloneLayer = pooling.clone();
    Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getWindowHeight(), windowHeight);
    ASSERT_EQ(clone->getWindowWidth(), windowWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), 0u);
    ASSERT_EQ(clone->getHorizontalPadding(), 0u);

    ASSERT_EQ(pooling.getId(), clone->getId());
    ASSERT_GT(pooling.getId(), 1u);

    ASSERT_TRUE(pooling == *clone);
    ASSERT_FALSE(pooling != *clone);
    ASSERT_FALSE(pooling > *clone);
    ASSERT_FALSE(pooling < *clone);
}

TEST(UtilityApiLayers, PoolingSpecifiedPaddingBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(10 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    uint32_t windowHeight = 1 + (rand() % dimensions[1]);
    uint32_t windowWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);
    uint32_t verticalPadding = rand() % windowHeight;
    uint32_t horizontalPadding = rand() % windowWidth;

    Tensor featureInput(dataType, dimensions);
    Pooling pooling = Pooling::Builder()
                          .network(network)
                          .featureInput(featureInput)
                          .type(Pooling::Type::MAX)
                          .windowHeight(windowHeight)
                          .windowWidth(windowWidth)
                          .verticalStride(verticalStride)
                          .horizontalStride(horizontalStride)
                          .verticalPadding(verticalPadding)
                          .horizontalPadding(horizontalPadding)
                          .build();

    ASSERT_TRUE(pooling.isInitialized());

    Optional<Tensor> actualInput = pooling.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    vector<uint64_t> outputDimensions;
    uint32_t outputHeight = Pooling::Builder::computeOutputDimension(dimensions[1], verticalStride, windowHeight, verticalPadding);
    uint32_t outputWidth = Pooling::Builder::computeOutputDimension(dimensions[2], horizontalStride, windowWidth, horizontalPadding);
    outputDimensions.push_back(dimensions[0]);
    outputDimensions.push_back(outputHeight);
    outputDimensions.push_back(outputWidth);

    Optional<Tensor> actualOutput = pooling.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
    ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
    ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
    ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(pooling.getVerticalPadding(), verticalPadding);
    ASSERT_EQ(pooling.getHorizontalPadding(), horizontalPadding);

    shared_ptr<Layer> cloneLayer = pooling.clone();
    Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getWindowHeight(), windowHeight);
    ASSERT_EQ(clone->getWindowWidth(), windowWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
    ASSERT_EQ(clone->getHorizontalPadding(), horizontalPadding);

    ASSERT_EQ(pooling.getId(), clone->getId());
    ASSERT_GT(pooling.getId(), 1u);

    ASSERT_TRUE(pooling == *clone);
    ASSERT_FALSE(pooling != *clone);
    ASSERT_FALSE(pooling > *clone);
    ASSERT_FALSE(pooling < *clone);
}

// TEST(UtilityApiLayers, PoolingSerializeDeserialize) {
//     srand(time(nullptr));
//
//     Network initialNetwork;
//     Stream stream(0);
//
//     Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
//     string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";
//     Pooling::Type poolingType = rand() % 2 ? Pooling::Type::AVERAGE : Pooling::Type::MAX;
//     FIXME
//     uint32_t windowHeight = 0;
//     uint32_t windowWidth = 0;
//     uint32_t verticalStride = 0;
//     uint32_t horizontalStride = 0;
//     uint32_t verticalPadding = 0;
//     uint32_t horizontalPadding = 0;
//
//
//         uint32_t numDimensions = 1 + (rand() % 4);
//
//     vector<uint64_t> dimensions;
//     for (uint32_t d = 0; d < numDimensions; ++d) {
//         dimensions.push_back(1 + (rand() % 5));
//     }
//
//     NetworkInput networkInput =
//         NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(dimensions).dataType(dataType).build();
//
//     Pooling pooling = Pooling::Builder()
//                           .network(initialNetwork)
//                           .type(poolingType)
//                           .windowHeight(windowHeight)
//                           .windowWidth(windowWidth)
//                           .verticalStride(verticalStride)
//                           .horizontalStride(horizontalStride)
//                           .verticalPadding(horizontalPadding)
//                           .horizontalPadding(horizontalPadding)
//                           .featureInput(networkInput.getFeatureOutput().get())
//                           .build();
//     ASSERT_TRUE(pooling.isInitialized());
//
//     NetworkOutput networkOutput = NetworkOutput::Builder()
//                                       .network(initialNetwork)
//                                       .name("testOutput")
//                                       .inputTensor(pooling.getFeatureOutput().get())
//                                       .dataType(dataType)
//                                       .build();
//
//     json poolingJ = pooling.serialize("/tmp/", stream);
//
//     // printf("%s\n", poolingJ.dump(4).c_str());
//
//     // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
//     Layer *layer = &pooling;
//     json fromLayerJ = layer->serialize("/tmp/", stream);
//     ASSERT_EQ(poolingJ, fromLayerJ);
//
//     json networkInputJ = networkInput.serialize("/tmp/", stream);
//     json networkOutputJ = networkOutput.serialize("/tmp/", stream);
//
//     ASSERT_EQ(poolingJ["factory"], Layer::Factory::Layer.value());
//     ASSERT_EQ(poolingJ["version"], "1.0.0");
//     ASSERT_EQ(poolingJ["layer_type"], "pooling");
//     ASSERT_EQ(poolingJ.at("drop_proportion").get<float>(), dropProportion);
//
//     const auto &input = poolingJ.at("feature_input");
//     ASSERT_TRUE(input.is_object());
//     EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
//     ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), dimensions);
//     ASSERT_TRUE(input.at("id").is_number_integer());
//
//     const auto &output = poolingJ.at("feature_output");
//     ASSERT_TRUE(output.is_object());
//     EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
//     ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), dimensions);
//     ASSERT_TRUE(output.at("id").is_number_integer());
//
//     ////////////////////////////
//     // Deserialize
//     ////////////////////////////
//     Network newNetwork;
//
//     Layer::deserialize(networkInputJ, &newNetwork);
//     Layer::deserialize(poolingJ, &newNetwork);
//     Layer::deserialize(networkOutputJ, &newNetwork);
//
//     uint32_t batchSize = 1 + (rand() % 16);
//     vector<Event> initDoneEvents;
//     Network::StatusCode statusCode;
//     statusCode = newNetwork.place(batchSize, initDoneEvents);
//     ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
//     for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
//         stream.waitEvent(initDoneEvents[i]);
//     }
//     initDoneEvents.clear();
//
//     vector<ThorImplementation::StampedNetwork> stampedNetworks = newNetwork.getStampedNetworks();
//     ASSERT_EQ(stampedNetworks.size(), 1UL);
//     ThorImplementation::StampedNetwork stampedNetwork = stampedNetworks[0];
//     vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
//     ASSERT_EQ(otherLayers.size(), 1U);
//     shared_ptr<ThorImplementation::Pooling> stampedPooling = dynamic_pointer_cast<ThorImplementation::Pooling>(otherLayers[0]);
//     ASSERT_NE(stampedPooling, nullptr);
//
//     vector<uint64_t> stampedDimensions = {batchSize};
//     for (uint32_t d = 0; d < numDimensions; ++d)
//         stampedDimensions.push_back(dimensions[d]);
//
//     vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
//     ASSERT_EQ(inputLayers.size(), 1U);
//     shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
//     ASSERT_NE(stampedInput, nullptr);
//     ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
//     ASSERT_EQ(stampedInput->getFeatureOutput().get().getDimensions(), stampedDimensions);
//
//     vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
//     ASSERT_EQ(outputLayers.size(), 1U);
//     shared_ptr<ThorImplementation::NetworkOutput> stampedOutput =
//     dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]); ASSERT_NE(outputLayers[0], nullptr);
//     ASSERT_TRUE(stampedOutput->getFeatureInput().isPresent());
//     ASSERT_EQ(stampedOutput->getFeatureOutput().get().getDimensions(), stampedDimensions);
//
//     // Ensure that they are all connected
//     EXPECT_EQ(stampedInput->getFeatureOutput().get(), stampedPooling->getFeatureInput().get());
//     ASSERT_EQ(stampedPooling->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());
//
//     ASSERT_EQ(stampedPooling->getFeatureInput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
//     ASSERT_EQ(stampedPooling->getFeatureOutput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
//
//     ASSERT_EQ(stampedPooling->getPoolingRate(), dropProportion);
// }
