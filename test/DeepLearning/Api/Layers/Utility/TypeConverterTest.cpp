#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, TypeConverterBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions = {2, 6, 4, 1};
    Tensor::DataType inputDataType = (Tensor::DataType)((int)Tensor::DataType::PACKED_BOOLEAN + (rand() % 13));
    Tensor::DataType outputDataType = (Tensor::DataType)((int)Tensor::DataType::PACKED_BOOLEAN + (rand() % 13));
    while (outputDataType == inputDataType)
        outputDataType = (Tensor::DataType)((int)Tensor::DataType::PACKED_BOOLEAN + (rand() % 13));

    Tensor featureInput(inputDataType, dimensions);
    TypeConverter typeConverter = TypeConverter::Builder().network(network).featureInput(featureInput).newDataType(outputDataType).build();

    ASSERT_TRUE(typeConverter.isInitialized());

    Optional<Tensor> actualInput = typeConverter.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), inputDataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = typeConverter.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), outputDataType);
    ASSERT_EQ(dimensions, actualOutput.get().getDimensions());

    shared_ptr<Layer> cloneLayer = typeConverter.clone();
    TypeConverter *clone = dynamic_cast<TypeConverter *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), inputDataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    ASSERT_EQ(typeConverter.getId(), clone->getId());
    ASSERT_GT(typeConverter.getId(), 1u);

    actualOutput = clone->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), outputDataType);
    ASSERT_EQ(dimensions, actualOutput.get().getDimensions());

    ASSERT_TRUE(typeConverter == *clone);
    ASSERT_FALSE(typeConverter != *clone);
    ASSERT_FALSE(typeConverter > *clone);
    ASSERT_FALSE(typeConverter < *clone);
}

TEST(UtilityApiLayers, TypeConverterSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Stream stream(0);

    Tensor::DataType fromDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    Tensor::DataType toDataType = fromDataType == Tensor::DataType::FP32 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    string fromDataTypeString = fromDataType == Tensor::DataType::FP32 ? "fp32" : "fp16";
    string toDataTypeString = toDataType == Tensor::DataType::FP32 ? "fp32" : "fp16";

    uint32_t numDimensions = 1 + (rand() % 4);

    vector<uint64_t> dimensions;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        dimensions.push_back(1 + (rand() % 5));
    }

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(dimensions).dataType(fromDataType).build();

    TypeConverter typeConverter = TypeConverter::Builder()
                                      .network(initialNetwork)
                                      .newDataType(toDataType)
                                      .featureInput(networkInput.getFeatureOutput().get())
                                      .build();
    ASSERT_TRUE(typeConverter.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(typeConverter.getFeatureOutput().get())
                                      .dataType(toDataType)
                                      .build();

    json typeConverterJ = typeConverter.serialize("/tmp/", stream);

    // printf("%s\n", typeConverterJ.dump(4).c_str());

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &typeConverter;
    json fromLayerJ = layer->serialize("/tmp/", stream);
    ASSERT_EQ(typeConverterJ, fromLayerJ);

    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    ASSERT_EQ(typeConverterJ["factory"], Layer::Factory::Layer.value());
    ASSERT_EQ(typeConverterJ["version"], "1.0.0");
    ASSERT_EQ(typeConverterJ["layer_type"], "type_converter");

    const auto &input = typeConverterJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    EXPECT_EQ(input.at("data_type").get<string>(), fromDataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = typeConverterJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), toDataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    Network newNetwork;

    Layer::deserialize(networkInputJ, &newNetwork);
    Layer::deserialize(typeConverterJ, &newNetwork);
    Layer::deserialize(networkOutputJ, &newNetwork);

    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode;
    statusCode = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    vector<ThorImplementation::StampedNetwork> stampedNetworks = newNetwork.getStampedNetworks();
    ASSERT_EQ(stampedNetworks.size(), 1UL);
    ThorImplementation::StampedNetwork stampedNetwork = stampedNetworks[0];
    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::TypeConversion> stampedTypeConverter =
        dynamic_pointer_cast<ThorImplementation::TypeConversion>(otherLayers[0]);
    ASSERT_NE(stampedTypeConverter, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    vector<uint64_t> stampedDimensions = {batchSize};
    for (uint32_t d = 0; d < numDimensions; ++d)
        stampedDimensions.push_back(dimensions[d]);

    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
    ASSERT_EQ(stampedInput->getFeatureOutput().get().getDimensions(), stampedDimensions);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
    ASSERT_EQ(stampedOutput->getFeatureOutput().get().getDimensions(), stampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().get(), stampedTypeConverter->getFeatureInput().get());
    ASSERT_EQ(stampedTypeConverter->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    ASSERT_EQ(stampedTypeConverter->getFeatureInput().get().getDataType(), Tensor::convertToImplementationDataType(fromDataType));
    ASSERT_EQ(stampedTypeConverter->getFeatureOutput().get().getDataType(), Tensor::convertToImplementationDataType(toDataType));
}
