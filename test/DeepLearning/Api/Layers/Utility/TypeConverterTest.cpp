#include <optional>
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, TypeConverterBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions = {2, 6, 4, 1};
    DataType inputDataType = (DataType)((int)DataType::FP16 + (rand() % 13));
    DataType outputDataType = (DataType)((int)DataType::FP16 + (rand() % 13));
    while (outputDataType == inputDataType)
        outputDataType = (DataType)((int)DataType::FP16 + (rand() % 13));

    Tensor featureInput(inputDataType, dimensions);
    TypeConverter typeConverter = TypeConverter::Builder().network(network).featureInput(featureInput).newDataType(outputDataType).build();

    ASSERT_TRUE(typeConverter.isInitialized());

    std::optional<Tensor> actualInput = typeConverter.getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), inputDataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = typeConverter.getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), outputDataType);
    ASSERT_EQ(dimensions, actualOutput.value().getDimensions());

    shared_ptr<Layer> cloneLayer = typeConverter.clone();
    TypeConverter *clone = dynamic_cast<TypeConverter *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    std::optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.has_value());
    ASSERT_EQ(cloneInput.value().getDataType(), inputDataType);
    ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);

    ASSERT_EQ(typeConverter.getId(), clone->getId());
    ASSERT_GT(typeConverter.getId(), 1u);

    actualOutput = clone->getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), outputDataType);
    ASSERT_EQ(dimensions, actualOutput.value().getDimensions());

    ASSERT_TRUE(typeConverter == *clone);
    ASSERT_FALSE(typeConverter != *clone);
    ASSERT_FALSE(typeConverter > *clone);
    ASSERT_FALSE(typeConverter < *clone);
}

TEST(UtilityApiLayers, TypeConverterSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    Stream stream(0);

    DataType fromDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
    DataType toDataType = fromDataType == DataType::FP32 ? DataType::FP16 : DataType::FP32;
    string fromDataTypeString = fromDataType == DataType::FP32 ? "fp32" : "fp16";
    string toDataTypeString = toDataType == DataType::FP32 ? "fp32" : "fp16";

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
                                      .featureInput(networkInput.getFeatureOutput().value())
                                      .build();
    ASSERT_TRUE(typeConverter.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(typeConverter.getFeatureOutput().value())
                                      .dataType(toDataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel");

    json typeConverterJ = typeConverter.serialize(archiveWriter, stream);

    // printf("%s\n", typeConverterJ.dump(4).c_str());

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &typeConverter;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(typeConverterJ, fromLayerJ);

    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

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
    Network newNetwork("newNetwork");

    // Write a dummy file with data into the archive since none of the layers wrote anything into it (no weights)
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::DataType::UINT8, {4});
    ThorImplementation::Tensor dummyData(cpuPlacement, descriptor);
    archiveWriter.addArchiveFile("dummy", dummyData);
    archiveWriter.createArchive("/tmp/", true);
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");

    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, typeConverterJ, &newNetwork);
    Layer::deserialize(archiveReader, networkOutputJ, &newNetwork);

    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork stampedNetwork = newPlacedNetwork->getStampedNetwork(0);
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
    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_EQ(stampedInput->getFeatureOutput().value().getDimensions(), stampedDimensions);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
    ASSERT_EQ(stampedOutput->getFeatureOutput().value().getDimensions(), stampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().value(), stampedTypeConverter->getFeatureInput().value());
    ASSERT_EQ(stampedTypeConverter->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());

    ASSERT_EQ(stampedTypeConverter->getFeatureInput().value().getDataType(), fromDataType);
    ASSERT_EQ(stampedTypeConverter->getFeatureOutput().value().getDataType(), toDataType);
}
