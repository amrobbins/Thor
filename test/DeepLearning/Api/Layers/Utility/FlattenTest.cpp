#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, FlattenBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> inputDimensions;
    int numInputDimensions = 2 + rand() % 6;
    for (int i = 0; i < numInputDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, inputDimensions);
    uint32_t numOutputDimensions = (rand() % (numInputDimensions - 1)) + 1;
    Flatten flatten = Flatten::Builder().network(network).featureInput(featureInput).numOutputDimensions(numOutputDimensions).build();

    ASSERT_TRUE(flatten.isInitialized());

    Optional<Tensor> actualInput = flatten.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), inputDimensions);

    Optional<Tensor> actualOutput = flatten.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    vector<uint64_t> outputDimensions = actualOutput.get().getDimensions();
    ASSERT_EQ(outputDimensions.size(), numOutputDimensions);
    uint64_t totalInputElements = 1;
    for (uint32_t i = 0; i < inputDimensions.size(); ++i)
        totalInputElements *= inputDimensions[i];
    uint64_t totalOutputElements = 1;
    for (uint32_t i = 0; i < outputDimensions.size(); ++i)
        totalOutputElements *= outputDimensions[i];
    ASSERT_EQ(totalInputElements, totalOutputElements);

    shared_ptr<Layer> cloneLayer = flatten.clone();
    Flatten *clone = dynamic_cast<Flatten *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), inputDimensions);

    ASSERT_EQ(flatten.getId(), clone->getId());
    ASSERT_GT(flatten.getId(), 1u);

    actualOutput = clone->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    outputDimensions = actualOutput.get().getDimensions();
    ASSERT_EQ(outputDimensions.size(), numOutputDimensions);
    totalInputElements = 1;
    for (uint32_t i = 0; i < inputDimensions.size(); ++i)
        totalInputElements *= inputDimensions[i];
    totalOutputElements = 1;
    for (uint32_t i = 0; i < outputDimensions.size(); ++i)
        totalOutputElements *= outputDimensions[i];
    ASSERT_EQ(totalInputElements, totalOutputElements);

    ASSERT_TRUE(flatten == *clone);
    ASSERT_FALSE(flatten != *clone);
    ASSERT_FALSE(flatten > *clone);
    ASSERT_FALSE(flatten < *clone);
}

TEST(UtilityApiLayers, FlattenSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Stream stream(0);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";

    uint32_t numDimensions = 2 + (rand() % 3);
    uint32_t numFlattenedDimensions = 1 + (rand() % (numDimensions - 1));

    vector<uint64_t> dimensions;
    vector<uint64_t> flattenedDimensions;
    uint64_t flattenedDimensionSize = 1;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        dimensions.push_back(1 + (rand() % 5));
        if ((d + 1) >= numFlattenedDimensions)
            flattenedDimensionSize *= dimensions.back();
        else
            flattenedDimensions.push_back(dimensions.back());
    }
    flattenedDimensions.push_back(flattenedDimensionSize);

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(dimensions).dataType(dataType).build();

    Flatten flatten = Flatten::Builder()
                          .network(initialNetwork)
                          .featureInput(networkInput.getFeatureOutput().get())
                          .numOutputDimensions(numFlattenedDimensions)
                          .build();
    ASSERT_TRUE(flatten.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(flatten.getFeatureOutput().get())
                                      .dataType(dataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    json flattenJ = flatten.serialize(archiveWriter, stream);

    // printf("%s\n", flattenJ.dump(4).c_str());

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &flatten;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(flattenJ, fromLayerJ);

    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    ASSERT_EQ(flattenJ["factory"], Layer::Factory::Layer.value());
    ASSERT_EQ(flattenJ["version"], "1.0.0");
    ASSERT_EQ(flattenJ["layer_type"], "flatten");

    const auto &input = flattenJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = flattenJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), flattenedDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    Network newNetwork;

    archiveWriter.finishArchive();
    thor_file::TarReader archiveReader("testModel", "/tmp/");
    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, flattenJ, &newNetwork);
    Layer::deserialize(archiveReader, networkOutputJ, &newNetwork);

    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode;
    statusCode = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork stampedNetwork = newNetwork.getStampedNetwork(0);
    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Flatten> stampedFlatten = dynamic_pointer_cast<ThorImplementation::Flatten>(otherLayers[0]);
    ASSERT_NE(stampedFlatten, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    vector<uint64_t> stampedDimensions = {batchSize};
    vector<uint64_t> flattenedStampedDimensions = {batchSize};
    for (uint32_t d = 0; d < numDimensions; ++d)
        stampedDimensions.push_back(dimensions[d]);
    for (uint32_t d = 0; d < flattenedDimensions.size(); ++d)
        flattenedStampedDimensions.push_back(flattenedDimensions[d]);

    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
    ASSERT_EQ(stampedInput->getFeatureOutput().get().getDimensions(), stampedDimensions);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
    ASSERT_EQ(stampedOutput->getFeatureOutput().get().getDimensions(), flattenedStampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().get(), stampedFlatten->getFeatureInput().get());
    ASSERT_EQ(stampedFlatten->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    ASSERT_EQ(stampedFlatten->getFeatureInput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
    ASSERT_EQ(stampedFlatten->getFeatureOutput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
}
