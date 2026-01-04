#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, DropOutBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    float dropProportion = ((rand() % 100) + 1) / 1000.0f;

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    DropOut dropOut = DropOut::Builder().network(network).featureInput(featureInput).dropProportion(dropProportion).build();

    ASSERT_TRUE(dropOut.isInitialized());

    Optional<Tensor> actualInput = dropOut.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = dropOut.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    float actualDropProportion = dropOut.getDropProportion();
    ASSERT_EQ(actualDropProportion, dropProportion);

    shared_ptr<Layer> cloneLayer = dropOut.clone();
    DropOut *clone = dynamic_cast<DropOut *>(cloneLayer.get());
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

    float cloneDropProportion = clone->getDropProportion();
    ASSERT_EQ(cloneDropProportion, dropProportion);

    ASSERT_EQ(dropOut.getId(), clone->getId());
    ASSERT_GT(dropOut.getId(), 1u);

    ASSERT_TRUE(dropOut == *clone);
    ASSERT_FALSE(dropOut != *clone);
    ASSERT_FALSE(dropOut > *clone);
    ASSERT_FALSE(dropOut < *clone);
}

TEST(UtilityApiLayers, DropOutSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Stream stream(0);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";
    float dropProportion = ((rand() % 999) + 1) / 1000.0f;

    uint32_t numDimensions = 1 + (rand() % 4);

    vector<uint64_t> dimensions;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        dimensions.push_back(1 + (rand() % 5));
    }

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(dimensions).dataType(dataType).build();

    DropOut dropOut = DropOut::Builder()
                          .network(initialNetwork)
                          .dropProportion(dropProportion)
                          .featureInput(networkInput.getFeatureOutput().get())
                          .build();
    ASSERT_TRUE(dropOut.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(dropOut.getFeatureOutput().get())
                                      .dataType(dataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    json dropOutJ = dropOut.serialize(archiveWriter, stream);

    // printf("%s\n", dropOutJ.dump(4).c_str());

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &dropOut;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(dropOutJ, fromLayerJ);

    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    ASSERT_EQ(dropOutJ["factory"], Layer::Factory::Layer.value());
    ASSERT_EQ(dropOutJ["version"], "1.0.0");
    ASSERT_EQ(dropOutJ["layer_type"], "drop_out");
    ASSERT_EQ(dropOutJ.at("drop_proportion").get<float>(), dropProportion);

    const auto &input = dropOutJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = dropOutJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    Network newNetwork;

    archiveWriter.finishArchive();
    thor_file::TarReader archiveReader("testModel", "/tmp/");
    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, dropOutJ, &newNetwork);
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
    shared_ptr<ThorImplementation::DropOut> stampedDropOut = dynamic_pointer_cast<ThorImplementation::DropOut>(otherLayers[0]);
    ASSERT_NE(stampedDropOut, nullptr);

    vector<uint64_t> stampedDimensions = {batchSize};
    for (uint32_t d = 0; d < numDimensions; ++d)
        stampedDimensions.push_back(dimensions[d]);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
    ASSERT_EQ(stampedInput->getFeatureOutput().get().getDimensions(), stampedDimensions);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
    ASSERT_TRUE(stampedOutput->getFeatureInput().isPresent());
    ASSERT_EQ(stampedOutput->getFeatureOutput().get().getDimensions(), stampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().get(), stampedDropOut->getFeatureInput().get());
    ASSERT_EQ(stampedDropOut->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    ASSERT_EQ(stampedDropOut->getFeatureInput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
    ASSERT_EQ(stampedDropOut->getFeatureOutput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));

    ASSERT_EQ(stampedDropOut->getDropOutRate(), dropProportion);
}
