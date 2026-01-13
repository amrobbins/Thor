#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, ReshapeBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> inputDimensions = {2, 6, 4, 1};
    vector<uint64_t> outputDimensions = {8, 1, 6};

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, inputDimensions);
    Reshape reshape = Reshape::Builder().network(network).featureInput(featureInput).newDimensions(outputDimensions).build();

    ASSERT_TRUE(reshape.isInitialized());

    Optional<Tensor> actualInput = reshape.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), inputDimensions);

    Optional<Tensor> actualOutput = reshape.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(outputDimensions, actualOutput.get().getDimensions());

    shared_ptr<Layer> cloneLayer = reshape.clone();
    Reshape *clone = dynamic_cast<Reshape *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), inputDimensions);

    ASSERT_EQ(reshape.getId(), clone->getId());
    ASSERT_GT(reshape.getId(), 1u);

    actualOutput = clone->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(outputDimensions, actualOutput.get().getDimensions());

    ASSERT_TRUE(reshape == *clone);
    ASSERT_FALSE(reshape != *clone);
    ASSERT_FALSE(reshape > *clone);
    ASSERT_FALSE(reshape < *clone);
}

TEST(UtilityApiLayers, ReshapeSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Stream stream(0);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";

    uint32_t numDimensions = 2 + (rand() % 3);
    uint32_t numReshapedDimensions = 1 + (rand() % (numDimensions - 1));

    vector<uint64_t> dimensions;
    vector<uint64_t> reshapedDimensions;
    uint64_t reshapedDimensionSize = 1;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        dimensions.push_back(1 + (rand() % 5));
        if ((d + 1) >= numReshapedDimensions)
            reshapedDimensionSize *= dimensions.back();
        else
            reshapedDimensions.push_back(dimensions.back());
    }
    reshapedDimensions.push_back(reshapedDimensionSize);

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(dimensions).dataType(dataType).build();

    Reshape reshape = Reshape::Builder()
                          .network(initialNetwork)
                          .featureInput(networkInput.getFeatureOutput().get())
                          .newDimensions(reshapedDimensions)
                          .build();
    ASSERT_TRUE(reshape.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(reshape.getFeatureOutput().get())
                                      .dataType(dataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    json reshapeJ = reshape.serialize(archiveWriter, stream);

    // printf("%s\n", reshapeJ.dump(4).c_str());

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &reshape;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(reshapeJ, fromLayerJ);

    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    ASSERT_EQ(reshapeJ["factory"], Layer::Factory::Layer.value());
    ASSERT_EQ(reshapeJ["version"], "1.0.0");
    ASSERT_EQ(reshapeJ["layer_type"], "reshape");

    const auto &input = reshapeJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = reshapeJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), reshapedDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    Network newNetwork;

    archiveWriter.finishArchive();
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");
    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, reshapeJ, &newNetwork);
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
    shared_ptr<ThorImplementation::Reshape> stampedReshape = dynamic_pointer_cast<ThorImplementation::Reshape>(otherLayers[0]);
    ASSERT_NE(stampedReshape, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    vector<uint64_t> stampedDimensions = {batchSize};
    vector<uint64_t> reshapedStampedDimensions = {batchSize};
    for (uint32_t d = 0; d < numDimensions; ++d)
        stampedDimensions.push_back(dimensions[d]);
    for (uint32_t d = 0; d < reshapedDimensions.size(); ++d)
        reshapedStampedDimensions.push_back(reshapedDimensions[d]);

    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
    ASSERT_EQ(stampedInput->getFeatureOutput().get().getDimensions(), stampedDimensions);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
    ASSERT_EQ(stampedOutput->getFeatureOutput().get().getDimensions(), reshapedStampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().get(), stampedReshape->getFeatureInput().get());
    ASSERT_EQ(stampedReshape->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    ASSERT_EQ(stampedReshape->getFeatureInput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
    ASSERT_EQ(stampedReshape->getFeatureOutput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
}
