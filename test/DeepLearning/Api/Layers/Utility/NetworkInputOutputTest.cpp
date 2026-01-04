#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, NetworkInputBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    NetworkInput networkInput = NetworkInput::Builder().network(network).dimensions(dimensions).dataType(dataType).build();

    ASSERT_TRUE(networkInput.isInitialized());

    Optional<Tensor> actualInput = networkInput.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = networkInput.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = networkInput.clone();
    NetworkInput *clone = dynamic_cast<NetworkInput *>(cloneLayer.get());
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

    ASSERT_EQ(networkInput.getId(), clone->getId());
    ASSERT_GT(networkInput.getId(), 1u);

    ASSERT_TRUE(networkInput == *clone);
    ASSERT_FALSE(networkInput != *clone);
    ASSERT_FALSE(networkInput > *clone);
    ASSERT_FALSE(networkInput < *clone);
}

TEST(UtilityApiLayers, NetworkOutputBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    Tensor::DataType outputDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    NetworkOutput networkOutput = NetworkOutput::Builder().network(network).inputTensor(featureInput).dataType(outputDataType).build();

    ASSERT_TRUE(networkOutput.isInitialized());

    Optional<Tensor> actualInput = networkOutput.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = networkOutput.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), outputDataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = networkOutput.clone();
    NetworkOutput *clone = dynamic_cast<NetworkOutput *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), outputDataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(networkOutput.getId(), clone->getId());
    ASSERT_GT(networkOutput.getId(), 1u);

    ASSERT_TRUE(networkOutput == *clone);
    ASSERT_FALSE(networkOutput != *clone);
    ASSERT_FALSE(networkOutput > *clone);
    ASSERT_FALSE(networkOutput < *clone);
}

TEST(UtilityApiLayers, NetworkInputOutputSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Stream stream(0);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";

    uint32_t numDimensions = 1 + (rand() % 4);

    vector<uint64_t> dimensions;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        dimensions.push_back(1 + (rand() % 5));
    }

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(dimensions).dataType(dataType).build();
    ASSERT_TRUE(networkInput.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(networkInput.getFeatureOutput().get())
                                      .dataType(dataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // printf("%s\n", networkInputJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &networkInput;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(networkInputJ, fromLayerJ);

    layer = &networkOutput;
    fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(networkOutputJ, fromLayerJ);

    ASSERT_EQ(networkInputJ["factory"], Layer::Factory::Layer.value());
    ASSERT_EQ(networkInputJ["version"], "1.0.0");
    ASSERT_EQ(networkInputJ["layer_type"], "network_input");
    const auto &input = networkInputJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = networkInputJ.at("feature_output");
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
    ASSERT_EQ(otherLayers.size(), 0U);

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
    ASSERT_EQ(stampedOutput->getFeatureInput().get().getDimensions(), stampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    ASSERT_EQ(stampedInput->getFeatureOutput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
    ASSERT_EQ(stampedOutput->getFeatureInput().get().getDataType(), Tensor::convertToImplementationDataType(dataType));
}
