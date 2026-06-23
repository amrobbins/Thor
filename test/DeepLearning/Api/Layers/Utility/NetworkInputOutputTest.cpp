#include <optional>
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, NetworkInputBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    NetworkInput networkInput = NetworkInput::Builder().network(network).dimensions(dimensions).dataType(dataType).build();

    ASSERT_TRUE(networkInput.isInitialized());

    std::optional<Tensor> actualInput = networkInput.getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), dataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = networkInput.getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), dataType);
    ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = networkInput.clone();
    NetworkInput *clone = dynamic_cast<NetworkInput *>(cloneLayer.get());
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

    ASSERT_EQ(networkInput.getId(), clone->getId());
    ASSERT_GT(networkInput.getId(), 1u);

    ASSERT_TRUE(networkInput == *clone);
    ASSERT_FALSE(networkInput != *clone);
    ASSERT_FALSE(networkInput > *clone);
    ASSERT_FALSE(networkInput < *clone);
}

TEST(UtilityApiLayers, NetworkOutputBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
    DataType outputDataType = rand() % 2 ? DataType::FP32 : DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    NetworkOutput networkOutput = NetworkOutput::Builder().network(network).inputTensor(featureInput).dataType(outputDataType).build();

    ASSERT_TRUE(networkOutput.isInitialized());

    std::optional<Tensor> actualInput = networkOutput.getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), dataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = networkOutput.getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), outputDataType);
    ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = networkOutput.clone();
    NetworkOutput *clone = dynamic_cast<NetworkOutput *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    std::optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.has_value());
    ASSERT_EQ(cloneInput.value().getDataType(), dataType);
    ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);

    std::optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.has_value());
    ASSERT_EQ(cloneOutput.value().getDataType(), outputDataType);
    ASSERT_EQ(cloneOutput.value().getDimensions(), dimensions);

    ASSERT_EQ(networkOutput.getId(), clone->getId());
    ASSERT_GT(networkOutput.getId(), 1u);

    ASSERT_TRUE(networkOutput == *clone);
    ASSERT_FALSE(networkOutput != *clone);
    ASSERT_FALSE(networkOutput > *clone);
    ASSERT_FALSE(networkOutput < *clone);
}

TEST(UtilityApiLayers, NetworkInputOutputSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    Stream stream(0);

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
    string dataTypeString = dataType == DataType::FP32 ? "fp32" : "fp16";

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
                                      .inputTensor(networkInput.getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel");

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
    Network newNetwork("newNetwork");

    // Write a dummy file with data into the archive since none of the layers wrote anything into it (no weights)
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::DataType::UINT8, {4});
    ThorImplementation::Tensor dummyData(cpuPlacement, descriptor);
    archiveWriter.addArchiveFile("dummy", dummyData);

    archiveWriter.createArchive("/tmp/", true);
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");

    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
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
    ASSERT_EQ(otherLayers.size(), 0U);

    vector<uint64_t> stampedDimensions = {batchSize};
    for (uint32_t d = 0; d < numDimensions; ++d)
        stampedDimensions.push_back(dimensions[d]);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_EQ(stampedInput->getFeatureOutput().value().getDimensions(), stampedDimensions);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
    ASSERT_TRUE(stampedOutput->getFeatureInput().has_value());
    ASSERT_EQ(stampedOutput->getFeatureInput().value().getDimensions(), stampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());

    ASSERT_EQ(stampedInput->getFeatureOutput().value().getDataType(), dataType);
    ASSERT_EQ(stampedOutput->getFeatureInput().value().getDataType(), dataType);
}

// Network composition primitive: a same-device NetworkOutput is an API boundary only.
// It must splice/alias the producer tensor instead of materializing a copy.
TEST(UtilityApiLayers, NetworkOutputAliasesProducerWhenOutputPlacementMatchesInputPlacement) {
    Network network("networkOutputAliasGpu");
    const uint32_t batchSize = 4;

    NetworkInput networkInput = NetworkInput::Builder()
                                    .network(network)
                                    .name("input")
                                    .dimensions({3})
                                    .dataType(DataType::FP32)
                                    .build();
    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(network)
                                      .name("output")
                                      .inputTensor(networkInput.getFeatureOutput().value())
                                      .dataType(DataType::FP32)
                                      .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize,
                                                            initDoneEvents,
                                                            /*inferenceOnly=*/true,
                                                            /*forcedDevices=*/{0},
                                                            /*forcedNumStampsPerGpu=*/1,
                                                            /*networkOutputsOnGpu=*/true);
    ASSERT_NE(placedNetwork, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(networkInput.getId()));
    auto stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(networkOutput.getId()));
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_NE(stampedOutput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureInput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().has_value());

    EXPECT_EQ(stampedInput->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());
    EXPECT_EQ(stampedOutput->getFeatureInput().value(), stampedOutput->getFeatureOutput().value());
    EXPECT_EQ(stampedOutput->getFeatureInput().value(), stampedOutput->getFeatureOutputForSlot(0).value());
    EXPECT_EQ(stampedOutput->getFeatureInput().value().getMemPtr<void>(), stampedOutput->getFeatureOutput().value().getMemPtr<void>());
    EXPECT_EQ(stampedOutput->getFeatureInput().value().getMemPtr<void>(), stampedOutput->getFeatureOutputForSlot(0).value().getMemPtr<void>());
    EXPECT_EQ(stampedOutput->getFeatureOutput().value().getPlacement().getMemDevice(), ThorImplementation::TensorPlacement::MemDevices::GPU);
}

// Network composition primitive: crossing a device boundary is still a real
// materialization boundary. GPU producer -> CPU NetworkOutput must copy.
TEST(UtilityApiLayers, NetworkOutputMaterializesWhenOutputPlacementDiffersFromInputPlacement) {
    Network network("networkOutputMaterializeCpu");
    const uint32_t batchSize = 4;

    NetworkInput networkInput = NetworkInput::Builder()
                                    .network(network)
                                    .name("input")
                                    .dimensions({3})
                                    .dataType(DataType::FP32)
                                    .build();
    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(network)
                                      .name("output")
                                      .inputTensor(networkInput.getFeatureOutput().value())
                                      .dataType(DataType::FP32)
                                      .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize,
                                                            initDoneEvents,
                                                            /*inferenceOnly=*/true,
                                                            /*forcedDevices=*/{0},
                                                            /*forcedNumStampsPerGpu=*/1,
                                                            /*networkOutputsOnGpu=*/false);
    ASSERT_NE(placedNetwork, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(networkInput.getId()));
    auto stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(networkOutput.getId()));
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_NE(stampedOutput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureInput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().has_value());

    EXPECT_EQ(stampedInput->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());
    EXPECT_NE(stampedOutput->getFeatureInput().value(), stampedOutput->getFeatureOutput().value());
    EXPECT_NE(stampedOutput->getFeatureInput().value(), stampedOutput->getFeatureOutputForSlot(0).value());
    EXPECT_NE(stampedOutput->getFeatureInput().value().getMemPtr<void>(), stampedOutput->getFeatureOutput().value().getMemPtr<void>());
    EXPECT_NE(stampedOutput->getFeatureInput().value().getMemPtr<void>(), stampedOutput->getFeatureOutputForSlot(0).value().getMemPtr<void>());
    EXPECT_EQ(stampedOutput->getFeatureOutput().value().getPlacement().getMemDevice(), ThorImplementation::TensorPlacement::MemDevices::CPU);
}
