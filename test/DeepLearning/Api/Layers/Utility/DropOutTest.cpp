#include <optional>
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

class TestableDropOut : public DropOut {
   public:
    explicit TestableDropOut(const DropOut &dropOut) : DropOut(dropOut) {}

    uint64_t reservedStateSizeInBytes(uint32_t batchSize) const { return getReservedStateSizeInBytes(batchSize); }
};

}  // namespace

TEST(UtilityApiLayers, DropOutReservedSpaceEstimateUsesFeatureInputDataType) {
    Network network("dropout_reserved_space_uses_dtype");
    const uint32_t batchSize = 7;
    const vector<uint64_t> dimensions = {3, 5};

    Tensor featureInput(DataType::FP32, dimensions);
    DropOut dropOut = DropOut::Builder().network(network).featureInput(featureInput).dropProportion(0.25f).build();

    vector<uint64_t> dimensionsWithBatchSize = {batchSize};
    dimensionsWithBatchSize.insert(dimensionsWithBatchSize.end(), dimensions.begin(), dimensions.end());

    const uint64_t expectedFp32ReserveBytes =
        ThorImplementation::DropOut::getReservedSpaceSizeInBytes(dimensionsWithBatchSize, DataType::FP32);
    const uint64_t fp16ReserveBytes =
        ThorImplementation::DropOut::getReservedSpaceSizeInBytes(dimensionsWithBatchSize, DataType::FP16);

    TestableDropOut testableDropOut(dropOut);
    EXPECT_EQ(testableDropOut.reservedStateSizeInBytes(batchSize), expectedFp32ReserveBytes);
    if (expectedFp32ReserveBytes != fp16ReserveBytes) {
        EXPECT_NE(testableDropOut.reservedStateSizeInBytes(batchSize), fp16ReserveBytes);
    }
}


TEST(UtilityApiLayers, Bfloat16DropOutUsesOneByteKeepMaskPerElement) {
    Network network("bfloat16_dropout_native_keep_mask");
    const uint32_t batchSize = 7;
    const vector<uint64_t> dimensions = {3, 5};

    Tensor featureInput(DataType::BF16, dimensions);
    DropOut dropOut = DropOut::Builder().network(network).featureInput(featureInput).dropProportion(0.25f).build();

    TestableDropOut testableDropOut(dropOut);
    EXPECT_EQ(testableDropOut.reservedStateSizeInBytes(batchSize), batchSize * 3 * 5);
}

TEST(UtilityApiLayers, DropOutBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    float dropProportion = ((rand() % 100) + 1) / 1000.0f;

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    DropOut dropOut = DropOut::Builder().network(network).featureInput(featureInput).dropProportion(dropProportion).build();

    ASSERT_TRUE(dropOut.isInitialized());

    std::optional<Tensor> actualInput = dropOut.getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), dataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = dropOut.getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), dataType);
    ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);

    float actualDropProportion = dropOut.getDropProportion();
    ASSERT_EQ(actualDropProportion, dropProportion);

    shared_ptr<Layer> cloneLayer = dropOut.clone();
    DropOut *clone = dynamic_cast<DropOut *>(cloneLayer.get());
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

    Network initialNetwork("initialNetwork");
    Stream stream(0);

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
    string dataTypeString = dataType == DataType::FP32 ? "fp32" : "fp16";
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
                          .featureInput(networkInput.getFeatureOutput().value())
                          .build();
    ASSERT_TRUE(dropOut.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(dropOut.getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel");

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
    Network newNetwork("newNetwork");

    // Write a dummy file with data into the archive since none of the layers wrote anything into it (no weights)
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::DataType::UINT8, {4});
    ThorImplementation::Tensor dummyData(cpuPlacement, descriptor);
    archiveWriter.addArchiveFile("dummy", dummyData);
    archiveWriter.createArchive("/tmp/", true);
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");
    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, dropOutJ, &newNetwork);
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
    shared_ptr<ThorImplementation::DropOut> stampedDropOut = dynamic_pointer_cast<ThorImplementation::DropOut>(otherLayers[0]);
    ASSERT_NE(stampedDropOut, nullptr);

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
    ASSERT_EQ(stampedOutput->getFeatureOutput().value().getDimensions(), stampedDimensions);

    // Ensure that they are all connected
    EXPECT_EQ(stampedInput->getFeatureOutput().value(), stampedDropOut->getFeatureInput().value());
    ASSERT_EQ(stampedDropOut->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());

    ASSERT_EQ(stampedDropOut->getFeatureInput().value().getDataType(), dataType);
    ASSERT_EQ(stampedDropOut->getFeatureOutput().value().getDataType(), dataType);

    ASSERT_EQ(stampedDropOut->getDropOutRate(), dropProportion);
}

TEST(UtilityApiLayers, NetworkAndPlacedNetworkControlTrainingDropoutWithoutSerializingRuntimePolicy) {
    Network network("dropout_runtime_training_control");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({4})
                             .dataType(DataType::FP16)
                             .build();
    DropOut dropOut = DropOut::Builder()
                          .network(network)
                          .featureInput(input.getFeatureOutput().value())
                          .dropProportion(0.5f)
                          .build();
    NetworkOutput output = NetworkOutput::Builder()
                               .network(network)
                               .name("output")
                               .inputTensor(dropOut.getFeatureOutput().value())
                               .dataType(DataType::FP16)
                               .build();
    (void)output;

    ASSERT_TRUE(dropOut.isTrainingDropoutEnabled());
    ASSERT_EQ(network.getNumTrainingDropoutControllableLayers(), 1U);
    ASSERT_TRUE(network.isTrainingDropoutEnabled());

    network.setTrainingDropoutEnabled(false);
    ASSERT_FALSE(dropOut.isTrainingDropoutEnabled());
    ASSERT_FALSE(network.isTrainingDropoutEnabled());

    const json architecture = dropOut.architectureJson();
    ASSERT_FALSE(architecture.contains("training_dropout_enabled"));

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placedNetwork = network.place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/false);
    ASSERT_NE(placedNetwork, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    ASSERT_EQ(placedNetwork->getNumTrainingDropoutControllableLayers(), 1U);
    ASSERT_FALSE(placedNetwork->isTrainingDropoutEnabled());
    auto physicalDropOut = dynamic_pointer_cast<ThorImplementation::DropOut>(
        placedNetwork->getStampedNetwork(0).getPhysicalLayerFromApiLayer(dropOut.getId()));
    ASSERT_NE(physicalDropOut, nullptr);
    ASSERT_FALSE(physicalDropOut->isTrainingDropoutEnabled());

    placedNetwork->setTrainingDropoutEnabled(true);
    ASSERT_TRUE(placedNetwork->isTrainingDropoutEnabled());
    ASSERT_TRUE(physicalDropOut->isTrainingDropoutEnabled());

    // The API graph remains an independently configurable template for future
    // placements; changing a placed network does not rewrite it.
    ASSERT_FALSE(network.isTrainingDropoutEnabled());
}


TEST(UtilityApiLayers, RaggedDropOutPreservesPartitionAndUsesPackedCapacityForReserveSpace) {
    Network network("ragged_dropout_builds");
    const uint64_t batchSize = 3;
    const uint64_t maxTotalValues = 9;
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("history")
                             .valuesDataType(DataType::FP32)
                             .trailingDimensions({4})
                             .maxTotalValues(maxTotalValues)
                             .batchSize(batchSize)
                             .offsetsDataType(DataType::UINT32)
                             .build();

    DropOut dropOut = DropOut::Builder().network(network).featureInput(input).dropProportion(0.25f).build();
    ASSERT_TRUE(dropOut.getUseRagged());
    ASSERT_TRUE(dropOut.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(dropOut.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(dropOut.getRaggedFeatureInput().value(), input);
    EXPECT_EQ(dropOut.getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(dropOut.getRaggedFeatureOutput()->getValuesDimensions(), input.getValuesDimensions());
    EXPECT_EQ(dropOut.getOutputTensorBytes(batchSize), input.getValues().getTotalSizeInBytes());

    TestableDropOut testableDropOut(dropOut);
    EXPECT_EQ(testableDropOut.reservedStateSizeInBytes(batchSize), maxTotalValues * 4);

    const json architecture = dropOut.architectureJson();
    ASSERT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>());
}

TEST(UtilityApiLayers, RaggedDropOutPlacesForInferenceAndPreservesLogicalOutput) {
    Network network("ragged_dropout_places");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("history")
                             .valuesDataType(DataType::BF16)
                             .trailingDimensions({3})
                             .maxTotalValues(9)
                             .batchSize(3)
                             .offsetsDataType(DataType::UINT32)
                             .build();
    DropOut dropOut = DropOut::Builder().network(network).featureInput(input).dropProportion(0.5f).build();
    RaggedNetworkOutput::Builder()
        .network(network)
        .name("history_out")
        .inputTensor(dropOut.getRaggedFeatureOutput().value())
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed = network.place(3, initDoneEvents, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();

    ThorImplementation::StampedNetwork stamped = placed->getStampedNetwork(0);
    auto physicalDropOut = dynamic_pointer_cast<ThorImplementation::DropOut>(
        stamped.getPhysicalLayerFromApiLayer(dropOut.getId()));
    ASSERT_NE(physicalDropOut, nullptr);
    EXPECT_TRUE(physicalDropOut->isRagged());
    EXPECT_FALSE(physicalDropOut->isTrainingMode());
}
