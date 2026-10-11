#include <optional>
#include <chrono>
#include <filesystem>
#include <limits>
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
    uint64_t firstInstanceMemRequirementInBytes(
        uint32_t batchSize, ThorImplementation::TensorPlacement placement) const {
        return getFirstInstanceMemRequirementInBytes(batchSize, placement);
    }
};

}  // namespace

TEST(UtilityApiLayers, NativeDropOutUsesNoReserveSpaceForSupportedStorageTypes) {
    const uint32_t batchSize = 7;
    const vector<uint64_t> dimensions = {3, 5};
    for (DataType dataType : {DataType::FP16, DataType::FP32, DataType::FP64, DataType::BF16}) {
        Network network("dropout_stateless_native_" + std::to_string(static_cast<int>(dataType)));
        Tensor featureInput(dataType, dimensions);
        DropOut dropOut = DropOut::Builder().network(network).featureInput(featureInput).dropProportion(0.25f).build();
        TestableDropOut testableDropOut(dropOut);
        EXPECT_TRUE(ThorImplementation::DropOut::usesNativeKernel(dataType));
        EXPECT_EQ(testableDropOut.reservedStateSizeInBytes(batchSize), 0u);
    }
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


TEST(UtilityApiLayers, RaggedDropOutPreservesPartitionWithoutReserveSpace) {
    Network network("ragged_dropout_builds");
    const uint64_t batchSize = 3;
    const uint64_t maxTotalValues = 9;
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("history")
                             .valuesDataType(DataType::FP32)
                             .trailingDimensions({4})
                             .maxValuesPerRow(3)
                             .offsetsDataType(DataType::UINT32)
                             .build();

    DropOut dropOut = DropOut::Builder().network(network).featureInput(input).dropProportion(0.25f).build();
    ASSERT_TRUE(dropOut.getUseRagged());
    ASSERT_TRUE(dropOut.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(dropOut.getRaggedFeatureOutput().has_value());
    ASSERT_EQ(dropOut.getAllInputTensors().size(), 2u);
    EXPECT_EQ(dropOut.getAllInputTensors()[0], input.getValues());
    EXPECT_EQ(dropOut.getAllInputTensors()[1], input.getOffsets());
    EXPECT_EQ(dropOut.getRaggedFeatureInput().value(), input);
    EXPECT_EQ(dropOut.getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(dropOut.getRaggedFeatureOutput()->getValuesDimensions(), input.getValuesDimensions());
    EXPECT_EQ(dropOut.getOutputTensorBytes(batchSize), batchSize * input.getValues().getTotalSizeInBytes());

    TestableDropOut testableDropOut(dropOut);
    EXPECT_EQ(testableDropOut.reservedStateSizeInBytes(batchSize), 0u);

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
                             .maxValuesPerRow(3)
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
    ASSERT_TRUE(physicalDropOut->getFeatureInput().has_value());
    ASSERT_TRUE(physicalDropOut->getFeatureOutput().has_value());
    EXPECT_EQ(physicalDropOut->getFeatureInput().value(), physicalDropOut->getFeatureOutput().value());
}


TEST(UtilityApiLayers, PlacementNeutralDropOutSerializesSemanticGeometryAndAccountsPlacementCapacity) {
    constexpr uint64_t maxValuesPerRow = 4;
    Network network("placement_neutral_dropout_json");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({2})
                             .maxValuesPerRow(maxValuesPerRow)
                             .build();

    DropOut dropOut = DropOut::Builder().network(network).featureInput(input).dropProportion(0.25f).build();
    ASSERT_TRUE(dropOut.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(dropOut.getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());
    EXPECT_EQ(dropOut.getRaggedFeatureOutput()->getValues().getDimensions(), (vector<uint64_t>{maxValuesPerRow, 2}));
    EXPECT_FALSE(dropOut.outputTensorDimensionsIncludeBatch(dropOut.getFeatureOutput().value()));
    EXPECT_EQ(dropOut.getOutputTensorBytes(2), 2U * maxValuesPerRow * 2U * sizeof(float));
    EXPECT_EQ(dropOut.getOutputTensorBytes(7), 7U * maxValuesPerRow * 2U * sizeof(float));
    EXPECT_EQ(dropOut.getRaggedPartitionRequirementForInput(input.getRowPartitionToken()),
              ThorImplementation::RaggedPartitionRequirement::HOST_EXTENT);

    TestableDropOut testable(dropOut);
    ThorImplementation::TensorPlacement gpuPlacement(ThorImplementation::TensorPlacement::MemDevices::GPU, 0);
    EXPECT_EQ(testable.firstInstanceMemRequirementInBytes(2, gpuPlacement),
              2U * 2U * maxValuesPerRow * 2U * sizeof(float));
    EXPECT_EQ(testable.firstInstanceMemRequirementInBytes(7, gpuPlacement),
              2U * 7U * maxValuesPerRow * 2U * sizeof(float));

    const json architecture = dropOut.architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_FALSE(architecture.contains("training_dropout_enabled"));
    const json& inputJson = architecture.at("ragged_feature_input");
    const json& outputJson = architecture.at("ragged_feature_output");
    EXPECT_FALSE(inputJson.contains("geometry_mode"));
    EXPECT_EQ(inputJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
    EXPECT_FALSE(inputJson.contains("batch_size"));
    EXPECT_FALSE(inputJson.contains("max_total_values"));
    EXPECT_FALSE(outputJson.contains("batch_size"));
    EXPECT_FALSE(outputJson.contains("max_total_values"));
    EXPECT_EQ(outputJson.at("offsets").at("id").get<uint64_t>(), inputJson.at("offsets").at("id").get<uint64_t>());
    EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(), architecture.at("feature_output").at("id").get<uint64_t>());

    json wrongVersion = architecture;
    wrongVersion["version"] = "9.0.0";
    EXPECT_THROW(DropOut::deserialize(wrongVersion, &network), runtime_error);

    json wrongPartition = architecture;
    wrongPartition["ragged_feature_output"]["offsets"]["id"] = input.getValues().getId();
    EXPECT_THROW(DropOut::deserialize(wrongPartition, &network), runtime_error);

    json wrongOutputValues = architecture;
    wrongOutputValues["ragged_feature_output"]["values"]["id"] = input.getValues().getId();
    EXPECT_THROW(DropOut::deserialize(wrongOutputValues, &network), runtime_error);

    Network fixedNetwork("fixed_dropout_compatibility");
    RaggedTensor fixedInput = RaggedNetworkInput::Builder()
                                  .network(fixedNetwork)
                                  .name("history")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({2})
                                  .maxValuesPerRow(4)
                                  .build();
    DropOut fixed = DropOut::Builder().network(fixedNetwork).featureInput(fixedInput).dropProportion(0.25f).build();
    (void)RaggedNetworkOutput::Builder()
        .network(fixedNetwork)
        .name("fixed_out")
        .inputTensor(fixed.getRaggedFeatureOutput().value())
        .build();
    const json fixedJson = fixed.architectureJson();
    EXPECT_EQ(fixedJson.at("version").get<string>(), "1.1.0");
    EXPECT_EQ(fixedJson.at("ragged_feature_input").at("max_values_per_row").get<uint64_t>(), 4U);
    EXPECT_FALSE(fixedJson.at("ragged_feature_input").contains("batch_size"));
    EXPECT_FALSE(fixedJson.at("ragged_feature_input").contains("max_total_values"));
    EXPECT_EQ(fixed.getRaggedFeatureOutput()->getDescriptorForPlacement(3).getMaxTotalValues(), 12U);
    EXPECT_EQ(fixedJson.at("ragged_feature_input").at("offsets").at("dimensions"),
              (vector<uint64_t>{1}));

    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path fixedArchive = filesystem::temp_directory_path() /
                                          ("thor_dropout_fixed_" + to_string(nonce));
    filesystem::remove_all(fixedArchive);
    fixedNetwork.save(fixedArchive.string(), true);
    Network fixedReloaded("fixed_dropout_compatibility");
    ASSERT_NO_THROW(fixedReloaded.load(fixedArchive.string()));
    shared_ptr<DropOut> reloadedDropOut;
    for (uint32_t i = 0; i < fixedReloaded.getNumLayers(); ++i) {
        auto candidate = dynamic_pointer_cast<DropOut>(fixedReloaded.getLayer(i));
        if (candidate != nullptr) reloadedDropOut = candidate;
    }
    ASSERT_NE(reloadedDropOut, nullptr);
    ASSERT_TRUE(reloadedDropOut->getRaggedFeatureInput().has_value());
        EXPECT_EQ(reloadedDropOut->getRaggedFeatureInput()->getMaxValuesPerRow(), 4U);
        EXPECT_EQ(reloadedDropOut->getRaggedFeatureInput()->getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(reloadedDropOut->architectureJson().at("version").get<string>(), "1.1.0");
    filesystem::remove_all(fixedArchive);
}

TEST(UtilityApiLayers, PlacementNeutralDropOutSaveLoadReplacesCapacityAtDifferentBatchesAndPreservesHostExtent) {
    constexpr uint64_t maxValuesPerRow = 4;
    constexpr uint64_t elementsPerValue = 2;
    const string networkName = "placement_neutral_dropout_round_trip";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_dropout_neutral_" + to_string(nonce));
    filesystem::remove_all(archive);

    uint64_t savedValuesOriginalId = 0;
    uint64_t savedPartitionOriginalId = 0;
    uint64_t savedOutputOriginalId = 0;
    {
        Network network(networkName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({elementsPerValue})
                                 .maxValuesPerRow(maxValuesPerRow)
                                 .build();
        DropOut dropOut = DropOut::Builder().network(network).featureInput(input).dropProportion(0.5f).build();
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("dropped")
            .inputTensor(dropOut.getRaggedFeatureOutput().value())
            .build();
        savedValuesOriginalId = input.getValues().getOriginalId();
        savedPartitionOriginalId = input.getRowPartitionToken().getOriginalId();
        savedOutputOriginalId = dropOut.getFeatureOutput()->getOriginalId();
        EXPECT_EQ(dropOut.architectureJson().at("version").get<string>(), "1.1.0");
        network.save(archive.string(), true);
    }

    auto makeBatch = [&](const RaggedTensor& logicalInput, uint32_t batchSize) {
        const auto descriptor = logicalInput.getDescriptorForPlacement(batchSize);
        const uint64_t capacity = descriptor.getMaxTotalValues();
        ThorImplementation::Tensor values(
            ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU),
            descriptor.getValuesDescriptor());
        float* valuesPtr = values.getMemPtr<float>();
        vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
        uint64_t activeValues = 0;
        for (uint32_t row = 0; row < batchSize; ++row) {
            const uint64_t rowLength = row == 0 ? 0 : 1 + ((row - 1) % 2);
            activeValues += rowLength;
            hostOffsets[row + 1] = activeValues;
        }
        for (uint64_t value = 0; value < capacity; ++value) {
            for (uint64_t element = 0; element < elementsPerValue; ++element) {
                valuesPtr[value * elementsPerValue + element] =
                    value < activeValues ? static_cast<float>(value * 10 + element + 1)
                                         : numeric_limits<float>::quiet_NaN();
            }
        }
        ThorImplementation::Tensor offsets(
            ThorImplementation::TensorPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU),
            descriptor.getOffsetsDescriptor());
        uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
        for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);
        ThorImplementation::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
        partition.setHostOffsets(hostOffsets);
        Batch batch;
        batch.insert("tokens", ThorImplementation::RaggedTensor(values, partition));
        return batch;
    };

    auto findDropOut = [](Network& network) {
        shared_ptr<DropOut> result;
        for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
            auto candidate = dynamic_pointer_cast<DropOut>(network.getLayer(i));
            if (candidate != nullptr) result = candidate;
        }
        return result;
    };

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto externalInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(externalInputs.size(), 1U);
        const RaggedTensor loadedInput = externalInputs.front().raggedTensor;
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionOriginalId);
        shared_ptr<DropOut> loadedDropOut = findDropOut(loaded);
        ASSERT_NE(loadedDropOut, nullptr);
        ASSERT_TRUE(loadedDropOut->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedDropOut->getFeatureOutput()->getOriginalId(), savedOutputOriginalId);
        EXPECT_EQ(loadedDropOut->getRaggedFeatureOutput()->getRowPartitionToken(), loadedInput.getRowPartitionToken());

        const json architectureBeforePlacement = loaded.architectureJson();
        vector<Event> initDoneEvents;
        shared_ptr<PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initDoneEvents) event.synchronize();

        auto& stamped = placed->getStampedNetwork(0);
        const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(loadedInput.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());
        EXPECT_TRUE(ThorImplementation::hasRaggedPartitionRequirement(
            requirements.value(), ThorImplementation::RaggedPartitionRequirement::HOST_EXTENT));
        EXPECT_EQ(stamped.getManagedPartitionActiveCountInputForTest(loadedInput.getRowPartitionId()), nullptr);
        EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId()), nullptr);

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        auto physicalValues = stamped.getNamedInput("tokens.values");
        ASSERT_NE(physicalValues, nullptr);
        ASSERT_TRUE(physicalValues->getFeatureOutput().has_value());
        EXPECT_EQ(physicalValues->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, elementsPerValue}));
        auto physicalDropOut = dynamic_pointer_cast<ThorImplementation::DropOut>(
            stamped.getPhysicalLayerFromApiLayer(loadedDropOut->getId()));
        ASSERT_NE(physicalDropOut, nullptr);
        EXPECT_TRUE(physicalDropOut->isRagged());
        EXPECT_FALSE(physicalDropOut->isTrainingMode());
        ASSERT_TRUE(physicalDropOut->getFeatureInput().has_value());
        ASSERT_TRUE(physicalDropOut->getFeatureOutput().has_value());
        EXPECT_EQ(physicalDropOut->getFeatureInput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, elementsPerValue}));
        EXPECT_EQ(physicalDropOut->getFeatureInput().value(), physicalDropOut->getFeatureOutput().value());
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        map<string, InferenceOutputValue> outputs;
        ASSERT_NO_THROW(outputs = placed->inferLogical(makeBatch(loadedInput, placementBatchSize)));
        ASSERT_TRUE(outputs.contains("dropped"));
        const auto dropped = get<ThorImplementation::RaggedTensor>(outputs.at("dropped"));
        EXPECT_EQ(dropped.getValues().getDimensions(), (vector<uint64_t>{expectedCapacity, elementsPerValue}));
        const auto hostOffsets = dropped.getHostOffsetsIfAvailable();
        ASSERT_TRUE(hostOffsets.has_value());
        ASSERT_GE(hostOffsets->size(), 2U);
        EXPECT_EQ(hostOffsets->at(0), 0U);
        EXPECT_EQ(hostOffsets->at(1), 0U);
    };

    try {
        loadPlaceAndRun(2);
        loadPlaceAndRun(7);
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

TEST(UtilityApiLayers, PlacementNeutralDropOutZeroRateAndTrainingPlacementPreserveRuntimePolicy) {
    constexpr uint32_t batchSize = 3;
    constexpr uint64_t maxValuesPerRow = 4;
    ThorImplementation::TensorPlacement gpuPlacement(ThorImplementation::TensorPlacement::MemDevices::GPU, 0);

    Network zeroNetwork("placement_neutral_dropout_zero_rate");
    RaggedTensor zeroInput = RaggedNetworkInput::Builder()
                                 .network(zeroNetwork)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .maxValuesPerRow(maxValuesPerRow)
                                 .build();
    DropOut zero = DropOut::Builder().network(zeroNetwork).featureInput(zeroInput).dropProportion(0.0f).build();
    (void)RaggedNetworkOutput::Builder()
        .network(zeroNetwork)
        .name("zero")
        .inputTensor(zero.getRaggedFeatureOutput().value())
        .build();
    TestableDropOut testableZero(zero);
    EXPECT_EQ(testableZero.firstInstanceMemRequirementInBytes(batchSize, gpuPlacement), 0U);
    EXPECT_FALSE(zero.architectureJson().contains("training_dropout_enabled"));
    vector<Event> zeroEvents;
    auto zeroPlaced = zeroNetwork.place(batchSize, zeroEvents, false);
    ASSERT_NE(zeroPlaced, nullptr);
    for (Event& event : zeroEvents) event.synchronize();
    auto physicalZero = dynamic_pointer_cast<ThorImplementation::DropOut>(
        zeroPlaced->getStampedNetwork(0).getPhysicalLayerFromApiLayer(zero.getId()));
    ASSERT_NE(physicalZero, nullptr);
    EXPECT_TRUE(physicalZero->isTrainingMode());
    ASSERT_TRUE(physicalZero->getFeatureInput().has_value());
    ASSERT_TRUE(physicalZero->getFeatureOutput().has_value());
    EXPECT_EQ(physicalZero->getFeatureInput().value(), physicalZero->getFeatureOutput().value());

    Network trainingNetwork("placement_neutral_dropout_training");
    RaggedTensor trainingInput = RaggedNetworkInput::Builder()
                                     .network(trainingNetwork)
                                     .name("tokens")
                                     .valuesDataType(DataType::FP32)
                                     .offsetsDataType(DataType::UINT32)
                                     .trailingDimensions({2})
                                     .maxValuesPerRow(maxValuesPerRow)
                                     .build();
    DropOut training = DropOut::Builder()
                           .network(trainingNetwork)
                           .featureInput(trainingInput)
                           .dropProportion(0.5f)
                           .build();
    (void)RaggedNetworkOutput::Builder()
        .network(trainingNetwork)
        .name("training")
        .inputTensor(training.getRaggedFeatureOutput().value())
        .build();
    trainingNetwork.setTrainingDropoutEnabled(false);
    EXPECT_FALSE(training.architectureJson().contains("training_dropout_enabled"));
    vector<Event> trainingEvents;
    auto trainingPlaced = trainingNetwork.place(batchSize, trainingEvents, false);
    ASSERT_NE(trainingPlaced, nullptr);
    for (Event& event : trainingEvents) event.synchronize();
    auto physicalTraining = dynamic_pointer_cast<ThorImplementation::DropOut>(
        trainingPlaced->getStampedNetwork(0).getPhysicalLayerFromApiLayer(training.getId()));
    ASSERT_NE(physicalTraining, nullptr);
    EXPECT_TRUE(physicalTraining->isRagged());
    EXPECT_TRUE(physicalTraining->isTrainingMode());
    EXPECT_FALSE(physicalTraining->isTrainingDropoutEnabled());
    ASSERT_TRUE(physicalTraining->getFeatureInput().has_value());
    ASSERT_TRUE(physicalTraining->getFeatureOutput().has_value());
    EXPECT_NE(physicalTraining->getFeatureInput().value(), physicalTraining->getFeatureOutput().value());
    trainingPlaced->setTrainingDropoutEnabled(true);
    EXPECT_TRUE(physicalTraining->isTrainingDropoutEnabled());
}
