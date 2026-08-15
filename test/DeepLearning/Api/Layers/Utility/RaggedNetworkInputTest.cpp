#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"

#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "Utilities/Common/Event.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;
using json = nlohmann::json;

TEST(RaggedNetworkInputApi, BuildsLogicalRaggedInputBackedByPhysicalNetworkInputs) {
    Network network("ragged_network_input_api");

    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(DataType::INT32)
                              .offsetsDataType(DataType::UINT32)
                              .trailingDimensions({})
                              .batchSize(3)
                              .maxTotalValues(7)
                              .build();

    ASSERT_TRUE(labels.isInitialized());
    EXPECT_EQ(labels.getValuesDimensions(), (std::vector<uint64_t>{7}));
    EXPECT_EQ(labels.getOffsetsDimensions(), (std::vector<uint64_t>{4}));
    EXPECT_EQ(labels.getBatchSize(), 3u);
    EXPECT_EQ(labels.getMaxTotalValues(), 7u);

    const json architecture = network.architectureJson();
    ASSERT_TRUE(architecture.contains("ragged_network_inputs"));
    ASSERT_EQ(architecture.at("ragged_network_inputs").size(), 1u);
    const json& raggedInput = architecture.at("ragged_network_inputs").at(0);
    EXPECT_EQ(raggedInput.at("name").get<std::string>(), "labels");
    EXPECT_EQ(raggedInput.at("values_input_name").get<std::string>(), "labels.values");
    EXPECT_EQ(raggedInput.at("offsets_input_name").get<std::string>(), "labels.offsets");
    EXPECT_EQ(raggedInput.at("values_tensor_id").get<uint64_t>(), labels.getValues().getId());
    EXPECT_EQ(raggedInput.at("offsets_tensor_id").get<uint64_t>(), labels.getOffsets().getId());
    EXPECT_FALSE(raggedInput.contains("ragged_tensor"));

    ASSERT_TRUE(architecture.contains("layers"));
    ASSERT_EQ(architecture.at("layers").size(), 2u);
    std::set<std::string> physicalInputNames;
    for (const json& layer : architecture.at("layers")) {
        ASSERT_EQ(layer.at("layer_type").get<std::string>(), "network_input");
        EXPECT_TRUE(layer.at("dimensions_include_batch").get<bool>());
        physicalInputNames.insert(layer.at("name").get<std::string>());
    }
    EXPECT_EQ(physicalInputNames, (std::set<std::string>{"labels.values", "labels.offsets"}));
}

TEST(RaggedNetworkInputApi, ArchitectureOnlySaveLoadRoundTripUsesCanonicalTensorReferences) {
    const std::string networkName = "ragged_network_input_architecture_round_trip";
    Network network(networkName);
    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(DataType::INT32)
                              .offsetsDataType(DataType::UINT32)
                              .trailingDimensions({2})
                              .batchSize(2)
                              .maxTotalValues(6)
                              .build();

    const json architecture = network.architectureJson();
    const json& logicalInput = architecture.at("ragged_network_inputs").at(0);
    EXPECT_FALSE(logicalInput.contains("ragged_tensor"));
    EXPECT_EQ(logicalInput.at("values_tensor_id").get<uint64_t>(), labels.getValues().getId());
    EXPECT_EQ(logicalInput.at("offsets_tensor_id").get<uint64_t>(), labels.getOffsets().getId());

    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archiveDir =
        std::filesystem::temp_directory_path() /
        (std::string("thor_ragged_network_input_architecture_round_trip_") + std::to_string(now));
    std::filesystem::remove_all(archiveDir);

    network.save(archiveDir.string(), /*overwrite=*/true);

    Network loaded(networkName);
    ASSERT_NO_THROW(loaded.load(archiveDir.string()));
    const std::vector<RaggedNetworkInputReference> loadedInputs = loaded.getExternalRaggedNetworkInputs();
    ASSERT_EQ(loadedInputs.size(), 1u);
    EXPECT_EQ(loadedInputs[0].name, "labels");
    EXPECT_EQ(loadedInputs[0].valuesInputName, "labels.values");
    EXPECT_EQ(loadedInputs[0].offsetsInputName, "labels.offsets");
    EXPECT_EQ(loadedInputs[0].raggedTensor.getValues().getDimensions(), (std::vector<uint64_t>{6, 2}));
    EXPECT_EQ(loadedInputs[0].raggedTensor.getOffsets().getDimensions(), (std::vector<uint64_t>{3}));

    const json loadedArchitecture = loaded.architectureJson();
    const json& loadedLogicalInput = loadedArchitecture.at("ragged_network_inputs").at(0);
    EXPECT_FALSE(loadedLogicalInput.contains("ragged_tensor"));
    EXPECT_EQ(loadedLogicalInput.at("values_tensor_id").get<uint64_t>(), loadedInputs[0].raggedTensor.getValues().getId());
    EXPECT_EQ(loadedLogicalInput.at("offsets_tensor_id").get<uint64_t>(), loadedInputs[0].raggedTensor.getOffsets().getId());

    std::filesystem::remove_all(archiveDir);
}

TEST(RaggedNetworkInputApi, PlacedNetworkExposesLogicalInputName) {
    constexpr uint32_t batchSize = 2;
    Network network("ragged_network_input_placement");

    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(DataType::INT32)
                              .offsetsDataType(DataType::UINT32)
                              .batchSize(batchSize)
                              .maxTotalValues(5)
                              .build();

    NetworkOutput::Builder().network(network).name("label_values").inputTensor(labels.getValues()).dataType(DataType::INT32).build();
    NetworkOutput::Builder().network(network).name("label_offsets").inputTensor(labels.getOffsets()).dataType(DataType::UINT32).build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    EXPECT_TRUE(placed->hasNetworkInput("labels"));
    EXPECT_FALSE(placed->hasNetworkInput("labels.values"));
    EXPECT_FALSE(placed->hasNetworkInput("labels.offsets"));

    std::vector<std::string> networkInputNames = placed->getNetworkInputNames();
    EXPECT_EQ(std::set<std::string>(networkInputNames.begin(), networkInputNames.end()), (std::set<std::string>{"labels"}));
}

TEST(RaggedNetworkInputApi, RejectsInvalidOffsetDType) {
    Network network("ragged_network_input_invalid_offsets");

    EXPECT_THROW((RaggedNetworkInput::Builder()
                      .network(network)
                      .name("labels")
                      .valuesDataType(DataType::INT32)
                      .offsetsDataType(DataType::INT32)
                      .batchSize(2)
                      .maxTotalValues(5)
                      .build()),
                 std::logic_error);
}

TEST(RaggedNetworkInputApi, NetworkInputDiscoveryReportsOnlyLogicalRaggedBoundary) {
    Network network("ragged_network_input_discovery");

    RaggedTensor labels = RaggedNetworkInput::Builder()
                              .network(network)
                              .name("labels")
                              .valuesDataType(DataType::INT32)
                              .offsetsDataType(DataType::UINT64)
                              .batchSize(2)
                              .maxTotalValues(5)
                              .build();

    NetworkOutput::Builder().network(network).name("label_values").inputTensor(labels.getValues()).dataType(DataType::INT32).build();
    NetworkOutput::Builder().network(network).name("label_offsets").inputTensor(labels.getOffsets()).dataType(DataType::UINT64).build();

    EXPECT_EQ(network.getExternalNetworkInputNames(), (std::vector<std::string>{"labels"}));
    EXPECT_EQ(network.getInferenceNetworkInputNames(), (std::vector<std::string>{"labels"}));
    EXPECT_EQ(network.getRequiredNetworkInputNamesForOutputs({"label_values"}, /*inferenceOnly=*/true),
              (std::vector<std::string>{"labels"}));
    EXPECT_EQ(network.getRequiredNetworkInputNamesForOutputs({"label_offsets"}, /*inferenceOnly=*/true),
              (std::vector<std::string>{"labels"}));
    EXPECT_EQ(network.getRequiredNetworkInputNamesForOutputs({"label_values", "label_offsets"}, /*inferenceOnly=*/true),
              (std::vector<std::string>{"labels"}));
}
