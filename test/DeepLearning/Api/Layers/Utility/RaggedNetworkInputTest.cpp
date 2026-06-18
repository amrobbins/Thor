#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"

#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "Utilities/Common/Event.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

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
