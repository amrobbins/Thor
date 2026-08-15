#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

using namespace Thor;
using json = nlohmann::json;

TEST(RaggedNetworkOutputApi, RegistersOneLogicalOutputBackedByInternalComponents) {
    Network network("ragged_network_output_api");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .trailingDimensions({2})
                             .batchSize(2)
                             .maxTotalValues(6)
                             .build();

    RaggedNetworkOutput output =
        RaggedNetworkOutput::Builder().network(network).name("tokens_out").inputTensor(input).build();

    EXPECT_EQ(output.getName(), "tokens_out");
    EXPECT_EQ(output.getInput(), input);
    EXPECT_TRUE(network.hasRaggedNetworkOutput("tokens_out"));
    EXPECT_TRUE(network.getExternalNetworkOutputNames().empty());

    const std::vector<RaggedNetworkOutputReference> logicalOutputs = network.getExternalRaggedNetworkOutputs();
    ASSERT_EQ(logicalOutputs.size(), 1u);
    EXPECT_EQ(logicalOutputs[0].name, "tokens_out");
    EXPECT_EQ(logicalOutputs[0].valuesOutputName, "__thor_ragged_output.tokens_out.values");
    EXPECT_EQ(logicalOutputs[0].offsetsOutputName, "__thor_ragged_output.tokens_out.offsets");
    EXPECT_EQ(logicalOutputs[0].raggedTensor, output.getFeatureOutput());
    EXPECT_EQ(network.getRequiredNetworkInputNamesForOutputs({"tokens_out"}, /*inferenceOnly=*/true),
              (std::vector<std::string>{"tokens"}));

    const json architecture = network.architectureJson();
    ASSERT_TRUE(architecture.contains("ragged_network_outputs"));
    ASSERT_EQ(architecture.at("ragged_network_outputs").size(), 1u);
    const json& logicalOutput = architecture.at("ragged_network_outputs").at(0);
    EXPECT_EQ(logicalOutput.at("name").get<std::string>(), "tokens_out");
    EXPECT_EQ(logicalOutput.at("values_tensor_id").get<uint64_t>(), output.getFeatureOutput().getValues().getId());
    EXPECT_EQ(logicalOutput.at("offsets_tensor_id").get<uint64_t>(), output.getFeatureOutput().getOffsets().getId());
    EXPECT_FALSE(logicalOutput.contains("ragged_tensor"));

    std::set<std::string> internalNames;
    for (const json& layer : architecture.at("layers")) {
        if (layer.at("layer_type").get<std::string>() != "network_output")
            continue;
        EXPECT_FALSE(layer.at("external").get<bool>());
        internalNames.insert(layer.at("name").get<std::string>());
    }
    EXPECT_EQ(internalNames,
              (std::set<std::string>{"__thor_ragged_output.tokens_out.values", "__thor_ragged_output.tokens_out.offsets"}));
}

TEST(RaggedNetworkOutputApi, RejectsDuplicateLogicalNames) {
    Network network("ragged_network_output_duplicate_name");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .trailingDimensions({2})
                             .batchSize(2)
                             .maxTotalValues(6)
                             .build();

    (void)RaggedNetworkOutput::Builder().network(network).name("tokens_out").inputTensor(input).build();
    EXPECT_THROW((RaggedNetworkOutput::Builder().network(network).name("tokens_out").inputTensor(input).build()),
                 std::logic_error);
}


TEST(RaggedNetworkOutputApi, ArchitectureOnlySaveLoadRoundTripUsesCanonicalTensorReferences) {
    const std::string networkName = "ragged_network_output_architecture_round_trip";
    Network network(networkName);
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .trailingDimensions({2})
                             .batchSize(2)
                             .maxTotalValues(6)
                             .build();
    RaggedNetworkOutput output =
        RaggedNetworkOutput::Builder().network(network).name("tokens_out").inputTensor(input).build();

    const json architecture = network.architectureJson();
    ASSERT_TRUE(architecture.contains("ragged_network_outputs"));
    ASSERT_EQ(architecture.at("ragged_network_outputs").size(), 1u);
    const json& logicalOutput = architecture.at("ragged_network_outputs").at(0);
    EXPECT_FALSE(logicalOutput.contains("ragged_tensor"));
    EXPECT_EQ(logicalOutput.at("values_tensor_id").get<uint64_t>(), output.getFeatureOutput().getValues().getId());
    EXPECT_EQ(logicalOutput.at("offsets_tensor_id").get<uint64_t>(), output.getFeatureOutput().getOffsets().getId());

    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path archiveDir =
        std::filesystem::temp_directory_path() /
        (std::string("thor_ragged_network_output_architecture_round_trip_") + std::to_string(now));
    std::filesystem::remove_all(archiveDir);

    network.save(archiveDir.string(), /*overwrite=*/true);

    Network loaded(networkName);
    ASSERT_NO_THROW(loaded.load(archiveDir.string()));
    const std::vector<RaggedNetworkOutputReference> loadedOutputs = loaded.getExternalRaggedNetworkOutputs();
    ASSERT_EQ(loadedOutputs.size(), 1u);
    EXPECT_EQ(loadedOutputs[0].name, "tokens_out");
    EXPECT_EQ(loadedOutputs[0].valuesOutputName, "__thor_ragged_output.tokens_out.values");
    EXPECT_EQ(loadedOutputs[0].offsetsOutputName, "__thor_ragged_output.tokens_out.offsets");
    EXPECT_EQ(loadedOutputs[0].raggedTensor.getValues().getDimensions(), (std::vector<uint64_t>{6, 2}));
    EXPECT_EQ(loadedOutputs[0].raggedTensor.getOffsets().getDimensions(), (std::vector<uint64_t>{3}));

    const json loadedArchitecture = loaded.architectureJson();
    const json& loadedLogicalOutput = loadedArchitecture.at("ragged_network_outputs").at(0);
    EXPECT_FALSE(loadedLogicalOutput.contains("ragged_tensor"));
    EXPECT_EQ(loadedLogicalOutput.at("values_tensor_id").get<uint64_t>(), loadedOutputs[0].raggedTensor.getValues().getId());
    EXPECT_EQ(loadedLogicalOutput.at("offsets_tensor_id").get<uint64_t>(), loadedOutputs[0].raggedTensor.getOffsets().getId());

    std::filesystem::remove_all(archiveDir);
}
