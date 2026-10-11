#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;
using DataType = ThorImplementation::DataType;
using json = nlohmann::json;

namespace {
Api::RaggedTensor neutralQuery(Api::Network& network, const char* name, uint64_t m = 5) {
    return Api::RaggedNetworkInput::Builder().network(network).name(name)
        .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
        .trailingDimensions({32}).maxValuesPerRow(m).build();
}

Api::RaggedTensor neutralKey(Api::Network& network, const char* name, uint64_t m = 7) {
    return Api::RaggedNetworkInput::Builder().network(network).name(name)
        .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
        .trailingDimensions({32}).maxValuesPerRow(m).build();
}

Api::Attention makeAttention(Api::Network& network, const Api::RaggedTensor& q, const Api::RaggedTensor& k) {
    auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
        .valuesDataType(DataType::FP16).trailingDimensions({32}).partition(k).build();
    return Api::Attention::Builder().network(network)
        .queryInput(q).keyInput(k).valueInput(v).numHeads(4).numKeyValueHeads(2)
        .headDim(8).valueDim(8).outputFeatures(32).build();
}

std::shared_ptr<Api::Attention> rehydrate(Api::Network& network, const json& architecture) {
    const uint32_t before = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> reader;
    Api::Attention::deserialize(reader, architecture, &network);
    EXPECT_EQ(network.getNumTrainableLayers(), before + 1);
    return std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(before));
}
}  // namespace

TEST(AttentionNeutral2D1, LogicalMIndependentPartitionsCloneAndStrictRoundTrip) {
    Api::Network network("attention_2d1_neutral_metadata");
    auto q = neutralQuery(network, "q", 5);
    auto k = neutralKey(network, "k", 7);
    auto attention = makeAttention(network, q, k);
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(attention.getRaggedFeatureOutput()->sharesPartitionWith(q));
    EXPECT_FALSE(attention.getRaggedFeatureOutput()->sharesPartitionWith(k));
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{5, 32}));
    EXPECT_TRUE(attention.getRaggedKeyInput()->sharesPartitionWith(*attention.getRaggedValueInput()));
    auto copied = std::dynamic_pointer_cast<Api::Attention>(attention.clone());
    ASSERT_NE(copied, nullptr);
    ASSERT_TRUE(copied->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(copied->getRaggedFeatureOutput()->sharesPartitionWith(*copied->getRaggedQueryInput()));

    const json schema = attention.architectureJson();
    ASSERT_EQ(schema.at("version"), "2.1.0");
    for (const char* field : {"ragged_query_input", "ragged_key_input", "ragged_value_input", "ragged_feature_output"}) {
        EXPECT_FALSE(schema.at(field).contains("geometry_mode"));
        EXPECT_FALSE(schema.at(field).contains("batch_size"));
        EXPECT_FALSE(schema.at(field).contains("max_total_values"));
    }
    EXPECT_EQ(schema.at("ragged_query_input").at("max_values_per_row"), 5);
    EXPECT_EQ(schema.at("ragged_key_input").at("max_values_per_row"), 7);
    EXPECT_FALSE(schema.contains("batch_size"));
    // Previously emitted canonical references carried a redundant marker.
    // Both encodings must reconstruct the same logical Q/KV partitions.
    json previouslyWritten = schema;
    for (const char* field : {"ragged_query_input", "ragged_key_input", "ragged_value_input", "ragged_feature_output"}) {
        previouslyWritten[field]["geometry_mode"] = "placement_neutral";
    }
    auto previousRestored = rehydrate(network, previouslyWritten);
    ASSERT_NE(previousRestored, nullptr);
    EXPECT_TRUE(previousRestored->getRaggedFeatureOutput()->sharesPartitionWith(*previousRestored->getRaggedQueryInput()));
    EXPECT_TRUE(previousRestored->getRaggedKeyInput()->sharesPartitionWith(*previousRestored->getRaggedValueInput()));
    EXPECT_FALSE(previousRestored->getRaggedQueryInput()->sharesPartitionWith(*previousRestored->getRaggedKeyInput()));

    EXPECT_FALSE(schema.contains("max_total_values"));

    auto restored = rehydrate(network, schema);
    ASSERT_NE(restored, nullptr);
    ASSERT_TRUE(restored->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(restored->getRaggedFeatureOutput()->sharesPartitionWith(*restored->getRaggedQueryInput()));
    EXPECT_TRUE(restored->getRaggedKeyInput()->sharesPartitionWith(*restored->getRaggedValueInput()));
    EXPECT_FALSE(restored->getRaggedQueryInput()->sharesPartitionWith(*restored->getRaggedKeyInput()));
    EXPECT_EQ(restored->getRaggedFeatureOutput()->getValuesDimensions(), (std::vector<uint64_t>{5, 32}));
    EXPECT_EQ(restored->getRaggedKeyInput()->getMaxValuesPerRow(), 7);
    EXPECT_EQ(restored->getRaggedKeyInput()->getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(restored->architectureJson().at("version"), "2.1.0");
}

TEST(AttentionNeutral2D1, RejectsCorruptOriginalIdsDtypesShapesPartitionsAndRuntimeGeometry) {
    Api::Network network("attention_2d1_bad_metadata");
    auto q = neutralQuery(network, "q");
    auto k = neutralKey(network, "k");
    auto attention = makeAttention(network, q, k);
    const json good = attention.architectureJson();
    auto rejects = [&](const json& candidate) {
        std::shared_ptr<thor_file::TarReader> reader;
        EXPECT_THROW(Api::Attention::deserialize(reader, candidate, &network), std::exception);
    };
    json invalid = good;
    invalid["query_input"]["data_type"] = DataType::BF16;
    rejects(invalid);
    invalid = good;
    invalid["key_input"]["dimensions"] = std::vector<uint64_t>{6, 32};
    rejects(invalid);
    invalid = good;
    invalid["ragged_query_input"]["offsets"]["id"] = good.at("ragged_key_input").at("offsets").at("id");
    rejects(invalid);
    invalid = good;
    invalid["ragged_value_input"]["max_values_per_row"] = 6;
    rejects(invalid);
    invalid = good;
    invalid["ragged_feature_output"]["values"]["id"] = good.at("query_input").at("id");
    rejects(invalid);
    invalid = good;
    invalid["ragged_query_input"]["batch_size"] = 2;
    rejects(invalid);
    invalid = good;
    invalid["batch_size"] = 7;
    rejects(invalid);
    invalid = good;
    invalid["version"] = "2.0.0";
    // Historical schema versions are input grammars, not execution modes.
    // An otherwise recoverable reference must preserve logical Q/KV geometry.
    auto historical = rehydrate(network, invalid);
    ASSERT_NE(historical, nullptr);
    ASSERT_TRUE(historical->getRaggedQueryInput().has_value());
    ASSERT_TRUE(historical->getRaggedKeyInput().has_value());
    EXPECT_EQ(historical->getRaggedQueryInput()->getMaxValuesPerRow(), 5);
    EXPECT_EQ(historical->getRaggedKeyInput()->getMaxValuesPerRow(), 7);
    EXPECT_TRUE(historical->getRaggedKeyInput()->sharesPartitionWith(*historical->getRaggedValueInput()));
    EXPECT_FALSE(historical->getRaggedQueryInput()->sharesPartitionWith(*historical->getRaggedKeyInput()));
    invalid = good;
    invalid["key_value_ragged"] = false;
    rejects(invalid);
}

TEST(AttentionNeutral2D1, MixedDirectionBuildersUseSemanticRaggedGeometry) {
    {
        Api::Network network("attention_2d1_q_neutral_kv_dense");
        auto q = neutralQuery(network, "q", 3);
        auto k = Api::NetworkInput::Builder().network(network).name("k")
            .dataType(DataType::FP16).dimensions({6, 32}).build();
        auto v = Api::NetworkInput::Builder().network(network).name("v")
            .dataType(DataType::FP16).dimensions({6, 32}).build();
        auto layer = Api::Attention::Builder().network(network)
            .queryInput(q).keyInput(k.getFeatureOutput().value())
            .valueInput(v.getFeatureOutput().value()).numHeads(4).headDim(8).build();
        EXPECT_EQ(layer.architectureJson().at("version"), "2.1.0");
        EXPECT_TRUE(layer.getRaggedFeatureOutput()->sharesPartitionWith(q));
        EXPECT_NE(rehydrate(network, layer.architectureJson()), nullptr);
    }
    {
        Api::Network network("attention_2d1_q_dense_kv_neutral");
        auto q = Api::NetworkInput::Builder().network(network).name("q")
            .dataType(DataType::FP16).dimensions({4, 32}).build();
        auto k = neutralKey(network, "k", 6);
        auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({32}).partition(k).build();
        auto layer = Api::Attention::Builder().network(network)
            .queryInput(q.getFeatureOutput().value()).keyInput(k)
            .valueInput(v).numHeads(4).headDim(8).build();
        EXPECT_EQ(layer.architectureJson().at("version"), "2.1.0");
        EXPECT_FALSE(layer.getRaggedFeatureOutput().has_value());
        EXPECT_NE(rehydrate(network, layer.architectureJson()), nullptr);
    }
}

TEST(AttentionNeutral2D1, RejectsMismatchedLogicalKvPartitionAtBuildTime) {
    Api::Network network("attention_2d1_wrong_partition");
    auto q = neutralQuery(network, "q");
    auto k = neutralKey(network, "k");
    auto wrongV = neutralKey(network, "wrong_v");
    EXPECT_THROW((Api::Attention::Builder().network(network)
        .queryInput(q).keyInput(k).valueInput(wrongV).numHeads(4).headDim(8).build()), std::invalid_argument);
}

TEST(AttentionNeutral2D1, ArchitectureOnlyDiskSaveReloadPreservesNeutralTopology) {
    const auto archive = std::filesystem::temp_directory_path() /
        ("attention_2d1_neutral_disk_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(archive);
    {
        Api::Network network("attention_2d1_neutral_saved");
        auto q = neutralQuery(network, "q", 5);
        auto k = neutralKey(network, "k", 7);
        auto attention = makeAttention(network, q, k);
        (void)Api::RaggedNetworkOutput::Builder().network(network).name("out")
            .inputTensor(*attention.getRaggedFeatureOutput()).build();
        ASSERT_NO_THROW(network.save(archive.string(), true));
    }
    {
        Api::Network restored("attention_2d1_neutral_saved");
        ASSERT_NO_THROW(restored.load(archive.string()));
        std::shared_ptr<Api::Attention> layer;
        for (uint32_t index = 0; index < restored.getNumTrainableLayers(); ++index) {
            if (auto candidate = std::dynamic_pointer_cast<Api::Attention>(restored.getTrainableLayer(index))) {
                layer = std::move(candidate);
                break;
            }
        }
        ASSERT_NE(layer, nullptr);
        EXPECT_EQ(layer->architectureJson().at("version"), "2.1.0");
        EXPECT_EQ(layer->getRaggedQueryInput()->getMaxValuesPerRow(), 5);
        EXPECT_EQ(layer->getRaggedKeyInput()->getMaxValuesPerRow(), 7);
        EXPECT_TRUE(layer->getRaggedFeatureOutput()->sharesPartitionWith(*layer->getRaggedQueryInput()));
        EXPECT_TRUE(layer->getRaggedKeyInput()->sharesPartitionWith(*layer->getRaggedValueInput()));
        EXPECT_FALSE(layer->getRaggedQueryInput()->sharesPartitionWith(*layer->getRaggedKeyInput()));
        EXPECT_EQ(layer->listParameters().size(), 4u);
    }
    std::filesystem::remove_all(archive);
}

TEST(AttentionNeutral2D1, ResidualMustReuseQueryPartition) {
    Api::Network network("attention_2d1_neutral_residual");
    auto q = neutralQuery(network, "q", 5);
    auto k = neutralKey(network, "k", 7);
    auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
        .valuesDataType(DataType::FP16).trailingDimensions({32}).partition(k).build();
    auto residual = Api::RaggedNetworkInput::Builder().network(network).name("residual")
        .valuesDataType(DataType::FP16).trailingDimensions({32}).partition(q).build();
    auto wrongResidual = neutralQuery(network, "wrong_residual", 5);
    EXPECT_THROW((Api::Attention::Builder().network(network)
        .queryInput(q).keyInput(k).valueInput(v).residualInput(wrongResidual)
        .numHeads(4).headDim(8).build()), std::invalid_argument);
    auto layer = Api::Attention::Builder().network(network)
        .queryInput(q).keyInput(k).valueInput(v).residualInput(residual)
        .numHeads(4).headDim(8).build();
    const json schema = layer.architectureJson();
    EXPECT_EQ(schema.at("version"), "2.1.0");
    EXPECT_FALSE(schema.at("ragged_residual_input").contains("geometry_mode"));
    EXPECT_TRUE(layer.getRaggedResidualInput()->sharesPartitionWith(q));
    auto restored = rehydrate(network, schema);
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->getRaggedResidualInput()->sharesPartitionWith(*restored->getRaggedQueryInput()));
}

TEST(AttentionNeutral2D1, HistoricalCompositeAttentionRecoversLogicalPartitionsAndOutputWithoutArchivedBatch) {
    Api::Network network("attention_2d1_legacy_logical_reconstruction");
    const auto q = neutralQuery(network, "q", 5);
    const auto k = neutralKey(network, "k", 7);
    const auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
        .valuesDataType(DataType::FP16).trailingDimensions({32}).partition(k).build();
    const auto residual = Api::RaggedNetworkInput::Builder().network(network).name("residual")
        .valuesDataType(DataType::FP16).trailingDimensions({32}).partition(q).build();
    const auto layer = Api::Attention::Builder().network(network)
        .queryInput(q).keyInput(k).valueInput(v).residualInput(residual)
        .numHeads(4).numKeyValueHeads(2).headDim(8).valueDim(8).outputFeatures(32).build();
    json legacy = layer.architectureJson();
    ASSERT_EQ(legacy.at("version"), "2.1.0");
    legacy["version"] = "2.0.0";
    // The older grammar recorded physical extents. Neither archived B nor
    // packed capacity is semantic, even when Q and KV have different M.
    for (const char* field : {"ragged_query_input", "ragged_key_input", "ragged_value_input",
                              "ragged_residual_input", "ragged_feature_output"}) {
        auto& ref = legacy.at(field);
        ref.erase("layer_ragged_reference_version");
        ref.erase("geometry_mode");
        ref.erase("max_values_per_row");  // M must be recovered from the graph.
        ref["batch_size"] = 97;
        ref["max_total_values"] = 3333;
        ref["values"]["dimensions"][0] = 3333;
        ref["offsets"]["dimensions"][0] = 98;
    }
    legacy["feature_output"]["dimensions"][0] = 3333;
    legacy["batch_size"] = 811;
    legacy["max_total_values"] = 45555;

    auto restored = rehydrate(network, legacy);
    ASSERT_NE(restored, nullptr);
    const auto rq = restored->getRaggedQueryInput().value();
    const auto rk = restored->getRaggedKeyInput().value();
    const auto rv = restored->getRaggedValueInput().value();
    const auto rr = restored->getRaggedResidualInput().value();
    const auto ro = restored->getRaggedFeatureOutput().value();
    EXPECT_EQ(rq.getMaxValuesPerRow(), 5u);
    EXPECT_EQ(rk.getMaxValuesPerRow(), 7u);
    EXPECT_TRUE(rk.sharesPartitionWith(rv));
    EXPECT_FALSE(rq.sharesPartitionWith(rk));
    EXPECT_TRUE(rq.sharesPartitionWith(rr));
    EXPECT_TRUE(rq.sharesPartitionWith(ro));
    EXPECT_EQ(ro.getValuesDimensions(), (std::vector<uint64_t>{5, 32}));
    EXPECT_EQ(restored->architectureJson().at("version"), "2.1.0");
    for (uint64_t b : {2u, 7u}) {
        EXPECT_EQ(rq.getDescriptorForPlacement(b).getValuesDimensions(), (std::vector<uint64_t>{b * 5, 32}));
        EXPECT_EQ(rk.getDescriptorForPlacement(b).getValuesDimensions(), (std::vector<uint64_t>{b * 7, 32}));
        EXPECT_EQ(ro.getDescriptorForPlacement(b).getValuesDimensions(), (std::vector<uint64_t>{b * 5, 32}));
    }

    // Corruption of *semantic* geometry still must fail, unlike archived B.
    json badSemanticM = legacy;
    badSemanticM["ragged_key_input"]["max_values_per_row"] = 6;
    std::shared_ptr<thor_file::TarReader> reader;
    EXPECT_THROW(Api::Attention::deserialize(reader, badSemanticM, &network), std::runtime_error);
    json badPartition = legacy;
    badPartition["ragged_value_input"]["offsets"]["id"] = q.getRowPartitionToken().getOriginalId();
    EXPECT_THROW(Api::Attention::deserialize(reader, badPartition, &network), std::runtime_error);
}
