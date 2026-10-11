#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {
namespace Api = Thor;
using DataType = ThorImplementation::DataType;
using json = nlohmann::json;

Api::RaggedTensor input(Api::Network& net, const char* name, uint64_t m, uint64_t heads,
                        DataType offsets = DataType::UINT64) {
    return Api::RaggedNetworkInput::Builder().network(net).name(name)
        .valuesDataType(DataType::FP16).offsetsDataType(offsets)
        .maxValuesPerRow(m).trailingDimensions({heads, 32}).build();
}

std::shared_ptr<Api::ScaledDotProductAttention> restore(Api::Network& network, const json& arch) {
    const uint32_t before = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> reader;
    Api::ScaledDotProductAttention::deserialize(reader, arch, &network);
    return std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getTrainableLayer(before));
}

struct CrossFixture {
    Api::Network network{"sdpa_neutral_c1"};
    Api::RaggedTensor q = input(network, "q", 3, 4, DataType::UINT32);
    Api::RaggedTensor kv = input(network, "kv", 7, 2, DataType::UINT64);
    Api::ScaledDotProductAttention sdpa = Api::ScaledDotProductAttention::Builder()
        .network(network).queryInput(q).keyInput(kv).valueInput(kv).build();
    json arch = sdpa.architectureJson();
};

void expectBad(CrossFixture& f, const json& bad) {
    EXPECT_ANY_THROW((void)restore(f.network, bad));
}
} // namespace

TEST(SdpaNeutralFoundationC1, VersionedSemanticQKvReferencesRoundTripWithoutRuntimeGeometry) {
    CrossFixture f;
    ASSERT_EQ(f.arch.at("version"), "2.2.0");
    EXPECT_FALSE(f.arch.contains("batch_size"));
    EXPECT_FALSE(f.arch.contains("max_total_values"));
    const auto& qr = f.arch.at("query_ragged_input");
    const auto& kr = f.arch.at("key_ragged_input");
    const auto& vr = f.arch.at("value_ragged_input");
    for (const json* ref : {&qr, &kr, &vr}) {
        ASSERT_FALSE(ref->contains("geometry_mode"));
        EXPECT_FALSE(ref->contains("batch_size"));
        EXPECT_FALSE(ref->contains("max_total_values"));
        EXPECT_EQ(ref->at("offsets").at("dimensions"), json::array({1}));
    }
    EXPECT_EQ(qr.at("max_values_per_row"), 3);
    EXPECT_EQ(kr.at("max_values_per_row"), 7);
    EXPECT_EQ(kr.at("offsets").at("id"), vr.at("offsets").at("id"));
    EXPECT_NE(qr.at("offsets").at("id"), kr.at("offsets").at("id"));

    auto restored = restore(f.network, f.arch);
    ASSERT_NE(restored, nullptr);
    ASSERT_TRUE(restored->getQueryRaggedInput().has_value());
    ASSERT_TRUE(restored->getKeyRaggedInput().has_value());
    EXPECT_TRUE(restored->getQueryRaggedInput()->sharesPartitionWith(f.q));
    EXPECT_TRUE(restored->getKeyRaggedInput()->sharesPartitionWith(f.kv));
    EXPECT_FALSE(restored->getQueryRaggedInput()->sharesPartitionWith(f.kv));
    EXPECT_EQ(restored->getOutput("output").getDimensions(), (std::vector<uint64_t>{3, 4, 32}));
    EXPECT_EQ(restored->architectureJson().at("version"), "2.2.0");
}

TEST(SdpaNeutralFoundationC1, SelfAttentionAndMixedDenseGeometryRoundTrip) {
    Api::Network network("sdpa_c1_modes");
    auto q = input(network, "q", 5, 4);
    auto kv = input(network, "kv", 7, 2);
    auto denseQ = Api::NetworkInput::Builder().network(network).name("dense_q")
        .dataType(DataType::FP16).dimensions({4, 4, 32}).build();
    auto denseKV = Api::NetworkInput::Builder().network(network).name("dense_kv")
        .dataType(DataType::FP16).dimensions({7, 2, 32}).build();
    for (const auto& layer : {
             Api::ScaledDotProductAttention::Builder().network(network).selfInput(q).build(),
             Api::ScaledDotProductAttention::Builder().network(network).queryInput(q)
                 .keyInput(denseKV.getFeatureOutput().value()).valueInput(denseKV.getFeatureOutput().value()).build(),
             Api::ScaledDotProductAttention::Builder().network(network).queryInput(denseQ.getFeatureOutput().value())
                 .keyInput(kv).valueInput(kv).build()}) {
        const json arch = layer.architectureJson();
        ASSERT_EQ(arch.at("version"), "2.2.0");
        auto loaded = restore(network, arch);
        ASSERT_NE(loaded, nullptr);
        EXPECT_EQ(loaded->getOutput("output").getDimensions(), layer.getOutput("output").getDimensions());
        EXPECT_EQ(loaded->getQueryIsRagged(), layer.getQueryIsRagged());
        EXPECT_EQ(loaded->getKeyValueIsRagged(), layer.getKeyValueIsRagged());
        EXPECT_EQ(loaded->getRaggedFeatureOutput().has_value(), layer.getRaggedFeatureOutput().has_value());
    }
}

TEST(SdpaNeutralFoundationC1, RejectsTamperedTensorIdentityDtypeAndDimensions) {
    CrossFixture f;
    auto bad = f.arch;
    bad["query_input"]["id"] = f.kv.getValues().getId();
    expectBad(f, bad);
    bad = f.arch;
    bad["query_input"]["dimensions"] = std::vector<uint64_t>{99, 4, 32};
    expectBad(f, bad);
    bad = f.arch;
    bad["query_ragged_input"]["values"]["data_type"] = DataType::BF16;
    expectBad(f, bad);
    bad = f.arch;
    bad["query_ragged_input"]["offsets"]["id"] = f.kv.getRowPartitionToken().getId();
    expectBad(f, bad);
    bad = f.arch;
    bad["output"]["dimensions"] = std::vector<uint64_t>{7, 4, 32};
    expectBad(f, bad);
    bad = f.arch;
    bad["output"]["data_type"] = DataType::BF16;
    expectBad(f, bad);
}

TEST(SdpaNeutralFoundationC1, RejectsMismatchedSemanticAndCanonicalKvMetadata) {
    CrossFixture f;
    auto bad = f.arch;
    bad["query_ragged_input"]["max_values_per_row"] = 4;
    expectBad(f, bad);
    bad = f.arch;
    bad["value_ragged_input"]["offsets"]["id"] = f.q.getRowPartitionToken().getId();
    expectBad(f, bad);
    bad = f.arch;
    bad["key_ragged_input"]["offsets"]["dimensions"] = std::vector<uint64_t>{8};
    expectBad(f, bad);
    bad = f.arch;
    bad["query_ragged_input"].erase("offsets");
    expectBad(f, bad);
}

TEST(SdpaNeutralFoundationC1, RejectsCanonicalRuntimeCapacityAndMalformedMetadata) {
    CrossFixture f;
    auto bad = f.arch;
    bad["batch_size"] = 7;
    expectBad(f, bad);
    bad = f.arch;
    bad["query_ragged_input"]["max_total_values"] = 21;
    expectBad(f, bad);
    bad = f.arch;
    bad["key_ragged_input"]["batch_size"] = 2;
    expectBad(f, bad);
    bad = f.arch;
    bad["query_ragged_input"]["layer_ragged_reference_version"] = "9.9.9";
    expectBad(f, bad);
    bad = f.arch;
    bad["version"] = "2.1.0";
    // Historical input grammars do not select fixed-batch model geometry.
    auto historical = restore(f.network, bad);
    ASSERT_NE(historical, nullptr);
    ASSERT_TRUE(historical->getQueryRaggedInput().has_value());
    ASSERT_TRUE(historical->getKeyRaggedInput().has_value());
    ASSERT_TRUE(historical->getValueRaggedInput().has_value());
    EXPECT_EQ(historical->getQueryRaggedInput()->getMaxValuesPerRow(), 3);
    EXPECT_EQ(historical->getKeyRaggedInput()->getMaxValuesPerRow(), 7);
    EXPECT_TRUE(historical->getKeyRaggedInput()->sharesPartitionWith(*historical->getValueRaggedInput()));
    EXPECT_FALSE(historical->getQueryRaggedInput()->sharesPartitionWith(*historical->getKeyRaggedInput()));
    bad = f.arch;
    bad["query_ragged"] = false;
    expectBad(f, bad);
}

TEST(SdpaNeutralFoundationC1, HistoricalQKvGrammarIgnoresPhysicalCapacityAndRecoversLogicalOutput) {
    CrossFixture f;
    json historical = f.arch;
    historical["version"] = "2.1.0";
    // The original query and KV graph tensors already establish independent
    // semantic maxima. Pretend an old writer recorded its packed B=4 geometry
    // in the layer references, and even supplied contradictory physical hints.
    for (const auto& [field, m] : std::vector<std::pair<const char*, uint64_t>>{
             {"query_ragged_input", 3}, {"key_ragged_input", 7}, {"value_ragged_input", 7}}) {
        auto& ref = historical[field];
        ref.erase("layer_ragged_reference_version");
        ref.erase("geometry_mode");
        ref.erase("max_values_per_row"); // Old grammar need not record semantic M.
        ref["values"]["dimensions"][0] = m * 4;
        ref["offsets"]["dimensions"][0] = 5;
        ref["batch_size"] = 99;           // These hints are not architectural.
        ref["max_total_values"] = 12345;
    }
    historical["output"]["dimensions"][0] = 12;
    historical["batch_size"] = 99;
    historical["max_total_values"] = 12345;
    auto loaded = restore(f.network, historical);
    ASSERT_NE(loaded, nullptr);
    ASSERT_TRUE(loaded->getQueryRaggedInput().has_value());
    ASSERT_TRUE(loaded->getKeyRaggedInput().has_value());
    ASSERT_TRUE(loaded->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(loaded->getRaggedFeatureOutput()->sharesPartitionWith(f.q));
    EXPECT_TRUE(loaded->getKeyRaggedInput()->sharesPartitionWith(f.kv));
    EXPECT_FALSE(loaded->getQueryRaggedInput()->sharesPartitionWith(f.kv));
    EXPECT_EQ(loaded->getOutput("output").getDimensions(), (std::vector<uint64_t>{3, 4, 32}));
    EXPECT_EQ(loaded->getQueryRaggedInput()->getDescriptorForPlacement(2).getValuesDimensions(),
              (std::vector<uint64_t>{6, 4, 32}));
    EXPECT_EQ(loaded->getQueryRaggedInput()->getDescriptorForPlacement(7).getValuesDimensions(),
              (std::vector<uint64_t>{21, 4, 32}));
    EXPECT_EQ(loaded->getKeyRaggedInput()->getDescriptorForPlacement(7).getValuesDimensions(),
              (std::vector<uint64_t>{49, 2, 32}));
    EXPECT_EQ(loaded->architectureJson().at("version"), "2.2.0");

    // The 2.0.0 all-ragged grammar has no separate Q/KV domain flags.
    auto historicalV2 = historical;
    historicalV2["version"] = "2.0.0";
    historicalV2.erase("query_ragged");
    historicalV2.erase("key_value_ragged");
    auto loadedV2 = restore(f.network, historicalV2);
    ASSERT_NE(loadedV2, nullptr);
    ASSERT_TRUE(loadedV2->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(loadedV2->getOutput("output").getDimensions(), (std::vector<uint64_t>{3, 4, 32}));
    EXPECT_TRUE(loadedV2->getRaggedFeatureOutput()->sharesPartitionWith(f.q));

    // Physical leading dimensions are discardable. Semantic trailing
    // geometry and the exact K/V partition identity are not.
    auto invalid = historical;
    invalid["key_ragged_input"]["values"]["dimensions"][1] = 3;
    expectBad(f, invalid);
    invalid = historical;
    invalid["value_ragged_input"]["offsets"]["id"] = f.q.getRowPartitionToken().getOriginalId();
    expectBad(f, invalid);
    invalid = historical;
    invalid["query_ragged_input"]["max_values_per_row"] = 999;
    expectBad(f, invalid);
}
