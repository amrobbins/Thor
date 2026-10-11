#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"

#include "gtest/gtest.h"

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

Api::RaggedTensor neutral(Api::Network& network,
                          const std::string& name,
                          uint64_t maxValuesPerRow,
                          uint64_t heads,
                          uint64_t headDim,
                          DataType offsetsDtype = DataType::UINT64) {
    return Api::RaggedNetworkInput::Builder()
        .network(network)
        .name(name)
        .valuesDataType(DataType::FP16)
        .offsetsDataType(offsetsDtype)
        .trailingDimensions({heads, headDim})
        .maxValuesPerRow(maxValuesPerRow)
        .build();
}

} // namespace

TEST(SdpaNeutralFoundationB1, DifferentLogicalPartitionsPreserveSemanticLengthsAndQPartition) {
    Api::Network network("sdpa_neutral_b1_cross");
    auto q = neutral(network, "q", 3, 4, 32, DataType::UINT32);
    auto kv = neutral(network, "kv", 7, 2, 32, DataType::UINT64);
    ASSERT_FALSE(q.sharesPartitionWith(kv));
    ASSERT_EQ(q.getOffsetsDimensions(), (std::vector<uint64_t>{1}));
    ASSERT_EQ(kv.getOffsetsDimensions(), (std::vector<uint64_t>{1}));

    const auto sdpa = Api::ScaledDotProductAttention::Builder()
                          .network(network).queryInput(q).keyInput(kv).valueInput(kv).build();
    ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(q));
    EXPECT_FALSE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(kv));
    EXPECT_EQ(sdpa.getOutput("output").getDimensions(), (std::vector<uint64_t>{3, 4, 32}));
    EXPECT_EQ(sdpa.getInputNames(),
              (std::vector<std::string>{"query", "key", "value", "query_ragged_offsets", "key_value_ragged_offsets"}));
}

TEST(SdpaNeutralFoundationB1, SelfAttentionReusesOneCanonicalPartition) {
    Api::Network network("sdpa_neutral_b1_self");
    auto input = neutral(network, "tokens", 5, 2, 64);
    const auto sdpa = Api::ScaledDotProductAttention::Builder().network(network).selfInput(input).build();
    ASSERT_TRUE(sdpa.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(sdpa.getRaggedFeatureOutput()->sharesPartitionWith(input));
    EXPECT_EQ(sdpa.getOutput("output").getDimensions(), (std::vector<uint64_t>{5, 2, 64}));
}

TEST(SdpaNeutralFoundationB1, MixedQueryAndKeyValueDomainsKeepTheCorrectLogicalOutput) {
    Api::Network network("sdpa_neutral_b1_mixed");
    auto q = neutral(network, "ragged_query", 3, 4, 32);
    auto kv = neutral(network, "ragged_kv", 7, 2, 32);
    auto denseQ = Api::NetworkInput::Builder().network(network).name("dense_q")
                      .dimensions({4, 4, 32}).dataType(DataType::FP16).build();
    auto denseK = Api::NetworkInput::Builder().network(network).name("dense_k")
                      .dimensions({7, 2, 32}).dataType(DataType::FP16).build();
    auto denseV = Api::NetworkInput::Builder().network(network).name("dense_v")
                      .dimensions({7, 2, 48}).dataType(DataType::FP16).build();

    const auto raggedQuery = Api::ScaledDotProductAttention::Builder()
                                 .network(network).queryInput(q)
                                 .keyInput(denseK.getFeatureOutput().value())
                                 .valueInput(denseV.getFeatureOutput().value()).build();
    EXPECT_EQ(raggedQuery.getOutput("output").getDimensions(), (std::vector<uint64_t>{3, 4, 48}));
    ASSERT_TRUE(raggedQuery.getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(raggedQuery.getRaggedFeatureOutput()->sharesPartitionWith(q));

    const auto raggedKv = Api::ScaledDotProductAttention::Builder()
                              .network(network).queryInput(denseQ.getFeatureOutput().value())
                              .keyInput(kv).valueInput(kv).build();
    EXPECT_EQ(raggedKv.getOutput("output").getDimensions(), (std::vector<uint64_t>{4, 4, 32}));
    EXPECT_FALSE(raggedKv.getRaggedFeatureOutput().has_value());
}

TEST(SdpaNeutralFoundationB1, IndependentQueryPartitionAndExactKeyValueSharing) {
    Api::Network network("sdpa_neutral_b1_reject");
    auto neutralQ = neutral(network, "neutral_q", 3, 4, 32);
    auto neutralK = neutral(network, "neutral_k", 7, 2, 32);
    auto separateV = neutral(network, "separate_v", 7, 2, 32);
    auto otherQ = Api::RaggedNetworkInput::Builder()
                      .network(network).name("other_q").valuesDataType(DataType::FP16)
                      .trailingDimensions({4, 32}).maxValuesPerRow(3).build();
    auto otherKv = Api::RaggedNetworkInput::Builder()
                       .network(network).name("other_kv").valuesDataType(DataType::FP16)
                       .trailingDimensions({2, 32}).maxValuesPerRow(7).build();
    EXPECT_NO_THROW((void)Api::ScaledDotProductAttention::Builder().network(network)
                     .queryInput(neutralQ).keyInput(otherKv).valueInput(otherKv).build());
    EXPECT_NO_THROW((void)Api::ScaledDotProductAttention::Builder().network(network)
                     .queryInput(otherQ).keyInput(neutralK).valueInput(neutralK).build());
    EXPECT_THROW((void)Api::ScaledDotProductAttention::Builder().network(network)
                     .queryInput(neutralQ).keyInput(neutralK).valueInput(separateV).build(), std::invalid_argument);
}
