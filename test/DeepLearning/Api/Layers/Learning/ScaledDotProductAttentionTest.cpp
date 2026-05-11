#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"

#include "gtest/gtest.h"

namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::TensorDescriptor::DataType;

TEST(AttentionApi, BuildsDenseMultiHeadSelfAttentionInterface) {
    Api::Network network("attention_api_builds_dense_multi_head_self_attention_interface");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("qkv").dimensions({4, 16, 32}).dataType(DataType::FP16).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .selfInput(input.getFeatureOutput().value())
                                                   .causal()
                                                   .attentionScale(1.0 / 8.0)
                                                   .build();

    EXPECT_EQ(attention.getLayerType(), "Attention");
    EXPECT_EQ(attention.getInputNames(), (std::vector<std::string>{"query", "key", "value"}));
    EXPECT_EQ(attention.getOutputNames(), (std::vector<std::string>{"output"}));
    EXPECT_EQ(attention.getOutput("output").getDataType(), DataType::FP16);
    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{4, 16, 32}));
    EXPECT_EQ(attention.getMaskKind(), Impl::AttentionMaskKind::CausalTopLeft);
    ASSERT_TRUE(attention.getAttentionScale().has_value());
    EXPECT_DOUBLE_EQ(attention.getAttentionScale().value(), 1.0 / 8.0);
}

TEST(AttentionApi, BuildsGqaAttentionWithAdditiveBias) {
    Api::Network network("attention_api_builds_gqa_attention_with_additive_bias");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({4, 8, 64}).dataType(DataType::BF16).build();
    Api::NetworkInput k = Api::NetworkInput::Builder().network(network).name("k").dimensions({2, 8, 64}).dataType(DataType::BF16).build();
    Api::NetworkInput v = Api::NetworkInput::Builder().network(network).name("v").dimensions({2, 8, 32}).dataType(DataType::BF16).build();
    Api::NetworkInput bias =
        Api::NetworkInput::Builder().network(network).name("bias").dimensions({4, 8, 8}).dataType(DataType::FP32).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .queryInput(q.getFeatureOutput().value())
                                                   .keyInput(k.getFeatureOutput().value())
                                                   .valueInput(v.getFeatureOutput().value())
                                                   .biasInput(bias.getFeatureOutput().value())
                                                   .outputDataType(DataType::BF16)
                                                   .build();

    EXPECT_EQ(attention.getInputNames(), (std::vector<std::string>{"query", "key", "value", "bias"}));
    EXPECT_EQ(attention.getOutput("output").getDataType(), DataType::BF16);
    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{4, 8, 32}));
}

TEST(AttentionApi, RejectsInvalidQueryKeyHeadDimMismatch) {
    Api::Network network("attention_api_rejects_invalid_query_key_head_dim_mismatch");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({4, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput k = Api::NetworkInput::Builder().network(network).name("k").dimensions({4, 8, 64}).dataType(DataType::FP16).build();
    Api::NetworkInput v = Api::NetworkInput::Builder().network(network).name("v").dimensions({4, 8, 32}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .queryInput(q.getFeatureOutput().value())
                     .keyInput(k.getFeatureOutput().value())
                     .valueInput(v.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsBiasDTypeThatWouldRequireHiddenConversion) {
    Api::Network network("attention_api_rejects_bias_dtype_that_would_require_hidden_conversion");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput bias =
        Api::NetworkInput::Builder().network(network).name("bias").dimensions({2, 8, 8}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .biasInput(bias.getFeatureOutput().value())
                     .computeDataType(DataType::FP32)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsBottomRightMaskWithAdditiveBias) {
    Api::Network network("attention_api_rejects_bottom_right_mask_with_additive_bias");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput bias =
        Api::NetworkInput::Builder().network(network).name("bias").dimensions({2, 8, 8}).dataType(DataType::FP32).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .biasInput(bias.getFeatureOutput().value())
                     .maskKind(Impl::AttentionMaskKind::CausalBottomRight)
                     .build(),
                 std::invalid_argument);
}
