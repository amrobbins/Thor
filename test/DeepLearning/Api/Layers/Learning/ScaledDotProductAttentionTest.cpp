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

    EXPECT_EQ(attention.getLayerType(), "ScaledDotProductAttention");
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

TEST(AttentionApi, SdpaRejectsBottomRightMaskWithAlibi) {
    Api::Network network("attention_api_rejects_bottom_right_mask_with_alibi_sdpa");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .maskKind(Impl::AttentionMaskKind::CausalBottomRight)
                     .useAlibiMask()
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, SdpaBuildsRaggedAttentionWithFullDenseAdditiveBias) {
    Api::Network network("attention_api_sdpa_builds_ragged_attention_with_full_dense_additive_bias");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({6, 2, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput k = Api::NetworkInput::Builder().network(network).name("k").dimensions({6, 2, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput v = Api::NetworkInput::Builder().network(network).name("v").dimensions({6, 2, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput bias =
        Api::NetworkInput::Builder().network(network).name("bias").dimensions({1, 6, 6}).dataType(DataType::FP32).build();
    Api::NetworkInput sequenceLengths =
        Api::NetworkInput::Builder().network(network).name("sequence_lengths").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput raggedOffsets =
        Api::NetworkInput::Builder().network(network).name("ragged_offsets").dimensions({2}).dataType(DataType::INT32).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .queryInput(q.getFeatureOutput().value())
                                                   .keyInput(k.getFeatureOutput().value())
                                                   .valueInput(v.getFeatureOutput().value())
                                                   .biasInput(bias.getFeatureOutput().value())
                                                   .sequenceLengthsInput(sequenceLengths.getFeatureOutput().value())
                                                   .raggedOffsetsInput(raggedOffsets.getFeatureOutput().value())
                                                   .bshdLayout()
                                                   .build();

    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"query",
                                        "key",
                                        "value",
                                        "bias",
                                        "query_sequence_lengths",
                                        "key_value_sequence_lengths",
                                        "query_ragged_offsets",
                                        "key_value_ragged_offsets"}));
    EXPECT_TRUE(attention.getUseSequenceLengths());
    EXPECT_TRUE(attention.getUseRaggedOffsets());
    ASSERT_TRUE(attention.getQuerySequenceLengthsInput().has_value());
    ASSERT_TRUE(attention.getKeyValueSequenceLengthsInput().has_value());
    ASSERT_TRUE(attention.getQueryRaggedOffsetsInput().has_value());
    ASSERT_TRUE(attention.getKeyValueRaggedOffsetsInput().has_value());
    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{6, 2, 32}));
}

TEST(AttentionApi, SdpaBuildsCrossAttentionWithSeparateRaggedMetadata) {
    Api::Network network("attention_api_sdpa_builds_cross_attention_with_separate_ragged_metadata");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({4, 4, 16}).dataType(DataType::BF16).build();
    Api::NetworkInput k = Api::NetworkInput::Builder().network(network).name("k").dimensions({5, 2, 16}).dataType(DataType::BF16).build();
    Api::NetworkInput v = Api::NetworkInput::Builder().network(network).name("v").dimensions({5, 2, 16}).dataType(DataType::BF16).build();
    Api::NetworkInput qSeq = Api::NetworkInput::Builder().network(network).name("q_seq").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput kvSeq =
        Api::NetworkInput::Builder().network(network).name("kv_seq").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput qOffsets =
        Api::NetworkInput::Builder().network(network).name("q_offsets").dimensions({2}).dataType(DataType::INT32).build();
    Api::NetworkInput kvOffsets =
        Api::NetworkInput::Builder().network(network).name("kv_offsets").dimensions({2}).dataType(DataType::INT32).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .queryInput(q.getFeatureOutput().value())
                                                   .keyInput(k.getFeatureOutput().value())
                                                   .valueInput(v.getFeatureOutput().value())
                                                   .querySequenceLengthsInput(qSeq.getFeatureOutput().value())
                                                   .keyValueSequenceLengthsInput(kvSeq.getFeatureOutput().value())
                                                   .queryRaggedOffsetsInput(qOffsets.getFeatureOutput().value())
                                                   .keyValueRaggedOffsetsInput(kvOffsets.getFeatureOutput().value())
                                                   .bshdLayout()
                                                   .build();

    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{4, 4, 16}));
    EXPECT_TRUE(attention.getUseSequenceLengths());
    EXPECT_TRUE(attention.getUseRaggedOffsets());
}

TEST(AttentionApi, SdpaBuildsPhiloxDropoutAndSerializesPublicSurface) {
    Api::Network network("attention_api_sdpa_builds_philox_dropout_and_serializes_public_surface");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({4, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput seq = Api::NetworkInput::Builder().network(network).name("seq").dimensions({1}).dataType(DataType::INT32).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .selfInput(q.getFeatureOutput().value())
                                                   .sequenceLengthsInput(seq.getFeatureOutput().value())
                                                   .dropout(0.125f, 1234, 5678)
                                                   .build();

    EXPECT_FLOAT_EQ(attention.getDropoutProbability(), 0.125f);
    EXPECT_EQ(attention.getDropoutSeed(), 1234);
    EXPECT_EQ(attention.getDropoutOffset(), 5678);

    nlohmann::json arch = attention.architectureJson();
    EXPECT_EQ(arch.at("layer_type").get<std::string>(), "scaled_dot_product_attention");
    EXPECT_EQ(arch.at("tensor_layout").get<std::string>(), "bhsd");
    EXPECT_EQ(arch.at("mask_kind").get<std::string>(), "none");
    EXPECT_TRUE(arch.at("attention_scale").is_null());
    EXPECT_FLOAT_EQ(arch.at("dropout_probability").get<float>(), 0.125f);
    EXPECT_EQ(arch.at("dropout_seed").get<int64_t>(), 1234);
    EXPECT_EQ(arch.at("dropout_offset").get<int64_t>(), 5678);
    EXPECT_FALSE(arch.at("use_bias").get<bool>());
    EXPECT_TRUE(arch.at("use_sequence_lengths").get<bool>());
    EXPECT_FALSE(arch.at("use_ragged_offsets").get<bool>());
    EXPECT_EQ(arch.at("query_sequence_lengths_input").at("id").get<uint64_t>(), seq.getFeatureOutput().value().getId());
    EXPECT_EQ(arch.at("key_value_sequence_lengths_input").at("id").get<uint64_t>(), seq.getFeatureOutput().value().getId());
    std::vector<uint64_t> outputDims = arch.at("output").at("dimensions").get<std::vector<uint64_t>>();
    EXPECT_EQ(outputDims, (std::vector<uint64_t>{4, 8, 32}));
}

TEST(AttentionApi, SdpaRejectsInvalidDropoutConfiguration) {
    Api::Network network("attention_api_sdpa_rejects_invalid_dropout_configuration");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .dropoutProbability(-0.01f)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .dropoutProbability(1.0f)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .dropout(0.1f, 7, -1)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .maskKind(Impl::AttentionMaskKind::CausalBottomRight)
                     .dropout(0.1f, 7, 11)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, SdpaRejectsRaggedOffsetsWithoutSequenceLengths) {
    Api::Network network("attention_api_sdpa_rejects_ragged_offsets_without_sequence_lengths");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput offsets =
        Api::NetworkInput::Builder().network(network).name("offsets").dimensions({2}).dataType(DataType::INT32).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .raggedOffsetsInput(offsets.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, SdpaRejectsRaggedOffsetsWhenValueDimDiffersFromQkDim) {
    Api::Network network("attention_api_sdpa_rejects_ragged_offsets_value_dim_mismatch");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput k = Api::NetworkInput::Builder().network(network).name("k").dimensions({2, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput v = Api::NetworkInput::Builder().network(network).name("v").dimensions({2, 8, 16}).dataType(DataType::FP16).build();
    Api::NetworkInput seq = Api::NetworkInput::Builder().network(network).name("seq").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput offsets =
        Api::NetworkInput::Builder().network(network).name("offsets").dimensions({2}).dataType(DataType::INT32).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .queryInput(q.getFeatureOutput().value())
                     .keyInput(k.getFeatureOutput().value())
                     .valueInput(v.getFeatureOutput().value())
                     .sequenceLengthsInput(seq.getFeatureOutput().value())
                     .raggedOffsetsInput(offsets.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, SdpaRejectsInvalidVariableLengthMetadata) {
    Api::Network network("attention_api_sdpa_rejects_invalid_variable_length_metadata");
    Api::NetworkInput q = Api::NetworkInput::Builder().network(network).name("q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput badSeqDtype =
        Api::NetworkInput::Builder().network(network).name("bad_seq_dtype").dimensions({1}).dataType(DataType::FP16).build();
    Api::NetworkInput badSeqShape =
        Api::NetworkInput::Builder().network(network).name("bad_seq_shape").dimensions({2}).dataType(DataType::INT32).build();
    Api::NetworkInput seq = Api::NetworkInput::Builder().network(network).name("seq").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput badOffsetsDtype =
        Api::NetworkInput::Builder().network(network).name("bad_offsets_dtype").dimensions({2}).dataType(DataType::FP16).build();
    Api::NetworkInput badOffsetsShape =
        Api::NetworkInput::Builder().network(network).name("bad_offsets_shape").dimensions({1}).dataType(DataType::INT32).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .sequenceLengthsInput(badSeqDtype.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .sequenceLengthsInput(badSeqShape.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .sequenceLengthsInput(seq.getFeatureOutput().value())
                     .raggedOffsetsInput(badOffsetsDtype.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .sequenceLengthsInput(seq.getFeatureOutput().value())
                     .raggedOffsetsInput(badOffsetsShape.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
}
