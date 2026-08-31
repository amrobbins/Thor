#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"

#include "gtest/gtest.h"

namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

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

TEST(AttentionApi, SdpaRejectsOutputStorageDtypeMismatchInsteadOfDeferringToCompiler) {
    Api::Network network("attention_api_sdpa_rejects_output_storage_dtype_mismatch");
    Api::NetworkInput q =
        Api::NetworkInput::Builder().network(network).name("q").dimensions({4, 8, 32}).dataType(DataType::BF16).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q.getFeatureOutput().value())
                     .outputDataType(DataType::FP16)
                     .build(),
                 std::invalid_argument);
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
    Api::RaggedTensor q = Api::RaggedNetworkInput::Builder()
                              .network(network).name("qkv").valuesDataType(DataType::FP16)
                              .trailingDimensions({2, 32}).maxTotalValues(6).batchSize(1).build();
    Api::NetworkInput bias =
        Api::NetworkInput::Builder().network(network).name("bias").dimensions({1, 6, 6}).dataType(DataType::FP32).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .selfInput(q)
                                                   .biasInput(bias.getFeatureOutput().value())
                                                   .build();

    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"query", "key", "value", "bias", "query_ragged_offsets", "key_value_ragged_offsets"}));
    EXPECT_FALSE(attention.getUseSequenceLengths());
    EXPECT_TRUE(attention.getUseRaggedInput());
    ASSERT_TRUE(attention.getQueryRaggedInput().has_value());
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getOffsets(), q.getOffsets());
    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{6, 2, 32}));
}

TEST(AttentionApi, SdpaBuildsCrossAttentionWithSeparateRaggedMetadata) {
    Api::Network network("attention_api_sdpa_builds_cross_attention_with_separate_ragged_metadata");
    Api::RaggedTensor q = Api::RaggedNetworkInput::Builder()
                              .network(network).name("q").valuesDataType(DataType::BF16)
                              .trailingDimensions({4, 16}).maxTotalValues(4).batchSize(1).offsetsDataType(DataType::UINT32).build();
    Api::RaggedTensor kv = Api::RaggedNetworkInput::Builder()
                               .network(network).name("kv").valuesDataType(DataType::BF16)
                               .trailingDimensions({2, 16}).maxTotalValues(5).batchSize(1).offsetsDataType(DataType::UINT64).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .queryInput(q)
                                                   .keyInput(kv)
                                                   .valueInput(kv)
                                                   .build();

    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{4, 4, 16}));
    EXPECT_FALSE(attention.getUseSequenceLengths());
    EXPECT_TRUE(attention.getUseRaggedInput());
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
    EXPECT_FALSE(arch.at("use_ragged_input").get<bool>());
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

TEST(AttentionApi, SdpaBuildsCanonicalRaggedOffsetsWithoutSequenceLengths) {
    Api::Network network("attention_api_sdpa_builds_canonical_ragged_offsets_without_sequence_lengths");
    Api::RaggedTensor q = Api::RaggedNetworkInput::Builder()
                              .network(network).name("q").valuesDataType(DataType::FP16)
                              .trailingDimensions({8, 32}).maxTotalValues(2).batchSize(1).build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .selfInput(q)
                                                   .build();
    EXPECT_TRUE(attention.getUseRaggedInput());
    EXPECT_FALSE(attention.getUseSequenceLengths());
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getOffsets(), q.getOffsets());
}

TEST(AttentionApi, SdpaAllowsRaggedOffsetsWhenValueDimDiffersFromQkDim) {
    Api::Network network("attention_api_sdpa_allows_ragged_offsets_value_dim_mismatch");
    Api::RaggedTensor qk = Api::RaggedNetworkInput::Builder()
                               .network(network).name("qk").valuesDataType(DataType::FP16)
                               .trailingDimensions({8, 32}).maxTotalValues(2).batchSize(1).offsetsDataType(DataType::UINT64).build();
    Api::NetworkInput vValues = Api::NetworkInput::Builder()
                                    .network(network).name("v.values")
                                    .dimensions({2, 8, 16}).dimensionsIncludeBatch(true).dataType(DataType::FP16).build();
    Api::RaggedTensor v(vValues.getFeatureOutput().value(), qk.getOffsets());

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .queryInput(qk)
                                                   .keyInput(qk)
                                                   .valueInput(v)
                                                   .build();
    EXPECT_TRUE(attention.getUseRaggedInput());
    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{2, 8, 16}));
}

TEST(AttentionApi, SdpaRejectsInvalidVariableLengthMetadata) {
    Api::Network network("attention_api_sdpa_rejects_invalid_variable_length_metadata");
    Api::RaggedTensor q = Api::RaggedNetworkInput::Builder()
                              .network(network).name("q").valuesDataType(DataType::FP16)
                              .trailingDimensions({8, 32}).maxTotalValues(2).batchSize(1).build();
    Api::NetworkInput denseQ =
        Api::NetworkInput::Builder().network(network).name("dense_q").dimensions({2, 8, 32}).dataType(DataType::FP16).build();
    Api::NetworkInput seq = Api::NetworkInput::Builder().network(network).name("seq").dimensions({1}).dataType(DataType::INT32).build();
    Api::NetworkInput badSeqDtype =
        Api::NetworkInput::Builder().network(network).name("bad_seq_dtype").dimensions({1}).dataType(DataType::FP16).build();
    Api::NetworkInput badSeqShape =
        Api::NetworkInput::Builder().network(network).name("bad_seq_shape").dimensions({2}).dataType(DataType::INT32).build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(denseQ.getFeatureOutput().value())
                     .sequenceLengthsInput(badSeqDtype.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(denseQ.getFeatureOutput().value())
                     .sequenceLengthsInput(badSeqShape.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q)
                     .sequenceLengthsInput(seq.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .selfInput(q)
                     .bhsdLayout()
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, SdpaDenseQueryRaggedKvMixedModeRoundTripsArchitecture) {
    Api::Network network("attention_api_sdpa_dense_query_ragged_kv_round_trip");
    Api::NetworkInput query = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .dimensions({5, 4, 16})
                                  .dataType(DataType::FP16)
                                  .build();
    Api::RaggedTensor context = Api::RaggedNetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .valuesDataType(DataType::FP16)
                                    .offsetsDataType(DataType::UINT64)
                                    .trailingDimensions({2, 16})
                                    .maxTotalValues(11)
                                    .batchSize(3)
                                    .build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .queryInput(query.getFeatureOutput().value())
                                                   .keyInput(context)
                                                   .valueInput(context)
                                                   .build();

    EXPECT_TRUE(attention.getUseRaggedInput());
    EXPECT_FALSE(attention.getQueryIsRagged());
    EXPECT_TRUE(attention.getKeyValueIsRagged());
    EXPECT_EQ(attention.getTensorLayout(), Impl::AttentionTensorLayout::BSHD);
    EXPECT_FALSE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{5, 4, 16}));
    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"query", "key", "value", "key_value_ragged_offsets"}));

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_EQ(arch.at("version").get<std::string>(), "2.1.0");
    EXPECT_TRUE(arch.at("use_ragged_input").get<bool>());
    EXPECT_FALSE(arch.at("query_ragged").get<bool>());
    EXPECT_TRUE(arch.at("key_value_ragged").get<bool>());
    EXPECT_FALSE(arch.contains("query_ragged_input"));
    EXPECT_TRUE(arch.contains("key_ragged_input"));
    EXPECT_TRUE(arch.contains("value_ragged_input"));

    auto cloned = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(attention.clone());
    ASSERT_NE(cloned, nullptr);
    EXPECT_FALSE(cloned->getQueryIsRagged());
    EXPECT_TRUE(cloned->getKeyValueIsRagged());
    EXPECT_FALSE(cloned->getRaggedFeatureOutput().has_value());

    const uint32_t beforeRestoreCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::ScaledDotProductAttention::deserialize(archiveReader, arch, &network);
    auto restored = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getTrainableLayer(beforeRestoreCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_FALSE(restored->getQueryIsRagged());
    EXPECT_TRUE(restored->getKeyValueIsRagged());
    EXPECT_FALSE(restored->getRaggedFeatureOutput().has_value());
    ASSERT_TRUE(restored->getKeyRaggedInput().has_value());
    EXPECT_EQ(restored->getKeyRaggedInput()->getOffsetsDataType(), DataType::UINT64);
}

TEST(AttentionApi, SdpaRaggedQueryDenseKvMixedModeRoundTripsArchitecture) {
    Api::Network network("attention_api_sdpa_ragged_query_dense_kv_round_trip");
    Api::RaggedTensor query = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .valuesDataType(DataType::BF16)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({4, 16})
                                  .maxTotalValues(9)
                                  .batchSize(3)
                                  .build();
    Api::NetworkInput context = Api::NetworkInput::Builder()
                                    .network(network)
                                    .name("context")
                                    .dimensions({7, 2, 16})
                                    .dataType(DataType::BF16)
                                    .build();

    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder()
                                                   .network(network)
                                                   .queryInput(query)
                                                   .keyInput(context.getFeatureOutput().value())
                                                   .valueInput(context.getFeatureOutput().value())
                                                   .build();

    EXPECT_TRUE(attention.getUseRaggedInput());
    EXPECT_TRUE(attention.getQueryIsRagged());
    EXPECT_FALSE(attention.getKeyValueIsRagged());
    EXPECT_EQ(attention.getTensorLayout(), Impl::AttentionTensorLayout::BSHD);
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(attention.getRaggedFeatureOutput()->getOffsets(), query.getOffsets());
    EXPECT_EQ(attention.getOutput("output").getDimensions(), (std::vector<uint64_t>{9, 4, 16}));
    EXPECT_EQ(attention.getInputNames(),
              (std::vector<std::string>{"query", "key", "value", "query_ragged_offsets"}));

    const nlohmann::json arch = attention.architectureJson();
    EXPECT_TRUE(arch.at("use_ragged_input").get<bool>());
    EXPECT_TRUE(arch.at("query_ragged").get<bool>());
    EXPECT_FALSE(arch.at("key_value_ragged").get<bool>());
    EXPECT_TRUE(arch.contains("query_ragged_input"));
    EXPECT_FALSE(arch.contains("key_ragged_input"));
    EXPECT_FALSE(arch.contains("value_ragged_input"));

    const uint32_t beforeRestoreCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::ScaledDotProductAttention::deserialize(archiveReader, arch, &network);
    auto restored = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getTrainableLayer(beforeRestoreCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->getQueryIsRagged());
    EXPECT_FALSE(restored->getKeyValueIsRagged());
    ASSERT_TRUE(restored->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(restored->getRaggedFeatureOutput()->getOffsetsDataType(), DataType::UINT32);
}

TEST(AttentionApi, SdpaMixedModeRejectsMismatchedKeyValueDomains) {
    Api::Network network("attention_api_sdpa_mixed_mode_rejects_mismatched_key_value_domains");
    Api::NetworkInput query = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("query")
                                  .dimensions({5, 4, 16})
                                  .dataType(DataType::FP16)
                                  .build();
    Api::NetworkInput denseContext = Api::NetworkInput::Builder()
                                         .network(network)
                                         .name("dense_context")
                                         .dimensions({7, 2, 16})
                                         .dataType(DataType::FP16)
                                         .build();
    Api::RaggedTensor raggedContext = Api::RaggedNetworkInput::Builder()
                                          .network(network)
                                          .name("ragged_context")
                                          .valuesDataType(DataType::FP16)
                                          .trailingDimensions({2, 16})
                                          .maxTotalValues(8)
                                          .batchSize(3)
                                          .build();

    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .queryInput(query.getFeatureOutput().value())
                     .keyInput(raggedContext)
                     .valueInput(denseContext.getFeatureOutput().value())
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW(Api::ScaledDotProductAttention::Builder()
                     .network(network)
                     .queryInput(query.getFeatureOutput().value())
                     .keyInput(denseContext.getFeatureOutput().value())
                     .valueInput(raggedContext)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, SdpaLegacyV2AllRaggedArchitectureStillDeserializes) {
    Api::Network network("attention_api_sdpa_legacy_v2_all_ragged_deserializes");
    Api::RaggedTensor q = Api::RaggedNetworkInput::Builder()
                              .network(network)
                              .name("q")
                              .valuesDataType(DataType::FP16)
                              .trailingDimensions({2, 16})
                              .maxTotalValues(6)
                              .batchSize(2)
                              .build();
    Api::ScaledDotProductAttention attention = Api::ScaledDotProductAttention::Builder().network(network).selfInput(q).build();

    nlohmann::json legacy = attention.architectureJson();
    legacy["version"] = "2.0.0";
    legacy.erase("query_ragged");
    legacy.erase("key_value_ragged");

    const uint32_t beforeRestoreCount = network.getNumTrainableLayers();
    std::shared_ptr<thor_file::TarReader> archiveReader;
    Api::ScaledDotProductAttention::deserialize(archiveReader, legacy, &network);
    auto restored = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(network.getTrainableLayer(beforeRestoreCount));
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->getQueryIsRagged());
    EXPECT_TRUE(restored->getKeyValueIsRagged());
    ASSERT_TRUE(restored->getRaggedFeatureOutput().has_value());
}
