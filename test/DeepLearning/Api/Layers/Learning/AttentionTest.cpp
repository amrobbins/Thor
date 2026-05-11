#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"

#include "gtest/gtest.h"

namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::TensorDescriptor::DataType;

TEST(AttentionApi, BuildsComposedCausalSelfAttention) {
    Api::Network network("attention_api_builds_composed_causal_self_attention");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({16, 64})
                                  .dataType(DataType::FP16)
                                  .build();

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(4)
                                   .causal()
                                   .build();

    EXPECT_EQ(attention.getLayerType(), "Attention");
    EXPECT_EQ(attention.getInputNames(), (std::vector<std::string>{"feature_input"}));
    EXPECT_EQ(attention.getOutputNames(), (std::vector<std::string>{"feature_output"}));
    EXPECT_EQ(attention.getOutput("feature_output").getDataType(), DataType::FP16);
    EXPECT_EQ(attention.getOutput("feature_output").getDimensions(), (std::vector<uint64_t>{16, 64}));
    EXPECT_EQ(attention.getNumHeads(), 4U);
    EXPECT_EQ(attention.getNumKeyValueHeads(), 4U);
    EXPECT_EQ(attention.getHeadDim(), 16U);
    EXPECT_EQ(attention.getValueDim(), 16U);
    EXPECT_EQ(attention.getOutputFeatures(), 64U);
    EXPECT_EQ(attention.getMaskKind(), Impl::AttentionMaskKind::CausalTopLeft);
}

TEST(AttentionApi, BuildsComposedGqaAttentionWithExplicitDimsBiasAndRope) {
    Api::Network network("attention_api_builds_composed_gqa_attention_with_explicit_dims_bias_and_rope");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({8, 96})
                                  .dataType(DataType::BF16)
                                  .build();

    Impl::RotaryPositionEmbeddingOptions rope;
    rope.rotary_dim = 16;
    rope.sequence_axis = 1;
    rope.head_dim_axis = 3;
    rope.output_dtype = DataType::BF16;
    rope.compute_dtype = DataType::FP32;

    Api::Attention attention = Api::Attention::Builder()
                                   .network(network)
                                   .featureInput(input.getFeatureOutput().value())
                                   .numHeads(6)
                                   .numKeyValueHeads(2)
                                   .headDim(16)
                                   .valueDim(12)
                                   .outputFeatures(80)
                                   .hasBias(true)
                                   .ropeOptions(rope)
                                   .attentionScale(0.25)
                                   .outputDataType(DataType::BF16)
                                   .build();

    EXPECT_EQ(attention.getOutput("feature_output").getDimensions(), (std::vector<uint64_t>{8, 80}));
    EXPECT_EQ(attention.getNumHeads(), 6U);
    EXPECT_EQ(attention.getNumKeyValueHeads(), 2U);
    EXPECT_EQ(attention.getHeadDim(), 16U);
    EXPECT_EQ(attention.getValueDim(), 12U);
    EXPECT_EQ(attention.getOutputFeatures(), 80U);
    EXPECT_TRUE(attention.getHasBias());
    EXPECT_TRUE(attention.getUseRope());
    ASSERT_TRUE(attention.getAttentionScale().has_value());
    EXPECT_DOUBLE_EQ(attention.getAttentionScale().value(), 0.25);
}

TEST(AttentionApi, RejectsInvalidHeadConfiguration) {
    Api::Network network("attention_api_rejects_invalid_head_configuration");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({8, 64})
                                  .dataType(DataType::FP16)
                                  .build();

    EXPECT_THROW(Api::Attention::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numHeads(6)
                     .numKeyValueHeads(4)
                     .headDim(16)
                     .build(),
                 std::invalid_argument);
}

TEST(AttentionApi, RejectsRank3FeatureInputForComposedAttention) {
    Api::Network network("attention_api_rejects_rank3_feature_input_for_composed_attention");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .dimensions({2, 8, 64})
                                  .dataType(DataType::FP16)
                                  .build();

    EXPECT_THROW(Api::Attention::Builder().network(network).featureInput(input.getFeatureOutput().value()).numHeads(4).build(),
                 std::invalid_argument);
}
