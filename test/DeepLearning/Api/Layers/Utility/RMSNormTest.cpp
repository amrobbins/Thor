#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, RMSNormDefaultsToLastFeatureDimension) {
    Network network("rms_norm_default_shape");
    Tensor input(Tensor::DataType::FP16, {4, 8, 16});

    RMSNorm layer = RMSNorm::Builder().network(network).featureInput(input).build();

    ASSERT_TRUE(layer.isInitialized());
    ASSERT_EQ(layer.getNormalizedShape(), vector<uint64_t>({16}));
    ASSERT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-5);
    ASSERT_EQ(layer.getParameterDataType(), Tensor::DataType::FP32);

    optional<Tensor> output = layer.getFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output.value().getDimensions(), input.getDimensions());
    EXPECT_EQ(output.value().getDataType(), input.getDataType());
}

TEST(UtilityApiLayers, RMSNormAcceptsExplicitTrailingNormalizedShape) {
    Network network("rms_norm_explicit_shape");
    Tensor input(Tensor::DataType::BF16, {2, 3, 4});

    RMSNorm layer = RMSNorm::Builder().network(network).featureInput(input).normalizedShape({3, 4}).epsilon(1.0e-4).build();

    EXPECT_EQ(layer.getNormalizedShape(), vector<uint64_t>({3, 4}));
    EXPECT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-4);
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, RMSNormRejectsBadNormalizedShape) {
    Network network("rms_norm_bad_shape");
    Tensor input(Tensor::DataType::FP16, {2, 3, 4});

    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(input).normalizedShape({4, 3}).build(), std::invalid_argument);
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(input).normalizedShape({0}).build(), std::invalid_argument);
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(input).normalizedShape({}).build(), std::invalid_argument);
}

TEST(UtilityApiLayers, RMSNormRejectsUnsupportedDtypes) {
    Network network("rms_norm_bad_dtype");
    Tensor intInput(Tensor::DataType::INT32, {2, 4});
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(intInput).build(), std::invalid_argument);

    Tensor fpInput(Tensor::DataType::FP16, {2, 4});
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(fpInput).parameterDataType(Tensor::DataType::FP16).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, RMSNormArchitectureJsonContainsWeightsOnlyAndNullableEpilogue) {
    Network network("rms_norm_architecture");
    Tensor input(Tensor::DataType::FP32, {8, 32});

    RMSNorm layer = RMSNorm::Builder().network(network).featureInput(input).normalizedShape({32}).build();
    json arch = layer.architectureJson();

    EXPECT_EQ(arch.at("layer_type").get<string>(), "rms_norm");
    EXPECT_EQ(arch.at("normalized_shape").get<vector<uint64_t>>(), vector<uint64_t>({32}));
    EXPECT_TRUE(arch.at("parameters").contains("weights"));
    EXPECT_FALSE(arch.at("parameters").contains("biases"));
    ASSERT_TRUE(arch.contains("epilogue"));
    EXPECT_TRUE(arch.at("epilogue").is_null());
    EXPECT_FALSE(arch.contains("fused_activation"));
    EXPECT_FALSE(arch.contains("rht_amax"));
    EXPECT_FALSE(arch.contains("amax_output"));
}

TEST(UtilityApiLayers, RMSNormAcceptsSwishEpilogueAndSerializesExpression) {
    Network network("rms_norm_swish_epilogue");
    Tensor input(Tensor::DataType::BF16, {8, 32});
    Swish swish;

    RMSNorm layer = RMSNorm::Builder()
                        .network(network)
                        .featureInput(input)
                        .normalizedShape({32})
                        .epilogue(swish.toExpression(RMSNorm::epilogueInput()))
                        .build();

    EXPECT_EQ(layer.getParameterDataType(), Tensor::DataType::FP32);
    json arch = layer.architectureJson();
    ASSERT_TRUE(arch.contains("epilogue"));
    EXPECT_FALSE(arch.at("epilogue").is_null());
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, RMSNormAcceptsBf16WeightsOnlyForSwishEpilogueFusionCandidate) {
    Network network("rms_norm_swish_epilogue_bf16_weights");
    Tensor input(Tensor::DataType::BF16, {8, 32});
    Swish swish;

    RMSNorm layer = RMSNorm::Builder()
                        .network(network)
                        .featureInput(input)
                        .normalizedShape({32})
                        .parameterDataType(Tensor::DataType::BF16)
                        .epilogue(swish)
                        .build();

    EXPECT_EQ(layer.getParameterDataType(), Tensor::DataType::BF16);

    Network badNetwork("rms_norm_bf16_weights_without_swish_epilogue");
    EXPECT_THROW(RMSNorm::Builder().network(badNetwork).featureInput(input).parameterDataType(Tensor::DataType::BF16).build(),
                 std::invalid_argument);

    Network badInputNetwork("rms_norm_swish_bf16_weights_bad_input");
    Tensor fp16Input(Tensor::DataType::FP16, {8, 32});
    EXPECT_THROW(RMSNorm::Builder()
                     .network(badInputNetwork)
                     .featureInput(fp16Input)
                     .parameterDataType(Tensor::DataType::BF16)
                     .epilogue(swish)
                     .build(),
                 std::invalid_argument);
}
