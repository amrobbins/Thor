#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, LayerNormDefaultsToLastFeatureDimension) {
    Network network("layer_norm_default_shape");
    Tensor input(DataType::FP16, {4, 8, 16});

    LayerNorm layer = LayerNorm::Builder().network(network).featureInput(input).build();

    ASSERT_TRUE(layer.isInitialized());
    ASSERT_EQ(layer.getNormalizedShape(), vector<uint64_t>({16}));
    ASSERT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-5);
    ASSERT_EQ(layer.getParameterDataType(), DataType::FP32);

    optional<Tensor> output = layer.getFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output.value().getDimensions(), input.getDimensions());
    EXPECT_EQ(output.value().getDataType(), input.getDataType());
}

TEST(UtilityApiLayers, LayerNormAcceptsExplicitTrailingNormalizedShape) {
    Network network("layer_norm_explicit_shape");
    Tensor input(DataType::BF16, {2, 3, 4});

    LayerNorm layer = LayerNorm::Builder().network(network).featureInput(input).normalizedShape({3, 4}).epsilon(1.0e-4).build();

    EXPECT_EQ(layer.getNormalizedShape(), vector<uint64_t>({3, 4}));
    EXPECT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-4);
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, LayerNormRejectsBadNormalizedShape) {
    Network network("layer_norm_bad_shape");
    Tensor input(DataType::FP16, {2, 3, 4});

    EXPECT_THROW(LayerNorm::Builder().network(network).featureInput(input).normalizedShape({4, 3}).build(), std::invalid_argument);
    EXPECT_THROW(LayerNorm::Builder().network(network).featureInput(input).normalizedShape({0}).build(), std::invalid_argument);
}

TEST(UtilityApiLayers, LayerNormRejectsUnsupportedDtypes) {
    Network network("layer_norm_bad_dtype");
    Tensor intInput(DataType::INT32, {2, 4});
    EXPECT_THROW(LayerNorm::Builder().network(network).featureInput(intInput).build(), std::invalid_argument);

    Tensor fpInput(DataType::FP16, {2, 4});
    EXPECT_THROW(LayerNorm::Builder().network(network).featureInput(fpInput).parameterDataType(DataType::FP16).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, LayerNormArchitectureJsonContainsParameters) {
    Network network("layer_norm_architecture");
    Tensor input(DataType::FP32, {8, 32});

    LayerNorm layer = LayerNorm::Builder().network(network).featureInput(input).normalizedShape({32}).build();
    json arch = layer.architectureJson();

    EXPECT_EQ(arch.at("layer_type").get<string>(), "layer_norm");
    EXPECT_EQ(arch.at("normalized_shape").get<vector<uint64_t>>(), vector<uint64_t>({32}));
    EXPECT_TRUE(arch.at("parameters").contains("weights"));
    EXPECT_TRUE(arch.at("parameters").contains("biases"));
}
