#include "DeepLearning/Api/Layers/Utility/AdaptiveLayerNorm.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, AdaptiveLayerNormConstructsDefaultLastDimAndOutputPreservesShapeDtype) {
    Network network("adaptive_layer_norm_default_shape");
    Tensor input(Tensor::DataType::FP16, {8, 16});
    Tensor scale(Tensor::DataType::FP32, {8, 16});
    Tensor bias(Tensor::DataType::FP32, {8, 16});

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).build();

    ASSERT_TRUE(layer.isInitialized());
    ASSERT_EQ(layer.getNormalizedShape(), vector<uint64_t>({16}));
    ASSERT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-5);
    ASSERT_EQ(layer.getScaleBiasDataType(), Tensor::DataType::FP32);

    optional<Tensor> output = layer.getFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output.value().getDimensions(), input.getDimensions());
    EXPECT_EQ(output.value().getDataType(), input.getDataType());
}

TEST(UtilityApiLayers, AdaptiveLayerNormAcceptsExplicitTrailingNormalizedShape) {
    Network network("adaptive_layer_norm_explicit_shape");
    Tensor input(Tensor::DataType::BF16, {2, 3, 4});
    Tensor scale(Tensor::DataType::FP32, {2, 3, 4});
    Tensor bias(Tensor::DataType::FP32, {2, 3, 4});

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .scaleInput(scale)
                                  .biasInput(bias)
                                  .normalizedShape({3, 4})
                                  .epsilon(1.0e-4)
                                  .build();

    EXPECT_EQ(layer.getNormalizedShape(), vector<uint64_t>({3, 4}));
    EXPECT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-4);
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, AdaptiveLayerNormRejectsBadNormalizedShape) {
    Network network("adaptive_layer_norm_bad_shape");
    Tensor input(Tensor::DataType::FP16, {2, 3, 4});
    Tensor scale(Tensor::DataType::FP32, {2, 3, 4});
    Tensor bias(Tensor::DataType::FP32, {2, 3, 4});

    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).normalizedShape({4, 3}).build(),
                 std::invalid_argument);
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).normalizedShape({0}).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, AdaptiveLayerNormRejectsUnsupportedDtypesOrShapes) {
    Network network("adaptive_layer_norm_bad_dtype");
    Tensor intInput(Tensor::DataType::INT32, {2, 4});
    Tensor scale(Tensor::DataType::FP32, {2, 4});
    Tensor bias(Tensor::DataType::FP32, {2, 4});
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(intInput).scaleInput(scale).biasInput(bias).build(), std::invalid_argument);

    Tensor fpInput(Tensor::DataType::FP16, {2, 4});
    Tensor halfScale(Tensor::DataType::FP16, {2, 4});
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(fpInput).scaleInput(halfScale).biasInput(bias).build(),
                 std::invalid_argument);

    Tensor wrongShapeBias(Tensor::DataType::FP32, {4});
    EXPECT_THROW(AdaptiveLayerNorm::Builder().network(network).featureInput(fpInput).scaleInput(scale).biasInput(wrongShapeBias).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, AdaptiveLayerNormArchitectureJsonContainsSideInputs) {
    Network network("adaptive_layer_norm_architecture");
    Tensor input(Tensor::DataType::FP32, {8, 32});
    Tensor scale(Tensor::DataType::FP32, {8, 32});
    Tensor bias(Tensor::DataType::FP32, {8, 32});

    AdaptiveLayerNorm layer = AdaptiveLayerNorm::Builder().network(network).featureInput(input).scaleInput(scale).biasInput(bias).normalizedShape({32}).build();
    json arch = layer.architectureJson();

    EXPECT_EQ(arch.at("layer_type").get<string>(), "adaptive_layer_norm");
    EXPECT_EQ(arch.at("normalized_shape").get<vector<uint64_t>>(), vector<uint64_t>({32}));
    ASSERT_EQ(arch.at("inputs").size(), 3);
    EXPECT_EQ(arch.at("inputs")[0].at("port").get<string>(), "feature_input");
    EXPECT_EQ(arch.at("inputs")[1].at("port").get<string>(), "scale_input");
    EXPECT_EQ(arch.at("inputs")[2].at("port").get<string>(), "bias_input");
}
