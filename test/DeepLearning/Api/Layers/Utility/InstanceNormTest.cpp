#include "DeepLearning/Api/Layers/Utility/InstanceNorm.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, InstanceNormDerivesChannelCountFromFirstFeatureDimension) {
    Network network("instance_norm_default_config");
    Tensor input(Tensor::DataType::FP16, {8, 16, 16});

    InstanceNorm layer = InstanceNorm::Builder().network(network).featureInput(input).build();

    ASSERT_TRUE(layer.isInitialized());
    ASSERT_EQ(layer.getChannelCount(), 8u);
    ASSERT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-5);
    ASSERT_EQ(layer.getParameterDataType(), Tensor::DataType::FP32);

    optional<Tensor> output = layer.getFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output.value().getDimensions(), input.getDimensions());
    EXPECT_EQ(output.value().getDataType(), input.getDataType());
}

TEST(UtilityApiLayers, InstanceNormAcceptsOneDimensionalSpatialInput) {
    Network network("instance_norm_1d_spatial");
    Tensor input(Tensor::DataType::BF16, {4, 32});

    InstanceNorm layer = InstanceNorm::Builder().network(network).featureInput(input).epsilon(1.0e-4).build();

    EXPECT_EQ(layer.getChannelCount(), 4u);
    EXPECT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-4);
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, InstanceNormRejectsBadInputShape) {
    Network network("instance_norm_bad_shape");
    Tensor rankOne(Tensor::DataType::FP16, {8});
    EXPECT_THROW(InstanceNorm::Builder().network(network).featureInput(rankOne).build(), std::invalid_argument);

}

TEST(UtilityApiLayers, InstanceNormRejectsUnsupportedDtypes) {
    Network network("instance_norm_bad_dtype");
    Tensor intInput(Tensor::DataType::INT32, {8, 16, 16});
    EXPECT_THROW(InstanceNorm::Builder().network(network).featureInput(intInput).build(), std::invalid_argument);

    Tensor fpInput(Tensor::DataType::FP16, {8, 16, 16});
    EXPECT_THROW(InstanceNorm::Builder().network(network).featureInput(fpInput).parameterDataType(Tensor::DataType::FP16).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, InstanceNormArchitectureJsonContainsParameters) {
    Network network("instance_norm_architecture");
    Tensor input(Tensor::DataType::FP32, {8, 32, 32});

    InstanceNorm layer = InstanceNorm::Builder().network(network).featureInput(input).build();
    json arch = layer.architectureJson();

    EXPECT_EQ(arch.at("layer_type").get<string>(), "instance_norm");
    EXPECT_EQ(arch.at("channel_count").get<uint64_t>(), 8u);
    EXPECT_TRUE(arch.at("parameters").contains("weights"));
    EXPECT_TRUE(arch.at("parameters").contains("biases"));
}
