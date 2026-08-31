#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Api/Layers/Utility/StopGradient.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using json = nlohmann::json;

TEST(UtilityApiLayers, RaggedStopGradientPreservesCanonicalPartition) {
    Network network("ragged_stop_gradient_build");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .trailingDimensions({5})
                             .batchSize(3)
                             .maxTotalValues(17)
                             .maxValuesPerRow(9)
                             .offsetsDataType(DataType::UINT64)
                             .build();

    StopGradient stop = StopGradient::Builder().network(network).featureInput(input).build();

    EXPECT_TRUE(stop.getUseRagged());
    ASSERT_TRUE(stop.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = stop.getRaggedFeatureOutput().value();
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getValues().getDimensions(), input.getValues().getDimensions());
    EXPECT_EQ(output.getValuesDataType(), input.getValuesDataType());
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), input.getMaxValuesPerRow());

    const json architecture = stop.architectureJson();
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>());
}

TEST(UtilityApiLayers, RaggedScaleGradientPreservesCanonicalPartitionAndScale) {
    Network network("ragged_scale_gradient_build");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::BF16)
                             .trailingDimensions({5})
                             .batchSize(3)
                             .maxTotalValues(17)
                             .maxValuesPerRow(9)
                             .offsetsDataType(DataType::UINT32)
                             .build();

    ScaleGradient scale = ScaleGradient::Builder().network(network).featureInput(input).scale(-0.25f).build();

    EXPECT_TRUE(scale.getUseRagged());
    EXPECT_FLOAT_EQ(scale.getScale(), -0.25f);
    ASSERT_TRUE(scale.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = scale.getRaggedFeatureOutput().value();
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getValues().getDimensions(), input.getValues().getDimensions());
    EXPECT_EQ(output.getValuesDataType(), input.getValuesDataType());
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), input.getMaxValuesPerRow());

    const json architecture = scale.architectureJson();
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_FLOAT_EQ(architecture.at("scale").get<float>(), -0.25f);
    EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>());
}
