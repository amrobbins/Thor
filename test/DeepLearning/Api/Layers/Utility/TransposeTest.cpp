#include "DeepLearning/Api/Layers/Utility/Transpose.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using json = nlohmann::json;

TEST(UtilityApiLayers, RaggedTransposePreservesPartitionAndSwapsOnlyFinalTrailingDimensions) {
    Network network("raggedTranspose");
    RaggedTensor input = RaggedTensor::logical(
        Tensor(DataType::FP32, {8, 2, 3, 4}), Tensor(DataType::UINT64, {1}), 8);

    Transpose transpose =
        Transpose::Builder().network(network).featureInput(input).outputDataType(DataType::FP16).build();

    ASSERT_TRUE(transpose.getUseRagged());
    ASSERT_TRUE(transpose.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(transpose.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = transpose.getRaggedFeatureOutput().value();
    EXPECT_EQ(output.getValuesDimensions(), std::vector<uint64_t>({8, 2, 4, 3}));
    EXPECT_EQ(output.getTrailingDimensions(), std::vector<uint64_t>({2, 4, 3}));
    EXPECT_EQ(output.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getMaxValuesPerRow(), input.getMaxValuesPerRow());
    EXPECT_TRUE(output.sharesPartitionWith(input));
    EXPECT_EQ(output.getDescriptorForPlacement(3).getMaxTotalValues(), 24u);
    EXPECT_EQ(output.getDescriptorForPlacement(7).getMaxTotalValues(), 56u);
    EXPECT_EQ(output.getOffsetsDataType(), DataType::UINT64);
    ASSERT_TRUE(transpose.getFeatureOutput().has_value());
    EXPECT_FALSE(transpose.outputTensorDimensionsIncludeBatch(transpose.getFeatureOutput().value()));
    EXPECT_EQ(transpose.getAllInputTensors(), std::vector<Tensor>({input.getValues(), input.getOffsets()}));

    const json architecture = transpose.architectureJson();
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(), input.getOffsets().getId());
}

TEST(UtilityApiLayers, RaggedTransposeRequiresAtLeastTwoTrailingDimensions) {
    Network network("raggedTransposeRejectsRank");
    RaggedTensor input = RaggedTensor::logical(
        Tensor(DataType::FP32, {8, 8}), Tensor(DataType::UINT32, {1}), 8);

    EXPECT_THROW((void)Transpose::Builder().network(network).featureInput(input).build(), std::invalid_argument);
}
