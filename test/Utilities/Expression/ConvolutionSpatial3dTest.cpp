#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/ConvolutionSpatial.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

const ExprNode& onlyNodeWithOp(const PhysicalOutputs& outputs, ExprOp op) {
    const ExprNode* found = nullptr;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op != op) {
            continue;
        }
        if (found != nullptr) {
            throw std::runtime_error("Expected exactly one convolution node in ConvolutionSpatial3dTest.");
        }
        found = &node;
    }
    if (found == nullptr) {
        throw std::runtime_error("Missing expected convolution node in ConvolutionSpatial3dTest.");
    }
    return *found;
}

ConvolutionSpatial3d makeLegacySpatial(int32_t strideD,
                                       int32_t strideH,
                                       int32_t strideW,
                                       int32_t padD,
                                       int32_t padH,
                                       int32_t padW) {
    ConvolutionSpatial3d spatial;
    spatial.stride_d = strideD;
    spatial.stride_h = strideH;
    spatial.stride_w = strideW;
    spatial.pre_padding_d = padD;
    spatial.post_padding_d = padD;
    spatial.pre_padding_h = padH;
    spatial.post_padding_h = padH;
    spatial.pre_padding_w = padW;
    spatial.post_padding_w = padW;
    return spatial;
}

PhysicalOutputs buildForward(const ConvolutionSpatial3d& spatial, uint64_t groups = 1) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    return Expression::outputs({{"output", Expression::conv3d(input, filter, spatial, DataType::FP32, DataType::FP32, groups)}})
        .physicalOutputs();
}

}  // namespace

TEST(ConvolutionSpatial3d, DefaultsAndLegacyBuilderPopulateCanonicalDescriptor) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    const PhysicalOutputs outputs =
        Expression::outputs({{"output", Expression::conv3d(input, filter, 2, 3, 4, 5, 6, 7)}}).physicalOutputs();

    const ConvolutionSpatial3d& spatial = onlyNodeWithOp(outputs, ExprOp::CONV3D).conv_spatial_3d;
    EXPECT_EQ(spatial.stride_d, 2);
    EXPECT_EQ(spatial.stride_h, 3);
    EXPECT_EQ(spatial.stride_w, 4);
    EXPECT_EQ(spatial.dilation_d, 1);
    EXPECT_EQ(spatial.dilation_h, 1);
    EXPECT_EQ(spatial.dilation_w, 1);
    EXPECT_EQ(spatial.pre_padding_d, 5);
    EXPECT_EQ(spatial.post_padding_d, 5);
    EXPECT_EQ(spatial.pre_padding_h, 6);
    EXPECT_EQ(spatial.post_padding_h, 6);
    EXPECT_EQ(spatial.pre_padding_w, 7);
    EXPECT_EQ(spatial.post_padding_w, 7);
}

TEST(ConvolutionSpatial3d, PhysicalExpressionSerializationRoundTripsCanonicalDescriptor) {
    ConvolutionSpatial3d expected;
    expected.stride_d = 2;
    expected.stride_h = 3;
    expected.stride_w = 4;
    expected.dilation_d = 2;
    expected.dilation_h = 1;
    expected.dilation_w = 3;
    expected.pre_padding_d = 1;
    expected.post_padding_d = 2;
    expected.pre_padding_h = 3;
    expected.post_padding_h = 4;
    expected.pre_padding_w = 5;
    expected.post_padding_w = 6;
    constexpr uint64_t groups = 2;

    PhysicalOutputs outputs = buildForward(expected, groups);
    const ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(outputs));
    const nlohmann::json payload = definition.architectureJson();

    bool found_conv = false;
    for (const auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() != "conv3d") {
            continue;
        }
        found_conv = true;
        EXPECT_EQ(node.at("conv_stride_d").get<int32_t>(), expected.stride_d);
        EXPECT_EQ(node.at("conv_stride_h").get<int32_t>(), expected.stride_h);
        EXPECT_EQ(node.at("conv_stride_w").get<int32_t>(), expected.stride_w);
        EXPECT_EQ(node.at("conv_dilation_d").get<int32_t>(), expected.dilation_d);
        EXPECT_EQ(node.at("conv_dilation_h").get<int32_t>(), expected.dilation_h);
        EXPECT_EQ(node.at("conv_dilation_w").get<int32_t>(), expected.dilation_w);
        EXPECT_EQ(node.at("conv_pre_padding_d").get<int32_t>(), expected.pre_padding_d);
        EXPECT_EQ(node.at("conv_post_padding_d").get<int32_t>(), expected.post_padding_d);
        EXPECT_EQ(node.at("conv_pre_padding_h").get<int32_t>(), expected.pre_padding_h);
        EXPECT_EQ(node.at("conv_post_padding_h").get<int32_t>(), expected.post_padding_h);
        EXPECT_EQ(node.at("conv_pre_padding_w").get<int32_t>(), expected.pre_padding_w);
        EXPECT_EQ(node.at("conv_post_padding_w").get<int32_t>(), expected.post_padding_w);
        EXPECT_EQ(node.at("conv_groups").get<uint64_t>(), groups);
        EXPECT_FALSE(node.contains("conv_pad_d"));
        EXPECT_FALSE(node.contains("conv_pad_h"));
        EXPECT_FALSE(node.contains("conv_pad_w"));
    }
    ASSERT_TRUE(found_conv);

    const ExpressionDefinition loaded = ExpressionDefinition::deserialize(payload);
    EXPECT_EQ(onlyNodeWithOp(loaded.outputs, ExprOp::CONV3D).conv_spatial_3d, expected);
    EXPECT_EQ(onlyNodeWithOp(loaded.outputs, ExprOp::CONV3D).conv_groups, groups);
    EXPECT_EQ(loaded.architectureJson(), payload);
}

TEST(ConvolutionSpatial3d, PhysicalExpressionDeserializerAcceptsLegacySymmetricSchema) {
    const ConvolutionSpatial3d expected = makeLegacySpatial(2, 3, 4, 1, 2, 3);
    constexpr uint64_t groups = 2;
    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(expected, groups)));
    nlohmann::json payload = definition.architectureJson();

    bool found_conv = false;
    for (auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() != "conv3d") {
            continue;
        }
        found_conv = true;
        node["conv_pad_d"] = expected.pre_padding_d;
        node["conv_pad_h"] = expected.pre_padding_h;
        node["conv_pad_w"] = expected.pre_padding_w;
        node.erase("conv_pre_padding_d");
        node.erase("conv_post_padding_d");
        node.erase("conv_pre_padding_h");
        node.erase("conv_post_padding_h");
        node.erase("conv_pre_padding_w");
        node.erase("conv_post_padding_w");
        node.erase("conv_dilation_d");
        node.erase("conv_dilation_h");
        node.erase("conv_dilation_w");
    }
    ASSERT_TRUE(found_conv);

    const ExpressionDefinition loaded = ExpressionDefinition::deserialize(payload);
    EXPECT_EQ(onlyNodeWithOp(loaded.outputs, ExprOp::CONV3D).conv_spatial_3d, expected);
    EXPECT_EQ(onlyNodeWithOp(loaded.outputs, ExprOp::CONV3D).conv_groups, groups);
}

TEST(ConvolutionSpatial3d, PhysicalExpressionDeserializerRejectsInvalidCanonicalDescriptor) {
    const ConvolutionSpatial3d expected = makeLegacySpatial(1, 1, 1, 1, 1, 1);
    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(expected)));

    nlohmann::json invalid_dilation = definition.architectureJson();
    for (auto& node : invalid_dilation.at("nodes")) {
        if (node.at("op").get<std::string>() == "conv3d") {
            node["conv_dilation_d"] = 0;
        }
    }
    EXPECT_THROW((void)ExpressionDefinition::deserialize(invalid_dilation), std::runtime_error);

    nlohmann::json invalid_padding = definition.architectureJson();
    for (auto& node : invalid_padding.at("nodes")) {
        if (node.at("op").get<std::string>() == "conv3d") {
            node["conv_post_padding_w"] = -1;
        }
    }
    EXPECT_THROW((void)ExpressionDefinition::deserialize(invalid_padding), std::runtime_error);

    nlohmann::json incomplete = definition.architectureJson();
    for (auto& node : incomplete.at("nodes")) {
        if (node.at("op").get<std::string>() == "conv3d") {
            node.erase("conv_dilation_h");
        }
    }
    EXPECT_THROW((void)ExpressionDefinition::deserialize(incomplete), std::runtime_error);
}

TEST(ConvolutionSpatial3d, AutoDiffCopiesDescriptorToDataAndFilterGradients) {
    ConvolutionSpatial3d expected = makeLegacySpatial(1, 1, 1, 1, 2, 1);
    expected.dilation_d = 2;
    expected.dilation_w = 2;
    expected.post_padding_d = 2;
    expected.post_padding_h = 1;
    expected.post_padding_w = 3;
    constexpr uint64_t groups = 2;
    PhysicalOutputs forward = buildForward(expected, groups);
    const PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"input", "filter"},
        std::unordered_map<std::string, std::string>{{"output", "doutput"}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"input", {2, 4, 9, 11, 13}},
            {"filter", {6, 2, 3, 3, 3}},
        });

    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV3D_BACKWARD_DATA).conv_spatial_3d, expected);
    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV3D_BACKWARD_FILTER).conv_spatial_3d, expected);
    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV3D_BACKWARD_DATA).conv_groups, groups);
    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV3D_BACKWARD_FILTER).conv_groups, groups);
}

TEST(ConvolutionSpatial3d, CompilerCarriesCanonicalDescriptor) {
    ConvolutionSpatial3d expected = makeLegacySpatial(2, 3, 1, 1, 2, 0);
    expected.dilation_d = 2;
    expected.dilation_h = 3;
    expected.post_padding_d = 4;
    expected.post_padding_h = 0;
    expected.pre_padding_w = 3;
    expected.post_padding_w = 1;
    constexpr uint64_t groups = 2;
    PhysicalOutputs outputs = buildForward(expected, groups);
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(outputs);

    const PhysicalExecutionStage* convolution_stage = nullptr;
    for (const PhysicalExecutionStage& stage : stages) {
        if (stage.kind == PhysicalExecutionStage::Kind::Convolution) {
            ASSERT_EQ(convolution_stage, nullptr);
            convolution_stage = &stage;
        }
    }
    ASSERT_NE(convolution_stage, nullptr);

    const std::shared_ptr<CompiledConvolution> compiled = EquationCompiler::compileConvolution(convolution_stage->expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_TRUE(compiled->is_3d);
    EXPECT_EQ(compiled->spatial_3d, expected);
    EXPECT_EQ(compiled->groups, groups);
}

TEST(ConvolutionSpatial3d, CompilerIdentityDistinguishesSpatialGeometry) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);

    ConvolutionSpatial3d first = makeLegacySpatial(1, 1, 1, 1, 1, 1);
    ConvolutionSpatial3d second = first;
    second.dilation_w = 2;

    PhysicalOutputs outputs =
        Expression::outputs({
                                {"first", Expression::conv3d(input, filter, first, DataType::FP32, DataType::FP32)},
                                {"second", Expression::conv3d(input, filter, second, DataType::FP32, DataType::FP32)},
                            })
            .physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(outputs);

    size_t convolution_stages = 0;
    for (const PhysicalExecutionStage& stage : stages) {
        if (stage.kind == PhysicalExecutionStage::Kind::Convolution) {
            ++convolution_stages;
        }
    }
    EXPECT_EQ(convolution_stages, 2u);
}

TEST(ConvolutionSpatial3d, LegacyPhysicalApiStillRejectsInvalidGeometry) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);

    EXPECT_THROW((void)Expression::conv3d(input, filter, 0, 1, 1, 0, 0, 0), std::runtime_error);
    EXPECT_THROW((void)Expression::conv3d(input, filter, 1, 1, 1, -1, 0, 0), std::runtime_error);
    EXPECT_THROW((void)Expression::conv3d(input, filter, 1, 1, 1, 0, 0, 0, std::nullopt, std::nullopt, 0),
                 std::runtime_error);
}

TEST(ConvolutionSpatial3d, DescriptorPhysicalApiAcceptsDilationAndAsymmetricPadding) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);

    ConvolutionSpatial3d spatial;
    spatial.stride_d = 2;
    spatial.stride_h = 1;
    spatial.stride_w = 3;
    spatial.dilation_d = 2;
    spatial.dilation_h = 3;
    spatial.dilation_w = 4;
    spatial.pre_padding_d = 1;
    spatial.post_padding_d = 2;
    spatial.pre_padding_h = 3;
    spatial.post_padding_h = 0;
    spatial.pre_padding_w = 4;
    spatial.post_padding_w = 5;

    EXPECT_NO_THROW((void)Expression::conv3d(input, filter, spatial, DataType::FP32, DataType::FP32, 2));

    ConvolutionSpatial3d invalid = spatial;
    invalid.stride_d = 0;
    EXPECT_THROW((void)Expression::conv3d(input, filter, invalid), std::runtime_error);

    invalid = spatial;
    invalid.dilation_h = 0;
    EXPECT_THROW((void)Expression::conv3d(input, filter, invalid), std::runtime_error);

    invalid = spatial;
    invalid.post_padding_w = -1;
    EXPECT_THROW((void)Expression::conv3d(input, filter, invalid), std::runtime_error);

    EXPECT_THROW((void)Expression::conv3d(input, filter, spatial, std::nullopt, std::nullopt, 0), std::runtime_error);
}
