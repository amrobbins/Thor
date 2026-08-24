#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/ConvolutionSpatial.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include "gtest/gtest.h"

#include <cstdint>
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
            throw std::runtime_error("Expected exactly one convolution node in ConvolutionSpatial2dTest.");
        }
        found = &node;
    }
    if (found == nullptr) {
        throw std::runtime_error("Missing expected convolution node in ConvolutionSpatial2dTest.");
    }
    return *found;
}


ConvolutionSpatial2d makeSpatial(int32_t strideH, int32_t strideW, int32_t padH, int32_t padW) {
    ConvolutionSpatial2d spatial;
    spatial.stride_h = strideH;
    spatial.stride_w = strideW;
    spatial.pre_padding_h = padH;
    spatial.post_padding_h = padH;
    spatial.pre_padding_w = padW;
    spatial.post_padding_w = padW;
    return spatial;
}

PhysicalOutputs buildForward(const ConvolutionSpatial2d& spatial, uint64_t groups = 1) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    return Expression::outputs({{"output", Expression::conv2d(input, filter, spatial, DataType::FP32, DataType::FP32, groups)}})
        .physicalOutputs();
}

}  // namespace

TEST(ConvolutionSpatial2d, DefaultsAndExplicitFieldsRepresentGeometryDirectly) {
    ConvolutionSpatial2d spatial;
    spatial.stride_h = 2;
    spatial.stride_w = 3;
    spatial.pre_padding_h = 4;
    spatial.post_padding_h = 4;
    spatial.pre_padding_w = 5;
    spatial.post_padding_w = 5;

    EXPECT_EQ(spatial.stride_h, 2);
    EXPECT_EQ(spatial.stride_w, 3);
    EXPECT_EQ(spatial.dilation_h, 1);
    EXPECT_EQ(spatial.dilation_w, 1);
    EXPECT_EQ(spatial.pre_padding_h, 4);
    EXPECT_EQ(spatial.post_padding_h, 4);
    EXPECT_EQ(spatial.pre_padding_w, 5);
    EXPECT_EQ(spatial.post_padding_w, 5);
}

TEST(ConvolutionSpatial2d, PhysicalExpressionSerializationRoundTripsFourSidedPadding) {
    ConvolutionSpatial2d expected;
    expected.stride_h = 2;
    expected.stride_w = 3;
    expected.dilation_h = 2;
    expected.dilation_w = 1;
    expected.pre_padding_h = 1;
    expected.post_padding_h = 2;
    expected.pre_padding_w = 3;
    expected.post_padding_w = 0;

    constexpr uint64_t groups = 2;
    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(expected, groups)));
    const nlohmann::json payload = definition.architectureJson();

    bool found_conv = false;
    for (const auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() != "conv2d") {
            continue;
        }
        found_conv = true;
        EXPECT_EQ(node.at("conv_stride_h").get<int32_t>(), 2);
        EXPECT_EQ(node.at("conv_stride_w").get<int32_t>(), 3);
        EXPECT_EQ(node.at("conv_pre_padding_h").get<int32_t>(), 1);
        EXPECT_EQ(node.at("conv_post_padding_h").get<int32_t>(), 2);
        EXPECT_EQ(node.at("conv_pre_padding_w").get<int32_t>(), 3);
        EXPECT_EQ(node.at("conv_post_padding_w").get<int32_t>(), 0);
        EXPECT_EQ(node.at("conv_dilation_h").get<int32_t>(), 2);
        EXPECT_EQ(node.at("conv_dilation_w").get<int32_t>(), 1);
        EXPECT_EQ(node.at("conv_groups").get<uint64_t>(), groups);
        EXPECT_FALSE(node.contains("conv_pad_h"));
        EXPECT_FALSE(node.contains("conv_pad_w"));
    }
    ASSERT_TRUE(found_conv);

    const ExpressionDefinition loaded = ExpressionDefinition::deserialize(payload);
    EXPECT_EQ(onlyNodeWithOp(loaded.outputs, ExprOp::CONV2D).conv_spatial_2d, expected);
    EXPECT_EQ(onlyNodeWithOp(loaded.outputs, ExprOp::CONV2D).conv_groups, groups);
    EXPECT_EQ(loaded.architectureJson(), payload);
}

TEST(ConvolutionSpatial2d, SerializedExpressionRejectsLegacySymmetricPaddingSchema) {
    ConvolutionSpatial2d spatial;
    spatial.pre_padding_h = 1;
    spatial.post_padding_h = 1;
    spatial.pre_padding_w = 2;
    spatial.post_padding_w = 2;

    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(spatial)));
    nlohmann::json payload = definition.architectureJson();
    for (auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() != "conv2d") {
            continue;
        }
        node.erase("conv_pre_padding_h");
        node.erase("conv_post_padding_h");
        node.erase("conv_pre_padding_w");
        node.erase("conv_post_padding_w");
        node["conv_pad_h"] = 1;
        node["conv_pad_w"] = 2;
    }

    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload), std::runtime_error);
}

TEST(ConvolutionSpatial2d, SerializedExpressionRejectsMissingGroupField) {
    ConvolutionSpatial2d spatial = makeSpatial(1, 1, 0, 0);
    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(spatial, 2)));
    nlohmann::json payload = definition.architectureJson();
    for (auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() == "conv2d") {
            node.erase("conv_groups");
        }
    }
    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload), std::runtime_error);
}

TEST(ConvolutionSpatial2d, AutoDiffCopiesDescriptorToDataAndFilterGradients) {
    ConvolutionSpatial2d expected = makeSpatial(2, 3, 1, 2);
    expected.dilation_h = 2;
    expected.dilation_w = 3;
    expected.post_padding_h = 3;
    expected.pre_padding_w = 4;
    expected.post_padding_w = 1;
    const PhysicalOutputs forward = buildForward(expected);
    const PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"input", "filter"},
        std::unordered_map<std::string, std::string>{{"output", "doutput"}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"input", {2, 4, 9, 11}},
            {"filter", {6, 4, 3, 5}},
        });

    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV2D_BACKWARD_DATA).conv_spatial_2d, expected);
    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV2D_BACKWARD_FILTER).conv_spatial_2d, expected);
}


TEST(ConvolutionSpatial2d, AutoDiffPreservesGroupedConvolutionGeometry) {
    ConvolutionSpatial2d spatial = makeSpatial(1, 1, 1, 1);
    constexpr uint64_t groups = 2;
    const PhysicalOutputs forward = buildForward(spatial, groups);
    const PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"input", "filter"},
        std::unordered_map<std::string, std::string>{{"output", "doutput"}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"input", {2, 4, 7, 7}},
            {"filter", {6, 2, 3, 3}},
        });

    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV2D_BACKWARD_DATA).conv_groups, groups);
    EXPECT_EQ(onlyNodeWithOp(backward, ExprOp::CONV2D_BACKWARD_FILTER).conv_groups, groups);
}

TEST(ConvolutionSpatial2d, PhysicalConv2dAcceptsPositiveDilationAndAsymmetricPadding) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);

    ConvolutionSpatial2d dilated = makeSpatial(1, 1, 0, 0);
    dilated.dilation_h = 2;
    dilated.dilation_w = 3;
    EXPECT_NO_THROW((void)Expression::conv2d(input, filter, dilated));

    dilated.dilation_h = 0;
    EXPECT_THROW((void)Expression::conv2d(input, filter, dilated), std::runtime_error);

    ConvolutionSpatial2d asymmetric = makeSpatial(1, 1, 1, 1);
    asymmetric.post_padding_h = 2;
    asymmetric.pre_padding_w = 3;
    asymmetric.post_padding_w = 0;
    EXPECT_NO_THROW((void)Expression::conv2d(input, filter, asymmetric));
}



TEST(ConvolutionSpatial2d, CompilerIdentityDistinguishesPostPadding) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);

    ConvolutionSpatial2d first = makeSpatial(2, 2, 1, 1);
    ConvolutionSpatial2d second = first;
    second.post_padding_w = 2;

    PhysicalOutputs outputs =
        Expression::outputs({
                                {"first", Expression::conv2d(input, filter, first, DataType::FP32, DataType::FP32)},
                                {"second", Expression::conv2d(input, filter, second, DataType::FP32, DataType::FP32)},
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

TEST(ConvolutionSpatial2d, SerializedExpressionRejectsNonPositiveDilation) {
    ConvolutionSpatial2d spatial = makeSpatial(1, 1, 0, 0);
    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(spatial)));
    nlohmann::json payload = definition.architectureJson();
    for (auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() == "conv2d") {
            node["conv_dilation_h"] = 0;
        }
    }
    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload), std::runtime_error);
}

TEST(ConvolutionSpatial2d, PhysicalExpressionIdentityDistinguishesDilation) {
    ConvolutionSpatial2d unit = makeSpatial(1, 1, 1, 1);
    ConvolutionSpatial2d dilated = unit;
    dilated.dilation_h = 2;

    const nlohmann::json unitJson =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(unit))).architectureJson();
    const nlohmann::json dilatedJson =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(dilated))).architectureJson();

    EXPECT_NE(unitJson, dilatedJson);
}

TEST(ConvolutionSpatial2d, PhysicalExpressionSerializationRoundTripsNonUnitDilation) {
    ConvolutionSpatial2d expected = makeSpatial(2, 1, 3, 2);
    expected.dilation_h = 2;
    expected.dilation_w = 4;

    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(buildForward(expected)));
    const nlohmann::json payload = definition.architectureJson();
    const ExpressionDefinition loaded = ExpressionDefinition::deserialize(payload);

    EXPECT_EQ(onlyNodeWithOp(loaded.outputs, ExprOp::CONV2D).conv_spatial_2d, expected);
}
