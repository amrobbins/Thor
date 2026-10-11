#include "Utilities/Expression/EquationCompiler.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <string>

using namespace ThorImplementation;

namespace {
PhysicalExpression makeBackwardExpr(bool packed = false) {
    PhysicalExpression expr;
    for (uint32_t slot = 0; slot < (packed ? 4u : 3u); ++slot) {
        expr.inputs.push_back(NamedInput{"arg" + std::to_string(slot), slot, NamedInput::Kind::Tensor});
        ExprNode arg;
        arg.op = ExprOp::INPUT;
        arg.input_slot = slot;
        arg.input_tensor_dtype = slot == 3 ? DataType::UINT32 : DataType::FP32;
        arg.output_dtype = arg.input_tensor_dtype;
        expr.nodes.push_back(arg);
    }
    ExprNode backward;
    backward.op = ExprOp::LAYERNORM_BACKWARD_X;
    backward.lhs = 0;
    backward.rhs = 1;
    backward.aux = 2;
    backward.layer_norm_normalized_feature_count = 8;
    backward.layer_norm_epsilon = 0.00031;
    backward.layer_norm_packed_row_capacity = packed ? 17 : 0;
    backward.ragged_runtime_offsets_input_slot = packed ? 3 : UINT32_MAX;
    backward.ragged_runtime_batch_size = packed ? 2 : 0;
    backward.ragged_runtime_max_active_values = packed ? 17 : 0;
    backward.compute_dtype = DataType::FP32;
    backward.output_dtype = DataType::FP32;
    expr.nodes.push_back(backward);
    expr.output_node = static_cast<uint32_t>(expr.nodes.size() - 1);
    return expr;
}
}  // namespace

TEST(LayerNormBackwardCompile1C2A, DenseThreeRoutesShareOneImmutableDescriptor) {
    PhysicalExpression expr = makeBackwardExpr();
    ExprNode gamma = expr.nodes.back();
    gamma.op = ExprOp::LAYERNORM_BACKWARD_SCALE;
    ExprNode beta = expr.nodes.back();
    beta.op = ExprOp::LAYERNORM_BACKWARD_BIAS;
    expr.nodes.push_back(gamma);
    expr.nodes.push_back(beta);

    const auto compiled = EquationCompiler::compileLayerNormBackward(expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->normalized_feature_count, 8u);
    EXPECT_DOUBLE_EQ(compiled->epsilon, 0.00031);
    EXPECT_TRUE(compiled->produces_dx);
    EXPECT_TRUE(compiled->produces_dscale);
    EXPECT_TRUE(compiled->produces_dbias);
    EXPECT_EQ(compiled->raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    EXPECT_EQ(compiled->outputDTypeFor(ExprOp::LAYERNORM_BACKWARD_X), DataType::FP32);
    EXPECT_EQ(compiled->outputDTypeFor(ExprOp::LAYERNORM_BACKWARD_SCALE), DataType::FP32);
    EXPECT_EQ(compiled->outputDTypeFor(ExprOp::LAYERNORM_BACKWARD_BIAS), DataType::FP32);
}

TEST(LayerNormBackwardCompile1C2A, OneRequestedRouteDoesNotInventLogicalGradients) {
    PhysicalExpression expr = makeBackwardExpr();
    expr.nodes.back().op = ExprOp::LAYERNORM_BACKWARD_BIAS;
    const auto compiled = EquationCompiler::compileLayerNormBackward(expr);
    EXPECT_FALSE(compiled->produces_dx);
    EXPECT_FALSE(compiled->produces_dscale);
    EXPECT_TRUE(compiled->produces_dbias);
}

TEST(LayerNormBackwardCompile1C2A, MismatchedMergedRouteRejected) {
    PhysicalExpression expr = makeBackwardExpr();
    ExprNode bad = expr.nodes.back();
    bad.op = ExprOp::LAYERNORM_BACKWARD_SCALE;
    bad.layer_norm_epsilon *= 2.0;
    expr.nodes.push_back(bad);
    EXPECT_THROW((void)EquationCompiler::compileLayerNormBackward(expr), std::runtime_error);
}

TEST(LayerNormBackwardCompile1C2A, PackedExtentContractDoesNotForceDeviceOffsets) {
    PhysicalExpression expr = makeBackwardExpr(true);
    const auto compiled = EquationCompiler::compileLayerNormBackward(expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->packed_row_capacity, 17u);
    EXPECT_EQ(compiled->ragged_offsets_input_slot, 3u);
    EXPECT_EQ(compiled->raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);
    expr.nodes.back().ragged_runtime_offsets_input_slot = UINT32_MAX;
    EXPECT_THROW((void)EquationCompiler::compileLayerNormBackward(expr), std::runtime_error);
}

TEST(LayerNormBackwardCompile1C2A, InvalidCuDnnParameterDTypeRejectedDuringCompilation) {
    PhysicalExpression expr = makeBackwardExpr();
    expr.nodes[1].input_tensor_dtype = DataType::FP16;
    expr.nodes.back().output_dtype = DataType::FP32;
    // Descriptor validation reports invalid caller configuration as invalid_argument.
    EXPECT_THROW((void)EquationCompiler::compileLayerNormBackward(expr), std::invalid_argument);
}

TEST(LayerNormBackwardCompile1C2A, MismatchedInputAndUpstreamDtypeIsExplicitlyUnsupported) {
    PhysicalExpression expr = makeBackwardExpr();
    expr.nodes[2].input_tensor_dtype = DataType::FP16;
    EXPECT_THROW((void)EquationCompiler::compileLayerNormBackward(expr), std::runtime_error);
}
