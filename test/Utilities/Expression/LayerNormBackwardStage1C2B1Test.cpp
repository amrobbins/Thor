#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/ExpressionTestHooks.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {
PhysicalOutputs makeBackward(const std::vector<std::string>& targets, bool packed = false) {
    Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression gamma = Expression::input("gamma", DataType::FP32, DataType::FP32);
    const Expression beta = Expression::input("beta", DataType::FP32, DataType::FP32);
    if (packed) {
        const Expression extent = Expression::input("extent", DataType::UINT32, DataType::UINT32);
        x = x.withRaggedRuntimeExtent(extent, 2, 14, 7, RaggedRuntimeExtentSource::HOST_EXTENT);
    }
    const Expression y = Expression::layerNorm(
        x, gamma, beta, 7, 0.00031, DataType::FP32, DataType::FP32,
        packed ? std::optional<uint64_t>(14) : std::nullopt);
    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    std::vector<DataType> forward_dtypes(forward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& in : forward.expr->inputs) {
        if (in.name == "extent") forward_dtypes.at(in.slot) = DataType::UINT32;
    }
    resolveOutputsDTypesInPlace(forward, forward_dtypes);
    PhysicalOutputs backward = buildBackwardOutputs(
        forward, targets,
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {packed ? 14u : 3u, 7}}, {"gamma", {7}}, {"beta", {7}}, {"extent", {3}}});
    std::vector<DataType> backward_dtypes(backward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& in : backward.expr->inputs) {
        if (in.name == "extent") backward_dtypes.at(in.slot) = DataType::UINT32;
    }
    resolveOutputsDTypesInPlace(backward, backward_dtypes);
    return backward;
}

const PhysicalExecutionStage& layerNormBackwardStage(const detail::EquationCompilerPlanForTests& plan) {
    for (const auto& stage : plan.stages) {
        if (stage.kind == PhysicalExecutionStage::Kind::LayerNormBackward) return stage;
    }
    throw std::runtime_error("Missing physical LayerNorm backward stage");
}
}  // namespace

TEST(LayerNormBackwardStage1C2B1, DenseThreeRoutesShareStageButRetainDistinctValues) {
    const auto plan = detail::planEquationCompilerForTests(makeBackward({"x", "gamma", "beta"}));
    size_t matched = 0;
    for (const auto& stage : plan.stages) matched += stage.kind == PhysicalExecutionStage::Kind::LayerNormBackward;
    ASSERT_EQ(matched, 1u);
    const auto& stage = layerNormBackwardStage(plan);
    ASSERT_EQ(stage.input_value_ids.size(), 3u);
    ASSERT_EQ(stage.outputs.size(), 3u);
    EXPECT_NE(stage.outputs[0].value_id, stage.outputs[1].value_id);
    EXPECT_NE(stage.outputs[0].value_id, stage.outputs[2].value_id);
    EXPECT_NE(stage.outputs[1].value_id, stage.outputs[2].value_id);
    EXPECT_NE(stage.outputs[0].local_node_idx, stage.outputs[1].local_node_idx);
    EXPECT_NE(stage.outputs[1].local_node_idx, stage.outputs[2].local_node_idx);
    const auto compiled = EquationCompiler::compileLayerNormBackward(stage.expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_TRUE(compiled->produces_dx);
    EXPECT_TRUE(compiled->produces_dscale);
    EXPECT_TRUE(compiled->produces_dbias);
}

TEST(LayerNormBackwardStage1C2B1, RequestedBiasRouteDoesNotFabricateOtherGradientOutputs) {
    const auto plan = detail::planEquationCompilerForTests(makeBackward({"beta"}));
    const auto& stage = layerNormBackwardStage(plan);
    ASSERT_EQ(stage.outputs.size(), 1u);
    ASSERT_EQ(stage.expr.nodes.at(stage.outputs.front().local_node_idx).op, ExprOp::LAYERNORM_BACKWARD_BIAS);
    const auto compiled = EquationCompiler::compileLayerNormBackward(stage.expr);
    EXPECT_FALSE(compiled->produces_dx);
    EXPECT_FALSE(compiled->produces_dscale);
    EXPECT_TRUE(compiled->produces_dbias);
}

TEST(LayerNormBackwardStage1C2B1, PackedRoutesKeepOneOriginalHostExtentBinding) {
    const auto plan = detail::planEquationCompilerForTests(makeBackward({"x", "gamma", "beta"}, true));
    const auto& stage = layerNormBackwardStage(plan);
    ASSERT_EQ(stage.input_value_ids.size(), 4u);
    ASSERT_EQ(stage.outputs.size(), 3u);
    const auto compiled = EquationCompiler::compileLayerNormBackward(stage.expr);
    EXPECT_EQ(compiled->raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(compiled->ragged_offsets_input_slot, 3u);
    EXPECT_EQ(compiled->packed_row_capacity, 14u);
    EXPECT_EQ(compiled->ragged_batch_size, 2u);
    for (const auto& out : stage.outputs) {
        const auto& route = stage.expr.nodes.at(out.local_node_idx);
        EXPECT_EQ(route.ragged_runtime_offsets_input_slot, 3u);
        EXPECT_EQ(route.ragged_runtime_max_active_values, 14u);
    }
}
