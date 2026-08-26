#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/RaggedExpression.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

bool containsOp(const PhysicalOutputs& outputs, ExprOp op) {
    if (!outputs.expr) {
        return false;
    }
    return std::any_of(outputs.expr->nodes.begin(), outputs.expr->nodes.end(), [op](const ExprNode& node) {
        return node.op == op;
    });
}

bool isZeroFill(const PhysicalExpression& expr, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }
    const ExprNode& node = expr.nodes.at(node_idx);
    return node.op == ExprOp::FILL && node.scalar_fp == 0.0;
}

bool containsZeroAddBroadcast(const PhysicalOutputs& outputs) {
    if (!outputs.expr) {
        return false;
    }
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op == ExprOp::ADD && (isZeroFill(*outputs.expr, node.lhs) || isZeroFill(*outputs.expr, node.rhs))) {
            return true;
        }
    }
    return false;
}

RaggedTensorDescriptor raggedDescriptor(std::vector<uint64_t> trailing_dims = {}) {
    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 8;
    return RaggedTensorDescriptor(
        DataType::FP32, trailing_dims, batch_size, max_total_values, max_total_values, DataType::UINT32);
}

}  // namespace

// The CMake production gate deliberately runs this disabled preflight with
// --gtest_also_run_disabled_tests. GPU-backed T9/materialization tests may skip
// without a CUDA device; the gate itself must never succeed vacuously that way.
TEST(MaterializationBroadcastProductionGate, DISABLED_RequiresCudaDevice) {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_GT(device_count, 0) << "M11 production qualification requires a CUDA device.";
}

TEST(MaterializationBroadcastProductionGate, DenseAndRaggedShapeTransformsRemainSemanticallyDistinct) {
    // Dense shape expansion is an ordinary BROADCAST_TO mathematical op.
    const Expression dense = Expression::input("dense", DataType::FP32, DataType::FP32);
    PhysicalOutputs dense_forward = Expression::outputs({{"y", dense.reduce_sum({0}, {})}}).physicalOutputs();
    resolveOutputsDTypesInPlace(dense_forward, {DataType::FP32});
    const PhysicalOutputs dense_backward = buildBackwardOutputs(
        dense_forward,
        {"dense"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"dense", {2, 3}}});

    ASSERT_EQ(dense_backward.outputs.size(), 1u);
    ASSERT_LT(dense_backward.outputs.front().node_idx, dense_backward.expr->nodes.size());
    const ExprNode& dense_root = dense_backward.expr->nodes.at(dense_backward.outputs.front().node_idx);
    EXPECT_EQ(dense_root.op, ExprOp::BROADCAST_TO);
    EXPECT_EQ(dense_root.broadcast_dims, (std::vector<uint64_t>{2, 3}));
    EXPECT_FALSE(containsZeroAddBroadcast(dense_backward));
    EXPECT_FALSE(containsOp(dense_backward, ExprOp::SEGMENTED_BROADCAST));

    // Ragged row-wise expansion retains explicit row-partition semantics and
    // must not be silently represented as ordinary dense broadcasting.
    const RaggedExpression ragged = RaggedExpression::input("ragged", raggedDescriptor());
    const PhysicalOutputs segment_sum_forward =
        Expression::outputs({{"segment_sum", ragged.segment_sum()}}).physicalOutputs();
    const PhysicalOutputs segment_sum_backward = buildBackwardOutputs(segment_sum_forward, {"ragged.values"});
    EXPECT_TRUE(containsOp(segment_sum_backward, ExprOp::SEGMENTED_BROADCAST));
    EXPECT_FALSE(containsOp(segment_sum_backward, ExprOp::BROADCAST_TO));
    EXPECT_FALSE(containsZeroAddBroadcast(segment_sum_backward));

    // A broadcast parameter gradient over authoritative packed rows must use
    // SEGMENTED_REDUCE_SUM before any ordinary shape-only reduction.
    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t width = 4;
    const Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input("offsets", std::nullopt, DataType::UINT32);
    const Expression scaled =
        (values * scale).withRaggedRuntimeExtent(offsets, batch_size, max_total_values, width);
    const PhysicalOutputs ragged_forward = Expression::outputs({{"scaled", scaled}}).physicalOutputs();
    PhysicalOutputs scale_backward = buildBackwardOutputs(
        ragged_forward,
        {"scale"},
        std::nullopt,
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"values", {max_total_values, width}},
            {"scale", {width}},
            {"offsets", {batch_size + 1}},
        });
    std::vector<DataType> ragged_input_dtypes(scale_backward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : scale_backward.expr->inputs) {
        if (input.name == "offsets") {
            ragged_input_dtypes.at(input.slot) = DataType::UINT32;
        }
    }
    resolveOutputsDTypesInPlace(scale_backward, ragged_input_dtypes);
    EXPECT_TRUE(containsOp(scale_backward, ExprOp::SEGMENTED_REDUCE_SUM));
    EXPECT_FALSE(containsZeroAddBroadcast(scale_backward));
}

TEST(MaterializationBroadcastProductionGate, OutputStorageDtypeAndOwnershipStayOutOfTheMathematicalDag) {
    const Expression lhs = Expression::input("lhs", DataType::BF16, DataType::BF16);
    const Expression rhs = Expression::input("rhs", DataType::BF16, DataType::BF16);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward = Expression::outputs({{"y", (lhs + rhs) * scale}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::BF16, DataType::BF16, DataType::FP32});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {4}},
            {"rhs", {4}},
            {"scale", {4}},
        });

    ASSERT_EQ(backward.outputs.size(), 2u);
    const NamedOutput& lhs_grad = backward.outputs[0];
    const NamedOutput& rhs_grad = backward.outputs[1];
    EXPECT_EQ(lhs_grad.node_idx, rhs_grad.node_idx);
    ASSERT_TRUE(lhs_grad.materialization.storage_dtype.has_value());
    ASSERT_TRUE(rhs_grad.materialization.storage_dtype.has_value());
    EXPECT_EQ(lhs_grad.materialization.storage_dtype.value(), DataType::BF16);
    EXPECT_EQ(rhs_grad.materialization.storage_dtype.value(), DataType::BF16);
    EXPECT_FALSE(lhs_grad.materialization.require_distinct_storage);
    EXPECT_TRUE(rhs_grad.materialization.require_distinct_storage);

    // Physical output requirements must not modify the persistent mathematics.
    EXPECT_FALSE(containsOp(backward, ExprOp::CAST));
    EXPECT_FALSE(containsOp(backward, ExprOp::FILL));
    EXPECT_FALSE(containsOp(backward, ExprOp::WHERE));
    EXPECT_FALSE(containsOp(backward, ExprOp::EQUAL));
    EXPECT_FALSE(containsZeroAddBroadcast(backward));

    std::vector<DataType> backward_input_dtypes(backward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : backward.expr->inputs) {
        if (input.name == "lhs" || input.name == "rhs") {
            backward_input_dtypes.at(input.slot) = DataType::BF16;
        } else if (input.name == "scale" || input.name == "dy") {
            backward_input_dtypes.at(input.slot) = DataType::FP32;
        } else {
            FAIL() << "Unexpected M11 backward input: " << input.name;
        }
    }
    resolveOutputsDTypesInPlace(backward, backward_input_dtypes);

    // The requested BF16 storage conversion appears exactly once, in the
    // compiler-local view. Distinct-output allocation remains an output-layer
    // concern and therefore needs no second mathematical terminal node.
    const auto stages = EquationCompiler::splitAtReductionBoundaries(backward);
    size_t compiler_local_cast_count = 0;
    for (const PhysicalExecutionStage& stage : stages) {
        for (const ExprNode& node : stage.expr.nodes) {
            if (node.op == ExprOp::CAST) {
                ++compiler_local_cast_count;
                ASSERT_TRUE(node.output_dtype.has_value());
                EXPECT_EQ(node.output_dtype.value(), DataType::BF16);
            }
        }
    }
    EXPECT_EQ(compiler_local_cast_count, 1u);

    // Planning is compile-local: the persistent DAG remains unmodified.
    EXPECT_FALSE(containsOp(backward, ExprOp::CAST));
    EXPECT_EQ(backward.outputs[0].node_idx, backward.outputs[1].node_idx);
}
