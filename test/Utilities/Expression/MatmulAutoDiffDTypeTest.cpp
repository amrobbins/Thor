#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/EquationCompiler.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ThorImplementation;

namespace {

const NamedOutput& namedOutput(const PhysicalOutputs& outputs, const std::string& name) {
    for (const NamedOutput& output : outputs.outputs) {
        if (output.name == name) {
            return output;
        }
    }
    throw std::runtime_error("Missing named output in MatmulAutoDiffDTypeTest: " + name);
}

std::vector<DataType> inputDTypes(const PhysicalOutputs& outputs,
                                  const std::unordered_map<std::string, DataType>& dtype_by_name) {
    if (!outputs.expr) {
        throw std::runtime_error("MatmulAutoDiffDTypeTest received null expression.");
    }

    std::vector<DataType> dtypes(outputs.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : outputs.expr->inputs) {
        auto it = dtype_by_name.find(input.name);
        if (it == dtype_by_name.end()) {
            throw std::runtime_error("Missing input dtype in MatmulAutoDiffDTypeTest for: " + input.name);
        }
        dtypes.at(input.slot) = it->second;
    }
    return dtypes;
}

std::vector<uint32_t> castNodesWithOutputDType(const PhysicalOutputs& outputs, DataType dtype) {
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < outputs.expr->nodes.size(); ++i) {
        const ExprNode& node = outputs.expr->nodes.at(i);
        if (node.op == ExprOp::CAST && node.output_dtype.has_value() && node.output_dtype.value() == dtype) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<uint32_t> matmulConsumers(const PhysicalOutputs& outputs, uint32_t source_node) {
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < outputs.expr->nodes.size(); ++i) {
        const ExprNode& node = outputs.expr->nodes.at(i);
        if ((node.op == ExprOp::MATMUL || node.op == ExprOp::GEMM) &&
            (node.lhs == source_node || node.rhs == source_node)) {
            result.push_back(i);
        }
    }
    return result;
}

bool dependsOnNode(const PhysicalExpression& expression, uint32_t root, uint32_t target) {
    if (root == target) {
        return true;
    }

    std::vector<uint32_t> pending{root};
    std::unordered_set<uint32_t> visited;
    while (!pending.empty()) {
        const uint32_t current = pending.back();
        pending.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        if (current >= expression.nodes.size()) {
            throw std::runtime_error("Invalid node index while traversing MatmulAutoDiffDTypeTest graph.");
        }

        const ExprNode& node = expression.nodes.at(current);
        const uint32_t children[] = {node.lhs, node.rhs, node.aux, node.alpha_node, node.beta_node, node.matmul_epilogue_aux};
        for (uint32_t child : children) {
            if (child == UINT32_MAX) {
                continue;
            }
            if (child == target) {
                return true;
            }
            pending.push_back(child);
        }
    }
    return false;
}

PhysicalOutputs buildRegularMatmulBackward(DataType operand_dtype,
                                           DataType output_dtype,
                                           std::optional<DataType> backward_output_dtype = std::nullopt) {
    const Expression lhs = Expression::input("lhs", operand_dtype, operand_dtype);
    const Expression rhs = Expression::input("rhs", operand_dtype, operand_dtype);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::matmul(lhs, rhs, false, false, DataType::FP32, output_dtype)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {operand_dtype, operand_dtype});
    if (backward_output_dtype.has_value()) {
        forward.expr->nodes.at(namedOutput(forward, "out").node_idx).backward_output_dtype =
            backward_output_dtype.value();
    }

    const DataType upstream_dtype = backward_output_dtype.value_or(output_dtype);
    return buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, DataType>{{"out", upstream_dtype}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
        });
}

void expectSharedLowPrecisionOutputGradientCast(DataType operand_dtype) {
    PhysicalOutputs backward = buildRegularMatmulBackward(operand_dtype, DataType::FP32);
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", operand_dtype},
                                                {"rhs", operand_dtype},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, operand_dtype);
    ASSERT_EQ(casts.size(), 1u);

    const std::vector<uint32_t> consumers = matmulConsumers(backward, casts.front());
    ASSERT_EQ(consumers.size(), 2u) << "dL/dlhs and dL/drhs must share one converted output gradient.";
    for (uint32_t consumer : consumers) {
        const ExprNode& node = backward.expr->nodes.at(consumer);
        ASSERT_TRUE(node.output_dtype.has_value());
        EXPECT_EQ(node.output_dtype.value(), operand_dtype);
    }
}

}  // namespace

TEST(MatmulAutoDiffDType, Bf16InputsFp32OutputShareOneConvertedOutputGradient) {
    expectSharedLowPrecisionOutputGradientCast(DataType::BF16);
}

TEST(MatmulAutoDiffDType, Fp16InputsFp32OutputShareOneConvertedOutputGradient) {
    expectSharedLowPrecisionOutputGradientCast(DataType::FP16);
}

TEST(MatmulAutoDiffDType, Bf16OutputWithFp32BackwardGradientSharesConvertedOutputGradient) {
    PhysicalOutputs backward =
        buildRegularMatmulBackward(DataType::BF16, DataType::BF16, DataType::FP32);
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::BF16},
                                                {"rhs", DataType::BF16},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, DataType::BF16);
    ASSERT_EQ(casts.size(), 1u);
    EXPECT_EQ(matmulConsumers(backward, casts.front()).size(), 2u);
}

TEST(MatmulAutoDiffDType, Fp16OutputWithFp32BackwardGradientSharesConvertedOutputGradient) {
    PhysicalOutputs backward =
        buildRegularMatmulBackward(DataType::FP16, DataType::FP16, DataType::FP32);
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::FP16},
                                                {"rhs", DataType::FP16},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, DataType::FP16);
    ASSERT_EQ(casts.size(), 1u);
    EXPECT_EQ(matmulConsumers(backward, casts.front()).size(), 2u);
}

TEST(MatmulAutoDiffDType, Bf16OutputUsesRuntimeFp32SeedDTypeWithoutForwardAnnotation) {
    const Expression lhs = Expression::input("lhs", DataType::BF16, DataType::BF16);
    const Expression rhs = Expression::input("rhs", DataType::BF16, DataType::BF16);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::BF16)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::BF16, DataType::BF16});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, DataType>{{"out", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
        });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::BF16},
                                                {"rhs", DataType::BF16},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, DataType::BF16);
    ASSERT_EQ(casts.size(), 1u);
    EXPECT_EQ(matmulConsumers(backward, casts.front()).size(), 2u);
}

TEST(MatmulAutoDiffDType, Fp16OutputUsesRuntimeFp32SeedDTypeWithoutForwardAnnotation) {
    const Expression lhs = Expression::input("lhs", DataType::FP16, DataType::FP16);
    const Expression rhs = Expression::input("rhs", DataType::FP16, DataType::FP16);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::FP16)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP16, DataType::FP16});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, DataType>{{"out", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
        });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::FP16},
                                                {"rhs", DataType::FP16},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, DataType::FP16);
    ASSERT_EQ(casts.size(), 1u);
    EXPECT_EQ(matmulConsumers(backward, casts.front()).size(), 2u);
}

void expectPromotedLogicalOperandUsesMaterializedLowPrecisionForBackwardCast(DataType operand_dtype) {
    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("rhs", operand_dtype, operand_dtype);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::FP32)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {operand_dtype, operand_dtype});

    const ExprNode& resolved_lhs = forward.expr->nodes.at(0);
    ASSERT_TRUE(resolved_lhs.input_tensor_dtype.has_value());
    ASSERT_TRUE(resolved_lhs.output_dtype.has_value());
    EXPECT_EQ(resolved_lhs.input_tensor_dtype.value(), operand_dtype);
    EXPECT_EQ(resolved_lhs.output_dtype.value(), DataType::FP32);

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, DataType>{{"out", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
        });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", operand_dtype},
                                                {"rhs", operand_dtype},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, operand_dtype);
    // One cast normalizes the FP32 upstream gradient and is shared by dL/dlhs and
    // dL/drhs.  The other converts the forward lhs's promoted FP32 logical value
    // back to its BF16/FP16 storage dtype before it is reused by dL/drhs.
    ASSERT_EQ(casts.size(), 2u);
    size_t shared_gradient_casts = 0;
    size_t promoted_operand_casts = 0;
    for (uint32_t cast : casts) {
        const size_t consumer_count = matmulConsumers(backward, cast).size();
        shared_gradient_casts += consumer_count == 2u ? 1u : 0u;
        promoted_operand_casts += consumer_count == 1u ? 1u : 0u;
    }
    EXPECT_EQ(shared_gradient_casts, 1u);
    EXPECT_EQ(promoted_operand_casts, 1u);

    for (const PhysicalExecutionStage& stage : EquationCompiler::splitAtReductionBoundaries(backward)) {
        if (stage.kind == PhysicalExecutionStage::Kind::Matmul) {
            EXPECT_NO_THROW((void)EquationCompiler::compileMatmul(stage.expr, stage.outputs));
        }
    }
}

TEST(MatmulAutoDiffDType, PromotedLogicalInputUsesMaterializedBf16ForBackwardCast) {
    expectPromotedLogicalOperandUsesMaterializedLowPrecisionForBackwardCast(DataType::BF16);
}

TEST(MatmulAutoDiffDType, PromotedLogicalInputUsesMaterializedFp16ForBackwardCast) {
    expectPromotedLogicalOperandUsesMaterializedLowPrecisionForBackwardCast(DataType::FP16);
}

void expectPromotedLogicalGemmOperandUsesMaterializedLowPrecisionForBackwardCast(DataType operand_dtype) {
    const Expression lhs = Expression::input("lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("rhs", operand_dtype, operand_dtype);
    const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward =
        Expression::outputs(
            {{"out", Expression::gemm(lhs, rhs, bias, 1.0, 1.0, false, false, false, DataType::FP32, DataType::FP32)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {operand_dtype, operand_dtype, DataType::FP32});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs", "bias"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, DataType>{{"out", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
            {"bias", {4}},
        });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", operand_dtype},
                                                {"rhs", operand_dtype},
                                                {"bias", DataType::FP32},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, operand_dtype);
    ASSERT_EQ(casts.size(), 2u);
    size_t shared_gradient_casts = 0;
    size_t promoted_operand_casts = 0;
    for (uint32_t cast : casts) {
        const size_t consumer_count = matmulConsumers(backward, cast).size();
        shared_gradient_casts += consumer_count == 2u ? 1u : 0u;
        promoted_operand_casts += consumer_count == 1u ? 1u : 0u;
    }
    EXPECT_EQ(shared_gradient_casts, 1u);
    EXPECT_EQ(promoted_operand_casts, 1u);

    const uint32_t bias_grad_idx = namedOutput(backward, "bias_grad").node_idx;
    for (uint32_t cast : casts) {
        EXPECT_FALSE(dependsOnNode(*backward.expr, bias_grad_idx, cast));
    }

    for (const PhysicalExecutionStage& stage : EquationCompiler::splitAtReductionBoundaries(backward)) {
        if (stage.kind == PhysicalExecutionStage::Kind::Matmul) {
            EXPECT_NO_THROW((void)EquationCompiler::compileMatmul(stage.expr, stage.outputs));
        }
    }
}

TEST(MatmulAutoDiffDType, PromotedLogicalGemmInputUsesMaterializedBf16ForBackwardCast) {
    expectPromotedLogicalGemmOperandUsesMaterializedLowPrecisionForBackwardCast(DataType::BF16);
}

TEST(MatmulAutoDiffDType, PromotedLogicalGemmInputUsesMaterializedFp16ForBackwardCast) {
    expectPromotedLogicalGemmOperandUsesMaterializedLowPrecisionForBackwardCast(DataType::FP16);
}

TEST(MatmulAutoDiffDType, UntypedBf16OutputSeedIsNormalizedBeforeRuntimeFp32Resolution) {
    const Expression lhs = Expression::input("lhs", DataType::BF16, DataType::BF16);
    const Expression rhs = Expression::input("rhs", DataType::BF16, DataType::BF16);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::BF16)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::BF16, DataType::BF16});

    // Match the deferred CustomLayer path: the synthetic upstream input has no dtype
    // annotation while autodiff builds the graph.  Its actual FP32 dtype is supplied
    // only when the backward expression is resolved/stamped.
    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
        });

    const std::vector<uint32_t> casts_before_resolution =
        castNodesWithOutputDType(backward, DataType::BF16);
    ASSERT_EQ(casts_before_resolution.size(), 1u);
    EXPECT_EQ(matmulConsumers(backward, casts_before_resolution.front()).size(), 2u);

    EXPECT_NO_THROW(resolveOutputsDTypesInPlace(backward,
                                                inputDTypes(backward,
                                                            {
                                                                {"lhs", DataType::BF16},
                                                                {"rhs", DataType::BF16},
                                                                {"dout", DataType::FP32},
                                                            })));
}

TEST(MatmulAutoDiffDType, UntypedFp16OutputSeedIsNormalizedBeforeRuntimeFp32Resolution) {
    const Expression lhs = Expression::input("lhs", DataType::FP16, DataType::FP16);
    const Expression rhs = Expression::input("rhs", DataType::FP16, DataType::FP16);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::FP16)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP16, DataType::FP16});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
        });

    const std::vector<uint32_t> casts_before_resolution =
        castNodesWithOutputDType(backward, DataType::FP16);
    ASSERT_EQ(casts_before_resolution.size(), 1u);
    EXPECT_EQ(matmulConsumers(backward, casts_before_resolution.front()).size(), 2u);

    EXPECT_NO_THROW(resolveOutputsDTypesInPlace(backward,
                                                inputDTypes(backward,
                                                            {
                                                                {"lhs", DataType::FP16},
                                                                {"rhs", DataType::FP16},
                                                                {"dout", DataType::FP32},
                                                            })));
}

TEST(MatmulAutoDiffDType, RuntimeBf16SeedAvoidsCastDespiteWiderForwardBackwardPreference) {
    const Expression lhs = Expression::input("lhs", DataType::BF16, DataType::BF16);
    const Expression rhs = Expression::input("rhs", DataType::BF16, DataType::BF16);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::BF16)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::BF16, DataType::BF16});
    forward.expr->nodes.at(namedOutput(forward, "out").node_idx).backward_output_dtype = DataType::FP32;

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"lhs", "rhs"},
        std::unordered_map<std::string, std::string>{{"out", "dout"}},
        std::unordered_map<std::string, DataType>{{"out", DataType::BF16}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"lhs", {2, 3}},
            {"rhs", {3, 4}},
        });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::BF16},
                                                {"rhs", DataType::BF16},
                                                {"dout", DataType::BF16},
                                            }));

    EXPECT_TRUE(castNodesWithOutputDType(backward, DataType::BF16).empty());
}

TEST(MatmulAutoDiffDType, ActivationBackwardRunsBeforeLowPrecisionOutputGradientCast) {
    const Expression lhs = Expression::input("lhs", DataType::BF16, DataType::BF16);
    const Expression rhs = Expression::input("rhs", DataType::BF16, DataType::BF16);
    const Expression preactivation = Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::FP32);
    PhysicalOutputs forward = Expression::outputs({{"out", preactivation.max(Expression(0.0))}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::BF16, DataType::BF16});

    PhysicalOutputs backward = buildBackwardOutputs(forward,
                                                    {"lhs", "rhs"},
                                                    std::optional<std::string>{"dout"},
                                                    std::unordered_map<std::string, std::vector<uint64_t>>{
                                                        {"lhs", {2, 3}},
                                                        {"rhs", {3, 4}},
                                                    });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::BF16},
                                                {"rhs", DataType::BF16},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, DataType::BF16);
    ASSERT_EQ(casts.size(), 1u);
    const ExprNode& cast = backward.expr->nodes.at(casts.front());
    ASSERT_NE(cast.lhs, UINT32_MAX);
    EXPECT_EQ(backward.expr->nodes.at(cast.lhs).op, ExprOp::MUL)
        << "The activation derivative must be applied in FP32 before dY is converted for the matrix-gradient GEMMs.";
}

TEST(MatmulAutoDiffDType, GemmBiasGradientRemainsFp32WhileMatrixGradientsShareBf16Cast) {
    const Expression lhs = Expression::input("lhs", DataType::BF16, DataType::BF16);
    const Expression rhs = Expression::input("rhs", DataType::BF16, DataType::BF16);
    const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward =
        Expression::outputs({{"out", Expression::gemm(lhs, rhs, bias, 1.0, 1.0, false, false, false, DataType::FP32, DataType::FP32)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::BF16, DataType::BF16, DataType::FP32});

    PhysicalOutputs backward = buildBackwardOutputs(forward,
                                                    {"lhs", "rhs", "bias"},
                                                    std::optional<std::string>{"dout"},
                                                    std::unordered_map<std::string, std::vector<uint64_t>>{
                                                        {"lhs", {2, 3}},
                                                        {"rhs", {3, 4}},
                                                        {"bias", {4}},
                                                    });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::BF16},
                                                {"rhs", DataType::BF16},
                                                {"bias", DataType::FP32},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, DataType::BF16);
    ASSERT_EQ(casts.size(), 1u);
    EXPECT_EQ(matmulConsumers(backward, casts.front()).size(), 2u);

    const ExprNode& bias_grad = backward.expr->nodes.at(namedOutput(backward, "bias_grad").node_idx);
    ASSERT_TRUE(bias_grad.output_dtype.has_value());
    EXPECT_EQ(bias_grad.output_dtype.value(), DataType::FP32);
    EXPECT_FALSE(dependsOnNode(*backward.expr, namedOutput(backward, "bias_grad").node_idx, casts.front()));
}

TEST(MatmulAutoDiffDType, GemmFp32ResidualGradientBypassesBf16MatrixGradientCast) {
    const Expression lhs = Expression::input("lhs", DataType::BF16, DataType::BF16);
    const Expression rhs = Expression::input("rhs", DataType::BF16, DataType::BF16);
    const Expression residual = Expression::input("residual", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward =
        Expression::outputs(
            {{"out", Expression::gemm(lhs, rhs, residual, 1.0, 1.0, false, false, false, DataType::FP32, DataType::FP32)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::BF16, DataType::BF16, DataType::FP32});

    PhysicalOutputs backward = buildBackwardOutputs(forward,
                                                    {"lhs", "rhs", "residual"},
                                                    std::optional<std::string>{"dout"},
                                                    std::unordered_map<std::string, std::vector<uint64_t>>{
                                                        {"lhs", {2, 3}},
                                                        {"rhs", {3, 4}},
                                                        {"residual", {2, 4}},
                                                    });
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::BF16},
                                                {"rhs", DataType::BF16},
                                                {"residual", DataType::FP32},
                                                {"dout", DataType::FP32},
                                            }));

    const std::vector<uint32_t> casts = castNodesWithOutputDType(backward, DataType::BF16);
    ASSERT_EQ(casts.size(), 1u);

    const uint32_t residual_grad_idx = namedOutput(backward, "residual_grad").node_idx;
    const ExprNode& residual_grad = backward.expr->nodes.at(residual_grad_idx);
    ASSERT_TRUE(residual_grad.output_dtype.has_value());
    EXPECT_EQ(residual_grad.output_dtype.value(), DataType::FP32);
    EXPECT_FALSE(dependsOnNode(*backward.expr, residual_grad_idx, casts.front()));
}

TEST(MatmulAutoDiffDType, OrdinaryBf16OutputDoesNotInsertAnExtraCast) {
    PhysicalOutputs backward = buildRegularMatmulBackward(DataType::BF16, DataType::BF16);
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::BF16},
                                                {"rhs", DataType::BF16},
                                                {"dout", DataType::BF16},
                                            }));
    EXPECT_TRUE(castNodesWithOutputDType(backward, DataType::BF16).empty());
}

TEST(MatmulAutoDiffDType, OrdinaryFp32MatmulBackwardDoesNotInsertAnExtraCast) {
    PhysicalOutputs backward = buildRegularMatmulBackward(DataType::FP32, DataType::FP32);
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"lhs", DataType::FP32},
                                                {"rhs", DataType::FP32},
                                                {"dout", DataType::FP32},
                                            }));
    EXPECT_TRUE(castNodesWithOutputDType(backward, DataType::BF16).empty());
    EXPECT_TRUE(castNodesWithOutputDType(backward, DataType::FP16).empty());
}

TEST(MatmulCompileDiagnostics, UnsupportedMixedOperandsIdentifyStageInputs) {
    PhysicalExpression expr;
    expr.inputs = {
        NamedInput{"dout", 0, NamedInput::Kind::Tensor},
        NamedInput{"weights", 1, NamedInput::Kind::Tensor},
    };

    ExprNode lhs;
    lhs.op = ExprOp::INPUT;
    lhs.input_slot = 0;
    lhs.input_tensor_dtype = DataType::FP32;
    lhs.output_dtype = DataType::FP32;
    lhs.compute_dtype = DataType::FP32;
    expr.nodes.push_back(lhs);

    ExprNode rhs;
    rhs.op = ExprOp::INPUT;
    rhs.input_slot = 1;
    rhs.input_tensor_dtype = DataType::BF16;
    rhs.output_dtype = DataType::BF16;
    rhs.compute_dtype = DataType::FP32;
    expr.nodes.push_back(rhs);

    ExprNode matmul;
    matmul.op = ExprOp::MATMUL;
    matmul.lhs = 0;
    matmul.rhs = 1;
    matmul.output_dtype = DataType::FP32;
    matmul.compute_dtype = DataType::FP32;
    expr.nodes.push_back(matmul);
    expr.output_node = 2;

    try {
        (void)EquationCompiler::compileMatmul(expr);
        FAIL() << "Expected mixed FP32/BF16 matmul compilation to fail.";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("name='dout'"), std::string::npos);
        EXPECT_NE(message.find("storage=fp32"), std::string::npos);
        EXPECT_NE(message.find("name='weights'"), std::string::npos);
        EXPECT_NE(message.find("storage=bf16"), std::string::npos);
        EXPECT_NE(message.find("output=fp32"), std::string::npos);
    }
}
