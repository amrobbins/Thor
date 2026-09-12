#include "Utilities/Expression/LogicalExpressionFwd.h"
#include "Utilities/Expression/CudaKernelExpression.h"

#include <memory>
#include <type_traits>
#include <vector>

#include "Utilities/Expression/ExpressionInternal.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

static_assert(std::is_same_v<LogicalExpression, std::shared_ptr<const LogicalExpressionNode>>);
static_assert(std::is_same_v<LogicalCudaKernelApplicationPtr, std::shared_ptr<const LogicalCudaKernelApplication>>);

template <typename T>
concept HasLegacyBuilderMaterializer = requires(const Expression& expression) {
    T::materializeForLegacyPhysicalBuilder(expression);
};

static_assert(!HasLegacyBuilderMaterializer<ExpressionInternalAccess>,
              "Expression must not expose a logical->physical legacy builder materialization API.");

template <typename T>
concept HasPublicPhysicalNodeImporter = requires(std::shared_ptr<PhysicalExpression> graph, uint32_t root) {
    T::fromPhysicalNode(graph, root);
};

template <typename T>
concept HasInternalSingleRootPhysicalImporter = requires(std::shared_ptr<PhysicalExpression> graph) {
    T::importPhysicalRoot(graph, uint32_t{0});
};

template <typename T>
concept HasInternalMultiRootPhysicalImporter = requires(std::shared_ptr<PhysicalExpression> graph,
                                                        const std::vector<uint32_t>& roots) {
    T::importPhysicalRoots(graph, roots);
};

static_assert(!HasPublicPhysicalNodeImporter<Expression>,
              "Expression must not expose a public root-by-root physical import seam.");
static_assert(!HasInternalSingleRootPhysicalImporter<ExpressionInternalAccess>,
              "Physical graph import must remain graph-scoped through Outputs::fromPhysicalOutputs().");
static_assert(!HasInternalMultiRootPhysicalImporter<ExpressionInternalAccess>,
              "Physical graph import must remain graph-scoped through Outputs::fromPhysicalOutputs().");

TEST(ExpressionInternalAccess, PhysicalSingleRootImportPreservesRootSemantics) {
    const Expression expression = Expression::input("x").reshape({2, 3});
    const PhysicalOutputs physical = Expression::outputs({{"output", expression}}).physicalOutputs();
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 1u);

    const Outputs importedOutputs = Outputs::fromPhysicalOutputs(physical);
    const Expression imported = importedOutputs.outputExpression("output");
    const ExprNode& semantics = ExpressionInternalAccess::rootSemantics(imported);

    EXPECT_EQ(ExpressionInternalAccess::rootOp(imported), ExprOp::RESHAPE);
    EXPECT_EQ(semantics.op, ExprOp::RESHAPE);
    EXPECT_EQ(semantics.reshape_dims, (std::vector<uint64_t>{2, 3}));
}

TEST(ExpressionInternalAccess, MultiRootPhysicalImportPreservesSharedLogicalAncestry) {
    const Expression x = Expression::input("x");
    const PhysicalOutputs physical = Expression::outputs({{"sin", x.sin()}, {"cos", x.cos()}}).physicalOutputs();
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 2u);

    const Outputs importedOutputs = Outputs::fromPhysicalOutputs(physical);
    const Expression importedSin = importedOutputs.outputExpression("sin");
    const Expression importedCos = importedOutputs.outputExpression("cos");

    const std::optional<Expression> first_lhs = ExpressionInternalAccess::lhsDependency(importedSin);
    const std::optional<Expression> second_lhs = ExpressionInternalAccess::lhsDependency(importedCos);
    ASSERT_TRUE(first_lhs.has_value());
    ASSERT_TRUE(second_lhs.has_value());
    EXPECT_TRUE(first_lhs->isSameLogicalNode(*second_lhs));
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*first_lhs), ExprOp::INPUT);
}

TEST(ExpressionInternalAccess, GraphScopedPhysicalImportKeepsDistinctRootsDistinct) {
    const Expression x = Expression::input("x");
    const PhysicalOutputs physical = Expression::outputs({{"sin", x.sin()}, {"cos", x.cos()}}).physicalOutputs();
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 2u);
    ASSERT_NE(physical.outputs[0].node_idx, physical.outputs[1].node_idx);

    const Outputs importedOutputs = Outputs::fromPhysicalOutputs(physical);
    const Expression importedSin = importedOutputs.outputExpression("sin");
    const Expression importedCos = importedOutputs.outputExpression("cos");

    EXPECT_FALSE(importedSin.isSameLogicalNode(importedCos));
    EXPECT_EQ(ExpressionInternalAccess::rootOp(importedSin), ExprOp::SIN);
    EXPECT_EQ(ExpressionInternalAccess::rootOp(importedCos), ExprOp::COS);
}

TEST(ExpressionInternalAccess, LhsDependencyReturnsTheAuthoredLogicalDependency) {
    const Expression input = Expression::input("x");
    const Expression expression = input.sin();
    const std::optional<Expression> lhs = ExpressionInternalAccess::lhsDependency(expression);
    ASSERT_TRUE(lhs.has_value());

    EXPECT_TRUE(lhs->isSameLogicalNode(input));
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*lhs), ExprOp::INPUT);
}


TEST(ExpressionInternalAccess, InputAnnotationsDoesNotRequirePhysicalMaterialization) {
    // Two logical bindings may intentionally share a textual name while carrying
    // different input kinds. Logical inspection can still reason about them, but
    // physical lowering cannot assign one PhysicalExpression input slot both kinds.
    // This makes the test a direct guard that inputAnnotations() stays logical-only.
    const Expression tensor = Expression::input("shared", DataType::FP32, DataType::BF16);
    const Expression runtime = Expression::runtimeScalar("shared", DataType::FP32, DataType::FP32);
    const Expression expression = tensor + runtime;

    const std::vector<ExpressionInputAnnotations> annotations =
        ExpressionInternalAccess::inputAnnotations(expression, "shared");
    ASSERT_EQ(annotations.size(), 2u);
    EXPECT_NE(annotations[0], annotations[1]);

    EXPECT_THROW((void)Expression::outputs({{"y", expression}}).physicalOutputs(), std::runtime_error);
}

TEST(ExpressionInternalAccess, InputAnnotationsTraverseLogicalDependenciesOnceAndReportMissingInputs) {
    const Expression input = Expression::input("x", DataType::FP32, DataType::BF16);
    const Expression shared = input.sin();
    const Expression expression = (shared + shared.cos()).tanh();

    const std::vector<ExpressionInputAnnotations> annotations =
        ExpressionInternalAccess::inputAnnotations(expression, "x");
    ASSERT_EQ(annotations.size(), 1u);
    EXPECT_EQ(annotations.front().computeDataType, DataType::FP32);
    EXPECT_EQ(annotations.front().outputDataType, DataType::BF16);
    EXPECT_TRUE(ExpressionInternalAccess::inputAnnotations(expression, "missing").empty());
}

TEST(ExpressionInternalAccess, InputAnnotationsExposeConsistentAndInconsistentDistinctBindings) {
    const Expression first = Expression::input("x", DataType::FP32, DataType::BF16);
    const Expression consistent = Expression::input("x", DataType::FP32, DataType::BF16);
    const Expression consistentExpression = first.sin() + consistent.cos();

    const std::vector<ExpressionInputAnnotations> consistentAnnotations =
        ExpressionInternalAccess::inputAnnotations(consistentExpression, "x");
    ASSERT_EQ(consistentAnnotations.size(), 2u);
    EXPECT_EQ(consistentAnnotations[0], consistentAnnotations[1]);

    const Expression inconsistent = Expression::input("x", DataType::BF16, DataType::BF16);
    const Expression inconsistentExpression = first.sin() + inconsistent.cos();
    const std::vector<ExpressionInputAnnotations> inconsistentAnnotations =
        ExpressionInternalAccess::inputAnnotations(inconsistentExpression, "x");
    ASSERT_EQ(inconsistentAnnotations.size(), 2u);
    EXPECT_NE(inconsistentAnnotations[0], inconsistentAnnotations[1]);
}

TEST(ExpressionInternalAccess, InputAnnotationsTraverseCustomCudaApplicationInputs) {
    const CudaKernelExpression kernel = CudaKernelExpression::builder("logical_input_annotation_probe")
        .source(R"cuda(
extern "C" __global__
void logical_input_annotation_probe_kernel(const float* x, float* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = x[i];
}
)cuda")
        .entry("logical_input_annotation_probe_kernel")
        .input("x", DataType::FP32)
        .outputLike("out", DataType::FP32, "x")
        .launchGrid1D(CudaKernelExpression::DimExpr::numel("out"), 64)
        .build();

    const Expression input = Expression::input("x", DataType::FP32, DataType::FP32);
    const Outputs applied = kernel.apply({{"x", input}});
    const Expression expression = applied.outputExpression("out").sin();

    const std::vector<ExpressionInputAnnotations> annotations =
        ExpressionInternalAccess::inputAnnotations(expression, "x");
    ASSERT_EQ(annotations.size(), 1u);
    EXPECT_EQ(annotations.front().computeDataType, DataType::FP32);
    EXPECT_EQ(annotations.front().outputDataType, DataType::FP32);
}
