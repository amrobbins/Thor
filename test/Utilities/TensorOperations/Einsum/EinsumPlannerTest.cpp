#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

std::vector<int32_t> labels(const std::string& text) {
    std::vector<int32_t> result;
    result.reserve(text.size());
    for (char c : text) {
        result.push_back(EinsumParser::labelId(c));
    }
    return result;
}

}  // namespace

TEST(EinsumPlanner, PlansDirectGemm) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ik,kj->ij", {{2, 3}, {3, 4}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::GEMM);
    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;

    EXPECT_TRUE(gemm.batch_labels.empty());
    EXPECT_EQ(gemm.lhs_free_labels, labels("i"));
    EXPECT_EQ(gemm.contraction_labels, labels("k"));
    EXPECT_EQ(gemm.rhs_free_labels, labels("j"));
    EXPECT_EQ(gemm.canonical_output_labels, labels("ij"));
    EXPECT_EQ(gemm.batch_count, 1u);
    EXPECT_EQ(gemm.m, 2u);
    EXPECT_EQ(gemm.n, 4u);
    EXPECT_EQ(gemm.k, 3u);
    EXPECT_FALSE(gemm.lhs.transpose);
    EXPECT_FALSE(gemm.rhs.transpose);
    EXPECT_FALSE(gemm.lhs.requires_materialized_permutation);
    EXPECT_FALSE(gemm.rhs.requires_materialized_permutation);
    EXPECT_FALSE(gemm.requires_output_permutation);
    EXPECT_TRUE(gemm.direct);

    EXPECT_EQ(plan.iteration_labels, labels("ijk"));
    EXPECT_EQ(plan.iteration_dimensions, (std::vector<uint64_t>{2, 4, 3}));
    EXPECT_EQ(plan.reduction_axes, (std::vector<uint32_t>{2}));

    ASSERT_EQ(plan.operands.size(), 2u);
    EXPECT_EQ(plan.operands[0].aligned_dimensions, (std::vector<uint64_t>{2, 1, 3}));
    EXPECT_EQ(plan.operands[0].inserted_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.operands[0].broadcast_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.operands[1].aligned_dimensions, (std::vector<uint64_t>{1, 4, 3}));
    EXPECT_EQ(plan.operands[1].permutation, (std::vector<uint32_t>{1, 0}));
    EXPECT_EQ(plan.operands[1].inserted_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(plan.operands[1].broadcast_axes, (std::vector<uint32_t>{0}));
}

TEST(EinsumPlanner, UsesGemmTransposeFlagsWithoutMaterializingPermutation) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ki,jk->ij", {{3, 2}, {4, 3}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(plan.kind, EinsumPlanKind::GEMM);
    EXPECT_TRUE(gemm.lhs.transpose);
    EXPECT_TRUE(gemm.rhs.transpose);
    EXPECT_FALSE(gemm.lhs.requires_materialized_permutation);
    EXPECT_FALSE(gemm.rhs.requires_materialized_permutation);
    EXPECT_TRUE(gemm.direct);
}

TEST(EinsumPlanner, PlansDirectBatchedGemm) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("bij,bjk->bik", {{5, 2, 3}, {5, 3, 4}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(plan.kind, EinsumPlanKind::BATCHED_GEMM);
    EXPECT_EQ(gemm.batch_labels, labels("b"));
    EXPECT_EQ(gemm.lhs_free_labels, labels("i"));
    EXPECT_EQ(gemm.contraction_labels, labels("j"));
    EXPECT_EQ(gemm.rhs_free_labels, labels("k"));
    EXPECT_EQ(gemm.batch_count, 5u);
    EXPECT_EQ(gemm.m, 2u);
    EXPECT_EQ(gemm.n, 4u);
    EXPECT_EQ(gemm.k, 3u);
    EXPECT_TRUE(gemm.direct);
}

TEST(EinsumPlanner, TreatsEllipsisAsBroadcastBatchDimensions) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("...ik,...kj->...ij", {{5, 2, 3}, {3, 4}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(plan.kind, EinsumPlanKind::BATCHED_GEMM);
    ASSERT_EQ(gemm.batch_labels.size(), 1u);
    EXPECT_TRUE(EinsumParser::isEllipsisLabel(gemm.batch_labels[0]));
    EXPECT_EQ(gemm.batch_count, 5u);
    EXPECT_TRUE(gemm.lhs.inserted_axes.empty());
    EXPECT_EQ(gemm.rhs.inserted_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(gemm.rhs.broadcast_axes, (std::vector<uint32_t>{0}));
    EXPECT_FALSE(gemm.direct);
}

TEST(EinsumPlanner, RecordsSharedContractionBroadcast) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ik,kj->ij", {{2, 1}, {3, 4}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.k, 3u);
    EXPECT_EQ(gemm.lhs.broadcast_axes, (std::vector<uint32_t>{1}));
    EXPECT_TRUE(gemm.rhs.broadcast_axes.empty());
    EXPECT_FALSE(gemm.direct);
}

TEST(EinsumPlanner, RecordsRequestedOutputPermutation) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ik,kj->ji", {{2, 3}, {3, 4}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.canonical_output_labels, labels("ij"));
    EXPECT_EQ(gemm.output_permutation, (std::vector<uint32_t>{1, 0}));
    EXPECT_TRUE(gemm.requires_output_permutation);
    EXPECT_FALSE(gemm.direct);
}

TEST(EinsumPlanner, DetectsMaterializedInputPermutation) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("acb,cd->abd", {{2, 3, 5}, {3, 7}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.lhs_free_labels, labels("ab"));
    EXPECT_EQ(gemm.contraction_labels, labels("c"));
    EXPECT_EQ(gemm.rhs_free_labels, labels("d"));
    EXPECT_EQ(gemm.m, 10u);
    EXPECT_EQ(gemm.n, 7u);
    EXPECT_EQ(gemm.k, 3u);
    EXPECT_TRUE(gemm.lhs.requires_materialized_permutation);
    EXPECT_FALSE(gemm.rhs.requires_materialized_permutation);
    EXPECT_FALSE(gemm.direct);
}

TEST(EinsumPlanner, FlattensMultipleLabelsWithinMatrixGroups) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("abc,cde->abde", {{2, 3, 5}, {5, 7, 11}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.lhs_free_labels, labels("ab"));
    EXPECT_EQ(gemm.contraction_labels, labels("c"));
    EXPECT_EQ(gemm.rhs_free_labels, labels("de"));
    EXPECT_EQ(gemm.m, 6u);
    EXPECT_EQ(gemm.n, 77u);
    EXPECT_EQ(gemm.k, 5u);
    EXPECT_TRUE(gemm.direct);
}


TEST(EinsumPlanner, ChoosesLhsPhysicalOrderForMultiAxisContraction) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("iacb,cbj->iaj", {{2, 3, 5, 7}, {5, 7, 11}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.lhs_free_labels, labels("ia"));
    EXPECT_EQ(gemm.contraction_labels, labels("cb"));
    EXPECT_EQ(gemm.rhs_free_labels, labels("j"));
    EXPECT_EQ(gemm.m, 6u);
    EXPECT_EQ(gemm.k, 35u);
    EXPECT_EQ(gemm.n, 11u);
    EXPECT_TRUE(gemm.direct);
}

TEST(EinsumPlanner, DiagonalExtractionCanFeedGemmButMakesItNonDirect) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("iik,kj->ij", {{2, 2, 3}, {3, 4}});

    ASSERT_EQ(plan.operands.size(), 2u);
    ASSERT_EQ(plan.operands[0].diagonals.size(), 1u);
    EXPECT_EQ(plan.operands[0].diagonals[0].label, EinsumParser::labelId('i'));
    EXPECT_EQ(plan.operands[0].diagonals[0].source_axes, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(plan.operands[0].diagonalized_labels, labels("ik"));
    EXPECT_EQ(plan.operands[0].diagonalized_dimensions, (std::vector<uint64_t>{2, 3}));

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    EXPECT_EQ(plan.kind, EinsumPlanKind::GEMM);
    EXPECT_FALSE(plan.matrix_multiply->direct);
}

TEST(EinsumPlanner, ClassifiesUnaryPermutation) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ij->ji", {{2, 3}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::UNARY);
    EXPECT_FALSE(plan.matrix_multiply.has_value());
    ASSERT_EQ(plan.operands.size(), 1u);
    EXPECT_EQ(plan.operands[0].permutation, (std::vector<uint32_t>{1, 0}));
    EXPECT_TRUE(plan.operands[0].requiresPermutation());
    EXPECT_TRUE(plan.reduction_axes.empty());
}

TEST(EinsumPlanner, ClassifiesUnaryReductionAndMakesReductionAxesTrailing) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ijk->ki", {{2, 3, 5}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::REDUCTION);
    EXPECT_EQ(plan.iteration_labels, labels("kij"));
    EXPECT_EQ(plan.iteration_dimensions, (std::vector<uint64_t>{5, 2, 3}));
    EXPECT_EQ(plan.reduction_axes, (std::vector<uint32_t>{2}));
    ASSERT_EQ(plan.operands.size(), 1u);
    EXPECT_EQ(plan.operands[0].permutation, (std::vector<uint32_t>{2, 0, 1}));
}

TEST(EinsumPlanner, ClassifiesOuterProductAsElementwiseBroadcast) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("i,j->ij", {{2}, {3}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::ELEMENTWISE);
    EXPECT_FALSE(plan.matrix_multiply.has_value());
    EXPECT_TRUE(plan.reduction_axes.empty());
    ASSERT_EQ(plan.operands.size(), 2u);
    EXPECT_EQ(plan.operands[0].aligned_dimensions, (std::vector<uint64_t>{2, 1}));
    EXPECT_EQ(plan.operands[0].inserted_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.operands[0].broadcast_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.operands[1].aligned_dimensions, (std::vector<uint64_t>{1, 3}));
    EXPECT_EQ(plan.operands[1].inserted_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(plan.operands[1].broadcast_axes, (std::vector<uint32_t>{0}));
}

TEST(EinsumPlanner, OneSidedReductionFallsBackToGeneralPlan) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("irk,kj->ij", {{2, 5, 3}, {3, 4}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::GENERAL);
    EXPECT_FALSE(plan.matrix_multiply.has_value());
    EXPECT_EQ(plan.equation.reduction_labels, labels("kr"));
    EXPECT_EQ(plan.reduction_axes, (std::vector<uint32_t>{2, 3}));
}

TEST(EinsumPlanner, MoreThanTwoOperandsFallsBackToGeneralPlan) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ik,kj,jl->il", {{2, 3}, {3, 4}, {4, 5}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::GENERAL);
    EXPECT_FALSE(plan.matrix_multiply.has_value());
    EXPECT_EQ(plan.equation.reduction_labels, labels("jk"));
}

TEST(EinsumPlanner, SharedOutputLabelsBecomeBatchGroupsEvenWithoutEllipsis) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("zab,zbc->zac", {{7, 2, 3}, {7, 3, 5}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.batch_labels, labels("z"));
    EXPECT_EQ(gemm.batch_count, 7u);
    EXPECT_EQ(plan.kind, EinsumPlanKind::BATCHED_GEMM);
    EXPECT_TRUE(gemm.direct);
}

TEST(EinsumPlanner, InterleavedRequestedBatchOutputRequiresOutputPermutation) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("bia,bac->ibc", {{5, 2, 3}, {5, 3, 7}});

    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.batch_labels, labels("b"));
    EXPECT_EQ(gemm.lhs_free_labels, labels("i"));
    EXPECT_EQ(gemm.rhs_free_labels, labels("c"));
    EXPECT_EQ(gemm.canonical_output_labels, labels("bic"));
    EXPECT_EQ(gemm.output_permutation, (std::vector<uint32_t>{1, 0, 2}));
    EXPECT_TRUE(gemm.requires_output_permutation);
    EXPECT_FALSE(gemm.direct);
}

TEST(EinsumPlanner, RejectsStaleResolvedShapes) {
    const ResolvedEinsumEquation resolved = EinsumParser::parseAndResolve("ik,kj->ij", {{2, 3}, {3, 4}});

    EXPECT_THROW((void)EinsumPlanner::plan(resolved, {{2, 1}, {1, 4}}), std::invalid_argument);
}

TEST(EinsumPlanner, DetectsFlattenedDimensionOverflow) {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    EXPECT_THROW((void)EinsumPlanner::parseAndPlan("abx,xc->abc", {{max, 2, 3}, {3, 4}}), std::overflow_error);
}
