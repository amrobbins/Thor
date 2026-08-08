#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
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

std::string repeatedOperandEquation(size_t operand_count) {
    std::string equation;
    for (size_t operand = 0; operand < operand_count; ++operand) {
        if (operand != 0) equation.push_back(',');
        equation.push_back('a');
    }
    equation += "->a";
    return equation;
}

std::vector<std::vector<uint64_t>> repeatedUnitShapes(size_t operand_count) {
    return std::vector<std::vector<uint64_t>>(operand_count, std::vector<uint64_t>{1});
}

EinsumLogicalOperandPlan denseLogicalOperand(const std::string& label_text,
                                             std::vector<uint64_t> dimensions,
                                             std::vector<uint32_t> provenance,
                                             DataType storage_dtype = DataType::BF16) {
    EinsumLogicalOperandPlan operand;
    operand.labels = labels(label_text);
    operand.dimensions = std::move(dimensions);
    operand.strides_elements.assign(operand.dimensions.size(), 1);
    for (size_t axis = operand.dimensions.size(); axis > 1; --axis) {
        operand.strides_elements[axis - 2] = operand.strides_elements[axis - 1] * operand.dimensions[axis - 1];
    }
    operand.storage_dtype = storage_dtype;
    operand.source_operand_indices = std::move(provenance);
    operand.dense_storage = true;
    return operand;
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

TEST(EinsumPlanner, FullyBroadcastSharedContractionNormalizesToPairProduct) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ik,kj->ij", {{2, 1}, {3, 4}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::PAIR_PRODUCT);
    EXPECT_FALSE(plan.matrix_multiply.has_value());
    ASSERT_TRUE(plan.pair_product.has_value());
    EXPECT_TRUE(plan.pair_product->lhs_reduction_labels.empty());
    EXPECT_EQ(plan.pair_product->lhs_broadcast_elision_labels, labels("k"));
    EXPECT_TRUE(plan.pair_product->rhs_broadcast_elision_labels.empty());
    EXPECT_EQ(plan.pair_product->rhs_reduction_labels, labels("k"));
}

TEST(EinsumPlanner, PartialSharedContractionBroadcastReducesBeforeRemainingGemm) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ikl,klj->ij", {{2, 1, 5}, {3, 5, 4}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::GEMM);
    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.contraction_labels, labels("l"));
    EXPECT_EQ(gemm.k, 5u);
    EXPECT_EQ(gemm.lhs_broadcast_elision_labels, labels("k"));
    EXPECT_TRUE(gemm.lhs_reduction_labels.empty());
    EXPECT_TRUE(gemm.rhs_broadcast_elision_labels.empty());
    EXPECT_EQ(gemm.rhs_reduction_labels, labels("k"));
    EXPECT_TRUE(gemm.lhs.broadcast_axes.empty());
    EXPECT_TRUE(gemm.rhs.broadcast_axes.empty());
    EXPECT_FALSE(gemm.direct);
}

TEST(EinsumPlanner, OppositeSharedBroadcastsCanNormalizeEveryKToPairProduct) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ikl,klj->ij", {{2, 1, 5}, {3, 1, 4}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::PAIR_PRODUCT);
    EXPECT_FALSE(plan.matrix_multiply.has_value());
    ASSERT_TRUE(plan.pair_product.has_value());
    EXPECT_EQ(plan.pair_product->lhs_reduction_labels, labels("l"));
    EXPECT_EQ(plan.pair_product->lhs_broadcast_elision_labels, labels("k"));
    EXPECT_EQ(plan.pair_product->rhs_broadcast_elision_labels, labels("l"));
    EXPECT_EQ(plan.pair_product->rhs_reduction_labels, labels("k"));
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
    ASSERT_TRUE(plan.pair_product.has_value());
    EXPECT_TRUE(plan.pair_product->lhs_reduction_labels.empty());
    EXPECT_TRUE(plan.pair_product->rhs_reduction_labels.empty());
    EXPECT_TRUE(plan.reduction_axes.empty());
    ASSERT_EQ(plan.operands.size(), 2u);
    EXPECT_EQ(plan.operands[0].aligned_dimensions, (std::vector<uint64_t>{2, 1}));
    EXPECT_EQ(plan.operands[0].inserted_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.operands[0].broadcast_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.operands[1].aligned_dimensions, (std::vector<uint64_t>{1, 3}));
    EXPECT_EQ(plan.operands[1].inserted_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(plan.operands[1].broadcast_axes, (std::vector<uint32_t>{0}));
}

TEST(EinsumPlanner, PlansIndependentReductionsBeforePairProduct) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ir,sj->ij", {{2, 3}, {5, 7}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::PAIR_PRODUCT);
    EXPECT_FALSE(plan.matrix_multiply.has_value());
    ASSERT_TRUE(plan.pair_product.has_value());
    EXPECT_EQ(plan.pair_product->lhs_reduction_labels, labels("r"));
    EXPECT_EQ(plan.pair_product->rhs_reduction_labels, labels("s"));
}

TEST(EinsumPlanner, SharedReductionLabelIsNotPairProduct) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ik,kj->ij", {{2, 3}, {3, 4}});

    EXPECT_TRUE(plan.matrix_multiply.has_value());
    EXPECT_FALSE(plan.pair_product.has_value());
}

TEST(EinsumPlanner, OneSidedReductionPlansPreReductionThenGemm) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("irk,kj->ij", {{2, 5, 3}, {3, 4}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::GEMM);
    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.lhs_free_labels, labels("i"));
    EXPECT_EQ(gemm.lhs_reduction_labels, labels("r"));
    EXPECT_EQ(gemm.contraction_labels, labels("k"));
    EXPECT_TRUE(gemm.rhs_reduction_labels.empty());
    EXPECT_EQ(gemm.rhs_free_labels, labels("j"));
    EXPECT_FALSE(gemm.lhs.requires_materialized_permutation);
    EXPECT_FALSE(gemm.rhs.requires_materialized_permutation);
    EXPECT_FALSE(gemm.direct);
    EXPECT_EQ(plan.equation.reduction_labels, labels("kr"));
    EXPECT_EQ(plan.reduction_axes, (std::vector<uint32_t>{2, 3}));
}

TEST(EinsumPlanner, BothOperandsMayPreReduceBeforeGemm) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("irk,ksj->ij", {{2, 5, 3}, {3, 7, 4}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::GEMM);
    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.lhs_reduction_labels, labels("r"));
    EXPECT_EQ(gemm.contraction_labels, labels("k"));
    EXPECT_EQ(gemm.rhs_reduction_labels, labels("s"));
    EXPECT_FALSE(gemm.direct);
}

TEST(EinsumPlanner, BatchedGemmMayConsumeAPreReducedOperand) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("birk,bkj->bij", {{2, 3, 5, 7}, {2, 7, 11}});

    EXPECT_EQ(plan.kind, EinsumPlanKind::BATCHED_GEMM);
    ASSERT_TRUE(plan.matrix_multiply.has_value());
    const EinsumMatrixMultiplyPlan& gemm = *plan.matrix_multiply;
    EXPECT_EQ(gemm.batch_labels, labels("b"));
    EXPECT_EQ(gemm.lhs_reduction_labels, labels("r"));
    EXPECT_EQ(gemm.contraction_labels, labels("k"));
    EXPECT_EQ(gemm.batch_count, 2u);
    EXPECT_FALSE(gemm.direct);
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

TEST(EinsumPlanner, RecordsPlannerSideLogicalInputPhysicalMetadata) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("iik,kj->ij", {{2, 2, 3}, {3, 4}});

    ASSERT_EQ(plan.logical_operands.size(), 2u);
    const EinsumLogicalOperandPlan& lhs = plan.logical_operands[0];
    EXPECT_EQ(lhs.labels, labels("ik"));
    EXPECT_EQ(lhs.dimensions, (std::vector<uint64_t>{2, 3}));
    // Dense [i,i,k] has source strides [6,3,1], so diagonal i advances
    // through both repeated source axes: 6 + 3 = 9.
    EXPECT_EQ(lhs.strides_elements, (std::vector<uint64_t>{9, 1}));
    EXPECT_FALSE(lhs.dense_storage);
    EXPECT_TRUE(lhs.diagonal_view);
    EXPECT_EQ(lhs.source_operand_indices, (std::vector<uint32_t>{0}));
    EXPECT_FALSE(lhs.storage_dtype.has_value());

    const EinsumLogicalOperandPlan& rhs = plan.logical_operands[1];
    EXPECT_EQ(rhs.labels, labels("kj"));
    EXPECT_EQ(rhs.dimensions, (std::vector<uint64_t>{3, 4}));
    EXPECT_EQ(rhs.strides_elements, (std::vector<uint64_t>{4, 1}));
    EXPECT_TRUE(rhs.dense_storage);
    EXPECT_FALSE(rhs.diagonal_view);
    EXPECT_EQ(rhs.source_operand_indices, (std::vector<uint32_t>{1}));
}

TEST(EinsumPlanner, ReusablePairPlannerAcceptsIntermediateOperands) {
    const EinsumLogicalOperandPlan lhs = denseLogicalOperand("ik", {2, 3}, {0, 1});
    const EinsumLogicalOperandPlan rhs = denseLogicalOperand("kj", {3, 4}, {2});

    const EinsumPairContractionPlan pair = EinsumPlanner::planPair(lhs, rhs, labels("ij"));

    EXPECT_EQ(pair.kind, EinsumPlanKind::GEMM);
    ASSERT_TRUE(pair.matrix_multiply.has_value());
    EXPECT_FALSE(pair.pair_product.has_value());
    EXPECT_EQ(pair.reduction_labels, labels("k"));
    EXPECT_EQ(pair.matrix_multiply->lhs_free_labels, labels("i"));
    EXPECT_EQ(pair.matrix_multiply->contraction_labels, labels("k"));
    EXPECT_EQ(pair.matrix_multiply->rhs_free_labels, labels("j"));
    EXPECT_EQ(pair.matrix_multiply->m, 2u);
    EXPECT_EQ(pair.matrix_multiply->k, 3u);
    EXPECT_EQ(pair.matrix_multiply->n, 4u);

    EXPECT_EQ(pair.result.labels, labels("ij"));
    EXPECT_EQ(pair.result.dimensions, (std::vector<uint64_t>{2, 4}));
    EXPECT_EQ(pair.result.strides_elements, (std::vector<uint64_t>{4, 1}));
    ASSERT_TRUE(pair.result.storage_dtype.has_value());
    EXPECT_EQ(*pair.result.storage_dtype, DataType::BF16);
    EXPECT_EQ(pair.result.source_operand_indices, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_TRUE(pair.result.dense_storage);
    EXPECT_FALSE(pair.result.diagonal_view);
    ASSERT_TRUE(pair.reduction_accumulation_dtype.has_value());
    EXPECT_EQ(*pair.reduction_accumulation_dtype, DataType::FP32);
}

TEST(EinsumPlanner, ReusablePairPlannerPreservesIndependentReductionPairProductSemantics) {
    const EinsumLogicalOperandPlan lhs = denseLogicalOperand("ir", {2, 3}, {0, 1});
    const EinsumLogicalOperandPlan rhs = denseLogicalOperand("sj", {5, 7}, {2, 3});

    const EinsumPairContractionPlan pair = EinsumPlanner::planPair(lhs, rhs, labels("ij"));

    EXPECT_EQ(pair.kind, EinsumPlanKind::PAIR_PRODUCT);
    EXPECT_FALSE(pair.matrix_multiply.has_value());
    ASSERT_TRUE(pair.pair_product.has_value());
    EXPECT_EQ(pair.pair_product->lhs_reduction_labels, labels("r"));
    EXPECT_EQ(pair.pair_product->rhs_reduction_labels, labels("s"));
    EXPECT_EQ(pair.result.dimensions, (std::vector<uint64_t>{2, 7}));
    EXPECT_EQ(pair.result.source_operand_indices, (std::vector<uint32_t>{0, 1, 2, 3}));
}

TEST(EinsumPlanner, ReusablePairPlannerRejectsOverlappingProvenanceAndMismatchedStorageDtypes) {
    EinsumLogicalOperandPlan lhs = denseLogicalOperand("ik", {2, 3}, {0, 1}, DataType::BF16);
    EinsumLogicalOperandPlan overlapping_rhs = denseLogicalOperand("kj", {3, 4}, {1, 2}, DataType::BF16);
    EXPECT_THROW((void)EinsumPlanner::planPair(lhs, overlapping_rhs, labels("ij")), std::invalid_argument);

    EinsumLogicalOperandPlan disjoint_rhs = denseLogicalOperand("kj", {3, 4}, {2}, DataType::FP16);
    EXPECT_THROW((void)EinsumPlanner::planPair(lhs, disjoint_rhs, labels("ij")), std::invalid_argument);
}

TEST(EinsumPlanner, ExistingTwoOperandPlanIsProducedThroughReusablePairPlanner) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("irk,kj->ij", {{2, 5, 3}, {3, 4}});

    ASSERT_TRUE(plan.pair_contraction.has_value());
    ASSERT_TRUE(plan.matrix_multiply.has_value());
    ASSERT_TRUE(plan.pair_contraction->matrix_multiply.has_value());
    EXPECT_EQ(plan.kind, plan.pair_contraction->kind);
    EXPECT_EQ(plan.matrix_multiply->lhs_reduction_labels,
              plan.pair_contraction->matrix_multiply->lhs_reduction_labels);
    EXPECT_EQ(plan.matrix_multiply->contraction_labels,
              plan.pair_contraction->matrix_multiply->contraction_labels);
    EXPECT_EQ(plan.matrix_multiply->rhs_free_labels,
              plan.pair_contraction->matrix_multiply->rhs_free_labels);
    EXPECT_EQ(plan.pair_contraction->result.labels, labels("ij"));
    EXPECT_EQ(plan.pair_contraction->result.dimensions, (std::vector<uint64_t>{2, 4}));
}


TEST(EinsumPlanner, PhysicalCandidatesPruneCopiedEquivalentWhenZeroCopyDenseGemmExists) {
    const EinsumLogicalOperandPlan lhs = denseLogicalOperand("ik", {2, 3}, {0});
    const EinsumLogicalOperandPlan rhs = denseLogicalOperand("kj", {3, 4}, {1});

    const EinsumPairContractionPlan pair = EinsumPlanner::planPair(lhs, rhs, labels("ij"));

    ASSERT_EQ(pair.physical_candidates.size(), 2u);
    ASSERT_LT(pair.preferred_physical_candidate, pair.physical_candidates.size());
    const EinsumPairPhysicalCandidate& preferred =
        pair.physical_candidates[pair.preferred_physical_candidate];
    EXPECT_EQ(preferred.kind, EinsumPlanKind::GEMM);
    EXPECT_FALSE(preferred.swapped_gemm_orientation);
    EXPECT_FALSE(preferred.lhs_materialized);
    EXPECT_FALSE(preferred.rhs_materialized);
    EXPECT_FALSE(preferred.output_materialized);
    EXPECT_TRUE(preferred.result.dense_storage);
    EXPECT_EQ(preferred.result.strides_elements, (std::vector<uint64_t>{4, 1}));
    EXPECT_EQ(preferred.cost.matmul_fma_count, 24u);
    EXPECT_EQ(preferred.cost.matmul_group_count, 1u);
    EXPECT_EQ(preferred.cost.fused_kernel_count, 0u);
    EXPECT_EQ(preferred.cost.reduction_op_count, 0u);
    EXPECT_EQ(preferred.cost.materialization_op_count, 0u);
    EXPECT_EQ(preferred.cost.result_elements, 8u);
    ASSERT_TRUE(preferred.cost.result_bytes.has_value());
    EXPECT_EQ(*preferred.cost.result_bytes, 16u);

    // The swapped GEMM orientation is retained because it produces a distinct
    // zero-copy physical layout that may be useful to a later contraction.
    // Its otherwise-equivalent copied-back dense [i,j] candidate is dominated
    // by the natural zero-copy GEMM and therefore is pruned.
    const EinsumPairPhysicalCandidate& alternate = pair.physical_candidates[1];
    EXPECT_TRUE(alternate.swapped_gemm_orientation);
    EXPECT_FALSE(alternate.output_materialized);
    EXPECT_FALSE(alternate.result.dense_storage);
    EXPECT_EQ(alternate.result.strides_elements, (std::vector<uint64_t>{1, 2}));
    for (const EinsumPairPhysicalCandidate& candidate : pair.physical_candidates) {
        EXPECT_FALSE(candidate.output_materialized);
    }
}

TEST(EinsumPlanner, PhysicalCostUsesActualDiagonalBlasAddressability) {
    const EinsumPlan addressable_top =
        EinsumPlanner::parseAndPlan("iik,kj->ij", {{2, 2, 3}, {3, 4}});
    ASSERT_EQ(addressable_top.logical_operands.size(), 2u);
    EinsumLogicalOperandPlan addressable_lhs = addressable_top.logical_operands[0];
    EinsumLogicalOperandPlan addressable_rhs = addressable_top.logical_operands[1];
    addressable_lhs.storage_dtype = DataType::BF16;
    addressable_rhs.storage_dtype = DataType::BF16;

    const EinsumPairContractionPlan addressable =
        EinsumPlanner::planPair(addressable_lhs, addressable_rhs, labels("ij"));
    const EinsumPairPhysicalCandidate& addressable_preferred =
        addressable.physical_candidates.at(addressable.preferred_physical_candidate);
    EXPECT_FALSE(addressable_preferred.lhs_materialized);
    EXPECT_EQ(addressable_preferred.cost.lhs_materialization_elements, 0u);
    ASSERT_TRUE(addressable_preferred.cost.lhs_materialization_bytes.has_value());
    EXPECT_EQ(*addressable_preferred.cost.lhs_materialization_bytes, 0u);

    const EinsumPlan materialized_top =
        EinsumPlanner::parseAndPlan("ikk,kj->ij", {{2, 3, 3}, {3, 4}});
    ASSERT_EQ(materialized_top.logical_operands.size(), 2u);
    EinsumLogicalOperandPlan materialized_lhs = materialized_top.logical_operands[0];
    EinsumLogicalOperandPlan materialized_rhs = materialized_top.logical_operands[1];
    materialized_lhs.storage_dtype = DataType::BF16;
    materialized_rhs.storage_dtype = DataType::BF16;

    const EinsumPairContractionPlan materialized =
        EinsumPlanner::planPair(materialized_lhs, materialized_rhs, labels("ij"));
    const EinsumPairPhysicalCandidate& materialized_preferred =
        materialized.physical_candidates.at(materialized.preferred_physical_candidate);
    EXPECT_TRUE(materialized_preferred.lhs_materialized);
    EXPECT_EQ(materialized_preferred.cost.lhs_materialization_elements, 6u);
    EXPECT_EQ(materialized_preferred.cost.materialization_op_count, 1u);
    ASSERT_TRUE(materialized_preferred.cost.lhs_materialization_bytes.has_value());
    EXPECT_EQ(*materialized_preferred.cost.lhs_materialization_bytes, 12u);
    EXPECT_FALSE(materialized_preferred.rhs_materialized);
}

TEST(EinsumPlanner, PhysicalCostRecordsSharedKBroadcastReductionWork) {
    const EinsumLogicalOperandPlan lhs = denseLogicalOperand("ik", {2, 1}, {0});
    const EinsumLogicalOperandPlan rhs = denseLogicalOperand("kj", {3, 4}, {1});

    const EinsumPairContractionPlan pair = EinsumPlanner::planPair(lhs, rhs, labels("ij"));

    EXPECT_EQ(pair.kind, EinsumPlanKind::PAIR_PRODUCT);
    ASSERT_EQ(pair.physical_candidates.size(), 1u);
    const EinsumPairPhysicalCandidate& candidate = pair.physical_candidates.front();
    EXPECT_EQ(candidate.kind, EinsumPlanKind::PAIR_PRODUCT);
    EXPECT_EQ(candidate.cost.reduction_input_elements, 12u);
    EXPECT_EQ(candidate.cost.fused_elementwise_count, 8u);
    EXPECT_EQ(candidate.cost.reduction_op_count, 1u);
    EXPECT_EQ(candidate.cost.fused_kernel_count, 1u);
    EXPECT_EQ(candidate.cost.matmul_fma_count, 0u);
    EXPECT_EQ(candidate.cost.result_elements, 8u);
    ASSERT_TRUE(candidate.cost.result_bytes.has_value());
    EXPECT_EQ(*candidate.cost.result_bytes, 16u);
}

TEST(EinsumPlanner, PhysicalCostComposesPartialKBroadcastReductionWithRemainingGemm) {
    const EinsumLogicalOperandPlan lhs = denseLogicalOperand("ikl", {2, 1, 5}, {0});
    const EinsumLogicalOperandPlan rhs = denseLogicalOperand("klj", {3, 5, 4}, {1});

    const EinsumPairContractionPlan pair = EinsumPlanner::planPair(lhs, rhs, labels("ij"));

    EXPECT_EQ(pair.kind, EinsumPlanKind::GEMM);
    ASSERT_TRUE(pair.matrix_multiply.has_value());
    EXPECT_EQ(pair.matrix_multiply->lhs_broadcast_elision_labels, labels("k"));
    EXPECT_EQ(pair.matrix_multiply->rhs_reduction_labels, labels("k"));
    EXPECT_EQ(pair.matrix_multiply->contraction_labels, labels("l"));

    const EinsumPairPhysicalCandidate& candidate =
        pair.physical_candidates.at(pair.preferred_physical_candidate);
    EXPECT_EQ(candidate.cost.reduction_input_elements, 60u);
    EXPECT_EQ(candidate.cost.reduction_op_count, 1u);
    EXPECT_EQ(candidate.cost.matmul_fma_count, 40u);
    EXPECT_EQ(candidate.cost.lhs_materialization_elements, 0u);
    EXPECT_EQ(candidate.cost.rhs_materialization_elements, 0u);
}

TEST(EinsumPlanner, PhysicalCandidatesPreserveUsefulGemmResultLayoutsWithoutEagerOutputCopies) {
    const EinsumLogicalOperandPlan lhs = denseLogicalOperand("bia", {5, 2, 3}, {0});
    const EinsumLogicalOperandPlan rhs = denseLogicalOperand("bac", {5, 3, 7}, {1});

    const EinsumPairContractionPlan pair = EinsumPlanner::planPair(lhs, rhs, labels("ibc"));

    ASSERT_EQ(pair.physical_candidates.size(), 3u);

    const EinsumPairPhysicalCandidate* natural = nullptr;
    const EinsumPairPhysicalCandidate* dense = nullptr;
    const EinsumPairPhysicalCandidate* swapped = nullptr;
    for (const EinsumPairPhysicalCandidate& candidate : pair.physical_candidates) {
        if (candidate.output_materialized) {
            dense = &candidate;
        } else if (candidate.swapped_gemm_orientation) {
            swapped = &candidate;
        } else {
            natural = &candidate;
        }
    }

    ASSERT_NE(natural, nullptr);
    ASSERT_NE(dense, nullptr);
    ASSERT_NE(swapped, nullptr);

    EXPECT_EQ(natural->result.labels, labels("ibc"));
    EXPECT_EQ(natural->result.dimensions, (std::vector<uint64_t>{2, 5, 7}));
    EXPECT_EQ(natural->result.strides_elements, (std::vector<uint64_t>{7, 14, 1}));
    EXPECT_FALSE(natural->result.dense_storage);
    EXPECT_EQ(natural->cost.output_materialization_elements, 0u);

    EXPECT_EQ(swapped->result.strides_elements, (std::vector<uint64_t>{1, 14, 2}));
    EXPECT_FALSE(swapped->result.dense_storage);
    EXPECT_EQ(swapped->cost.output_materialization_elements, 0u);

    EXPECT_EQ(dense->result.strides_elements, (std::vector<uint64_t>{35, 7, 1}));
    EXPECT_TRUE(dense->result.dense_storage);
    EXPECT_TRUE(dense->output_materialized);
    EXPECT_EQ(dense->cost.output_materialization_elements, 70u);
    ASSERT_TRUE(dense->cost.output_materialization_bytes.has_value());
    EXPECT_EQ(*dense->cost.output_materialization_bytes, 140u);
    EXPECT_EQ(pair.preferred_physical_candidate,
              static_cast<uint32_t>(dense - pair.physical_candidates.data()));
}


TEST(EinsumPlanner, ExactThreeOperandPlannerChoosesGloballyCheaperRightToLeftChain) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("ab,bc,cd->ad", {{100, 2}, {2, 100}, {100, 2}});

    // The whole-equation algebraic kind remains GENERAL; runtime may now
    // execute this selected exact tree through the Expression pair lowerer.
    EXPECT_EQ(plan.kind, EinsumPlanKind::GENERAL);
    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);
    EXPECT_EQ(exact.cost.matmul_group_count, 2u);
    EXPECT_EQ(exact.cost.fused_kernel_count, 0u);
    EXPECT_EQ(exact.cost.reduction_op_count, 0u);
    EXPECT_EQ(exact.cost.materialization_op_count, 0u);
    EXPECT_EQ(exact.cost.estimated_execution_units, 13856u);

    // B@C first costs 2*100*2 FMAs and leaves only a 2x2 matrix for A.
    EXPECT_EQ(exact.steps[0].lhs_source_mask, 2u);
    EXPECT_EQ(exact.steps[0].rhs_source_mask, 4u);
    EXPECT_EQ(exact.steps[0].result_source_mask, 6u);
    EXPECT_EQ(exact.steps[0].physical_candidate.kind, EinsumPlanKind::GEMM);
    EXPECT_EQ(exact.steps[1].result_source_mask, 7u);
    EXPECT_EQ(exact.cost.matmul_fma_count, 800u);
    EXPECT_EQ(exact.result.labels, labels("ad"));
    EXPECT_TRUE(exact.result.dense_storage);
}

TEST(EinsumPlanner, ExactPlannerRetainsUsefulNonDenseIntermediatePhysicalLayout) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("ab,bc,cd->ad", {{100, 2}, {2, 100}, {100, 2}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);

    // The first pair's logical surviving order follows the whole equation's
    // stable iteration order, but its GEMM backing is useful in the opposite
    // physical axis order. Preserve that as a zero-copy view for the second
    // contraction instead of materializing it.
    const EinsumPairPhysicalCandidate& first = exact.steps[0].physical_candidate;
    EXPECT_FALSE(first.result.dense_storage);
    EXPECT_FALSE(first.output_materialized);
    EXPECT_EQ(first.cost.output_materialization_elements, 0u);
    EXPECT_EQ(exact.cost.materialization_elements, 0u);
}

TEST(EinsumPlanner, ExactFourOperandPlannerBuildsThreeStepTree) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan(
        "ab,bc,cd,de->ae", {{100, 2}, {2, 100}, {100, 2}, {2, 100}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 3u);
    EXPECT_EQ(exact.steps.back().result_source_mask, 15u);
    EXPECT_EQ(exact.result.source_operand_indices,
              (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(exact.result.labels, labels("ae"));
    EXPECT_TRUE(exact.result.dense_storage);
}

TEST(EinsumPlanner, ExactFiveOperandPlannerBuildsFourStepTree) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan(
        "ab,bc,cd,de,ef->af",
        {{100, 2}, {2, 100}, {100, 2}, {2, 100}, {100, 2}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 4u);
    EXPECT_EQ(exact.steps.back().result_source_mask, 31u);
    EXPECT_EQ(exact.result.source_operand_indices,
              (std::vector<uint32_t>{0, 1, 2, 3, 4}));
    for (const EinsumExactContractionStep& step : exact.steps) {
        EXPECT_TRUE(step.physical_candidate.kind == EinsumPlanKind::GEMM ||
                    step.physical_candidate.kind == EinsumPlanKind::BATCHED_GEMM);
    }
}

TEST(EinsumPlanner, ExactPlannerComposesBatchedGemms) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("bij,bjk,bkl->bil", {{5, 2, 3}, {5, 3, 4}, {5, 4, 6}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);
    EXPECT_EQ(exact.steps[0].physical_candidate.kind, EinsumPlanKind::BATCHED_GEMM);
    EXPECT_EQ(exact.steps[1].physical_candidate.kind, EinsumPlanKind::BATCHED_GEMM);
    EXPECT_GT(exact.cost.matmul_group_count, 0u);
}

TEST(EinsumPlanner, ExactPlannerComposesOperandLocalReductionBeforeGemm) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("irk,kj,jl->il", {{2, 5, 3}, {3, 4}, {4, 7}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);
    ASSERT_TRUE(exact.steps[0].physical_candidate.matrix_multiply.has_value());
    EXPECT_EQ(exact.steps[0].physical_candidate.matrix_multiply->lhs_reduction_labels,
              labels("r"));
    EXPECT_GT(exact.cost.reduction_input_elements, 0u);
}

TEST(EinsumPlanner, ExactPlannerComposesSharedKBroadcastNormalization) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("ik,kj,jl->il", {{2, 1}, {3, 4}, {4, 5}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);
    EXPECT_EQ(exact.steps[0].result_source_mask, 3u);
    EXPECT_EQ(exact.steps[0].physical_candidate.kind, EinsumPlanKind::PAIR_PRODUCT);
    EXPECT_GT(exact.steps[0].physical_candidate.cost.reduction_input_elements, 0u);
    EXPECT_EQ(exact.steps[1].physical_candidate.kind, EinsumPlanKind::GEMM);
}

TEST(EinsumPlanner, ExactCostCanPreferMoreFlopsToAvoidLargeMaterialization) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("acb,cd,be->ade", {{1, 100, 2}, {100, 1}, {2, 1}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);

    // Contracting operands 0 and 1 first has fewer arithmetic FMAs:
    //   200 for acb,cd plus 2 for the final b contraction.
    // But the interleaved acb matrix groups require materializing all 200 lhs
    // elements. The backend-aware cost policy therefore chooses operand 0 with
    // operand 2 first: 300 total FMAs, zero materialization.
    const EinsumPairContractionPlan lower_flop_first =
        EinsumPlanner::planPair(plan.logical_operands[0],
                                plan.logical_operands[1],
                                labels("adb"));
    ASSERT_FALSE(lower_flop_first.physical_candidates.empty());
    uint64_t minimum_first_materialization = std::numeric_limits<uint64_t>::max();
    for (const EinsumPairPhysicalCandidate& candidate :
         lower_flop_first.physical_candidates) {
        const uint64_t materialization =
            candidate.cost.lhs_materialization_elements +
            candidate.cost.rhs_materialization_elements +
            candidate.cost.output_materialization_elements;
        minimum_first_materialization =
            std::min(minimum_first_materialization, materialization);
    }
    EXPECT_EQ(minimum_first_materialization, 200u);

    EXPECT_EQ(exact.cost.matmul_fma_count, 300u);
    EXPECT_EQ(exact.cost.materialization_elements, 0u);
    EXPECT_EQ(exact.steps[0].result_source_mask, 5u);
}

TEST(EinsumPlanner, ExactPlannerIsDeterministic) {
    const EinsumPlan first =
        EinsumPlanner::parseAndPlan("ab,bc,cd,de,ef->af",
                                    {{17, 3}, {3, 11}, {11, 5}, {5, 13}, {13, 7}});
    const EinsumPlan second =
        EinsumPlanner::parseAndPlan("ab,bc,cd,de,ef->af",
                                    {{17, 3}, {3, 11}, {11, 5}, {5, 13}, {13, 7}});

    ASSERT_TRUE(first.exact_contraction.has_value());
    ASSERT_TRUE(second.exact_contraction.has_value());
    const EinsumExactContractionPlan& lhs = *first.exact_contraction;
    const EinsumExactContractionPlan& rhs = *second.exact_contraction;
    ASSERT_EQ(lhs.steps.size(), rhs.steps.size());
    EXPECT_EQ(lhs.cost.estimated_execution_units, rhs.cost.estimated_execution_units);
    for (size_t step = 0; step < lhs.steps.size(); ++step) {
        EXPECT_EQ(lhs.steps[step].lhs_source_mask, rhs.steps[step].lhs_source_mask);
        EXPECT_EQ(lhs.steps[step].rhs_source_mask, rhs.steps[step].rhs_source_mask);
        EXPECT_EQ(lhs.steps[step].result_source_mask, rhs.steps[step].result_source_mask);
        EXPECT_EQ(lhs.steps[step].surviving_labels, rhs.steps[step].surviving_labels);
        EXPECT_EQ(lhs.steps[step].physical_candidate.result.strides_elements,
                  rhs.steps[step].physical_candidate.result.strides_elements);
        EXPECT_EQ(lhs.steps[step].physical_candidate.swapped_gemm_orientation,
                  rhs.steps[step].physical_candidate.swapped_gemm_orientation);
        EXPECT_EQ(lhs.steps[step].physical_candidate.output_materialized,
                  rhs.steps[step].physical_candidate.output_materialized);
    }
}

TEST(EinsumPlanner, SixOperandBridgeBuildsFiveStepTreeWithOriginalProvenance) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("ab,bc,cd,de,ef,fg->ag",
                                    {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}});

    EXPECT_EQ(EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS, 5u);
    EXPECT_EQ(EinsumPlanner::MAX_BRIDGED_ACTIVE_OPERANDS, 6u);
    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& bridged = *plan.exact_contraction;
    EXPECT_EQ(bridged.planning_mode, EinsumContractionPlanningMode::SIX_OPERAND_BRIDGE);
    ASSERT_EQ(bridged.steps.size(), 5u);
    EXPECT_NE(bridged.bridge_seed_pair_mask, 0u);
    const uint64_t seed_without_lowest =
        bridged.bridge_seed_pair_mask & (bridged.bridge_seed_pair_mask - 1);
    EXPECT_NE(seed_without_lowest, 0u);
    EXPECT_EQ(seed_without_lowest & (seed_without_lowest - 1), 0u);
    EXPECT_EQ(bridged.steps.back().result_source_mask, 63u);
    EXPECT_EQ(bridged.result.source_operand_indices,
              (std::vector<uint32_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(bridged.result.labels, labels("ag"));
    EXPECT_TRUE(bridged.result.dense_storage);

    bool found_seed_step = false;
    for (const EinsumExactContractionStep& step : bridged.steps) {
        EXPECT_EQ(step.lhs_source_mask & step.rhs_source_mask, 0u);
        EXPECT_EQ(step.lhs_source_mask | step.rhs_source_mask,
                  step.result_source_mask);
        if (step.result_source_mask == bridged.bridge_seed_pair_mask) {
            found_seed_step = true;
        }
    }
    EXPECT_TRUE(found_seed_step);

    const std::string description = EinsumPlanner::describeExactContraction(plan);
    EXPECT_NE(description.find("exact_contraction operands=6 steps=5 mode=six_operand_bridge"),
              std::string::npos);
    EXPECT_NE(description.find("bridge_seed_pair_mask="), std::string::npos);
}

TEST(EinsumPlanner, SixOperandBridgeIsDeterministic) {
    const auto plan_once = [] {
        return EinsumPlanner::parseAndPlan(
            "ab,bc,cd,de,ef,fg->ag",
            {{17, 3}, {3, 11}, {11, 5}, {5, 13}, {13, 7}, {7, 19}});
    };
    const EinsumPlan first = plan_once();
    const EinsumPlan second = plan_once();

    ASSERT_TRUE(first.exact_contraction.has_value());
    ASSERT_TRUE(second.exact_contraction.has_value());
    const EinsumExactContractionPlan& lhs = *first.exact_contraction;
    const EinsumExactContractionPlan& rhs = *second.exact_contraction;
    EXPECT_EQ(lhs.bridge_seed_pair_mask, rhs.bridge_seed_pair_mask);
    EXPECT_EQ(lhs.cost.estimated_execution_units, rhs.cost.estimated_execution_units);
    ASSERT_EQ(lhs.steps.size(), rhs.steps.size());
    for (size_t step = 0; step < lhs.steps.size(); ++step) {
        EXPECT_EQ(lhs.steps[step].lhs_source_mask, rhs.steps[step].lhs_source_mask);
        EXPECT_EQ(lhs.steps[step].rhs_source_mask, rhs.steps[step].rhs_source_mask);
        EXPECT_EQ(lhs.steps[step].result_source_mask, rhs.steps[step].result_source_mask);
        EXPECT_EQ(lhs.steps[step].physical_candidate.result.strides_elements,
                  rhs.steps[step].physical_candidate.result.strides_elements);
    }
}

TEST(EinsumPlanner, MaximumOperandCountAllowsSixtyThreeOperandsPastCountGuard) {
    EXPECT_EQ(EinsumPlanner::MAX_SOURCE_OPERANDS, 63u);

    const std::string equation = repeatedOperandEquation(EinsumPlanner::MAX_SOURCE_OPERANDS);
    const auto valid_shapes = repeatedUnitShapes(EinsumPlanner::MAX_SOURCE_OPERANDS);
    const ResolvedEinsumEquation resolved = EinsumParser::parseAndResolve(equation, valid_shapes);

    auto invalid_shapes = valid_shapes;
    invalid_shapes.pop_back();
    try {
        (void)EinsumPlanner::plan(resolved, invalid_shapes);
        FAIL() << "Expected normal operand-count validation to reject mismatched shapes.";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(),
                     "Einsum planner operand count does not match the supplied input shapes.");
    }
}

TEST(EinsumPlanner, MaximumOperandCountRejectsSixtyFourOperandsExplicitly) {
    constexpr size_t unsupported_operand_count = EinsumPlanner::MAX_SOURCE_OPERANDS + 1;
    const std::string equation = repeatedOperandEquation(unsupported_operand_count);
    const auto shapes = repeatedUnitShapes(unsupported_operand_count);

    try {
        (void)EinsumPlanner::parseAndPlan(equation, shapes);
        FAIL() << "Expected einsum operand-count limit to reject 64 operands.";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "Einsum supports at most 63 operands; received 64.");
    }
}

TEST(EinsumPlanner, BeamPlannerBuildsSevenOperandTreeWithExactFiveOperandTail) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("ab,bc,cd,de,ef,fg,gh->ah",
                                    {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}});

    EXPECT_EQ(EinsumPlanner::DEFAULT_BEAM_WIDTH, 32u);
    EXPECT_FALSE(plan.exact_contraction.has_value());
    ASSERT_TRUE(plan.beam_contraction.has_value());
    const EinsumBeamContractionPlan& beam = *plan.beam_contraction;
    EXPECT_EQ(beam.beam_width, EinsumPlanner::DEFAULT_BEAM_WIDTH);
    EXPECT_EQ(beam.exact_tail_active_operands, EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS);
    EXPECT_EQ(beam.beam_levels, 2u);
    EXPECT_GT(beam.expanded_state_count, 0u);
    EXPECT_GT(beam.generated_state_count, 0u);
    EXPECT_GT(beam.deferred_disconnected_pair_count, 0u);
    EXPECT_GT(beam.truncated_state_count, 0u);
    EXPECT_GT(beam.retained_state_count, 0u);
    EXPECT_EQ(beam.generated_state_count,
              beam.deduplicated_state_count + beam.truncated_state_count +
                  beam.retained_state_count);
    EXPECT_GT(beam.exact_tail_count, 0u);
    EXPECT_LE(beam.exact_tail_count, EinsumPlanner::DEFAULT_BEAM_WIDTH);

    ASSERT_EQ(beam.steps.size(), 6u);
    EXPECT_EQ(beam.steps.back().result_source_mask, 127u);
    EXPECT_EQ(beam.result.source_operand_indices,
              (std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(beam.result.labels, labels("ah"));
    EXPECT_TRUE(beam.result.dense_storage);
    for (const EinsumExactContractionStep& step : beam.steps) {
        EXPECT_EQ(step.lhs_source_mask & step.rhs_source_mask, 0u);
        EXPECT_EQ(step.lhs_source_mask | step.rhs_source_mask, step.result_source_mask);
        EXPECT_EQ(step.physical_candidate.kind, EinsumPlanKind::GEMM);
    }

    const std::string description = EinsumPlanner::describeBeamContraction(plan);
    EXPECT_NE(description.find("beam_contraction operands=7 steps=6 beam_width=32"),
              std::string::npos);
    EXPECT_NE(description.find("exact_tail_active_operands=5 beam_levels=2"),
              std::string::npos);
    EXPECT_NE(description.find("deferred_disconnected_pairs="), std::string::npos);
    EXPECT_NE(description.find("truncated_states="), std::string::npos);
    EXPECT_NE(description.find("weights={fma:1,fused:128,reduction:64,materialization:128,writes:64}"),
              std::string::npos);
}

TEST(EinsumPlanner, DiagnosticBeamWidthDoesNotChangeProductionDefault) {
    const std::string equation = "ab,bc,cd,de,ef,fg,gh->ah";
    const std::vector<std::vector<uint64_t>> shapes =
        {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}};

    const EinsumPlan production = EinsumPlanner::parseAndPlan(equation, shapes);
    const EinsumPlan diagnostic =
        EinsumPlanner::parseAndPlanWithBeamWidthForDiagnostics(equation, shapes, 64);

    ASSERT_TRUE(production.beam_contraction.has_value());
    ASSERT_TRUE(diagnostic.beam_contraction.has_value());
    EXPECT_EQ(production.beam_contraction->beam_width,
              EinsumPlanner::DEFAULT_BEAM_WIDTH);
    EXPECT_EQ(diagnostic.beam_contraction->beam_width, 64u);
    EXPECT_EQ(EinsumPlanner::DEFAULT_BEAM_WIDTH, 32u);
}

TEST(EinsumPlanner, DiagnosticBeamWidthRejectsZero) {
    try {
        (void)EinsumPlanner::parseAndPlanWithBeamWidthForDiagnostics(
            "ab,bc,cd,de,ef,fg,gh->ah",
            {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}},
            0);
        FAIL() << "Expected zero diagnostic beam width to be rejected.";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "Einsum beam width must be greater than zero.");
    }
}

TEST(EinsumPlanner, BeamPlannerDisconnectedPruningPreservesScalarFactors) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan(
        "ab,bc,cd,de,ef,fg,->a",
        {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {}});

    ASSERT_TRUE(plan.beam_contraction.has_value());
    const EinsumBeamContractionPlan& beam = *plan.beam_contraction;
    EXPECT_GT(beam.deferred_disconnected_pair_count, 0u);
    ASSERT_EQ(beam.steps.size(), 6u);
    EXPECT_EQ(beam.steps.back().result_source_mask, 127u);
    EXPECT_TRUE(beam.result.dense_storage);
    EXPECT_EQ(beam.result.labels, labels("a"));
}

TEST(EinsumPlanner, BeamPlannerDisconnectedPruningPreservesSharedElementwiseMerges) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan(
        "a,a,a,a,a,a,a->a",
        {{17}, {17}, {17}, {17}, {17}, {17}, {17}});

    ASSERT_TRUE(plan.beam_contraction.has_value());
    const EinsumBeamContractionPlan& beam = *plan.beam_contraction;
    // All live dimensions are shared passthrough labels. There is no
    // independent-axis expansion to defer, so ordinary elementwise pair
    // products must remain eligible.
    EXPECT_EQ(beam.deferred_disconnected_pair_count, 0u);
    ASSERT_EQ(beam.steps.size(), 6u);
    for (const EinsumExactContractionStep& step : beam.steps) {
        EXPECT_EQ(step.physical_candidate.kind, EinsumPlanKind::ELEMENTWISE);
    }
}

TEST(EinsumPlanner, BeamPlannerDisconnectedPruningPreservesClosedComponentChoices) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan(
        "ab,bc,de,ef,gh,hi,jk->acdfgijk",
        {{2, 3}, {3, 5}, {7, 11}, {11, 13}, {17, 19}, {19, 23}, {29, 31}});

    ASSERT_TRUE(plan.beam_contraction.has_value());
    const EinsumBeamContractionPlan& beam = *plan.beam_contraction;
    EXPECT_GT(beam.deferred_disconnected_pair_count, 0u);
    EXPECT_EQ(beam.cost.estimated_execution_units, uint64_t{26'661'878'206});
    ASSERT_EQ(beam.steps.size(), 6u);
    EXPECT_EQ(beam.steps[0].result_source_mask, 3u);
    EXPECT_EQ(beam.steps[1].result_source_mask, 7u);
    EXPECT_EQ(beam.steps[2].result_source_mask, 24u);
    EXPECT_EQ(beam.steps[3].result_source_mask, 31u);
    EXPECT_EQ(beam.steps[4].result_source_mask, 96u);
    EXPECT_EQ(beam.steps[5].result_source_mask, 127u);
}

TEST(EinsumPlanner, BeamPlannerIsDeterministic) {
    const auto plan_once = [] {
        return EinsumPlanner::parseAndPlan(
            "ab,bc,cd,de,ef,fg,gh->ah",
            {{17, 3}, {3, 11}, {11, 5}, {5, 13}, {13, 7}, {7, 19}, {19, 2}});
    };
    const EinsumPlan first = plan_once();
    const EinsumPlan second = plan_once();

    ASSERT_TRUE(first.beam_contraction.has_value());
    ASSERT_TRUE(second.beam_contraction.has_value());
    const EinsumBeamContractionPlan& lhs = *first.beam_contraction;
    const EinsumBeamContractionPlan& rhs = *second.beam_contraction;
    EXPECT_EQ(lhs.cost.estimated_execution_units, rhs.cost.estimated_execution_units);
    EXPECT_EQ(lhs.expanded_state_count, rhs.expanded_state_count);
    EXPECT_EQ(lhs.generated_state_count, rhs.generated_state_count);
    EXPECT_EQ(lhs.deduplicated_state_count, rhs.deduplicated_state_count);
    EXPECT_EQ(lhs.deferred_disconnected_pair_count, rhs.deferred_disconnected_pair_count);
    EXPECT_EQ(lhs.truncated_state_count, rhs.truncated_state_count);
    EXPECT_EQ(lhs.retained_state_count, rhs.retained_state_count);
    ASSERT_EQ(lhs.steps.size(), rhs.steps.size());
    for (size_t step = 0; step < lhs.steps.size(); ++step) {
        EXPECT_EQ(lhs.steps[step].lhs_source_mask, rhs.steps[step].lhs_source_mask);
        EXPECT_EQ(lhs.steps[step].rhs_source_mask, rhs.steps[step].rhs_source_mask);
        EXPECT_EQ(lhs.steps[step].result_source_mask, rhs.steps[step].result_source_mask);
        EXPECT_EQ(lhs.steps[step].physical_candidate.result.strides_elements,
                  rhs.steps[step].physical_candidate.result.strides_elements);
        EXPECT_EQ(lhs.steps[step].physical_candidate.swapped_gemm_orientation,
                  rhs.steps[step].physical_candidate.swapped_gemm_orientation);
    }
}

TEST(EinsumPlanner, BeamPlannerComposesBatchedGemms) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan(
        "bij,bjk,bkl,blm,bmn,bno,bop->bip",
        {{3, 2, 4},
         {3, 4, 5},
         {3, 5, 6},
         {3, 6, 7},
         {3, 7, 8},
         {3, 8, 9},
         {3, 9, 10}});

    ASSERT_TRUE(plan.beam_contraction.has_value());
    const EinsumBeamContractionPlan& beam = *plan.beam_contraction;
    // Nonadjacent operands share only the persistent batch label. That label
    // survives the pair, so those batched outer products should be deferred
    // while each side still has true chain-contraction work available.
    EXPECT_GT(beam.deferred_disconnected_pair_count, 0u);
    ASSERT_EQ(beam.steps.size(), 6u);
    EXPECT_EQ(beam.steps.back().result_source_mask, 127u);
    EXPECT_GT(beam.cost.matmul_group_count, 0u);
    for (const EinsumExactContractionStep& step : beam.steps) {
        EXPECT_EQ(step.physical_candidate.kind, EinsumPlanKind::BATCHED_GEMM);
    }
}

TEST(EinsumPlanner, DenseBinaryRankOneToThreeExplicitEquationsAlwaysHaveOptimizedPairLowering) {
    const std::string alphabet = "abc";
    std::vector<std::string> operand_subscripts;
    for (size_t rank = 1; rank <= 3; ++rank) {
        size_t count = 1;
        for (size_t axis = 0; axis < rank; ++axis) count *= alphabet.size();
        for (size_t encoded = 0; encoded < count; ++encoded) {
            size_t remaining = encoded;
            std::string subscript(rank, 'a');
            for (size_t axis = rank; axis > 0; --axis) {
                subscript[axis - 1] = alphabet[remaining % alphabet.size()];
                remaining /= alphabet.size();
            }
            operand_subscripts.push_back(std::move(subscript));
        }
    }

    size_t audited_equations = 0;
    for (const std::string& lhs : operand_subscripts) {
        for (const std::string& rhs : operand_subscripts) {
            std::set<char> present_labels(lhs.begin(), lhs.end());
            present_labels.insert(rhs.begin(), rhs.end());
            const std::vector<char> present(present_labels.begin(), present_labels.end());

            std::vector<std::string> outputs{std::string{}};
            for (size_t mask = 1; mask < (size_t{1} << present.size()); ++mask) {
                std::string output;
                for (size_t label_index = 0; label_index < present.size(); ++label_index) {
                    if ((mask & (size_t{1} << label_index)) != 0) {
                        output.push_back(present[label_index]);
                    }
                }
                std::sort(output.begin(), output.end());
                do {
                    outputs.push_back(output);
                } while (std::next_permutation(output.begin(), output.end()));
            }

            for (const std::string& output : outputs) {
                const std::string equation = lhs + "," + rhs + "->" + output;
                const EinsumPlan plan = EinsumPlanner::parseAndPlan(
                    equation,
                    {std::vector<uint64_t>(lhs.size(), 2), std::vector<uint64_t>(rhs.size(), 2)});
                ++audited_equations;
                EXPECT_TRUE(plan.matrix_multiply.has_value() || plan.pair_product.has_value())
                    << "equation=" << equation;
                EXPECT_NE(plan.kind, EinsumPlanKind::GENERAL) << "equation=" << equation;
            }
        }
    }

    // This is deliberately broad enough to include reductions, diagonals,
    // transposed free axes, shared surviving labels, true contractions, and
    // outer/pair products rather than a hand-selected set of friendly forms.
    EXPECT_EQ(audited_equations, 18084u);
}

TEST(EinsumPlanner, ExactDescriptionExplainsSelectedPhysicalTreeAndCosts) {
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("ab,bc,cd->ad", {{100, 2}, {2, 100}, {100, 2}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    ASSERT_EQ(exact.steps.size(), 2u);
    const EinsumExactContractionStep& first = exact.steps.front();
    EXPECT_EQ(first.lhs.source_operand_indices, (std::vector<uint32_t>{1}));
    EXPECT_EQ(first.rhs.source_operand_indices, (std::vector<uint32_t>{2}));
    EXPECT_EQ(first.eliminated_labels, labels("c"));
    EXPECT_EQ(first.cumulative_cost.estimated_execution_units,
              first.incremental_estimated_execution_units);
    EXPECT_EQ(exact.steps.back().cumulative_cost.estimated_execution_units,
              exact.cost.estimated_execution_units);

    const std::string description = EinsumPlanner::describeExactContraction(plan);
    EXPECT_NE(description.find("exact_contraction operands=3 steps=2"), std::string::npos);
    EXPECT_NE(description.find("lhs_mask=2 rhs_mask=4 result_mask=6"), std::string::npos);
    EXPECT_NE(description.find("surviving=[d,b] eliminated=[c] kind=GEMM"), std::string::npos);
    EXPECT_NE(description.find("orientation="), std::string::npos);
    EXPECT_NE(description.find("materialize={lhs:"), std::string::npos);
    EXPECT_NE(description.find("pair_cost={estimated:"), std::string::npos);
    EXPECT_NE(description.find("cumulative={estimated:"), std::string::npos);
    EXPECT_NE(description.find("weights={fma:1,fused:128,reduction:64,materialization:128,writes:64}"),
              std::string::npos);
    EXPECT_NE(description.find("ops={gemm_groups:2,fused:0,reduction:0,materialization:0}"),
              std::string::npos);
}

TEST(EinsumPlanner, ExactDescriptionIsEmptyWhenExactPlanningDoesNotApply) {
    const EinsumPlan plan = EinsumPlanner::parseAndPlan("ab,bc->ac", {{2, 3}, {3, 4}});
    EXPECT_TRUE(EinsumPlanner::describeExactContraction(plan).empty());
}

TEST(EinsumPlanner, ExactPlannerHandlesLargeDimensionsNearUint64CostBoundary) {
    constexpr uint64_t q = uint64_t{1} << 20;
    const EinsumPlan plan =
        EinsumPlanner::parseAndPlan("ab,bc,cd->ad", {{q, q}, {q, q}, {q, q}});

    ASSERT_TRUE(plan.exact_contraction.has_value());
    const EinsumExactContractionCost& cost = plan.exact_contraction->cost;
    EXPECT_EQ(cost.matmul_fma_count, uint64_t{1} << 61);
    EXPECT_EQ(cost.result_write_elements, uint64_t{1} << 41);
    EXPECT_EQ(cost.estimated_execution_units,
              (uint64_t{1} << 61) + (uint64_t{1} << 47));
}

TEST(EinsumPlanner, ExactPlannerRejectsMatmulCostOverflowInsteadOfWrapping) {
    constexpr uint64_t q = uint64_t{1} << 22;
    try {
        (void)EinsumPlanner::parseAndPlan("ab,bc,cd->ad", {{q, q}, {q, q}, {q, q}});
        FAIL() << "Expected exact planner cost overflow.";
    } catch (const std::overflow_error& error) {
        EXPECT_NE(std::string(error.what()).find("no complete contraction tree representable"),
                  std::string::npos);
    }
}
