#include "Utilities/TensorOperations/Einsum/EinsumBackwardPlanner.h"

#include "gtest/gtest.h"

#include <cstdint>
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

std::vector<int32_t> physicalLabels(const std::string& text) {
    std::vector<int32_t> result{EinsumLayerBatchContract::kImplicitBatchLabel};
    const std::vector<int32_t> feature = labels(text);
    result.insert(result.end(), feature.begin(), feature.end());
    return result;
}

}  // namespace

TEST(EinsumBackwardPlanner, DirectMatrixMultiplyProducesReverseContractionsPerOperandOccurrence) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("ik,kj->ij", {{2, 3}, {3, 4}});

    ASSERT_EQ(plan.operand_gradients.size(), 2u);
    EXPECT_EQ(plan.physical_forward_input_axis_labels[0], physicalLabels("ik"));
    EXPECT_EQ(plan.physical_forward_input_axis_labels[1], physicalLabels("kj"));
    EXPECT_EQ(plan.physical_forward_output_axis_labels, physicalLabels("ij"));

    const EinsumOperandBackwardPlan& lhs = plan.operand_gradients[0];
    EXPECT_EQ(lhs.target_unique_feature_labels, labels("ik"));
    EXPECT_EQ(lhs.contraction.upstream_gradient_feature_labels, labels("ij"));
    EXPECT_EQ(lhs.contraction.other_operand_indices, (std::vector<uint32_t>{1}));
    ASSERT_EQ(lhs.contraction.other_operand_feature_axis_labels.size(), 1u);
    EXPECT_EQ(lhs.contraction.other_operand_feature_axis_labels[0], labels("kj"));
    EXPECT_EQ(lhs.contraction.output_feature_labels, labels("ik"));
    EXPECT_EQ(lhs.contraction.output_feature_dimensions, (std::vector<uint64_t>{2, 3}));
    EXPECT_EQ(lhs.contraction.physical_input_axis_labels[0], physicalLabels("ij"));
    EXPECT_EQ(lhs.contraction.physical_input_axis_labels[1], physicalLabels("kj"));
    EXPECT_EQ(lhs.contraction.physical_output_axis_labels, physicalLabels("ik"));
    EXPECT_TRUE(lhs.broadcast_reductions.empty());
    EXPECT_TRUE(lhs.missing_axis_expansions.empty());
    EXPECT_TRUE(lhs.diagonal_scatters.empty());

    const EinsumOperandBackwardPlan& rhs = plan.operand_gradients[1];
    EXPECT_EQ(rhs.contraction.other_operand_indices, (std::vector<uint32_t>{0}));
    EXPECT_EQ(rhs.contraction.other_operand_feature_axis_labels[0], labels("ik"));
    EXPECT_EQ(rhs.contraction.output_feature_labels, labels("kj"));
}

TEST(EinsumBackwardPlanner, ForwardReducedTargetOnlyLabelsBecomeBackwardBroadcastExpansions) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("ij->i", {{2, 3}});
    ASSERT_EQ(plan.operand_gradients.size(), 1u);

    const EinsumOperandBackwardPlan& gradient = plan.operand_gradients[0];
    EXPECT_EQ(gradient.target_unique_feature_labels, labels("ij"));
    EXPECT_EQ(gradient.contraction.upstream_gradient_feature_labels, labels("i"));
    EXPECT_TRUE(gradient.contraction.other_operand_indices.empty());
    EXPECT_EQ(gradient.contraction.output_feature_labels, labels("i"));
    EXPECT_EQ(gradient.contraction.output_feature_dimensions, (std::vector<uint64_t>{2}));
    ASSERT_EQ(gradient.missing_axis_expansions.size(), 1u);
    EXPECT_EQ(gradient.missing_axis_expansions[0].label, EinsumParser::labelId('j'));
    EXPECT_EQ(gradient.missing_axis_expansions[0].target_unique_feature_axis, 1u);
    EXPECT_EQ(gradient.missing_axis_expansions[0].target_dimension, 3u);
    EXPECT_EQ(gradient.final_feature_dimensions, (std::vector<uint64_t>{2, 3}));
}

TEST(EinsumBackwardPlanner, FullReductionExpandsEveryTargetAxisAfterScalarGradientContraction) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("ij->", {{2, 3}});
    const EinsumOperandBackwardPlan& gradient = plan.operand_gradients[0];

    EXPECT_TRUE(gradient.contraction.upstream_gradient_feature_labels.empty());
    EXPECT_TRUE(gradient.contraction.output_feature_labels.empty());
    EXPECT_EQ(gradient.contraction.physical_input_axis_labels,
              (std::vector<std::vector<int32_t>>{{EinsumLayerBatchContract::kImplicitBatchLabel}}));
    EXPECT_EQ(gradient.contraction.physical_output_axis_labels,
              (std::vector<int32_t>{EinsumLayerBatchContract::kImplicitBatchLabel}));
    ASSERT_EQ(gradient.missing_axis_expansions.size(), 2u);
    EXPECT_EQ(gradient.missing_axis_expansions[0].label, EinsumParser::labelId('i'));
    EXPECT_EQ(gradient.missing_axis_expansions[0].target_unique_feature_axis, 0u);
    EXPECT_EQ(gradient.missing_axis_expansions[0].target_dimension, 2u);
    EXPECT_EQ(gradient.missing_axis_expansions[1].label, EinsumParser::labelId('j'));
    EXPECT_EQ(gradient.missing_axis_expansions[1].target_unique_feature_axis, 1u);
    EXPECT_EQ(gradient.missing_axis_expansions[1].target_dimension, 3u);
}

TEST(EinsumBackwardPlanner, SingletonForwardBroadcastReducesRawGradientBackToLocalExtent) {
    const EinsumLayerBackwardPlan plan =
        EinsumBackwardPlanner::parseAndPlan("bij,bjk->bik", {{1, 2, 3}, {5, 3, 4}});

    const EinsumOperandBackwardPlan& lhs = plan.operand_gradients[0];
    EXPECT_EQ(lhs.contraction.output_feature_labels, labels("bij"));
    EXPECT_EQ(lhs.contraction.output_feature_dimensions, (std::vector<uint64_t>{5, 2, 3}));
    ASSERT_EQ(lhs.broadcast_reductions.size(), 1u);
    EXPECT_EQ(lhs.broadcast_reductions[0].label, EinsumParser::labelId('b'));
    EXPECT_EQ(lhs.broadcast_reductions[0].contraction_output_feature_axis, 0u);
    EXPECT_EQ(lhs.broadcast_reductions[0].source_dimension, 5u);
    EXPECT_EQ(lhs.broadcast_reductions[0].target_dimension, 1u);

    const EinsumOperandBackwardPlan& rhs = plan.operand_gradients[1];
    EXPECT_TRUE(rhs.broadcast_reductions.empty());
}

TEST(EinsumBackwardPlanner, SingletonBackwardInputExpandsBackToNonSingletonTargetExtent) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("ij,j->i", {{2, 3}, {1}});

    const EinsumOperandBackwardPlan& matrix = plan.operand_gradients[0];
    EXPECT_EQ(matrix.contraction.output_feature_labels, labels("ij"));
    EXPECT_EQ(matrix.contraction.output_feature_dimensions, (std::vector<uint64_t>{2, 1}));
    ASSERT_EQ(matrix.existing_axis_expansions.size(), 1u);
    EXPECT_EQ(matrix.existing_axis_expansions[0].label, EinsumParser::labelId('j'));
    EXPECT_EQ(matrix.existing_axis_expansions[0].contraction_output_feature_axis, 1u);
    EXPECT_EQ(matrix.existing_axis_expansions[0].target_unique_feature_axis, 1u);
    EXPECT_EQ(matrix.existing_axis_expansions[0].source_dimension, 1u);
    EXPECT_EQ(matrix.existing_axis_expansions[0].target_dimension, 3u);
    EXPECT_TRUE(matrix.broadcast_reductions.empty());
    EXPECT_TRUE(matrix.missing_axis_expansions.empty());

    const EinsumOperandBackwardPlan& vector = plan.operand_gradients[1];
    EXPECT_EQ(vector.contraction.output_feature_dimensions, (std::vector<uint64_t>{3}));
    ASSERT_EQ(vector.broadcast_reductions.size(), 1u);
    EXPECT_EQ(vector.broadcast_reductions[0].source_dimension, 3u);
    EXPECT_EQ(vector.broadcast_reductions[0].target_dimension, 1u);
}

TEST(EinsumBackwardPlanner, BroadcastedContractionLabelIsReducedBackToSingletonTarget) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("ij,jk->ik", {{2, 1}, {3, 4}});

    const EinsumOperandBackwardPlan& lhs = plan.operand_gradients[0];
    EXPECT_EQ(lhs.contraction.output_feature_labels, labels("ij"));
    EXPECT_EQ(lhs.contraction.output_feature_dimensions, (std::vector<uint64_t>{2, 3}));
    ASSERT_EQ(lhs.broadcast_reductions.size(), 1u);
    EXPECT_EQ(lhs.broadcast_reductions[0].label, EinsumParser::labelId('j'));
    EXPECT_EQ(lhs.broadcast_reductions[0].contraction_output_feature_axis, 1u);
    EXPECT_EQ(lhs.broadcast_reductions[0].source_dimension, 3u);
}

TEST(EinsumBackwardPlanner, RepeatedTargetLabelProducesDiagonalScatterMetadata) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("ii->", {{4, 4}});
    const EinsumOperandBackwardPlan& gradient = plan.operand_gradients[0];

    EXPECT_EQ(gradient.target_unique_feature_labels, labels("i"));
    EXPECT_EQ(gradient.target_unique_feature_dimensions, (std::vector<uint64_t>{4}));
    ASSERT_EQ(gradient.missing_axis_expansions.size(), 1u);
    EXPECT_EQ(gradient.missing_axis_expansions[0].label, EinsumParser::labelId('i'));
    ASSERT_EQ(gradient.diagonal_scatters.size(), 1u);
    EXPECT_EQ(gradient.diagonal_scatters[0].label, EinsumParser::labelId('i'));
    EXPECT_EQ(gradient.diagonal_scatters[0].source_unique_feature_axis, 0u);
    EXPECT_EQ(gradient.diagonal_scatters[0].target_feature_axes, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(gradient.diagonal_scatters[0].dimension, 4u);
    EXPECT_EQ(gradient.final_feature_dimensions, (std::vector<uint64_t>{4, 4}));
}

TEST(EinsumBackwardPlanner, RepeatedTargetLabelThatSurvivesOutputStillScattersWithoutExpansion) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("iii->i", {{3, 3, 3}});
    const EinsumOperandBackwardPlan& gradient = plan.operand_gradients[0];

    EXPECT_EQ(gradient.contraction.output_feature_labels, labels("i"));
    EXPECT_TRUE(gradient.missing_axis_expansions.empty());
    ASSERT_EQ(gradient.diagonal_scatters.size(), 1u);
    EXPECT_EQ(gradient.diagonal_scatters[0].target_feature_axes, (std::vector<uint32_t>{0, 1, 2}));
}

TEST(EinsumBackwardPlanner, EllipsisRemainsFeatureMetadataDistinctFromImplicitThorBatch) {
    const EinsumLayerBackwardPlan plan =
        EinsumBackwardPlanner::parseAndPlan("...ik,...kj->...ij", {{5, 2, 3}, {1, 3, 4}});

    ASSERT_EQ(plan.feature_equation.ellipsis_rank, 1u);
    const int32_t ellipsis_label = plan.feature_equation.output_labels.front();
    ASSERT_TRUE(EinsumParser::isEllipsisLabel(ellipsis_label));
    EXPECT_NE(ellipsis_label, EinsumLayerBatchContract::kImplicitBatchLabel);

    ASSERT_EQ(plan.physical_forward_input_axis_labels[0].size(), 4u);
    EXPECT_EQ(plan.physical_forward_input_axis_labels[0][0], EinsumLayerBatchContract::kImplicitBatchLabel);
    EXPECT_EQ(plan.physical_forward_input_axis_labels[0][1], ellipsis_label);

    const EinsumOperandBackwardPlan& rhs = plan.operand_gradients[1];
    ASSERT_EQ(rhs.broadcast_reductions.size(), 1u);
    EXPECT_EQ(rhs.broadcast_reductions[0].label, ellipsis_label);
    EXPECT_EQ(rhs.broadcast_reductions[0].source_dimension, 5u);
    EXPECT_EQ(rhs.broadcast_reductions[0].target_dimension, 1u);
    ASSERT_FALSE(rhs.contraction.physical_output_axis_labels.empty());
    EXPECT_EQ(rhs.contraction.physical_output_axis_labels[0], EinsumLayerBatchContract::kImplicitBatchLabel);
}

TEST(EinsumBackwardPlanner, MultiOperandGradientUsesOutputGradientAndEveryOtherOperand) {
    const EinsumLayerBackwardPlan plan =
        EinsumBackwardPlanner::parseAndPlan("ab,bc,cd,de,ef,fg,gh->ah",
                                            {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}});

    ASSERT_EQ(plan.operand_gradients.size(), 7u);
    const EinsumOperandBackwardPlan& middle = plan.operand_gradients[3];
    EXPECT_EQ(middle.target_unique_feature_labels, labels("de"));
    EXPECT_EQ(middle.contraction.upstream_gradient_feature_labels, labels("ah"));
    EXPECT_EQ(middle.contraction.other_operand_indices,
              (std::vector<uint32_t>{0, 1, 2, 4, 5, 6}));
    EXPECT_EQ(middle.contraction.output_feature_labels, labels("de"));
    EXPECT_TRUE(middle.broadcast_reductions.empty());
    EXPECT_TRUE(middle.missing_axis_expansions.empty());
    EXPECT_TRUE(middle.diagonal_scatters.empty());
}

TEST(EinsumBackwardPlanner, TenOperandChainKeepsOneGradientPlanPerOccurrence) {
    const EinsumLayerBackwardPlan plan =
        EinsumBackwardPlanner::parseAndPlan("ab,bc,cd,de,ef,fg,gh,hi,ij,jk->ak",
                                            {{2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7},
                                             {7, 8}, {8, 9}, {9, 10}, {10, 11}, {11, 12}});
    ASSERT_EQ(plan.operand_gradients.size(), 10u);
    EXPECT_EQ(plan.operand_gradients[0].contraction.other_operand_indices.size(), 9u);
    EXPECT_EQ(plan.operand_gradients[9].contraction.other_operand_indices.size(), 9u);
}

TEST(EinsumBackwardPlanner, DuplicateOperandOccurrencesRemainSeparateProductRuleTerms) {
    const EinsumLayerBackwardPlan plan = EinsumBackwardPlanner::parseAndPlan("ij,ij->", {{2, 3}, {2, 3}});

    ASSERT_EQ(plan.operand_gradients.size(), 2u);
    EXPECT_EQ(plan.operand_gradients[0].operand_index, 0u);
    EXPECT_EQ(plan.operand_gradients[0].contraction.other_operand_indices, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.operand_gradients[0].contraction.output_feature_labels, labels("ij"));
    EXPECT_EQ(plan.operand_gradients[1].operand_index, 1u);
    EXPECT_EQ(plan.operand_gradients[1].contraction.other_operand_indices, (std::vector<uint32_t>{0}));
    EXPECT_EQ(plan.operand_gradients[1].contraction.output_feature_labels, labels("ij"));
}

TEST(EinsumBackwardPlanner, ImplicitBatchHelpersPreserveFeatureShapeAndRejectZeroBatch) {
    EXPECT_EQ(EinsumLayerBatchContract::prependImplicitBatchLabel(labels("ij")), physicalLabels("ij"));
    EXPECT_EQ(EinsumLayerBatchContract::prependBatchDimension(7, {2, 3}),
              (std::vector<uint64_t>{7, 2, 3}));
    EXPECT_THROW((void)EinsumLayerBatchContract::prependBatchDimension(0, {2, 3}), std::invalid_argument);
}
