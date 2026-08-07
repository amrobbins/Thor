#include "Utilities/Expression/BatchedMatmulPlan.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

MatmulTensorLayout layout(std::vector<uint64_t> dims, std::vector<uint64_t> strides, uint64_t offset = 0) {
    return MatmulTensorLayout{std::move(dims), std::move(strides), offset};
}

}  // namespace

TEST(ExpressionBatchedMatmulPlan, ShapeInferencePreservesRankTwoSemantics) {
    const BatchedMatmulShapePlan plan = planBatchedMatmulShape({3, 4}, {4, 5});
    EXPECT_TRUE(plan.batch_dimensions.empty());
    EXPECT_EQ(plan.output_dimensions, (std::vector<uint64_t>{3, 5}));
    EXPECT_EQ(plan.m, 3);
    EXPECT_EQ(plan.k, 4);
    EXPECT_EQ(plan.n, 5);
    EXPECT_EQ(plan.batch_count, 1);
}

TEST(ExpressionBatchedMatmulPlan, ShapeInferenceBroadcastsLeadingDimensionsFromTheRight) {
    const BatchedMatmulShapePlan plan = planBatchedMatmulShape({2, 1, 3, 4}, {1, 5, 4, 6});
    EXPECT_EQ(plan.lhs_aligned_batch_dimensions, (std::vector<uint64_t>{2, 1}));
    EXPECT_EQ(plan.rhs_aligned_batch_dimensions, (std::vector<uint64_t>{1, 5}));
    EXPECT_EQ(plan.batch_dimensions, (std::vector<uint64_t>{2, 5}));
    EXPECT_EQ(plan.output_dimensions, (std::vector<uint64_t>{2, 5, 3, 6}));
    EXPECT_EQ(plan.batch_count, 10);
}

TEST(ExpressionBatchedMatmulPlan, ShapeInferenceAppliesTransposeOnlyToFinalMatrixAxes) {
    const BatchedMatmulShapePlan plan = planBatchedMatmulShape({7, 4, 3}, {1, 5, 4}, true, true);
    EXPECT_EQ(plan.batch_dimensions, (std::vector<uint64_t>{7}));
    EXPECT_EQ(plan.output_dimensions, (std::vector<uint64_t>{7, 3, 5}));
    EXPECT_EQ(plan.m, 3);
    EXPECT_EQ(plan.k, 4);
    EXPECT_EQ(plan.n, 5);
}

TEST(ExpressionBatchedMatmulPlan, ShapeInferenceRejectsInvalidRanksContractionsAndBroadcasts) {
    EXPECT_THROW((void)planBatchedMatmulShape({4}, {4, 5}), std::runtime_error);
    EXPECT_THROW((void)planBatchedMatmulShape({3, 4}, {5, 6}), std::runtime_error);
    EXPECT_THROW((void)planBatchedMatmulShape({2, 3, 4}, {5, 4, 6}), std::runtime_error);
}

TEST(ExpressionBatchedMatmulPlan, ShapeInferenceRejectsBatchCountOverflow) {
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    EXPECT_THROW((void)planBatchedMatmulShape({maximum, 2, 3, 4}, {maximum, 2, 4, 5}), std::runtime_error);
}

TEST(ExpressionBatchedMatmulPlan, DenseRegularBatchBecomesOneStridedBatch) {
    const auto lhs = denseMatmulTensorLayout({2, 3, 4, 5});
    const auto rhs = denseMatmulTensorLayout({2, 3, 5, 6});
    const auto output = denseMatmulTensorLayout({2, 3, 4, 6});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    ASSERT_EQ(plan.batch_axes.size(), 2);
    EXPECT_EQ(plan.grouping.varying_axes, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(plan.grouping.batch_count, 6);
    EXPECT_EQ(plan.grouping.group_count, 1);
    EXPECT_EQ(plan.grouping.lhs_batch_stride_elements, 20);
    EXPECT_EQ(plan.grouping.rhs_batch_stride_elements, 30);
    EXPECT_EQ(plan.grouping.output_batch_stride_elements, 24);
    EXPECT_TRUE(plan.grouping.isSingleStridedBatch());
    EXPECT_TRUE(plan.canLowerWithoutMaterialization());
}

TEST(ExpressionBatchedMatmulPlan, EntireUnbatchedOperandUsesZeroStrideInOneBatch) {
    const auto lhs = denseMatmulTensorLayout({2, 3, 4, 5});
    const auto rhs = denseMatmulTensorLayout({5, 6});
    const auto output = denseMatmulTensorLayout({2, 3, 4, 6});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    ASSERT_EQ(plan.batch_axes.size(), 2);
    EXPECT_TRUE(plan.batch_axes[0].rhs_broadcast);
    EXPECT_TRUE(plan.batch_axes[1].rhs_broadcast);
    EXPECT_EQ(plan.grouping.batch_count, 6);
    EXPECT_EQ(plan.grouping.group_count, 1);
    EXPECT_EQ(plan.grouping.rhs_batch_stride_elements, 0);
}

TEST(ExpressionBatchedMatmulPlan, CrossBroadcastChoosesLargestRegularAxisWithoutMaterializing) {
    const auto lhs = denseMatmulTensorLayout({2, 1, 3, 4});
    const auto rhs = denseMatmulTensorLayout({1, 5, 4, 6});
    const auto output = denseMatmulTensorLayout({2, 5, 3, 6});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    ASSERT_EQ(plan.batch_axes.size(), 2);
    EXPECT_EQ(plan.grouping.varying_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.grouping.batch_count, 5);
    EXPECT_EQ(plan.grouping.group_count, 2);
    EXPECT_EQ(plan.grouping.lhs_batch_stride_elements, 0);
    EXPECT_EQ(plan.grouping.rhs_batch_stride_elements, 24);
    EXPECT_EQ(plan.grouping.output_batch_stride_elements, 18);
    EXPECT_TRUE(plan.canLowerWithoutMaterialization());

    const std::vector<MatmulBatchGroup> groups = materializeBatchedMatmulGroups(plan);
    ASSERT_EQ(groups.size(), 2);
    EXPECT_EQ(groups[0], (MatmulBatchGroup{5, 0, 0, 0, 0, 24, 18}));
    EXPECT_EQ(groups[1], (MatmulBatchGroup{5, 12, 0, 90, 0, 24, 18}));
}

TEST(ExpressionBatchedMatmulPlan, CrossBroadcastCanChooseOuterAxisWhenItProducesFewerGroups) {
    const auto lhs = denseMatmulTensorLayout({7, 1, 3, 4});
    const auto rhs = denseMatmulTensorLayout({1, 2, 4, 6});
    const auto output = denseMatmulTensorLayout({7, 2, 3, 6});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    EXPECT_EQ(plan.grouping.varying_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(plan.grouping.batch_count, 7);
    EXPECT_EQ(plan.grouping.group_count, 2);
    EXPECT_EQ(plan.grouping.lhs_batch_stride_elements, 12);
    EXPECT_EQ(plan.grouping.rhs_batch_stride_elements, 0);
    EXPECT_EQ(plan.grouping.output_batch_stride_elements, 36);

    const std::vector<MatmulBatchGroup> groups = materializeBatchedMatmulGroups(plan);
    ASSERT_EQ(groups.size(), 2);
    EXPECT_EQ(groups[0].lhs_relative_element_offset, 0);
    EXPECT_EQ(groups[0].rhs_relative_element_offset, 0);
    EXPECT_EQ(groups[0].output_relative_element_offset, 0);
    EXPECT_EQ(groups[1].lhs_relative_element_offset, 0);
    EXPECT_EQ(groups[1].rhs_relative_element_offset, 24);
    EXPECT_EQ(groups[1].output_relative_element_offset, 18);
}

TEST(ExpressionBatchedMatmulPlan, NonContiguousBatchAxesCanFormOneRegularGroupWithoutPermutation) {
    const auto lhs = layout({2, 3, 4, 2, 3}, {40, 100, 10, 3, 1});
    const auto rhs = layout({2, 3, 4, 3, 5}, {80, 200, 20, 5, 1});
    const auto output = layout({2, 3, 4, 2, 5}, {48, 120, 12, 5, 1});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    EXPECT_EQ(plan.grouping.varying_axes, (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(plan.grouping.batch_count, 8);
    EXPECT_EQ(plan.grouping.group_count, 3);
    EXPECT_EQ(plan.grouping.lhs_batch_stride_elements, 10);
    EXPECT_EQ(plan.grouping.rhs_batch_stride_elements, 20);
    EXPECT_EQ(plan.grouping.output_batch_stride_elements, 12);

    const std::vector<MatmulBatchGroup> groups = materializeBatchedMatmulGroups(plan);
    ASSERT_EQ(groups.size(), 3);
    EXPECT_EQ(groups[0].lhs_relative_element_offset, 0);
    EXPECT_EQ(groups[1].lhs_relative_element_offset, 100);
    EXPECT_EQ(groups[2].lhs_relative_element_offset, 200);
    EXPECT_EQ(groups[1].rhs_relative_element_offset, 200);
    EXPECT_EQ(groups[1].output_relative_element_offset, 120);
}

TEST(ExpressionBatchedMatmulPlan, PaddedBatchLayoutSplitsOnlyWhereConstantStrideBreaks) {
    const auto lhs = layout({2, 3, 4, 5}, {100, 20, 5, 1});
    const auto rhs = denseMatmulTensorLayout({2, 3, 5, 6});
    const auto output = denseMatmulTensorLayout({2, 3, 4, 6});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    EXPECT_EQ(plan.grouping.varying_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(plan.grouping.batch_count, 3);
    EXPECT_EQ(plan.grouping.group_count, 2);
    EXPECT_EQ(plan.grouping.lhs_batch_stride_elements, 20);
}

TEST(ExpressionBatchedMatmulPlan, RowMajorPaddedMatrixPlaneIsAddressableWithoutMaterialization) {
    const auto lhs = layout({2, 3, 4}, {20, 6, 1}, 17);
    const auto rhs = denseMatmulTensorLayout({2, 4, 5}, 23);
    const auto output = layout({2, 3, 5}, {24, 7, 1}, 31);
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    EXPECT_EQ(plan.lhs_layout.storage_element_offset, 17);
    EXPECT_EQ(plan.rhs_layout.storage_element_offset, 23);
    EXPECT_EQ(plan.output_layout.storage_element_offset, 31);
    EXPECT_EQ(plan.lhs_matrix.storage_kind, MatmulMatrixStorageKind::RowMajor);
    EXPECT_EQ(plan.lhs_matrix.leading_dimension, 6);
    EXPECT_EQ(plan.lhs_matrix.storage_span_elements, 16);
    EXPECT_EQ(plan.output_matrix.leading_dimension, 7);
    EXPECT_TRUE(plan.canLowerWithoutMaterialization());
}

TEST(ExpressionBatchedMatmulPlan, TransposedPhysicalMatrixViewBecomesBackendTransposeInsteadOfMaterialization) {
    const auto lhs = layout({2, 3, 4}, {12, 1, 3});
    const auto rhs = denseMatmulTensorLayout({2, 4, 5});
    const auto output = denseMatmulTensorLayout({2, 3, 5});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    EXPECT_EQ(plan.lhs_matrix.storage_kind, MatmulMatrixStorageKind::TransposedRowMajor);
    EXPECT_EQ(plan.lhs_matrix.stored_rows, 4);
    EXPECT_EQ(plan.lhs_matrix.stored_cols, 3);
    EXPECT_EQ(plan.lhs_matrix.leading_dimension, 3);
    EXPECT_TRUE(plan.lhs_matrix.backend_transpose);
    EXPECT_TRUE(plan.canLowerWithoutMaterialization());
}

TEST(ExpressionBatchedMatmulPlan, LogicalTransposeCancelsTransposedPhysicalView) {
    const auto lhs = layout({2, 3, 4}, {12, 1, 3});
    const auto rhs = denseMatmulTensorLayout({2, 3, 5});
    const auto output = denseMatmulTensorLayout({2, 4, 5});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output, true, false);

    EXPECT_EQ(plan.lhs_matrix.storage_kind, MatmulMatrixStorageKind::TransposedRowMajor);
    EXPECT_FALSE(plan.lhs_matrix.backend_transpose);
    EXPECT_EQ(plan.shape.output_dimensions, (std::vector<uint64_t>{2, 4, 5}));
}

TEST(ExpressionBatchedMatmulPlan, NonBlasMatrixPlaneIsMarkedForLaterMaterializationDecision) {
    const auto lhs = layout({2, 3, 4}, {50, 7, 2});
    const auto rhs = denseMatmulTensorLayout({2, 4, 5});
    const auto output = denseMatmulTensorLayout({2, 3, 5});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    EXPECT_EQ(plan.lhs_matrix.storage_kind, MatmulMatrixStorageKind::Unsupported);
    EXPECT_FALSE(plan.canAddressOperandsWithoutMaterialization());
    EXPECT_FALSE(plan.canLowerWithoutMaterialization());
}

TEST(ExpressionBatchedMatmulPlan, TransposedOutputViewIsMarkedForPostprocessRatherThanSilentlyAccepted) {
    const auto lhs = denseMatmulTensorLayout({2, 3, 4});
    const auto rhs = denseMatmulTensorLayout({2, 4, 5});
    const auto output = layout({2, 3, 5}, {15, 1, 3});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);

    EXPECT_EQ(plan.output_matrix.storage_kind, MatmulMatrixStorageKind::TransposedRowMajor);
    EXPECT_FALSE(plan.canWriteOutputWithoutPostprocess());
    EXPECT_FALSE(plan.canLowerWithoutMaterialization());
}

TEST(ExpressionBatchedMatmulPlan, TensorAdapterPreservesCustomStridesAndStorageOffset) {
    TensorPlacement cpu(TensorPlacement::MemDevices::CPU, 0);
    Tensor storage(cpu, TensorDescriptor(DataType::FP32, {256}));
    Tensor view = storage.aliasView({2, 3, 4}, {20, 6, 1}, 9);

    const MatmulTensorLayout adapted = matmulTensorLayout(view);
    EXPECT_EQ(adapted.dimensions, (std::vector<uint64_t>{2, 3, 4}));
    EXPECT_EQ(adapted.strides_elements, (std::vector<uint64_t>{20, 6, 1}));
    EXPECT_EQ(adapted.storage_element_offset, 9);
}

TEST(ExpressionBatchedMatmulPlan, RejectsOutputShapeAndLayoutContractViolations) {
    const auto lhs = denseMatmulTensorLayout({2, 3, 4});
    const auto rhs = denseMatmulTensorLayout({2, 4, 5});
    EXPECT_THROW((void)planBatchedMatmulLayout(lhs, rhs, denseMatmulTensorLayout({2, 3, 6})), std::runtime_error);
    EXPECT_THROW((void)planBatchedMatmulLayout(lhs, rhs, layout({2, 3, 5}, {15, 5})), std::runtime_error);
}

TEST(ExpressionBatchedMatmulPlan, GroupMaterializationLimitPreventsAccidentalHugeExpansion) {
    const auto lhs = denseMatmulTensorLayout({8, 1, 3, 4});
    const auto rhs = denseMatmulTensorLayout({1, 8, 4, 5});
    const auto output = denseMatmulTensorLayout({8, 8, 3, 5});
    const BatchedMatmulLayoutPlan plan = planBatchedMatmulLayout(lhs, rhs, output);
    ASSERT_EQ(plan.grouping.group_count, 8);
    EXPECT_THROW((void)materializeBatchedMatmulGroups(plan, 7), std::runtime_error);
}
