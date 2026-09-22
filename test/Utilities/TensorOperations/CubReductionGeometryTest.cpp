#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/Exceptions.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

TEST(CubReductionGeometry, SelectsBestSingleAxisPath) {
    const CubReductionGeometry scalar = CubReduction::analyzeGeometry({257}, 0);
    EXPECT_EQ(scalar.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_TRUE(scalar.reduced_axes_are_contiguous);
    EXPECT_EQ(scalar.outer_size, 1U);
    EXPECT_EQ(scalar.reduction_size, 257U);
    EXPECT_EQ(scalar.inner_size, 1U);
    EXPECT_EQ(scalar.output_elements, 1U);
    EXPECT_EQ(scalar.output_dimensions, (std::vector<uint64_t>{1}));

    const CubReductionGeometry contiguous = CubReduction::analyzeGeometry({2, 3, 4}, 2);
    EXPECT_EQ(contiguous.path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_TRUE(contiguous.reduced_axes_are_contiguous);
    EXPECT_EQ(contiguous.outer_size, 6U);
    EXPECT_EQ(contiguous.reduction_size, 4U);
    EXPECT_EQ(contiguous.inner_size, 1U);
    EXPECT_EQ(contiguous.output_elements, 6U);
    EXPECT_EQ(contiguous.output_dimensions, (std::vector<uint64_t>{2, 3, 1}));

    const CubReductionGeometry tiled = CubReduction::analyzeGeometry({2, 3, 4}, 1);
    EXPECT_EQ(tiled.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(tiled.reduced_axes_are_contiguous);
    EXPECT_EQ(tiled.outer_size, 2U);
    EXPECT_EQ(tiled.reduction_size, 3U);
    EXPECT_EQ(tiled.inner_size, 4U);
    EXPECT_EQ(tiled.output_elements, 8U);
    EXPECT_EQ(tiled.output_dimensions, (std::vector<uint64_t>{2, 1, 4}));
}

TEST(CubReductionGeometry, RejectsInvalidSingleAxisGeometry) {
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({}, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 3}, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 0, 3}, 1)), std::invalid_argument);
}

TEST(CubReductionGeometry, SelectsBestMultiAxisPathAndShapes) {
    const CubReductionGeometry all_axes = CubReduction::analyzeGeometry({2, 3, 4}, std::vector<uint32_t>{0, 1, 2});
    EXPECT_EQ(all_axes.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_TRUE(all_axes.reduced_axes_are_contiguous);
    EXPECT_EQ(all_axes.outer_size, 1U);
    EXPECT_EQ(all_axes.inner_size, 1U);
    EXPECT_EQ(all_axes.input_elements, 24U);
    EXPECT_EQ(all_axes.reduction_size, 24U);
    EXPECT_EQ(all_axes.output_elements, 1U);
    EXPECT_EQ(all_axes.output_dimensions, (std::vector<uint64_t>{1, 1, 1}));
    EXPECT_EQ(all_axes.squeezed_output_dimensions, (std::vector<uint64_t>{1}));

    const CubReductionGeometry suffix =
        CubReduction::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{2, 3});
    EXPECT_EQ(suffix.path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_TRUE(suffix.reduced_axes_are_contiguous);
    EXPECT_EQ(suffix.outer_size, 6U);
    EXPECT_EQ(suffix.reduction_size, 20U);
    EXPECT_EQ(suffix.inner_size, 1U);
    EXPECT_EQ(suffix.output_elements, 6U);
    EXPECT_EQ(suffix.output_dimensions, (std::vector<uint64_t>{2, 3, 1, 1}));
    EXPECT_EQ(suffix.squeezed_output_dimensions, (std::vector<uint64_t>{2, 3}));

    const CubReductionGeometry middle =
        CubReduction::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{1, 2});
    EXPECT_EQ(middle.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(middle.reduced_axes_are_contiguous);
    EXPECT_EQ(middle.outer_size, 2U);
    EXPECT_EQ(middle.reduction_size, 12U);
    EXPECT_EQ(middle.inner_size, 5U);
    EXPECT_EQ(middle.output_elements, 10U);
    EXPECT_EQ(middle.output_dimensions, (std::vector<uint64_t>{2, 1, 1, 5}));
    EXPECT_EQ(middle.squeezed_output_dimensions, (std::vector<uint64_t>{2, 5}));

    const CubReductionGeometry disjoint =
        CubReduction::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{1, 3});
    EXPECT_EQ(disjoint.path, CubReductionPath::ComposedDense);
    EXPECT_FALSE(disjoint.reduced_axes_are_contiguous);
    EXPECT_EQ(disjoint.outer_size, 0U);
    EXPECT_EQ(disjoint.reduction_size, 15U);
    EXPECT_EQ(disjoint.inner_size, 0U);
    EXPECT_EQ(disjoint.output_elements, 8U);
    EXPECT_EQ(disjoint.output_dimensions, (std::vector<uint64_t>{2, 1, 4, 1}));
    EXPECT_EQ(disjoint.squeezed_output_dimensions, (std::vector<uint64_t>{2, 4}));

    const CubReductionGeometry leading =
        CubReduction::analyzeGeometry({2, 3, 4}, std::vector<uint32_t>{0, 1});
    EXPECT_EQ(leading.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(leading.reduced_axes_are_contiguous);
    EXPECT_EQ(leading.outer_size, 1U);
    EXPECT_EQ(leading.reduction_size, 6U);
    EXPECT_EQ(leading.inner_size, 4U);
    EXPECT_EQ(leading.output_dimensions, (std::vector<uint64_t>{1, 1, 4}));
    EXPECT_EQ(leading.squeezed_output_dimensions, (std::vector<uint64_t>{4}));

    const CubReductionGeometry singleton_trailing =
        CubReduction::analyzeGeometry({2, 3, 1}, std::vector<uint32_t>{1});
    EXPECT_EQ(singleton_trailing.path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_TRUE(singleton_trailing.reduced_axes_are_contiguous);
    EXPECT_EQ(singleton_trailing.outer_size, 2U);
    EXPECT_EQ(singleton_trailing.reduction_size, 3U);
    EXPECT_EQ(singleton_trailing.inner_size, 1U);
    EXPECT_EQ(singleton_trailing.output_elements, 2U);

    const CubReductionGeometry singleton_retained =
        CubReduction::analyzeGeometry({1, 3, 1}, std::vector<uint32_t>{1});
    EXPECT_EQ(singleton_retained.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_TRUE(singleton_retained.reduced_axes_are_contiguous);
    EXPECT_EQ(singleton_retained.outer_size, 1U);
    EXPECT_EQ(singleton_retained.reduction_size, 3U);
    EXPECT_EQ(singleton_retained.inner_size, 1U);
    EXPECT_EQ(singleton_retained.output_elements, 1U);
}

TEST(CubReductionGeometry, RejectsInvalidMultiAxisGeometry) {
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 3}, std::vector<uint32_t>{})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 3}, std::vector<uint32_t>{0, 0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 3}, std::vector<uint32_t>{1, 0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 3}, std::vector<uint32_t>{0, 2})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 0, 3}, std::vector<uint32_t>{0, 2})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{1, 0})),
                 std::invalid_argument);
}

TEST(CubReductionGeometry, SupportsRankBeyondFormerCudnnDescriptorLimit) {
    const std::vector<uint64_t> dimensions{2, 1, 2, 1, 2, 1, 2, 1, 2};
    const std::vector<uint32_t> axes{0, 2, 4, 6};
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, axes);

    EXPECT_EQ(geometry.rank, 9U);
    EXPECT_EQ(geometry.path, CubReductionPath::ComposedDense);
    EXPECT_EQ(geometry.input_elements, 32U);
    EXPECT_EQ(geometry.reduction_size, 16U);
    EXPECT_EQ(geometry.output_elements, 2U);
    EXPECT_EQ(geometry.output_dimensions, (std::vector<uint64_t>{1, 1, 1, 1, 1, 1, 1, 1, 2}));
}

TEST(CubReductionGeometry, SelectsPermutationAwareTiledPathForDensePhysicalPermutation) {
    // Logical [K,I,J] view over physically dense [I,J,K], matching einsum("ijk->ki") after its zero-copy permutation.
    constexpr uint64_t I = 7;
    constexpr uint64_t J = 5;
    constexpr uint64_t K = 32;
    const CubReductionGeometry permuted = CubReduction::analyzeGeometry(
        {K, I, J}, {1, J * K, K}, std::vector<uint32_t>{2});

    EXPECT_EQ(permuted.path, CubReductionPath::TiledFixedSegment);
    EXPECT_FALSE(permuted.device_transform_uses_affine_stride);
    EXPECT_TRUE(permuted.physical_layout_is_dense_permutation);
    EXPECT_EQ(permuted.physical_non_singleton_axis_order, (std::vector<uint32_t>{1, 2, 0}));
    ASSERT_TRUE(permuted.permutation_aware_tiled_geometry.has_value());

    const CubReductionPermutationAwareTiledGeometry& tiled = permuted.permutation_aware_tiled_geometry.value();
    EXPECT_EQ(tiled.physical_outer_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(tiled.physical_reduction_axes, (std::vector<uint32_t>{2}));
    EXPECT_EQ(tiled.physical_inner_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(tiled.logical_non_singleton_retained_axes, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(tiled.outer_size, I);
    EXPECT_EQ(tiled.reduction_size, J);
    EXPECT_EQ(tiled.inner_size, K);
    EXPECT_EQ(tiled.retained_output_order, CubReductionTiledRetainedOutputOrder::PermutedInnerOuter);
    EXPECT_EQ(permuted.outer_size, I);
    EXPECT_EQ(permuted.reduction_size, J);
    EXPECT_EQ(permuted.inner_size, K);
    EXPECT_TRUE(permuted.tiled_output_permuted);
    EXPECT_TRUE(permuted.tiled_output_shared_transpose);
    EXPECT_EQ(permuted.tiled_output_outer_stride, 1U);
    EXPECT_EQ(permuted.tiled_output_inner_stride, I);
}

TEST(CubReductionGeometry, DetectsNaturalRetainedOrderForDensePhysicalPermutation) {
    // Logical [J,I,K] over physical [I,J,K]. Reducing J leaves the natural physical [I,K] output order.
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(
        {5, 7, 32}, {32, 160, 1}, std::vector<uint32_t>{0});

    EXPECT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
    EXPECT_EQ(geometry.physical_non_singleton_axis_order, (std::vector<uint32_t>{1, 0, 2}));
    ASSERT_TRUE(geometry.permutation_aware_tiled_geometry.has_value());
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->physical_outer_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->physical_reduction_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->physical_inner_axes, (std::vector<uint32_t>{2}));
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->retained_output_order,
              CubReductionTiledRetainedOutputOrder::NaturalOuterInner);
    EXPECT_EQ(geometry.outer_size, 7U);
    EXPECT_EQ(geometry.reduction_size, 5U);
    EXPECT_EQ(geometry.inner_size, 32U);
    EXPECT_FALSE(geometry.tiled_output_permuted);
    EXPECT_FALSE(geometry.tiled_output_shared_transpose);
    EXPECT_EQ(geometry.tiled_output_outer_stride, 32U);
    EXPECT_EQ(geometry.tiled_output_inner_stride, 1U);
}

TEST(CubReductionGeometry, DensePhysicalPermutationPlanningIsStorageOffsetInvariant) {
    Tensor storage(TensorPlacement(TensorPlacement::MemDevices::CPU),
                   TensorDescriptor(DataType::FP32, {8, 5, 32}));
    Tensor offset_view = storage.aliasView({32, 7, 5}, {1, 160, 32}, 160);
    ASSERT_EQ(offset_view.getStorageElementOffset(), 160U);

    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(
        offset_view.getDimensions(), offset_view.getStridesElements(), std::vector<uint32_t>{2});
    EXPECT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
    ASSERT_TRUE(geometry.permutation_aware_tiled_geometry.has_value());
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->outer_size, 7U);
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->reduction_size, 5U);
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->inner_size, 32U);
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->retained_output_order,
              CubReductionTiledRetainedOutputOrder::PermutedInnerOuter);
    EXPECT_TRUE(geometry.tiled_output_shared_transpose);
    EXPECT_EQ(geometry.tiled_output_outer_stride, 1U);
    EXPECT_EQ(geometry.tiled_output_inner_stride, 7U);
}

TEST(CubReductionGeometry, DensePhysicalPermutationDetectionHandlesSingletonAxes) {
    // Singleton axes are memory-layout neutral and must not prevent recognizing a genuine dense physical permutation.
    const CubReductionGeometry permuted = CubReduction::analyzeGeometry(
        {4, 1, 2, 3}, {1, 12, 12, 4}, std::vector<uint32_t>{3});

    EXPECT_EQ(permuted.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(permuted.physical_layout_is_dense_permutation);
    EXPECT_EQ(permuted.physical_non_singleton_axis_order, (std::vector<uint32_t>{2, 3, 0}));
    ASSERT_TRUE(permuted.permutation_aware_tiled_geometry.has_value());
    const CubReductionPermutationAwareTiledGeometry& tiled = permuted.permutation_aware_tiled_geometry.value();
    EXPECT_EQ(tiled.physical_outer_axes, (std::vector<uint32_t>{2}));
    EXPECT_EQ(tiled.physical_reduction_axes, (std::vector<uint32_t>{3}));
    EXPECT_EQ(tiled.physical_inner_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(tiled.logical_non_singleton_retained_axes, (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(tiled.retained_output_order, CubReductionTiledRetainedOutputOrder::PermutedInnerOuter);
    EXPECT_EQ(permuted.outer_size, 2U);
    EXPECT_EQ(permuted.reduction_size, 3U);
    EXPECT_EQ(permuted.inner_size, 4U);
    EXPECT_TRUE(permuted.tiled_output_permuted);
    EXPECT_TRUE(permuted.tiled_output_shared_transpose);
    EXPECT_EQ(permuted.tiled_output_outer_stride, 1U);
    EXPECT_EQ(permuted.tiled_output_inner_stride, 2U);
}

TEST(CubReductionGeometry, PhysicallyContiguousReductionBlockCanUseTiledPathWhenLogicalAxesAreDisjoint) {
    // Physical order is [logical 1, logical 0, logical 2, logical 3]. Logical reduction axes {0,2}
    // are disjoint, but physically they form one contiguous reduction block in dense [2,3,5,7] storage.
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(
        {3, 2, 5, 7}, {35, 105, 7, 1}, std::vector<uint32_t>{0, 2});

    EXPECT_FALSE(geometry.reduced_axes_are_contiguous);
    EXPECT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
    ASSERT_TRUE(geometry.permutation_aware_tiled_geometry.has_value());
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->physical_outer_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->physical_reduction_axes,
              (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(geometry.permutation_aware_tiled_geometry->physical_inner_axes, (std::vector<uint32_t>{3}));
    EXPECT_EQ(geometry.outer_size, 2U);
    EXPECT_EQ(geometry.reduction_size, 15U);
    EXPECT_EQ(geometry.inner_size, 7U);
    EXPECT_FALSE(geometry.tiled_output_permuted);
}

TEST(CubReductionGeometry, RejectsGappedOverlappingAndBroadcastStorageAfterDelete) {
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {2, 2, 3}, {1, 7, 2}, std::vector<uint32_t>{2}),
                 NotImplementedException);
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {2, 2, 3}, {1, 1, 2}, std::vector<uint32_t>{2}),
                 NotImplementedException);
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {2, 2, 3}, {0, 6, 2}, std::vector<uint32_t>{2}),
                 NotImplementedException);
}

TEST(CubReductionGeometry, SplitPhysicalReductionIsUnsupportedAndDirect2BOwnsRetainedRotation) {
    // Physical order is [2,0,1,3]. Reduction axes {1,2} are split by retained axis 0 in physical order.
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {3, 5, 2, 7}, {35, 7, 105, 1}, std::vector<uint32_t>{1, 2}),
                 NotImplementedException);

    // Reducing axis 0 is the ordained VIEW-DIRECT-2B [A,R,B,P] -> [B,A,P] retained rotation.
    const CubReductionGeometry direct2b = CubReduction::analyzeGeometry(
        {3, 5, 2, 7}, {35, 7, 105, 1}, std::vector<uint32_t>{0});
    EXPECT_TRUE(direct2b.physical_layout_is_dense_permutation);
    EXPECT_EQ(direct2b.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(direct2b.payload_transpose_tiled_geometry.has_value());
}

TEST(CubReductionGeometry, CompactPhysicalPermutationTrailingReductionUsesContiguousSegmentsWhenOutputOrderMatches) {
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(
        {2, 3, 4}, {1, 8, 2}, std::vector<uint32_t>{0});

    EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
    EXPECT_EQ(geometry.physical_non_singleton_axis_order, (std::vector<uint32_t>{1, 2, 0}));
    EXPECT_EQ(geometry.output_elements, 12U);
    EXPECT_EQ(geometry.reduction_size, 2U);
    EXPECT_TRUE(geometry.permutation_aware_contiguous_segments);
    EXPECT_EQ(geometry.path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_FALSE(geometry.permutation_aware_tiled_geometry.has_value());
}

TEST(CubReductionGeometry, CompactPhysicalPermutationTrailingRetainedOrderMismatchIsUnsupported) {
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {2, 3, 4}, {4, 8, 1}, std::vector<uint32_t>{2}),
                 NotImplementedException);
}

TEST(CubReductionGeometry, CompactPhysicalPermutationFullReductionUsesDeviceTransformReduce) {
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(
        {2, 3, 4}, {1, 8, 2}, std::vector<uint32_t>{0, 1, 2});

    EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
    EXPECT_EQ(geometry.physical_non_singleton_axis_order, (std::vector<uint32_t>{1, 2, 0}));
    EXPECT_EQ(geometry.output_elements, 1U);
    EXPECT_EQ(geometry.reduction_size, 24U);
    EXPECT_EQ(geometry.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_FALSE(geometry.device_transform_uses_affine_stride);
    EXPECT_FALSE(geometry.permutation_aware_tiled_geometry.has_value());
}

TEST(CubReductionGeometry, DenseAndAffinePathSelectionRemainsUnchanged) {
    const CubReductionGeometry dense = CubReduction::analyzeGeometry(
        {2, 3, 4}, std::vector<uint32_t>{1});
    EXPECT_EQ(dense.path, CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(dense.physical_layout_is_dense_permutation);
    EXPECT_FALSE(dense.permutation_aware_tiled_geometry.has_value());

    const CubReductionGeometry diagonal = CubReduction::analyzeGeometry(
        {3}, {4}, std::vector<uint32_t>{0});
    EXPECT_EQ(diagonal.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_TRUE(diagonal.device_transform_uses_affine_stride);
    EXPECT_EQ(diagonal.affine_input_stride, 4U);
    EXPECT_FALSE(diagonal.physical_layout_is_dense_permutation);
    EXPECT_FALSE(diagonal.permutation_aware_tiled_geometry.has_value());
}

TEST(CubReductionGeometry, StampsCollapsedDenseRunsForAlexNetBiasGradient) {
    const CubReductionGeometry geometry =
        CubReduction::analyzeGeometry({512, 64, 55, 55}, std::vector<uint32_t>{0, 2, 3});

    // Geometry analysis owns the final execution family. This dense R|K|R layout is therefore classified as
    // ComposedDense immediately; value/ARG planners only choose the legal stage order within that family.
    EXPECT_EQ(geometry.path, CubReductionPath::ComposedDense);
    ASSERT_TRUE(geometry.dense_run_geometry.has_value());
    const CubReductionDenseRunGeometry& runs = geometry.dense_run_geometry.value();
    ASSERT_EQ(runs.runs.size(), 3U);
    EXPECT_EQ(runs.reduced_run_count, 2U);
    EXPECT_EQ(runs.retained_run_count, 1U);

    EXPECT_EQ(runs.runs[0].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(runs.runs[0].extent, 512U);
    EXPECT_EQ(runs.runs[0].physical_stride, 64U * 55U * 55U);
    EXPECT_EQ(runs.runs[0].domain_stride, 55U * 55U);

    EXPECT_EQ(runs.runs[1].kind, CubReductionDenseRunKind::Retained);
    EXPECT_EQ(runs.runs[1].extent, 64U);
    EXPECT_EQ(runs.runs[1].physical_stride, 55U * 55U);
    EXPECT_EQ(runs.runs[1].domain_stride, 1U);

    EXPECT_EQ(runs.runs[2].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(runs.runs[2].extent, 55U * 55U);
    EXPECT_EQ(runs.runs[2].physical_stride, 1U);
    EXPECT_EQ(runs.runs[2].domain_stride, 1U);
}

TEST(CubReductionGeometry, ValuePlanningSelectsGeneralComposedDenseForRkrBeforeStamping) {
    const std::vector<uint64_t> dimensions{512, 64, 55, 55};
    const std::vector<uint32_t> axes{0, 2, 3};

    for (CubReductionOp op : {CubReductionOp::Sum,
                              CubReductionOp::Min,
                              CubReductionOp::Max,
                              CubReductionOp::Product,
                              CubReductionOp::Mean,
                              CubReductionOp::L1Norm,
                              CubReductionOp::L2Norm,
                              CubReductionOp::SumSquares}) {
        const CubReductionGeometry value = CubReduction::analyzeValueGeometry(op, dimensions, axes);
        EXPECT_EQ(value.path, CubReductionPath::ComposedDense);
    }
}

TEST(CubReductionGeometry, StampsGeneralAlternatingDenseRunGeometry) {
    const CubReductionGeometry geometry =
        CubReduction::analyzeGeometry({31, 17, 29, 19, 23}, std::vector<uint32_t>{0, 2, 4});

    ASSERT_TRUE(geometry.dense_run_geometry.has_value());
    const CubReductionDenseRunGeometry& dense = geometry.dense_run_geometry.value();
    ASSERT_EQ(dense.runs.size(), 5U);
    EXPECT_EQ(dense.reduced_run_count, 3U);
    EXPECT_EQ(dense.retained_run_count, 2U);

    EXPECT_EQ(dense.runs[0].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(dense.runs[0].extent, 31U);
    EXPECT_EQ(dense.runs[0].physical_stride, 17U * 29U * 19U * 23U);
    EXPECT_EQ(dense.runs[0].domain_stride, 29U * 23U);

    EXPECT_EQ(dense.runs[1].kind, CubReductionDenseRunKind::Retained);
    EXPECT_EQ(dense.runs[1].extent, 17U);
    EXPECT_EQ(dense.runs[1].physical_stride, 29U * 19U * 23U);
    EXPECT_EQ(dense.runs[1].domain_stride, 19U);

    EXPECT_EQ(dense.runs[2].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(dense.runs[2].extent, 29U);
    EXPECT_EQ(dense.runs[2].physical_stride, 19U * 23U);
    EXPECT_EQ(dense.runs[2].domain_stride, 23U);

    EXPECT_EQ(dense.runs[3].kind, CubReductionDenseRunKind::Retained);
    EXPECT_EQ(dense.runs[3].extent, 19U);
    EXPECT_EQ(dense.runs[3].physical_stride, 23U);
    EXPECT_EQ(dense.runs[3].domain_stride, 1U);

    EXPECT_EQ(dense.runs[4].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(dense.runs[4].extent, 23U);
    EXPECT_EQ(dense.runs[4].physical_stride, 1U);
    EXPECT_EQ(dense.runs[4].domain_stride, 1U);
}

TEST(CubReductionGeometry, ValuePlanningSelectsGeneralComposedDenseForRkrkrBeforeStamping) {
    const std::vector<uint64_t> dimensions{31, 17, 29, 19, 23};
    const std::vector<uint32_t> axes{0, 2, 4};

    for (CubReductionOp op : {CubReductionOp::Sum,
                              CubReductionOp::Min,
                              CubReductionOp::Max,
                              CubReductionOp::Product,
                              CubReductionOp::Mean,
                              CubReductionOp::L1Norm,
                              CubReductionOp::L2Norm,
                              CubReductionOp::SumSquares}) {
        const CubReductionGeometry value = CubReduction::analyzeValueGeometry(op, dimensions, axes);
        EXPECT_EQ(value.path, CubReductionPath::ComposedDense);
    }
}

TEST(CubReductionGeometry, ValuePlanningSelectsComposedDenseForTrailingRetainedTwoRunTopologies) {
    for (CubReductionOp op : {CubReductionOp::Sum,
                              CubReductionOp::Min,
                              CubReductionOp::Max,
                              CubReductionOp::Product,
                              CubReductionOp::Mean,
                              CubReductionOp::L1Norm,
                              CubReductionOp::L2Norm,
                              CubReductionOp::SumSquares}) {
        const CubReductionGeometry rkrk =
            CubReduction::analyzeValueGeometry(op, {2, 3, 5, 7}, {0, 2});
        EXPECT_EQ(rkrk.path, CubReductionPath::ComposedDense);

        const CubReductionGeometry krkrk =
            CubReduction::analyzeValueGeometry(op, {2, 3, 5, 7, 11}, {1, 3});
        EXPECT_EQ(krkrk.path, CubReductionPath::ComposedDense);
    }
}

TEST(CubReductionGeometry, DenseValuePlannerHandlesArbitraryAlternatingRunsAndSingletonSeparators) {
    for (CubReductionOp op : {CubReductionOp::Sum,
                              CubReductionOp::Min,
                              CubReductionOp::Max,
                              CubReductionOp::Product,
                              CubReductionOp::Mean,
                              CubReductionOp::L1Norm,
                              CubReductionOp::L2Norm,
                              CubReductionOp::SumSquares}) {
        const CubReductionGeometry trailing_reduced = CubReduction::analyzeValueGeometry(
            op, {2, 3, 5, 7, 11, 13, 17}, {0, 2, 4, 6});
        EXPECT_EQ(trailing_reduced.path, CubReductionPath::ComposedDense);

        const CubReductionGeometry trailing_retained = CubReduction::analyzeValueGeometry(
            op, {2, 3, 5, 7, 11, 13, 17, 19}, {0, 2, 4, 6});
        EXPECT_EQ(trailing_retained.path, CubReductionPath::ComposedDense);

        // Axis 1 is a retained singleton between reduced non-singleton axes 0 and 2. The generic planner absorbs it
        // into one direct contiguous stage instead of treating the public non-contiguous axis list as strided.
        const CubReductionGeometry singleton_separator =
            CubReduction::analyzeValueGeometry(op, {2, 1, 3, 5}, {0, 2});
        EXPECT_EQ(singleton_separator.path, CubReductionPath::ComposedDense);

        // Reducing only singleton axes still needs one direct pass for transform/finalize/conversion/scaling.
        const CubReductionGeometry singleton_only =
            CubReduction::analyzeValueGeometry(op, {3, 1, 5, 1, 7}, {1, 3});
        EXPECT_EQ(singleton_only.path, CubReductionPath::ComposedDense);
    }
}

TEST(CubReductionGeometry, DenseValueCompositionValidatesPerStageRatherThanMonolithicReductionSize) {
    const uint64_t run_extent = 65536;
    const CubReductionGeometry structural =
        CubReduction::analyzeGeometry({run_extent, 2, run_extent, 2}, std::vector<uint32_t>{0, 2});
    EXPECT_EQ(structural.path, CubReductionPath::ComposedDense);
    EXPECT_GT(structural.reduction_size, static_cast<uint64_t>(std::numeric_limits<int>::max()));

    // The logical reduction domain is > INT_MAX, but each TiledFixedSegment stage reduces only 65536 elements.
    const CubReductionGeometry sum = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, {run_extent, 2, run_extent, 2}, {0, 2});
    EXPECT_EQ(sum.path, CubReductionPath::ComposedDense);

    // Every dense value operation validates its direct component stages independently rather than inheriting any
    // obsolete monolithic fallback limit.
    for (CubReductionOp op : {CubReductionOp::Min,
                              CubReductionOp::Max,
                              CubReductionOp::Product,
                              CubReductionOp::Mean,
                              CubReductionOp::L1Norm,
                              CubReductionOp::L2Norm,
                              CubReductionOp::SumSquares}) {
        EXPECT_EQ(CubReduction::analyzeValueGeometry(
                      op, {run_extent, 2, run_extent, 2}, {0, 2})
                      .path,
                  CubReductionPath::ComposedDense);
    }
}

TEST(CubReductionGeometry, OnlyCubFixedSizeSegmentedPathsInheritTheIntSegmentSizeLimit) {
    const uint64_t above_int_max = static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1ULL;

    // TiledFixedSegment is Thor-owned. Its geometry and kernel loops are uint64_t, so a middle reduced block may be
    // larger than INT_MAX even though CUB's fixed-size segmented API cannot represent such a segment size.
    const CubReductionGeometry tiled =
        CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {2, above_int_max, 2}, {1});
    EXPECT_EQ(tiled.path, CubReductionPath::TiledFixedSegment);
    EXPECT_EQ(tiled.reduction_size, above_int_max);

    // The same must remain true when such a Tiled stage is one pass inside a disjoint-axis dense composition.
    const CubReductionGeometry composed =
        CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {above_int_max, 2, 3, 2}, {0, 2});
    EXPECT_EQ(composed.path, CubReductionPath::ComposedDense);
    EXPECT_GT(composed.reduction_size, static_cast<uint64_t>(std::numeric_limits<int>::max()));

    // Full reductions use DeviceReduce with an int64_t item count and likewise do not inherit the fixed-segment cap.
    const CubReductionGeometry full =
        CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {above_int_max}, {0});
    EXPECT_EQ(full.path, CubReductionPath::DeviceTransformReduce);

    // ContiguousFixedSegment still calls DeviceSegmentedReduce's fixed-size overload, whose segment_size parameter is
    // an int. Keep rejecting that geometry until we deliberately replace or widen that primitive.
    EXPECT_THROW((void)CubReduction::analyzeValueGeometry(
                     CubReductionOp::Sum, {2, above_int_max}, {1}),
                 std::invalid_argument);
}

TEST(CubReductionGeometry, DenseRunPlanningCollapsesAdjacentRolesAndIgnoresSingletonSeparators) {
    const CubReductionGeometry adjacent =
        CubReduction::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{0, 1, 3});
    ASSERT_TRUE(adjacent.dense_run_geometry.has_value());
    ASSERT_EQ(adjacent.dense_run_geometry->runs.size(), 3U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].extent, 6U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].physical_stride, 20U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].domain_stride, 5U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].kind, CubReductionDenseRunKind::Retained);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].extent, 4U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].physical_stride, 5U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].domain_stride, 1U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].extent, 5U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].physical_stride, 1U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].domain_stride, 1U);

    // The retained singleton does not consume storage or an output coordinate, so it must not split the two reduced
    // spans into separate runtime runs. This gives the executor the physically meaningful R(2*3)|K(5) traversal.
    const CubReductionGeometry singleton_separator =
        CubReduction::analyzeGeometry({2, 1, 3, 5}, std::vector<uint32_t>{0, 2});
    ASSERT_TRUE(singleton_separator.dense_run_geometry.has_value());
    ASSERT_EQ(singleton_separator.dense_run_geometry->runs.size(), 2U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].kind, CubReductionDenseRunKind::Reduced);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].extent, 6U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].physical_stride, 5U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].domain_stride, 1U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].kind, CubReductionDenseRunKind::Retained);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].extent, 5U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].physical_stride, 1U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].domain_stride, 1U);
}

TEST(CubReductionGeometry, DenseRunPlanningDoesNotPretendArbitraryStridesAreDense) {
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {2, 3, 4}, {20, 4, 1}, std::vector<uint32_t>{0, 2}),
                 NotImplementedException);
}


TEST(CubReductionGeometry, DeleteRejectsFormerLogicalIndexMapperDomains) {
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {65535, 3, 32768}, {1, 1, 1}, std::vector<uint32_t>{0, 2}),
                 NotImplementedException);
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {2, 3, 4}, {uint64_t{1} << 32, 4, 1}, std::vector<uint32_t>{0, 2}),
                 NotImplementedException);
}
