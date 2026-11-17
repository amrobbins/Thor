#include "Utilities/TensorOperations/Reduction/ArgReducer.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"
#include "Utilities/Exceptions.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

TEST(ReducerGeometry, SelectsBestSingleAxisPath) {
    const ReductionGeometry scalar = Reducer::analyzeGeometry({257}, 0);
    EXPECT_EQ(scalar.path, ReductionLayout::SingleOutput);
    EXPECT_TRUE(scalar.reduced_axes_are_contiguous);
    EXPECT_EQ(scalar.outer_size, 1U);
    EXPECT_EQ(scalar.reduction_size, 257U);
    EXPECT_EQ(scalar.inner_size, 1U);
    EXPECT_EQ(scalar.output_elements, 1U);
    EXPECT_EQ(scalar.output_dimensions, (std::vector<uint64_t>{1}));

    const ReductionGeometry contiguous = Reducer::analyzeGeometry({2, 3, 4}, 2);
    EXPECT_EQ(contiguous.path, ReductionLayout::TrailingBlock);
    EXPECT_TRUE(contiguous.reduced_axes_are_contiguous);
    EXPECT_EQ(contiguous.outer_size, 6U);
    EXPECT_EQ(contiguous.reduction_size, 4U);
    EXPECT_EQ(contiguous.inner_size, 1U);
    EXPECT_EQ(contiguous.output_elements, 6U);
    EXPECT_EQ(contiguous.output_dimensions, (std::vector<uint64_t>{2, 3, 1}));

    const ReductionGeometry tiled = Reducer::analyzeGeometry({2, 3, 4}, 1);
    EXPECT_EQ(tiled.path, ReductionLayout::MiddleBlock);
    EXPECT_TRUE(tiled.reduced_axes_are_contiguous);
    EXPECT_EQ(tiled.outer_size, 2U);
    EXPECT_EQ(tiled.reduction_size, 3U);
    EXPECT_EQ(tiled.inner_size, 4U);
    EXPECT_EQ(tiled.output_elements, 8U);
    EXPECT_EQ(tiled.output_dimensions, (std::vector<uint64_t>{2, 1, 4}));
}

TEST(ReducerGeometry, RejectsInvalidSingleAxisGeometry) {
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({}, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({2, 3}, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({2, 0, 3}, 1)), std::invalid_argument);
}

TEST(ReducerGeometry, SelectsBestMultiAxisPathAndShapes) {
    const ReductionGeometry all_axes = Reducer::analyzeGeometry({2, 3, 4}, std::vector<uint32_t>{0, 1, 2});
    EXPECT_EQ(all_axes.path, ReductionLayout::SingleOutput);
    EXPECT_TRUE(all_axes.reduced_axes_are_contiguous);
    EXPECT_EQ(all_axes.outer_size, 1U);
    EXPECT_EQ(all_axes.inner_size, 1U);
    EXPECT_EQ(all_axes.input_elements, 24U);
    EXPECT_EQ(all_axes.reduction_size, 24U);
    EXPECT_EQ(all_axes.output_elements, 1U);
    EXPECT_EQ(all_axes.output_dimensions, (std::vector<uint64_t>{1, 1, 1}));
    EXPECT_EQ(all_axes.squeezed_output_dimensions, (std::vector<uint64_t>{1}));

    const ReductionGeometry suffix =
        Reducer::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{2, 3});
    EXPECT_EQ(suffix.path, ReductionLayout::TrailingBlock);
    EXPECT_TRUE(suffix.reduced_axes_are_contiguous);
    EXPECT_EQ(suffix.outer_size, 6U);
    EXPECT_EQ(suffix.reduction_size, 20U);
    EXPECT_EQ(suffix.inner_size, 1U);
    EXPECT_EQ(suffix.output_elements, 6U);
    EXPECT_EQ(suffix.output_dimensions, (std::vector<uint64_t>{2, 3, 1, 1}));
    EXPECT_EQ(suffix.squeezed_output_dimensions, (std::vector<uint64_t>{2, 3}));

    const ReductionGeometry middle =
        Reducer::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{1, 2});
    EXPECT_EQ(middle.path, ReductionLayout::MiddleBlock);
    EXPECT_TRUE(middle.reduced_axes_are_contiguous);
    EXPECT_EQ(middle.outer_size, 2U);
    EXPECT_EQ(middle.reduction_size, 12U);
    EXPECT_EQ(middle.inner_size, 5U);
    EXPECT_EQ(middle.output_elements, 10U);
    EXPECT_EQ(middle.output_dimensions, (std::vector<uint64_t>{2, 1, 1, 5}));
    EXPECT_EQ(middle.squeezed_output_dimensions, (std::vector<uint64_t>{2, 5}));

    const ReductionGeometry disjoint =
        Reducer::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{1, 3});
    EXPECT_EQ(disjoint.path, ReductionLayout::DisjointRuns);
    EXPECT_FALSE(disjoint.reduced_axes_are_contiguous);
    EXPECT_EQ(disjoint.outer_size, 0U);
    EXPECT_EQ(disjoint.reduction_size, 15U);
    EXPECT_EQ(disjoint.inner_size, 0U);
    EXPECT_EQ(disjoint.output_elements, 8U);
    EXPECT_EQ(disjoint.output_dimensions, (std::vector<uint64_t>{2, 1, 4, 1}));
    EXPECT_EQ(disjoint.squeezed_output_dimensions, (std::vector<uint64_t>{2, 4}));

    const ReductionGeometry leading =
        Reducer::analyzeGeometry({2, 3, 4}, std::vector<uint32_t>{0, 1});
    EXPECT_EQ(leading.path, ReductionLayout::MiddleBlock);
    EXPECT_TRUE(leading.reduced_axes_are_contiguous);
    EXPECT_EQ(leading.outer_size, 1U);
    EXPECT_EQ(leading.reduction_size, 6U);
    EXPECT_EQ(leading.inner_size, 4U);
    EXPECT_EQ(leading.output_dimensions, (std::vector<uint64_t>{1, 1, 4}));
    EXPECT_EQ(leading.squeezed_output_dimensions, (std::vector<uint64_t>{4}));

    const ReductionGeometry singleton_trailing =
        Reducer::analyzeGeometry({2, 3, 1}, std::vector<uint32_t>{1});
    EXPECT_EQ(singleton_trailing.path, ReductionLayout::TrailingBlock);
    EXPECT_TRUE(singleton_trailing.reduced_axes_are_contiguous);
    EXPECT_EQ(singleton_trailing.outer_size, 2U);
    EXPECT_EQ(singleton_trailing.reduction_size, 3U);
    EXPECT_EQ(singleton_trailing.inner_size, 1U);
    EXPECT_EQ(singleton_trailing.output_elements, 2U);

    const ReductionGeometry singleton_retained =
        Reducer::analyzeGeometry({1, 3, 1}, std::vector<uint32_t>{1});
    EXPECT_EQ(singleton_retained.path, ReductionLayout::SingleOutput);
    EXPECT_TRUE(singleton_retained.reduced_axes_are_contiguous);
    EXPECT_EQ(singleton_retained.outer_size, 1U);
    EXPECT_EQ(singleton_retained.reduction_size, 3U);
    EXPECT_EQ(singleton_retained.inner_size, 1U);
    EXPECT_EQ(singleton_retained.output_elements, 1U);
}

TEST(ReducerGeometry, RejectsInvalidMultiAxisGeometry) {
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({2, 3}, std::vector<uint32_t>{})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({2, 3}, std::vector<uint32_t>{0, 0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({2, 3}, std::vector<uint32_t>{1, 0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({2, 3}, std::vector<uint32_t>{0, 2})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer::analyzeGeometry({2, 0, 3}, std::vector<uint32_t>{0, 2})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer(ReductionOp::Sum, std::vector<uint32_t>{})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer(ReductionOp::Sum, std::vector<uint32_t>{0, 0})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Reducer(ReductionOp::Sum, std::vector<uint32_t>{1, 0})),
                 std::invalid_argument);
}

TEST(ReducerGeometry, SupportsRankBeyondFormerCudnnDescriptorLimit) {
    const std::vector<uint64_t> dimensions{2, 1, 2, 1, 2, 1, 2, 1, 2};
    const std::vector<uint32_t> axes{0, 2, 4, 6};
    const ReductionGeometry geometry = Reducer::analyzeGeometry(dimensions, axes);

    EXPECT_EQ(geometry.rank, 9U);
    EXPECT_EQ(geometry.path, ReductionLayout::DisjointRuns);
    EXPECT_EQ(geometry.input_elements, 32U);
    EXPECT_EQ(geometry.reduction_size, 16U);
    EXPECT_EQ(geometry.output_elements, 2U);
    EXPECT_EQ(geometry.output_dimensions, (std::vector<uint64_t>{1, 1, 1, 1, 1, 1, 1, 1, 2}));
}

namespace {

// The reducer refuses a rotated-output view (F / G) and splitRefusedView says how a caller runs it: the dense reduction
// of [a, r, b x p] over axis 1, then the copy of that output read as [b, a, p].
void expectRotatedSplit(const std::vector<uint64_t>& dims,
                        const std::vector<uint64_t>& strides,
                        const std::vector<uint32_t>& axes,
                        uint64_t a,
                        uint64_t r,
                        uint64_t b,
                        uint64_t p) {
    EXPECT_THROW((void)Reducer::analyzeGeometry(dims, strides, axes), NotImplementedException);
    for (ReductionOp op : {ReductionOp::Sum, ReductionOp::Max, ReductionOp::Mean, ReductionOp::L2Norm})
        EXPECT_THROW((void)Reducer::analyzeValueGeometry(op, dims, strides, axes), NotImplementedException);
    const std::optional<ViewSplit> split = Reducer::splitRefusedView(dims, strides, axes);
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->kind, ViewSplit::Kind::RotatedOutput);
    EXPECT_EQ(split->dense_dimensions, (std::vector<uint64_t>{a, r, b * p}));
    EXPECT_EQ(split->dense_axes, std::vector<uint32_t>{1});
    EXPECT_EQ(split->a, a);
    EXPECT_EQ(split->b, b);
    EXPECT_EQ(split->p, p);
}

}  // namespace

TEST(ReducerGeometry, RefusesARotatedOutputPermutationAndSplitsIt) {
    // Logical [K,I,J] view over physically dense [I,J,K], matching einsum("ijk->ki") after its zero-copy permutation:
    // the output [K, I] is the rotation of the dense [I, K].
    constexpr uint64_t I = 7;
    constexpr uint64_t J = 5;
    constexpr uint64_t K = 32;
    expectRotatedSplit({K, I, J}, {1, J * K, K}, {2}, I, J, K, 1);
}

TEST(ReducerGeometry, DetectsNaturalRetainedOrderForDensePhysicalPermutation) {
    // Logical [J,I,K] over physical [I,J,K]. Reducing J leaves the natural physical [I,K] output order.
    const ReductionGeometry geometry = Reducer::analyzeGeometry(
        {5, 7, 32}, {32, 160, 1}, std::vector<uint32_t>{0});

    EXPECT_EQ(geometry.path, ReductionLayout::MiddleBlock);
    EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
    EXPECT_EQ(geometry.physical_non_singleton_axis_order, (std::vector<uint32_t>{1, 0, 2}));
    ASSERT_TRUE(geometry.permuted_layout.has_value());
    EXPECT_EQ(geometry.permuted_layout->physical_outer_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(geometry.permuted_layout->physical_reduction_axes, (std::vector<uint32_t>{0}));
    EXPECT_EQ(geometry.permuted_layout->physical_inner_axes, (std::vector<uint32_t>{2}));
    EXPECT_EQ(geometry.permuted_layout->retained_output_order,
              RetainedOutputOrder::NaturalOuterInner);
    EXPECT_EQ(geometry.outer_size, 7U);
    EXPECT_EQ(geometry.reduction_size, 5U);
    EXPECT_EQ(geometry.inner_size, 32U);
    EXPECT_FALSE(geometry.tiled_output_permuted);
    EXPECT_FALSE(geometry.tiled_output_shared_transpose);
    EXPECT_EQ(geometry.tiled_output_outer_stride, 32U);
    EXPECT_EQ(geometry.tiled_output_inner_stride, 1U);
}

TEST(ReducerGeometry, RotatedOutputSplitIsStorageOffsetInvariant) {
    Tensor storage(TensorPlacement(TensorPlacement::MemDevices::CPU),
                   TensorDescriptor(DataType::FP32, {8, 5, 32}));
    StridedSpan offset_view = StridedSpan(storage, 160, {32, 7, 5}, {1, 160, 32});
    ASSERT_EQ(offset_view.getStorage().offset, 160U);
    expectRotatedSplit(offset_view.getDimensions(), offset_view.getStridesElements(), {2}, 7, 5, 32, 1);
    // The caller reads the view's bytes as the dense [a, r, b] from the same first element.
    const StridedSpan dense = offset_view.asDense({7, 5, 32});
    EXPECT_EQ(dense.getStorage().offset, 160U);
    EXPECT_EQ(dense.getStridesElements(), (std::vector<uint64_t>{160, 32, 1}));
}

TEST(ReducerGeometry, RotatedOutputSplitIgnoresSingletonAxes) {
    // Singleton axes are memory-layout neutral and must not prevent recognizing a genuine dense physical permutation:
    // physical [2, 3, 4] (logical axes 2, 3, 0) read as [4, 1, 2, 3], reducing logical 3.
    expectRotatedSplit({4, 1, 2, 3}, {1, 12, 12, 4}, {3}, 2, 3, 4, 1);
}

TEST(ReducerGeometry, PhysicallyContiguousReductionBlockCanUseTiledPathWhenLogicalAxesAreDisjoint) {
    // Physical order is [logical 1, logical 0, logical 2, logical 3]. Logical reduction axes {0,2}
    // are disjoint, but physically they form one contiguous reduction block in dense [2,3,5,7] storage.
    const ReductionGeometry geometry = Reducer::analyzeGeometry(
        {3, 2, 5, 7}, {35, 105, 7, 1}, std::vector<uint32_t>{0, 2});

    EXPECT_FALSE(geometry.reduced_axes_are_contiguous);
    EXPECT_EQ(geometry.path, ReductionLayout::MiddleBlock);
    ASSERT_TRUE(geometry.permuted_layout.has_value());
    EXPECT_EQ(geometry.permuted_layout->physical_outer_axes, (std::vector<uint32_t>{1}));
    EXPECT_EQ(geometry.permuted_layout->physical_reduction_axes,
              (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(geometry.permuted_layout->physical_inner_axes, (std::vector<uint32_t>{3}));
    EXPECT_EQ(geometry.outer_size, 2U);
    EXPECT_EQ(geometry.reduction_size, 15U);
    EXPECT_EQ(geometry.inner_size, 7U);
    EXPECT_FALSE(geometry.tiled_output_permuted);
}

TEST(ReducerGeometry, RejectsGappedOverlappingAndBroadcastStorageAfterDelete) {
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {2, 2, 3}, {1, 7, 2}, std::vector<uint32_t>{2}),
                 NotImplementedException);
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {2, 2, 3}, {1, 1, 2}, std::vector<uint32_t>{2}),
                 NotImplementedException);
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {2, 2, 3}, {0, 6, 2}, std::vector<uint32_t>{2}),
                 NotImplementedException);
}

TEST(ReducerGeometry, SplitPhysicalReductionIsComposedDenseAndDirect2BIsRefused) {
    // Physical order is [2,0,1,3]. Reduction axes {1,2} are split by retained axis 0 in physical order; the retained
    // axes {0,3} keep their physical order. A dense [2,3,5,7] reduced over its physical axes {0,2} (DisjointRuns, run
    // on the dense planner).
    const ReductionGeometry split = Reducer::analyzeGeometry(
        {3, 5, 2, 7}, {35, 7, 105, 1}, std::vector<uint32_t>{1, 2});
    EXPECT_TRUE(split.physical_layout_is_dense_permutation);
    EXPECT_EQ(split.path, ReductionLayout::DisjointRuns);
    EXPECT_EQ(split.output_elements, 21U);
    EXPECT_EQ(split.reduction_size, 10U);
    for (ReductionOp op : {ReductionOp::Sum, ReductionOp::Max, ReductionOp::Mean, ReductionOp::L2Norm}) {
        const ReductionGeometry value =
            Reducer::analyzeValueGeometry(op, {3, 5, 2, 7}, {35, 7, 105, 1}, std::vector<uint32_t>{1, 2});
        EXPECT_EQ(value.path, ReductionLayout::DisjointRuns);
    }
    // The same split with the retained axes out of physical order still has no reducer: [2,3,5,7] read as
    // [7,5,2,3] (logical axis 0 is physical 3, logical 3 is physical 1) reducing logical {1,2}.
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {7, 5, 2, 3}, {1, 7, 105, 35}, std::vector<uint32_t>{1, 2}),
                 NotImplementedException);

    // Reducing axis 0 is case G, the [A,R,B,P] -> [B,A,P] rotation of the kept axes (physical [2, 3, 5, 7]: A 2, R 3, B
    // 5, P 7). The reducer refuses it, and the caller rotates the dense [A, B x P] output.
    expectRotatedSplit({3, 5, 2, 7}, {35, 7, 105, 1}, {0}, 2, 3, 5, 7);
}

TEST(ReducerGeometry, CompactPhysicalPermutationTrailingReductionUsesContiguousSegmentsWhenOutputOrderMatches) {
    const ReductionGeometry geometry = Reducer::analyzeGeometry(
        {2, 3, 4}, {1, 8, 2}, std::vector<uint32_t>{0});

    EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
    EXPECT_EQ(geometry.physical_non_singleton_axis_order, (std::vector<uint32_t>{1, 2, 0}));
    EXPECT_EQ(geometry.output_elements, 12U);
    EXPECT_EQ(geometry.reduction_size, 2U);
    EXPECT_TRUE(geometry.permutation_aware_contiguous_segments);
    EXPECT_EQ(geometry.path, ReductionLayout::TrailingBlock);
    EXPECT_FALSE(geometry.permuted_layout.has_value());
}

TEST(ReducerGeometry, CompactPhysicalPermutationTrailingRetainedOrderMismatchIsUnsupported) {
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {2, 3, 4}, {4, 8, 1}, std::vector<uint32_t>{2}),
                 NotImplementedException);
}

TEST(ReducerGeometry, CompactPhysicalPermutationFullReductionUsesDeviceTransformReduce) {
    const ReductionGeometry geometry = Reducer::analyzeGeometry(
        {2, 3, 4}, {1, 8, 2}, std::vector<uint32_t>{0, 1, 2});

    EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
    EXPECT_EQ(geometry.physical_non_singleton_axis_order, (std::vector<uint32_t>{1, 2, 0}));
    EXPECT_EQ(geometry.output_elements, 1U);
    EXPECT_EQ(geometry.reduction_size, 24U);
    EXPECT_EQ(geometry.path, ReductionLayout::SingleOutput);
    EXPECT_FALSE(geometry.device_transform_uses_affine_stride);
    EXPECT_FALSE(geometry.permuted_layout.has_value());
}

TEST(ReducerGeometry, DenseAndAffinePathSelectionRemainsUnchanged) {
    const ReductionGeometry dense = Reducer::analyzeGeometry(
        {2, 3, 4}, std::vector<uint32_t>{1});
    EXPECT_EQ(dense.path, ReductionLayout::MiddleBlock);
    EXPECT_TRUE(dense.physical_layout_is_dense_permutation);
    EXPECT_FALSE(dense.permuted_layout.has_value());

    const ReductionGeometry diagonal = Reducer::analyzeGeometry(
        {3}, {4}, std::vector<uint32_t>{0});
    EXPECT_EQ(diagonal.path, ReductionLayout::SingleOutput);
    EXPECT_TRUE(diagonal.device_transform_uses_affine_stride);
    EXPECT_EQ(diagonal.affine_input_stride, 4U);
    EXPECT_FALSE(diagonal.physical_layout_is_dense_permutation);
    EXPECT_FALSE(diagonal.permuted_layout.has_value());
}

TEST(ReducerGeometry, StampsCollapsedDenseRunsForAlexNetBiasGradient) {
    const ReductionGeometry geometry =
        Reducer::analyzeGeometry({512, 64, 55, 55}, std::vector<uint32_t>{0, 2, 3});

    // Geometry analysis owns the final execution family. This dense R|K|R layout is therefore classified as
    // DisjointRuns immediately; value/ARG planners only choose the legal stage order within that family.
    EXPECT_EQ(geometry.path, ReductionLayout::DisjointRuns);
    ASSERT_TRUE(geometry.dense_run_geometry.has_value());
    const DenseRunGeometry& runs = geometry.dense_run_geometry.value();
    ASSERT_EQ(runs.runs.size(), 3U);
    EXPECT_EQ(runs.reduced_run_count, 2U);
    EXPECT_EQ(runs.retained_run_count, 1U);

    EXPECT_EQ(runs.runs[0].kind, DenseRunKind::Reduced);
    EXPECT_EQ(runs.runs[0].extent, 512U);
    EXPECT_EQ(runs.runs[0].physical_stride, 64U * 55U * 55U);
    EXPECT_EQ(runs.runs[0].domain_stride, 55U * 55U);

    EXPECT_EQ(runs.runs[1].kind, DenseRunKind::Retained);
    EXPECT_EQ(runs.runs[1].extent, 64U);
    EXPECT_EQ(runs.runs[1].physical_stride, 55U * 55U);
    EXPECT_EQ(runs.runs[1].domain_stride, 1U);

    EXPECT_EQ(runs.runs[2].kind, DenseRunKind::Reduced);
    EXPECT_EQ(runs.runs[2].extent, 55U * 55U);
    EXPECT_EQ(runs.runs[2].physical_stride, 1U);
    EXPECT_EQ(runs.runs[2].domain_stride, 1U);
}

TEST(ReducerGeometry, ValuePlanningSelectsGeneralComposedDenseForRkrBeforeStamping) {
    const std::vector<uint64_t> dimensions{512, 64, 55, 55};
    const std::vector<uint32_t> axes{0, 2, 3};

    for (ReductionOp op : {ReductionOp::Sum,
                           ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        const ReductionGeometry value = Reducer::analyzeValueGeometry(op, dimensions, axes);
        EXPECT_EQ(value.path, ReductionLayout::DisjointRuns);
    }
}

TEST(ReducerGeometry, StampsGeneralAlternatingDenseRunGeometry) {
    const ReductionGeometry geometry =
        Reducer::analyzeGeometry({31, 17, 29, 19, 23}, std::vector<uint32_t>{0, 2, 4});

    ASSERT_TRUE(geometry.dense_run_geometry.has_value());
    const DenseRunGeometry& dense = geometry.dense_run_geometry.value();
    ASSERT_EQ(dense.runs.size(), 5U);
    EXPECT_EQ(dense.reduced_run_count, 3U);
    EXPECT_EQ(dense.retained_run_count, 2U);

    EXPECT_EQ(dense.runs[0].kind, DenseRunKind::Reduced);
    EXPECT_EQ(dense.runs[0].extent, 31U);
    EXPECT_EQ(dense.runs[0].physical_stride, 17U * 29U * 19U * 23U);
    EXPECT_EQ(dense.runs[0].domain_stride, 29U * 23U);

    EXPECT_EQ(dense.runs[1].kind, DenseRunKind::Retained);
    EXPECT_EQ(dense.runs[1].extent, 17U);
    EXPECT_EQ(dense.runs[1].physical_stride, 29U * 19U * 23U);
    EXPECT_EQ(dense.runs[1].domain_stride, 19U);

    EXPECT_EQ(dense.runs[2].kind, DenseRunKind::Reduced);
    EXPECT_EQ(dense.runs[2].extent, 29U);
    EXPECT_EQ(dense.runs[2].physical_stride, 19U * 23U);
    EXPECT_EQ(dense.runs[2].domain_stride, 23U);

    EXPECT_EQ(dense.runs[3].kind, DenseRunKind::Retained);
    EXPECT_EQ(dense.runs[3].extent, 19U);
    EXPECT_EQ(dense.runs[3].physical_stride, 23U);
    EXPECT_EQ(dense.runs[3].domain_stride, 1U);

    EXPECT_EQ(dense.runs[4].kind, DenseRunKind::Reduced);
    EXPECT_EQ(dense.runs[4].extent, 23U);
    EXPECT_EQ(dense.runs[4].physical_stride, 1U);
    EXPECT_EQ(dense.runs[4].domain_stride, 1U);
}

TEST(ReducerGeometry, ValuePlanningSelectsGeneralComposedDenseForRkrkrBeforeStamping) {
    const std::vector<uint64_t> dimensions{31, 17, 29, 19, 23};
    const std::vector<uint32_t> axes{0, 2, 4};

    for (ReductionOp op : {ReductionOp::Sum,
                           ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        const ReductionGeometry value = Reducer::analyzeValueGeometry(op, dimensions, axes);
        EXPECT_EQ(value.path, ReductionLayout::DisjointRuns);
    }
}

TEST(ReducerGeometry, ValuePlanningSelectsComposedDenseForTrailingRetainedTwoRunTopologies) {
    for (ReductionOp op : {ReductionOp::Sum,
                           ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        const ReductionGeometry rkrk =
            Reducer::analyzeValueGeometry(op, {2, 3, 5, 7}, {0, 2});
        EXPECT_EQ(rkrk.path, ReductionLayout::DisjointRuns);

        const ReductionGeometry krkrk =
            Reducer::analyzeValueGeometry(op, {2, 3, 5, 7, 11}, {1, 3});
        EXPECT_EQ(krkrk.path, ReductionLayout::DisjointRuns);
    }
}

TEST(ReducerGeometry, DenseValuePlannerHandlesArbitraryAlternatingRunsAndSingletonSeparators) {
    for (ReductionOp op : {ReductionOp::Sum,
                           ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        const ReductionGeometry trailing_reduced = Reducer::analyzeValueGeometry(
            op, {2, 3, 5, 7, 11, 13, 17}, {0, 2, 4, 6});
        EXPECT_EQ(trailing_reduced.path, ReductionLayout::DisjointRuns);

        const ReductionGeometry trailing_retained = Reducer::analyzeValueGeometry(
            op, {2, 3, 5, 7, 11, 13, 17, 19}, {0, 2, 4, 6});
        EXPECT_EQ(trailing_retained.path, ReductionLayout::DisjointRuns);

        // Axis 1 is a retained singleton between reduced non-singleton axes 0 and 2. The generic planner absorbs it
        // into one direct contiguous stage instead of treating the public non-contiguous axis list as strided.
        const ReductionGeometry singleton_separator =
            Reducer::analyzeValueGeometry(op, {2, 1, 3, 5}, {0, 2});
        EXPECT_EQ(singleton_separator.path, ReductionLayout::DisjointRuns);

        // Reducing only singleton axes still needs one direct pass for transform/finalize/conversion/scaling.
        const ReductionGeometry singleton_only =
            Reducer::analyzeValueGeometry(op, {3, 1, 5, 1, 7}, {1, 3});
        EXPECT_EQ(singleton_only.path, ReductionLayout::DisjointRuns);
    }
}

TEST(ReducerGeometry, DenseValueCompositionValidatesPerStageRatherThanMonolithicReductionSize) {
    const uint64_t run_extent = 65536;
    const ReductionGeometry structural =
        Reducer::analyzeGeometry({run_extent, 2, run_extent, 2}, std::vector<uint32_t>{0, 2});
    EXPECT_EQ(structural.path, ReductionLayout::DisjointRuns);
    EXPECT_GT(structural.reduction_size, static_cast<uint64_t>(std::numeric_limits<int>::max()));

    // The logical reduction domain is > INT_MAX, but each MiddleBlock stage reduces only 65536 elements.
    const ReductionGeometry sum = Reducer::analyzeValueGeometry(
        ReductionOp::Sum, {run_extent, 2, run_extent, 2}, {0, 2});
    EXPECT_EQ(sum.path, ReductionLayout::DisjointRuns);

    // Every dense value operation validates its direct component stages independently rather than inheriting any
    // obsolete monolithic fallback limit.
    for (ReductionOp op : {ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        EXPECT_EQ(Reducer::analyzeValueGeometry(
                      op, {run_extent, 2, run_extent, 2}, {0, 2})
                      .path,
                  ReductionLayout::DisjointRuns);
    }
}

// No geometry has a 32-bit segment-size limit (the dense and StridedR kernels carry 64-bit geometry). The GPU runs past
// the limit are in ReducerViewPhysicalTest.cpp.
TEST(ReducerGeometry, NoGeometryInheritsTheCubIntSegmentSizeLimit) {
    const uint64_t above_int_max = static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1ULL;

    // A middle reduced block larger than INT_MAX (MiddleBlock).
    const ReductionGeometry tiled =
        Reducer::analyzeValueGeometry(ReductionOp::Sum, {2, above_int_max, 2}, {1});
    EXPECT_EQ(tiled.path, ReductionLayout::MiddleBlock);
    EXPECT_EQ(tiled.reduction_size, above_int_max);

    // The same as one pass inside a disjoint-axis dense composition.
    const ReductionGeometry composed =
        Reducer::analyzeValueGeometry(ReductionOp::Sum, {above_int_max, 2, 3, 2}, {0, 2});
    EXPECT_EQ(composed.path, ReductionLayout::DisjointRuns);
    EXPECT_GT(composed.reduction_size, static_cast<uint64_t>(std::numeric_limits<int>::max()));

    // A full reduction.
    const ReductionGeometry full =
        Reducer::analyzeValueGeometry(ReductionOp::Sum, {above_int_max}, {0});
    EXPECT_EQ(full.path, ReductionLayout::SingleOutput);

    // Trailing rows longer than INT_MAX (TrailingBlock), for every value op and for ARG.
    for (ReductionOp op : {ReductionOp::Sum,
                           ReductionOp::Min,
                           ReductionOp::Max,
                           ReductionOp::Product,
                           ReductionOp::Mean,
                           ReductionOp::L1Norm,
                           ReductionOp::L2Norm,
                           ReductionOp::SumSquares}) {
        const ReductionGeometry rows = Reducer::analyzeValueGeometry(op, {2, above_int_max}, {1});
        EXPECT_EQ(rows.path, ReductionLayout::TrailingBlock);
        EXPECT_EQ(rows.reduction_size, above_int_max);
    }
    const ReductionGeometry arg_rows = ArgReducer::analyzeDenseGeometry({2, above_int_max}, {1});
    EXPECT_EQ(arg_rows.path, ReductionLayout::TrailingBlock);
    EXPECT_EQ(arg_rows.reduction_size, above_int_max);

    // Case B past the limit: [2, A, B] read as [2, B, A], the reduced axes the physical suffix.
    const uint64_t a = 65537, b = 32771;
    ASSERT_GT(a * b, static_cast<uint64_t>(std::numeric_limits<int>::max()));
    const ReductionGeometry view =
        Reducer::analyzeValueGeometry(ReductionOp::Sum, {2, b, a}, {a * b, 1, b}, {1, 2});
    EXPECT_EQ(view.path, ReductionLayout::TrailingBlock);
    EXPECT_TRUE(view.permutation_aware_contiguous_segments);
    EXPECT_EQ(view.reduction_size, a * b);
}

TEST(ReducerGeometry, DenseRunPlanningCollapsesAdjacentRolesAndIgnoresSingletonSeparators) {
    const ReductionGeometry adjacent =
        Reducer::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{0, 1, 3});
    ASSERT_TRUE(adjacent.dense_run_geometry.has_value());
    ASSERT_EQ(adjacent.dense_run_geometry->runs.size(), 3U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].kind, DenseRunKind::Reduced);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].extent, 6U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].physical_stride, 20U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[0].domain_stride, 5U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].kind, DenseRunKind::Retained);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].extent, 4U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].physical_stride, 5U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[1].domain_stride, 1U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].kind, DenseRunKind::Reduced);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].extent, 5U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].physical_stride, 1U);
    EXPECT_EQ(adjacent.dense_run_geometry->runs[2].domain_stride, 1U);

    // The retained singleton does not consume storage or an output coordinate, so it must not split the two reduced
    // spans into separate runtime runs. This gives the executor the physically meaningful R(2*3)|K(5) traversal.
    const ReductionGeometry singleton_separator =
        Reducer::analyzeGeometry({2, 1, 3, 5}, std::vector<uint32_t>{0, 2});
    ASSERT_TRUE(singleton_separator.dense_run_geometry.has_value());
    ASSERT_EQ(singleton_separator.dense_run_geometry->runs.size(), 2U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].kind, DenseRunKind::Reduced);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].extent, 6U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].physical_stride, 5U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[0].domain_stride, 1U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].kind, DenseRunKind::Retained);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].extent, 5U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].physical_stride, 1U);
    EXPECT_EQ(singleton_separator.dense_run_geometry->runs[1].domain_stride, 1U);
}

TEST(ReducerGeometry, DenseRunPlanningDoesNotPretendArbitraryStridesAreDense) {
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {2, 3, 4}, {20, 4, 1}, std::vector<uint32_t>{0, 2}),
                 NotImplementedException);
}


TEST(ReducerGeometry, DeleteRejectsFormerLogicalIndexMapperDomains) {
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {65535, 3, 32768}, {1, 1, 1}, std::vector<uint32_t>{0, 2}),
                 NotImplementedException);
    EXPECT_THROW((void)Reducer::analyzeGeometry(
                     {2, 3, 4}, {uint64_t{1} << 32, 4, 1}, std::vector<uint32_t>{0, 2}),
                 NotImplementedException);
}
