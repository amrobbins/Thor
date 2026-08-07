#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
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
    EXPECT_EQ(disjoint.path, CubReductionPath::StridedFixedSegment);
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
    EXPECT_EQ(geometry.path, CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(geometry.input_elements, 32U);
    EXPECT_EQ(geometry.reduction_size, 16U);
    EXPECT_EQ(geometry.output_elements, 2U);
    EXPECT_EQ(geometry.output_dimensions, (std::vector<uint64_t>{1, 1, 1, 1, 1, 1, 1, 1, 2}));
    EXPECT_EQ(geometry.indexing.input_strides.size(), dimensions.size());
    EXPECT_EQ(geometry.indexing.reduced_axes, axes);

    EXPECT_EQ(CubReduction::mapLogicalReductionIndexToPhysicalIndex(geometry, 0, 15), 30U);
    EXPECT_EQ(CubReduction::mapLogicalReductionIndexToPhysicalIndex(geometry, 1, 15), 31U);
}

TEST(CubReductionGeometry, LogicalIndexMappingIsBijectiveAndMatchesRowMajorCoordinates) {
    const std::vector<std::vector<uint64_t>> shapes = {
        {2},
        {2, 3},
        {2, 1, 3},
        {2, 3, 2, 1},
    };

    for (const std::vector<uint64_t>& dimensions : shapes) {
        const uint32_t rank = static_cast<uint32_t>(dimensions.size());
        const uint32_t subset_count = 1U << rank;
        for (uint32_t mask = 1; mask < subset_count; ++mask) {
            std::vector<uint32_t> axes;
            std::vector<uint32_t> retained_axes;
            for (uint32_t axis = 0; axis < rank; ++axis) {
                if ((mask & (1U << axis)) != 0) {
                    axes.push_back(axis);
                } else {
                    retained_axes.push_back(axis);
                }
            }

            const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, axes);
            std::vector<bool> visited(geometry.input_elements, false);

            for (uint64_t output_index = 0; output_index < geometry.output_elements; ++output_index) {
                for (uint64_t reduction_index = 0; reduction_index < geometry.reduction_size; ++reduction_index) {
                    std::vector<uint64_t> coordinates(rank, 0);
                    uint64_t remaining_output = output_index;
                    for (int32_t retained = static_cast<int32_t>(retained_axes.size()) - 1; retained >= 0; --retained) {
                        const uint32_t axis = retained_axes[retained];
                        coordinates[axis] = remaining_output % dimensions[axis];
                        remaining_output /= dimensions[axis];
                    }
                    uint64_t remaining_reduction = reduction_index;
                    for (int32_t reduced = static_cast<int32_t>(axes.size()) - 1; reduced >= 0; --reduced) {
                        const uint32_t axis = axes[reduced];
                        coordinates[axis] = remaining_reduction % dimensions[axis];
                        remaining_reduction /= dimensions[axis];
                    }

                    uint64_t expected_physical_index = 0;
                    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
                        expected_physical_index = expected_physical_index * dimensions[dimension] + coordinates[dimension];
                    }

                    const uint64_t actual_physical_index = CubReduction::mapLogicalReductionIndexToPhysicalIndex(
                        geometry, output_index, reduction_index);
                    EXPECT_EQ(actual_physical_index, expected_physical_index)
                        << "rank=" << rank << " mask=" << mask << " output=" << output_index
                        << " reduction=" << reduction_index;
                    ASSERT_LT(actual_physical_index, visited.size());
                    EXPECT_FALSE(visited[actual_physical_index]);
                    visited[actual_physical_index] = true;
                }
            }
            EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](bool value) { return value; }));
        }
    }

    const CubReductionGeometry geometry = CubReduction::analyzeGeometry({2, 3, 4}, std::vector<uint32_t>{0, 2});
    EXPECT_THROW(static_cast<void>(CubReduction::mapLogicalReductionIndexToPhysicalIndex(
                     geometry, geometry.output_elements, 0)),
                 std::out_of_range);
    EXPECT_THROW(static_cast<void>(CubReduction::mapLogicalReductionIndexToPhysicalIndex(
                     geometry, 0, geometry.reduction_size)),
                 std::out_of_range);
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

TEST(CubReductionGeometry, RejectsGappedOverlappingAndBroadcastStorageAsDensePhysicalPermutations) {
    const CubReductionGeometry gapped = CubReduction::analyzeGeometry(
        {2, 2, 3}, {1, 7, 2}, std::vector<uint32_t>{2});
    EXPECT_EQ(gapped.path, CubReductionPath::StridedFixedSegment);
    EXPECT_FALSE(gapped.physical_layout_is_dense_permutation);
    EXPECT_FALSE(gapped.permutation_aware_tiled_geometry.has_value());

    const CubReductionGeometry overlapping = CubReduction::analyzeGeometry(
        {2, 2, 3}, {1, 1, 2}, std::vector<uint32_t>{2});
    EXPECT_EQ(overlapping.path, CubReductionPath::StridedFixedSegment);
    EXPECT_FALSE(overlapping.physical_layout_is_dense_permutation);
    EXPECT_FALSE(overlapping.permutation_aware_tiled_geometry.has_value());

    const CubReductionGeometry broadcast = CubReduction::analyzeGeometry(
        {2, 2, 3}, {0, 6, 2}, std::vector<uint32_t>{2});
    EXPECT_EQ(broadcast.path, CubReductionPath::StridedFixedSegment);
    EXPECT_FALSE(broadcast.physical_layout_is_dense_permutation);
    EXPECT_FALSE(broadcast.permutation_aware_tiled_geometry.has_value());
}

TEST(CubReductionGeometry, RequiresPhysicallyContiguousReductionBlockAndSupportedRetainedOrder) {
    // Physical order is [2,0,1,3]. Reducing logical axes 1 and 2 leaves a retained axis between them physically.
    const CubReductionGeometry split_reduction = CubReduction::analyzeGeometry(
        {3, 5, 2, 7}, {35, 7, 105, 1}, std::vector<uint32_t>{1, 2});
    EXPECT_TRUE(split_reduction.physical_layout_is_dense_permutation);
    EXPECT_EQ(split_reduction.physical_non_singleton_axis_order, (std::vector<uint32_t>{2, 0, 1, 3}));
    EXPECT_FALSE(split_reduction.permutation_aware_tiled_geometry.has_value());

    // Reducing logical axis 0 is physically contiguous, but requested logical retained order [1,2,3] is neither
    // natural [2,1,3] nor the flattened inner/outer rotation [1,3,2].
    const CubReductionGeometry unsupported_output_order = CubReduction::analyzeGeometry(
        {3, 5, 2, 7}, {35, 7, 105, 1}, std::vector<uint32_t>{0});
    EXPECT_TRUE(unsupported_output_order.physical_layout_is_dense_permutation);
    EXPECT_FALSE(unsupported_output_order.permutation_aware_tiled_geometry.has_value());
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
    EXPECT_EQ(CubReduction::mapLogicalReductionIndexToPhysicalIndex(diagonal, 0, 2), 8U);
}
