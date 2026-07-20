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
    EXPECT_EQ(scalar.outer_size, 1U);
    EXPECT_EQ(scalar.reduction_size, 257U);
    EXPECT_EQ(scalar.inner_size, 1U);
    EXPECT_EQ(scalar.output_elements, 1U);
    EXPECT_EQ(scalar.output_dimensions, (std::vector<uint64_t>{1}));

    const CubReductionGeometry contiguous = CubReduction::analyzeGeometry({2, 3, 4}, 2);
    EXPECT_EQ(contiguous.path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_EQ(contiguous.outer_size, 6U);
    EXPECT_EQ(contiguous.reduction_size, 4U);
    EXPECT_EQ(contiguous.inner_size, 1U);
    EXPECT_EQ(contiguous.output_elements, 6U);
    EXPECT_EQ(contiguous.output_dimensions, (std::vector<uint64_t>{2, 3, 1}));

    const CubReductionGeometry strided = CubReduction::analyzeGeometry({2, 3, 4}, 1);
    EXPECT_EQ(strided.path, CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(strided.outer_size, 2U);
    EXPECT_EQ(strided.reduction_size, 3U);
    EXPECT_EQ(strided.inner_size, 4U);
    EXPECT_EQ(strided.output_elements, 8U);
    EXPECT_EQ(strided.output_dimensions, (std::vector<uint64_t>{2, 1, 4}));
}

TEST(CubReductionGeometry, RejectsInvalidSingleAxisGeometry) {
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({}, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 3}, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction::analyzeGeometry({2, 0, 3}, 1)), std::invalid_argument);
}

TEST(CubReductionGeometry, SelectsBestMultiAxisPathAndShapes) {
    const CubReductionGeometry all_axes = CubReduction::analyzeGeometry({2, 3, 4}, std::vector<uint32_t>{0, 1, 2});
    EXPECT_EQ(all_axes.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_EQ(all_axes.input_elements, 24U);
    EXPECT_EQ(all_axes.reduction_size, 24U);
    EXPECT_EQ(all_axes.output_elements, 1U);
    EXPECT_EQ(all_axes.output_dimensions, (std::vector<uint64_t>{1, 1, 1}));
    EXPECT_EQ(all_axes.squeezed_output_dimensions, (std::vector<uint64_t>{1}));

    const CubReductionGeometry suffix =
        CubReduction::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{2, 3});
    EXPECT_EQ(suffix.path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_EQ(suffix.reduction_size, 20U);
    EXPECT_EQ(suffix.output_elements, 6U);
    EXPECT_EQ(suffix.output_dimensions, (std::vector<uint64_t>{2, 3, 1, 1}));
    EXPECT_EQ(suffix.squeezed_output_dimensions, (std::vector<uint64_t>{2, 3}));

    const CubReductionGeometry disjoint =
        CubReduction::analyzeGeometry({2, 3, 4, 5}, std::vector<uint32_t>{1, 3});
    EXPECT_EQ(disjoint.path, CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(disjoint.reduction_size, 15U);
    EXPECT_EQ(disjoint.output_elements, 8U);
    EXPECT_EQ(disjoint.output_dimensions, (std::vector<uint64_t>{2, 1, 4, 1}));
    EXPECT_EQ(disjoint.squeezed_output_dimensions, (std::vector<uint64_t>{2, 4}));

    const CubReductionGeometry leading =
        CubReduction::analyzeGeometry({2, 3, 4}, std::vector<uint32_t>{0, 1});
    EXPECT_EQ(leading.path, CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(leading.output_dimensions, (std::vector<uint64_t>{1, 1, 4}));
    EXPECT_EQ(leading.squeezed_output_dimensions, (std::vector<uint64_t>{4}));

    const CubReductionGeometry singleton_retained =
        CubReduction::analyzeGeometry({1, 3, 1}, std::vector<uint32_t>{1});
    EXPECT_EQ(singleton_retained.path, CubReductionPath::DeviceTransformReduce);
    EXPECT_EQ(singleton_retained.reduction_size, 3U);
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
