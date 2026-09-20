#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

uint64_t flattenReducedCoordinate(const std::vector<uint64_t>& extents, const std::vector<uint64_t>& coordinates) {
    EXPECT_EQ(extents.size(), coordinates.size());
    uint64_t index = 0;
    for (size_t i = 0; i < extents.size(); ++i) {
        EXPECT_LT(coordinates[i], extents[i]);
        index = index * extents[i] + coordinates[i];
    }
    return index;
}

std::vector<std::vector<size_t>> allEndEliminationOrders(size_t run_count) {
    std::vector<std::vector<size_t>> orders;
    std::vector<size_t> current;
    std::function<void(size_t, size_t)> visit = [&](size_t left, size_t right) {
        if (left == right) {
            current.push_back(left);
            orders.push_back(current);
            current.pop_back();
            return;
        }
        current.push_back(left);
        visit(left + 1, right);
        current.pop_back();
        current.push_back(right);
        visit(left, right - 1);
        current.pop_back();
    };
    visit(0, run_count - 1);
    return orders;
}

std::vector<const CubReductionDenseRun*> reducedRuns(const CubReductionGeometry& geometry) {
    std::vector<const CubReductionDenseRun*> runs;
    EXPECT_TRUE(geometry.dense_run_geometry.has_value());
    for (const CubReductionDenseRun& run : geometry.dense_run_geometry->runs) {
        if (run.kind == CubReductionDenseRunKind::Reduced) {
            runs.push_back(&run);
        }
    }
    return runs;
}

void verifyAllRunCoordinatesAndOrders(const std::vector<uint64_t>& dimensions, const std::vector<uint32_t>& axes) {
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, axes);
    const std::vector<const CubReductionDenseRun*> runs = reducedRuns(geometry);
    ASSERT_FALSE(runs.empty());

    std::vector<uint64_t> extents;
    std::vector<uint64_t> strides;
    for (const CubReductionDenseRun* run : runs) {
        extents.push_back(run->extent);
        strides.push_back(run->domain_stride);
    }

    const std::vector<std::vector<size_t>> orders = allEndEliminationOrders(runs.size());
    std::vector<uint64_t> coordinates(runs.size(), 0);
    std::function<void(size_t)> enumerate = [&](size_t dimension) {
        if (dimension == runs.size()) {
            const uint64_t expected = flattenReducedCoordinate(extents, coordinates);
            for (const std::vector<size_t>& order : orders) {
                uint64_t composed = 0;
                for (size_t run_index : order) {
                    composed = CubArgReduction::composeOriginalArgIndex(composed, coordinates[run_index], strides[run_index]);
                }
                EXPECT_EQ(composed, expected);
            }
            return;
        }
        for (uint64_t coordinate = 0; coordinate < extents[dimension]; ++coordinate) {
            coordinates[dimension] = coordinate;
            enumerate(dimension + 1);
        }
    };
    enumerate(0);
}

}  // namespace

TEST(CubArgReductionPlan, GeometrySelectorOwnsDenseComposedFamilyWhileStructuralPlannerRemainsHostOnly) {
    const std::vector<uint64_t> dimensions{2, 3, 5, 7, 11};
    const std::vector<uint32_t> axes{0, 2, 4};

    const CubReductionGeometry structural_geometry = CubReduction::analyzeGeometry(dimensions, axes);
    EXPECT_EQ(structural_geometry.path, CubReductionPath::ComposedDense);
    const CubReductionGeometry production_geometry = CubArgReduction::analyzeDenseGeometry(dimensions, axes);
    EXPECT_EQ(production_geometry.path, structural_geometry.path);

    const std::optional<CubArgReductionDenseCompositionPlan> arg_plan = CubArgReduction::analyzeDenseCompositionPlan(dimensions, axes);
    ASSERT_TRUE(arg_plan.has_value());
    ASSERT_EQ(arg_plan->topology.stages.size(), 3U);
    EXPECT_EQ(arg_plan->carried_index_dtype, DataType::UINT32);

    EXPECT_EQ(arg_plan->topology.stages[0].role, CubReductionDenseCompositionStageRole::First);
    EXPECT_EQ(arg_plan->topology.stages[1].role, CubReductionDenseCompositionStageRole::Intermediate);
    EXPECT_EQ(arg_plan->topology.stages[2].role, CubReductionDenseCompositionStageRole::Final);

    // The structural, no-calibration planner deterministically breaks equal-cost choices to the left. Production
    // planning supplies the ARG-specific cost function without changing this interval topology machinery.
    EXPECT_EQ(arg_plan->topology.stages[0].reduced_run_ordinal, 0U);
    EXPECT_EQ(arg_plan->topology.stages[1].reduced_run_ordinal, 1U);
    EXPECT_EQ(arg_plan->topology.stages[2].reduced_run_ordinal, 2U);
}

TEST(CubArgReductionPlan, ExposesOriginalRunDomainMetadataAndDirectStageGeometry) {
    const std::vector<uint64_t> dimensions{3, 2, 5, 7, 11, 13};
    const std::vector<uint32_t> axes{1, 2, 4, 5};  // K RR K RR
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, axes);
    ASSERT_TRUE(geometry.dense_run_geometry.has_value());

    const std::optional<CubArgReductionDenseCompositionPlan> arg_plan = CubArgReduction::analyzeDenseCompositionPlan(dimensions, axes);
    ASSERT_TRUE(arg_plan.has_value());
    ASSERT_EQ(arg_plan->topology.stages.size(), 2U);

    const auto& first = arg_plan->topology.stages[0];
    const auto& second = arg_plan->topology.stages[1];
    EXPECT_EQ(first.reduction_axes, (std::vector<uint32_t>{1, 2}));
    EXPECT_EQ(first.reduction_extent, 10U);
    EXPECT_EQ(first.domain_stride, 143U);
    EXPECT_EQ(first.stage_reduction_size, first.reduction_extent);
    EXPECT_EQ(first.expected_path, CubReductionPath::TiledFixedSegment);
    EXPECT_EQ(first.input_dimensions, dimensions);
    EXPECT_EQ(first.output_dimensions, (std::vector<uint64_t>{3, 1, 1, 7, 11, 13}));
    ASSERT_TRUE(first.dense_run_index.has_value());
    EXPECT_EQ(geometry.dense_run_geometry->runs[first.dense_run_index.value()].domain_stride, first.domain_stride);

    EXPECT_EQ(second.reduction_axes, (std::vector<uint32_t>{4, 5}));
    EXPECT_EQ(second.reduction_extent, 143U);
    EXPECT_EQ(second.domain_stride, 1U);
    EXPECT_EQ(second.stage_reduction_size, second.reduction_extent);
    EXPECT_EQ(second.expected_path, CubReductionPath::ContiguousFixedSegment);
    EXPECT_EQ(second.input_dimensions, first.output_dimensions);
    EXPECT_EQ(second.output_dimensions, geometry.output_dimensions);
}

TEST(CubArgReductionPlan, AbsorbsRetainedSingletonSeparatorsWithoutChangingOriginalRunIndexing) {
    const std::vector<uint64_t> dimensions{2, 1, 3, 5};
    const std::vector<uint32_t> axes{0, 2};
    const std::optional<CubArgReductionDenseCompositionPlan> arg_plan = CubArgReduction::analyzeDenseCompositionPlan(dimensions, axes);
    ASSERT_TRUE(arg_plan.has_value());
    ASSERT_EQ(arg_plan->topology.stages.size(), 1U);

    const auto& stage = arg_plan->topology.stages.front();
    EXPECT_EQ(stage.role, CubReductionDenseCompositionStageRole::Complete);
    EXPECT_EQ(stage.reduction_axes, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(stage.reduction_extent, 6U);
    EXPECT_EQ(stage.domain_stride, 1U);
    EXPECT_EQ(stage.stage_reduction_size, 6U);
    EXPECT_EQ(stage.output_dimensions, (std::vector<uint64_t>{1, 1, 1, 5}));
}

TEST(CubArgReductionPlan, KeepsExplicitExtentOneStageWhenAllReducedAxesAreSingleton) {
    const std::vector<uint64_t> dimensions{3, 1, 5, 1, 7};
    const std::vector<uint32_t> axes{1, 3};
    const std::optional<CubArgReductionDenseCompositionPlan> arg_plan = CubArgReduction::analyzeDenseCompositionPlan(dimensions, axes);
    ASSERT_TRUE(arg_plan.has_value());
    ASSERT_EQ(arg_plan->topology.stages.size(), 1U);

    const auto& stage = arg_plan->topology.stages.front();
    EXPECT_EQ(stage.role, CubReductionDenseCompositionStageRole::Complete);
    EXPECT_FALSE(stage.dense_run_index.has_value());
    EXPECT_EQ(stage.reduction_extent, 1U);
    EXPECT_EQ(stage.domain_stride, 1U);
    EXPECT_EQ(stage.stage_reduction_size, 1U);
    EXPECT_EQ(stage.output_dimensions, dimensions);
}

TEST(CubArgReductionPlan, OriginalFlattenedIndexIsIndependentOfEveryLegalEndEliminationOrder) {
    // Two runs: R K R.
    verifyAllRunCoordinatesAndOrders({2, 3, 3}, {0, 2});
    // Three runs: R K R K R.
    verifyAllRunCoordinatesAndOrders({2, 2, 3, 2, 2}, {0, 2, 4});
    // Four runs, rank > 5, with leading/trailing retained regions: K R K R K R K R K.
    verifyAllRunCoordinatesAndOrders({2, 2, 2, 2, 2, 2, 2, 2, 2}, {1, 3, 5, 7});
    // Multi-axis reduced runs: K RR K RR.
    verifyAllRunCoordinatesAndOrders({2, 2, 2, 2, 2, 2}, {1, 2, 4, 5});
}

TEST(CubArgReductionPlan, ExplicitRunOrderPlannerAcceptsEveryLegalEndEliminationOrder) {
    const std::vector<uint64_t> dimensions{2, 3, 5, 7, 11, 13, 17};  // R K R K R K R
    const std::vector<uint32_t> axes{0, 2, 4, 6};
    const std::vector<std::vector<size_t>> orders = allEndEliminationOrders(4);
    ASSERT_EQ(orders.size(), 8U);

    for (const std::vector<size_t>& order : orders) {
        std::vector<uint32_t> run_order;
        run_order.reserve(order.size());
        for (size_t ordinal : order) {
            run_order.push_back(static_cast<uint32_t>(ordinal));
        }
        const auto plan = CubArgReduction::analyzeDenseCompositionPlanForRunOrder(dimensions, axes, run_order);
        ASSERT_TRUE(plan.has_value());
        ASSERT_EQ(plan->topology.stages.size(), run_order.size());
        for (size_t i = 0; i < run_order.size(); ++i) {
            EXPECT_EQ(plan->topology.stages[i].reduced_run_ordinal, run_order[i]);
        }
    }

    // Run 1 is an interior run in the initial [0,3] interval, so it is not a legal first elimination.
    EXPECT_FALSE(CubArgReduction::analyzeDenseCompositionPlanForRunOrder(dimensions, axes, {1, 0, 2, 3}).has_value());
}

TEST(CubArgReductionPlan, ChoosesCarriedIndexWidthFromCompleteOriginalReductionDomain) {
    const auto small = CubArgReduction::analyzeDenseCompositionPlan({65535, 2, 65536, 2}, {0, 2});
    ASSERT_TRUE(small.has_value());
    EXPECT_EQ(small->carried_index_dtype, DataType::UINT32);

    const auto wide = CubArgReduction::analyzeDenseCompositionPlan({65536, 2, 65536, 2}, {0, 2});
    ASSERT_TRUE(wide.has_value());
    EXPECT_EQ(wide->carried_index_dtype, DataType::UINT64);
    ASSERT_EQ(wide->topology.stages.size(), 2U);
    EXPECT_EQ(wide->topology.stages[0].stage_reduction_size, 65536U);
    EXPECT_EQ(wide->topology.stages[1].stage_reduction_size, 65536U);
}

TEST(CubArgReductionPlan, ExecutionCostModelAvoidsKnownExpensiveLeftFirstRkrkPlan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<uint64_t> dimensions{31, 17, 29, 19};  // R K R K
    const std::vector<uint32_t> axes{0, 2};
    const auto structural = CubArgReduction::analyzeDenseCompositionPlan(dimensions, axes);
    ASSERT_TRUE(structural.has_value());
    ASSERT_EQ(structural->topology.stages.size(), 2U);
    EXPECT_EQ(structural->topology.stages.front().reduced_run_ordinal, 0U);  // deterministic structural tie-break

    CubArgReductionOutputOptions outputs;
    outputs.produce_value = false;
    outputs.produce_index = true;
    outputs.index_output_dtype = DataType::UINT32;
    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        const auto calibrated =
            CubArgReduction::analyzeDenseCompositionPlanForExecution(dimensions, axes, dtype, outputs, stream);
        ASSERT_TRUE(calibrated.has_value());
        ASSERT_EQ(calibrated->topology.stages.size(), 2U);
        EXPECT_EQ(calibrated->topology.stages.front().reduced_run_ordinal, 1U);
        EXPECT_EQ(calibrated->topology.stages.front().reduction_axes, (std::vector<uint32_t>{2}));
    }
}

TEST(CubArgReductionPlan, ExecutionCostModelAvoidsMeasuredShortTrailingStageTroughs) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    CubArgReductionOutputOptions outputs;
    outputs.produce_value = false;
    outputs.produce_index = true;
    outputs.index_output_dtype = DataType::UINT32;

    struct Case {
        std::vector<uint64_t> dimensions;
        std::vector<uint32_t> axes;
        uint32_t expected_first_run;
    };
    const std::vector<Case> cases{
        // RR K RR K RR: reducing the trailing extent-7 contiguous run first is several times slower in the forced-order
        // census than eliminating the leading reduced run first.
        {{13, 5, 11, 17, 3, 19, 7}, {0, 1, 3, 4, 6}, 0},
        // K R K R K R: the large cold extent-23 contiguous first pass is a measured short-segment trough.
        {{7, 31, 11, 29, 13, 23}, {1, 3, 5}, 0},
        // Rank-9 alternating: the saturated short-narrow trailing pass is slower than beginning with the leading tiled
        // pass even though the latter moves more logical bytes.
        {{5, 7, 3, 11, 2, 13, 3, 17, 5}, {1, 3, 5, 7}, 0},
        // Four reduced runs with a trailing extent-7 contiguous run exercise the same calibrated regime at another
        // interval depth.
        {{13, 5, 11, 7, 9, 11, 7}, {0, 2, 4, 6}, 0},
    };

    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        for (const Case& test_case : cases) {
            const auto calibrated = CubArgReduction::analyzeDenseCompositionPlanForExecution(
                test_case.dimensions, test_case.axes, dtype, outputs, stream);
            ASSERT_TRUE(calibrated.has_value());
            ASSERT_FALSE(calibrated->topology.stages.empty());
            EXPECT_EQ(calibrated->topology.stages.front().reduced_run_ordinal, test_case.expected_first_run);
        }
    }
}

TEST(CubArgReductionPlan, ComposeOriginalArgIndexChecksUint64Arithmetic) {
    EXPECT_EQ(CubArgReduction::composeOriginalArgIndex(7, 3, 11), 40U);
    EXPECT_THROW((void)CubArgReduction::composeOriginalArgIndex(0, std::numeric_limits<uint64_t>::max(), 2), std::overflow_error);
    EXPECT_THROW((void)CubArgReduction::composeOriginalArgIndex(std::numeric_limits<uint64_t>::max(), 1, 1), std::overflow_error);
}
