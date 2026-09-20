#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr std::array<CubArgReductionOp, 2> ALL_ARG_OPERATIONS = {
    CubArgReductionOp::ArgMin,
    CubArgReductionOp::ArgMax,
};

[[nodiscard]] bool isOrdainedDenseArgPath(CubReductionPath path) {
    return path == CubReductionPath::DeviceTransformReduce || path == CubReductionPath::ContiguousFixedSegment
           || path == CubReductionPath::TiledFixedSegment || path == CubReductionPath::ComposedDense;
}

[[nodiscard]] std::vector<uint32_t> axesFromMask(uint32_t rank, uint32_t mask) {
    std::vector<uint32_t> axes;
    axes.reserve(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if ((mask & (1U << axis)) != 0) {
            axes.push_back(axis);
        }
    }
    return axes;
}

[[nodiscard]] const char* argOperationName(CubArgReductionOp op) {
    return op == CubArgReductionOp::ArgMin ? "ArgMin" : "ArgMax";
}

[[nodiscard]] std::string denseArgGateContext(CubArgReductionOp op,
                                               const std::vector<uint64_t>& dimensions,
                                               const std::vector<uint32_t>& axes) {
    std::ostringstream context;
    context << "op=" << argOperationName(op) << " dimensions=[";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0) {
            context << ',';
        }
        context << dimensions[i];
    }
    context << "] axes=[";
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i != 0) {
            context << ',';
        }
        context << axes[i];
    }
    context << ']';
    return context.str();
}

void expectEveryDenseArgMaskUsesOrdainedPath(const std::vector<uint64_t>& dimensions) {
    ASSERT_FALSE(dimensions.empty());
    ASSERT_LT(dimensions.size(), 32U);

    const uint32_t rank = static_cast<uint32_t>(dimensions.size());
    const uint32_t mask_count = 1U << rank;
    for (CubArgReductionOp op : ALL_ARG_OPERATIONS) {
        for (uint32_t mask = 1; mask < mask_count; ++mask) {
            const std::vector<uint32_t> axes = axesFromMask(rank, mask);
            SCOPED_TRACE(denseArgGateContext(op, dimensions, axes));

            // ArgMin and ArgMax intentionally share one geometry selector; constructing the public operation here keeps
            // this gate tied to both supported public ARG operations rather than merely testing an internal helper.
            CubArgReduction reduction(op, axes);
            EXPECT_EQ(reduction.getOperation(), op);
            const CubReductionGeometry structural_geometry = CubReduction::analyzeGeometry(dimensions, axes);
            const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry(dimensions, axes);
            EXPECT_EQ(geometry.path, structural_geometry.path);
            EXPECT_TRUE(isOrdainedDenseArgPath(geometry.path));
            EXPECT_NE(geometry.path, CubReductionPath::StridedFixedSegment);
        }
    }
}

}  // namespace

TEST(CubArgReductionDenseGate, EveryNonemptyMaskAcrossDenseRanksOneThroughNineAvoidsStridedFallback) {
    // Exhaustive masks naturally cover prefix/suffix/middle reductions, alternating R/K, adjacent multi-axis reduced
    // runs, and both leading and trailing retained regions.
    const std::vector<std::vector<uint64_t>> dense_shapes = {
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 2, 5},
        {2, 3, 2, 5, 3},
        {2, 3, 2, 5, 3, 2},
        {2, 3, 2, 5, 3, 2, 3},
        {2, 3, 2, 5, 3, 2, 3, 2},
        {2, 3, 2, 5, 3, 2, 3, 2, 5},
    };

    for (const std::vector<uint64_t>& dimensions : dense_shapes) {
        expectEveryDenseArgMaskUsesOrdainedPath(dimensions);
    }
}

TEST(CubArgReductionDenseGate, SingletonHeavyDenseRanksAvoidStridedFallbackForEveryMask) {
    expectEveryDenseArgMaskUsesOrdainedPath({2, 1, 3, 1, 5, 1, 2, 1, 3});
    expectEveryDenseArgMaskUsesOrdainedPath({1, 2, 1, 3, 1, 5, 1, 2, 1});
}

TEST(CubArgReductionDenseGate, RepresentativeDisjointRunsSelectComposedDense) {
    struct Case {
        std::vector<uint64_t> dimensions;
        std::vector<uint32_t> axes;
    };
    const std::vector<Case> cases = {
        {{31, 17, 29}, {0, 2}},                         // R K R
        {{31, 17, 29, 19}, {0, 2}},                     // R K R K
        {{17, 31, 19, 29}, {1, 3}},                     // K R K R
        {{31, 17, 29, 19, 23}, {0, 2, 4}},              // R K R K R
        {{7, 31, 11, 29, 13, 23}, {1, 3, 5}},           // K R K R K R
        {{13, 5, 11, 7, 9, 11, 7}, {0, 2, 4, 6}},       // four reduced runs
        {{17, 31, 7, 19, 29, 5}, {1, 2, 4, 5}},         // adjacent multi-axis reduced runs
        {{31, 1, 29, 19}, {0, 2}},                      // singleton separator
        {{5, 7, 3, 11, 2, 13, 3, 17, 5}, {1, 3, 5, 7}},// rank 9 alternating
    };

    for (CubArgReductionOp op : ALL_ARG_OPERATIONS) {
        for (const Case& test_case : cases) {
            SCOPED_TRACE(denseArgGateContext(op, test_case.dimensions, test_case.axes));
            const CubReductionGeometry structural_geometry =
                CubReduction::analyzeGeometry(test_case.dimensions, test_case.axes);
            const CubReductionGeometry geometry =
                CubArgReduction::analyzeDenseGeometry(test_case.dimensions, test_case.axes);
            EXPECT_EQ(geometry.path, structural_geometry.path);
            EXPECT_EQ(geometry.path, CubReductionPath::ComposedDense);
        }
    }
}

TEST(CubArgReductionDenseGate, GenuineIrregularViewRemainsOnIrregularFallback) {
    // VIEW-ARG is deliberately later in the sequence. The dense ARG gate must not erase the negative control that
    // proves a genuinely gapped/non-dense view is still classified as irregular rather than as ordinary dense work.
    const std::vector<uint64_t> dimensions{2, 3, 4};
    const std::vector<uint64_t> strides{20, 4, 1};
    const std::vector<uint32_t> axes{0, 2};
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(dimensions, strides, axes);
    EXPECT_EQ(geometry.path, CubReductionPath::StridedFixedSegment);
    EXPECT_FALSE(geometry.dense_run_geometry.has_value());
}
