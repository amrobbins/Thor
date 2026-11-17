#include "Utilities/TensorOperations/Reduction/ArgReducer.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"
#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"

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

// The original-index arithmetic the planned dense ARG passes use to carry indices across reduced runs.
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

std::vector<const DenseRun*> reducedRuns(const ReductionGeometry& geometry) {
    std::vector<const DenseRun*> runs;
    EXPECT_TRUE(geometry.dense_run_geometry.has_value());
    for (const DenseRun& run : geometry.dense_run_geometry->runs) {
        if (run.kind == DenseRunKind::Reduced) {
            runs.push_back(&run);
        }
    }
    return runs;
}

void verifyAllRunCoordinatesAndOrders(const std::vector<uint64_t>& dimensions, const std::vector<uint32_t>& axes) {
    const ReductionGeometry geometry = Reducer::analyzeGeometry(dimensions, axes);
    const std::vector<const DenseRun*> runs = reducedRuns(geometry);
    ASSERT_FALSE(runs.empty());

    std::vector<uint64_t> extents;
    std::vector<uint64_t> strides;
    for (const DenseRun* run : runs) {
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
                    composed = ArgReducer::composeOriginalArgIndex(composed, coordinates[run_index], strides[run_index]);
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

TEST(ArgReducerPlan, OriginalFlattenedIndexIsIndependentOfEveryLegalEndEliminationOrder) {
    // Two runs: R K R.
    verifyAllRunCoordinatesAndOrders({2, 3, 3}, {0, 2});
    // Three runs: R K R K R.
    verifyAllRunCoordinatesAndOrders({2, 2, 3, 2, 2}, {0, 2, 4});
    // Four runs, rank > 5, with leading/trailing retained regions: K R K R K R K R K.
    verifyAllRunCoordinatesAndOrders({2, 2, 2, 2, 2, 2, 2, 2, 2}, {1, 3, 5, 7});
    // Multi-axis reduced runs: K RR K RR.
    verifyAllRunCoordinatesAndOrders({2, 2, 2, 2, 2, 2}, {1, 2, 4, 5});
}

TEST(ArgReducerPlan, ComposeOriginalArgIndexChecksUint64Arithmetic) {
    EXPECT_EQ(ArgReducer::composeOriginalArgIndex(7, 3, 11), 40U);
    EXPECT_THROW((void)ArgReducer::composeOriginalArgIndex(0, std::numeric_limits<uint64_t>::max(), 2), std::overflow_error);
    EXPECT_THROW((void)ArgReducer::composeOriginalArgIndex(std::numeric_limits<uint64_t>::max(), 1, 1), std::overflow_error);
}
