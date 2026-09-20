#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ThorImplementation;

namespace {

constexpr std::array<CubReductionOp, 8> ALL_VALUE_OPERATIONS = {
    CubReductionOp::Sum,
    CubReductionOp::Min,
    CubReductionOp::Max,
    CubReductionOp::Product,
    CubReductionOp::Mean,
    CubReductionOp::L1Norm,
    CubReductionOp::L2Norm,
    CubReductionOp::SumSquares,
};

const char* valueOperationName(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
            return "sum";
        case CubReductionOp::Min:
            return "min";
        case CubReductionOp::Max:
            return "max";
        case CubReductionOp::Product:
            return "product";
        case CubReductionOp::Mean:
            return "mean";
        case CubReductionOp::L1Norm:
            return "l1";
        case CubReductionOp::L2Norm:
            return "l2";
        case CubReductionOp::SumSquares:
            return "sum_squares";
    }
    return "unknown";
}

struct ViewPathCase {
    const char* name;
    std::vector<uint64_t> dimensions;
    std::vector<uint64_t> strides;
    std::vector<uint32_t> axes;
    CubReductionPath expected_path;
    bool expect_dense_physical_permutation = false;
};

void expectPathForEveryValueOperation(const ViewPathCase& test_case) {
    const CubReductionGeometry structural =
        CubReduction::analyzeGeometry(test_case.dimensions, test_case.strides, test_case.axes);
    EXPECT_EQ(structural.path, test_case.expected_path) << test_case.name;
    EXPECT_EQ(structural.physical_layout_is_dense_permutation, test_case.expect_dense_physical_permutation)
        << test_case.name;

    for (CubReductionOp op : ALL_VALUE_OPERATIONS) {
        SCOPED_TRACE(std::string(test_case.name) + "/" + valueOperationName(op));
        const CubReductionGeometry value =
            CubReduction::analyzeValueGeometry(op, test_case.dimensions, test_case.strides, test_case.axes);
        EXPECT_EQ(value.path, structural.path);
        EXPECT_EQ(value.path, test_case.expected_path);
    }
}

}  // namespace

TEST(CubReductionViewGate, RemainingRankGreaterThanOneViewsStayOnLegacyFallbackUntilView1) {
    // VIEW-1A deliberately freezes the population that VIEW-1 must replace. These cases are semantically valid views,
    // but none can currently be represented by an ordained dense/permutation/rank-1-affine direct path.
    const std::vector<ViewPathCase> cases = {
        {"gapped_storage", {2, 2, 3}, {1, 7, 2}, {2}, CubReductionPath::StridedFixedSegment, false},
        {"overlapping_storage", {2, 2, 3}, {1, 1, 2}, {2}, CubReductionPath::StridedFixedSegment, false},
        {"broadcast_zero_stride", {2, 2, 3}, {0, 6, 2}, {2}, CubReductionPath::StridedFixedSegment, false},
        // Physical order is [2,0,1,3]. Reduction axes {1,2} are split by retained axis 0 in physical order.
        {"dense_permutation_split_reduction",
         {3, 5, 2, 7},
         {35, 7, 105, 1},
         {1, 2},
         CubReductionPath::StridedFixedSegment,
         true},
        // Axis 0 is one physical block, but retained logical order [1,2,3] is neither supported tiled output order.
        {"dense_permutation_unsupported_output_order",
         {3, 5, 2, 7},
         {35, 7, 105, 1},
         {0},
         CubReductionPath::StridedFixedSegment,
         true},
        // Physical order is [1,2,0]. Reducing logical axis 0 is physically trailing (inner_size == 1), for which the
        // current permutation-aware tiled family intentionally has no candidate.
        {"dense_permutation_physically_trailing_reduction",
         {2, 3, 4},
         {1, 8, 2},
         {0},
         CubReductionPath::StridedFixedSegment,
         true},
        // A rank>1 compact permutation reduced to one scalar cannot use the rank-1 affine DeviceTransformReduce path.
        {"dense_permutation_full_reduction",
         {2, 3, 4},
         {1, 8, 2},
         {0, 1, 2},
         CubReductionPath::StridedFixedSegment,
         true},
        {"singleton_heavy_gapped",
         {2, 1, 3, 1, 4},
         {20, 20, 4, 4, 1},
         {0, 2},
         CubReductionPath::StridedFixedSegment,
         false},
        // A huge stride on a singleton axis does not enlarge the allocation span, but it deliberately forces the legacy
        // value mapper to retain UINT64 indexing metadata. The non-singleton axes remain genuinely gapped.
        {"uint64_indexing_singleton_stride",
         {2, 1, 3, 4},
         {20, (1ULL << 32), 4, 1},
         {0, 2},
         CubReductionPath::StridedFixedSegment,
         false},
    };

    for (const ViewPathCase& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expectPathForEveryValueOperation(test_case);
    }
}

TEST(CubReductionViewGate, ExistingOrdainedViewAndDensePathsRemainOutsideLegacyFallback) {
    const std::vector<ViewPathCase> controls = {
        // Rank-1 arbitrary affine storage already has an intentional single-output direct path.
        {"rank1_affine", {4096}, {3}, {0}, CubReductionPath::DeviceTransformReduce, false},
        // Logical [K,I,J] over physical [I,J,K]; reducing J is supported by permutation-aware TiledFixedSegment.
        {"supported_dense_permutation", {32, 7, 5}, {1, 160, 32}, {2}, CubReductionPath::TiledFixedSegment, true},
        // Ordinary dense disjoint reductions are already classified as ComposedDense by analyzeGeometry().
        {"ordinary_dense_disjoint",
         {31, 17, 29, 19, 23},
         {17ULL * 29 * 19 * 23, 29ULL * 19 * 23, 19ULL * 23, 23, 1},
         {0, 2, 4},
         CubReductionPath::ComposedDense,
         true},
    };

    for (const ViewPathCase& test_case : controls) {
        SCOPED_TRACE(test_case.name);
        expectPathForEveryValueOperation(test_case);
        const CubReductionGeometry geometry =
            CubReduction::analyzeGeometry(test_case.dimensions, test_case.strides, test_case.axes);
        EXPECT_NE(geometry.path, CubReductionPath::StridedFixedSegment);
    }
}

TEST(CubReductionViewGate, LegacyFallbackBaselineCoversBothValueIndexWidths) {
    const CubReductionGeometry compact_indexing =
        CubReduction::analyzeValueGeometry(CubReductionOp::Sum, {2, 2, 3}, {1, 7, 2}, {2});
    ASSERT_EQ(compact_indexing.path, CubReductionPath::StridedFixedSegment);
    EXPECT_TRUE(compact_indexing.strided_value_indexing_fits_uint32);

    const CubReductionGeometry wide_indexing = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, {2, 1, 3, 4}, {20, (1ULL << 32), 4, 1}, {0, 2});
    ASSERT_EQ(wide_indexing.path, CubReductionPath::StridedFixedSegment);
    EXPECT_FALSE(wide_indexing.strided_value_indexing_fits_uint32);
}
