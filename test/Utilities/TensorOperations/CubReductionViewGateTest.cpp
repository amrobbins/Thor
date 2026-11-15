#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/Exceptions.h"

#include <array>
#include <cstdint>
#include <optional>
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
    std::optional<CubReductionPath> expected_path;
    bool expect_dense_physical_permutation = false;
    bool expect_permutation_contiguous_segments = false;
};

void expectPathForEveryValueOperation(const ViewPathCase& test_case) {
    if (!test_case.expected_path.has_value()) {
        EXPECT_THROW((void)CubReduction::analyzeGeometry(test_case.dimensions, test_case.strides, test_case.axes),
                     NotImplementedException)
            << test_case.name;
        for (CubReductionOp op : ALL_VALUE_OPERATIONS) {
            SCOPED_TRACE(std::string(test_case.name) + "/" + valueOperationName(op));
            EXPECT_THROW((void)CubReduction::analyzeValueGeometry(
                             op, test_case.dimensions, test_case.strides, test_case.axes),
                         NotImplementedException);
        }
        return;
    }

    const CubReductionGeometry structural =
        CubReduction::analyzeGeometry(test_case.dimensions, test_case.strides, test_case.axes);
    EXPECT_EQ(structural.path, test_case.expected_path.value()) << test_case.name;
    EXPECT_EQ(structural.physical_layout_is_dense_permutation, test_case.expect_dense_physical_permutation)
        << test_case.name;
    EXPECT_EQ(structural.permutation_aware_contiguous_segments, test_case.expect_permutation_contiguous_segments)
        << test_case.name;

    for (CubReductionOp op : ALL_VALUE_OPERATIONS) {
        SCOPED_TRACE(std::string(test_case.name) + "/" + valueOperationName(op));
        const CubReductionGeometry value =
            CubReduction::analyzeValueGeometry(op, test_case.dimensions, test_case.strides, test_case.axes);
        EXPECT_EQ(value.path, structural.path);
        EXPECT_EQ(value.path, test_case.expected_path.value());
    }
}

}  // namespace

TEST(CubReductionViewGate, UnsupportedRankGreaterThanOneViewsAreRejectedAfterDelete) {
    // DELETE removes the arbitrary mixed-radix mapper entirely. Unsupported rank>1 views fail during structural
    // analysis itself rather than receiving a catch-all path that could accidentally become executable later.
    const std::vector<ViewPathCase> cases = {
        {"gapped_storage", {2, 2, 3}, {1, 7, 2}, {2}, std::nullopt, false},
        {"overlapping_storage", {2, 2, 3}, {1, 1, 2}, {2}, std::nullopt, false},
        {"broadcast_zero_stride", {2, 2, 3}, {0, 6, 2}, {2}, std::nullopt, false},
        // Physical order is [1,0,2]. The reduced axis is physically trailing, but retained physical order [1,0]
        // differs from logical output order [0,1]. This is not the middle-reduction [A,R,B,P] rotation owned by 2B.
        {"dense_permutation_physically_trailing_output_reorder",
         {2, 3, 4},
         {4, 8, 1},
         {2},
         std::nullopt,
         true},
        {"singleton_heavy_gapped",
         {2, 1, 3, 1, 4},
         {20, 20, 4, 4, 1},
         {0, 2},
         std::nullopt,
         false},
        // A huge stride on a singleton axis does not enlarge the allocation span. It remains a useful regression that
        // unsupported geometry cannot regain a hidden wide-index catch-all path. The non-singleton axes remain gapped.
        {"uint64_indexing_singleton_stride",
         {2, 1, 3, 4},
         {20, (1ULL << 32), 4, 1},
         {0, 2},
         std::nullopt,
         false},
    };

    for (const ViewPathCase& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expectPathForEveryValueOperation(test_case);
    }

    // 1376: a split physical reduction whose retained axes keep their physical order is the dense tensor of its bytes
    // over disjoint runs. Physical order is [2,0,1,3]; reduction axes {1,2} are split by retained axis 0.
    expectPathForEveryValueOperation(
        {"dense_permutation_split_reduction", {3, 5, 2, 7}, {35, 7, 105, 1}, {1, 2}, CubReductionPath::ComposedDense,
         true});
}

TEST(CubReductionViewGate, ExistingOrdainedViewAndDensePathsRetainTheirOwnersAfterDelete) {
    const std::vector<ViewPathCase> controls = {
        // Rank-1 arbitrary affine storage already has an intentional single-output direct path.
        {"rank1_affine", {4096}, {3}, {0}, CubReductionPath::DeviceTransformReduce, false},
        // VIEW-DIRECT-1: reducing every axis of a compact physical permutation is one contiguous physical domain.
        {"dense_permutation_full_reduction",
         {2, 3, 4},
         {1, 8, 2},
         {0, 1, 2},
         CubReductionPath::DeviceTransformReduce,
         true},
        // VIEW-DIRECT-2A: physical order is [1,2,0], so reducing axis 0 gives direct contiguous segments whose
        // physical retained order [1,2] already equals Thor's logical dense output order.
        {"dense_permutation_physically_trailing_reduction",
         {2, 3, 4},
         {1, 8, 2},
         {0},
         CubReductionPath::ContiguousFixedSegment,
         true,
         true},
        // The same direct family also applies when logically disjoint reduced axes become one physical trailing block.
        {"dense_permutation_physically_trailing_disjoint_reduction",
         {2, 3, 4, 5},
         {4, 40, 1, 8},
         {0, 2},
         CubReductionPath::ContiguousFixedSegment,
         true,
         true},
        // Logical [J,I,K] over physical [I,J,K]; reducing J leaves the natural output order (the dense planner on the
        // view's bytes since 1376). 1390: [K,I,J] reducing J, with the output rotated, is refused (the test below).
        {"natural_dense_permutation", {5, 7, 32}, {32, 160, 1}, {0}, CubReductionPath::TiledFixedSegment, true},
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
    }
}


// 1390: the reducer refuses the rotated-output views (F and VIEW-DIRECT-2B) and the pitched views (H); a caller runs
// them in two parts (CubReduction::splitRefusedView, CubReductionViewSplitTest).
static void expectRefusedForEveryValueOperation(const std::vector<uint64_t>& dimensions,
                                                const std::vector<uint64_t>& strides,
                                                const std::vector<uint32_t>& axes) {
    expectPathForEveryValueOperation({"refused", dimensions, strides, axes, std::nullopt});
}

TEST(CubReductionViewGate, RotatedOutputViewsAreRefusedAndSplitAsADenseReductionThenARotation) {
    // F: logical [K,I,J] over physical [I,J,K], reducing J: the output [K, I] is the rotation of the dense [I, K].
    expectRefusedForEveryValueOperation({32, 7, 5}, {1, 160, 32}, {2});
    std::optional<CubReductionViewSplit> split = CubReduction::splitRefusedView({32, 7, 5}, {1, 160, 32}, {2});
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->kind, CubReductionViewSplit::Kind::RotatedOutput);
    EXPECT_EQ(split->dense_dimensions, (std::vector<uint64_t>{7, 5, 32}));
    EXPECT_EQ(split->a, 7U);
    EXPECT_EQ(split->b, 32U);
    EXPECT_EQ(split->p, 1U);

    // VIEW-DIRECT-2B: physical [A 2, R 3, B 5, P 7] read as [3, 5, 2, 7] (logical axes R, B, A, P), reducing R: the
    // output [B, A, P] is the rotation of payload vectors of the dense [A, B x P].
    const std::vector<uint64_t> dimensions = {3, 5, 2, 7};
    const std::vector<uint64_t> strides = {35, 7, 105, 1};
    expectRefusedForEveryValueOperation(dimensions, strides, {0});
    split = CubReduction::splitRefusedView(dimensions, strides, {0});
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->kind, CubReductionViewSplit::Kind::RotatedOutput);
    EXPECT_EQ(split->dense_dimensions, (std::vector<uint64_t>{2, 3, 35}));
    EXPECT_EQ(split->dense_axes, std::vector<uint32_t>{1});
    EXPECT_EQ(split->a, 2U);
    EXPECT_EQ(split->b, 5U);
    EXPECT_EQ(split->p, 7U);

    // 1376: a split physical reduction with the retained axes in physical order is ComposedDense (the dense planner on
    // its bytes), not a 2B rotation.
    EXPECT_EQ(CubReduction::analyzeGeometry({3, 5, 2, 7}, {35, 7, 105, 1}, {1, 2}).path,
              CubReductionPath::ComposedDense);

    // A physically trailing reduction with retained-order mismatch is not [A,R,B,P] and remains explicitly unsupported.
    EXPECT_THROW((void)CubReduction::analyzeGeometry({2, 3, 4}, {4, 8, 1}, {2}),
                 NotImplementedException);
    EXPECT_FALSE(CubReduction::splitRefusedView({2, 3, 4}, {4, 8, 1}, {2}).has_value());
}

TEST(CubReductionViewGate, PitchedViewsOfNonOverlappingAffineSlabsAreRefusedAndSplitAsACopyThenADenseReduction) {
    struct PitchedCase {
        const char* name;
        std::vector<uint64_t> dimensions;
        std::vector<uint64_t> strides;
        std::vector<uint32_t> axes;
    };

    const std::vector<PitchedCase> cases = {
        // Collapsed repeated-label view from iirk,kj->ij.
        {"einsum_diagonal_pre_gemm", {2, 2, 3}, {18, 3, 1}, {1}},
        // Collapsed repeated-label view from iir,j->rj.
        {"einsum_diagonal_pair_product", {2, 3}, {9, 1}, {0}},
        // Multiple adjacent reduced axes flatten to one pitched reduction coordinate.
        {"multi_axis_pitched", {2, 2, 3, 4}, {100, 12, 4, 1}, {1, 2}},
        // Multiple outer axes may also flatten as long as that group is internally row-major affine.
        {"multi_outer_pitched", {2, 3, 2, 4}, {300, 100, 4, 1}, {2}},
    };

    for (const PitchedCase& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        // 1390: refused, and split as the dense copy of the view's rows, then the reduction of the copy.
        expectRefusedForEveryValueOperation(test_case.dimensions, test_case.strides, test_case.axes);
        const std::optional<CubReductionViewSplit> split =
            CubReduction::splitRefusedView(test_case.dimensions, test_case.strides, test_case.axes);
        ASSERT_TRUE(split.has_value());
        EXPECT_EQ(split->kind, CubReductionViewSplit::Kind::PackedInput);
        EXPECT_EQ(split->dense_dimensions, test_case.dimensions);
        EXPECT_EQ(split->dense_axes, test_case.axes);
    }

    // Views the pitched classification excludes have no split either: no reducer and no caller owns them.
    const std::vector<ViewPathCase> exclusions = {
        // A non-contiguous trailing payload would turn adjacent component lanes into strided scalar traffic.
        {"noncontiguous_inner_payload",
         {2048, 127, 65},
         {16510, 130, 2},
         {1},
         std::nullopt,
         false},
        // A true broadcast reduction is not a one-to-one pitched source traversal.
        {"broadcast_reduced_axis",
         {8192, 127, 65},
         {0, 65, 1},
         {0},
         std::nullopt,
         false},
        // Adjacent outer entries must own disjoint reduction slabs.
        {"overlapping_outer_slabs",
         {4096, 127, 65},
         {65, 65, 1},
         {1},
         std::nullopt,
         false},
    };
    for (const ViewPathCase& test_case : exclusions) {
        SCOPED_TRACE(test_case.name);
        expectPathForEveryValueOperation(test_case);
        EXPECT_FALSE(
            CubReduction::splitRefusedView(test_case.dimensions, test_case.strides, test_case.axes).has_value());
    }
}

TEST(CubReductionViewGate, DeleteRejectsFormerUint32AndUint64LegacyIndexDomains) {
    EXPECT_THROW((void)CubReduction::analyzeGeometry({2, 2, 3}, {1, 7, 2}, {2}), NotImplementedException);
    EXPECT_THROW((void)CubReduction::analyzeGeometry(
                     {2, 1, 3, 4}, {20, (1ULL << 32), 4, 1}, {0, 2}),
                 NotImplementedException);
}
