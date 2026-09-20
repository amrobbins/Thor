#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

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

static_assert(static_cast<uint8_t>(CubReductionOp::Sum) == 0);
static_assert(static_cast<uint8_t>(CubReductionOp::Min) == 1);
static_assert(static_cast<uint8_t>(CubReductionOp::Max) == 2);
static_assert(static_cast<uint8_t>(CubReductionOp::Product) == 3);
static_assert(static_cast<uint8_t>(CubReductionOp::Mean) == 4);
static_assert(static_cast<uint8_t>(CubReductionOp::L1Norm) == 5);
static_assert(static_cast<uint8_t>(CubReductionOp::L2Norm) == 6);
static_assert(static_cast<uint8_t>(CubReductionOp::SumSquares) == 7);

[[nodiscard]] bool isOrdainedDenseValuePath(CubReductionPath path) {
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

[[nodiscard]] const char* valueOperationName(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
            return "Sum";
        case CubReductionOp::Min:
            return "Min";
        case CubReductionOp::Max:
            return "Max";
        case CubReductionOp::Product:
            return "Product";
        case CubReductionOp::Mean:
            return "Mean";
        case CubReductionOp::L1Norm:
            return "L1Norm";
        case CubReductionOp::L2Norm:
            return "L2Norm";
        case CubReductionOp::SumSquares:
            return "SumSquares";
    }
    return "Unknown";
}

[[nodiscard]] std::string denseValueGateContext(CubReductionOp op,
                                                 const std::vector<uint64_t>& dimensions,
                                                 const std::vector<uint32_t>& axes) {
    std::ostringstream context;
    context << "op=" << valueOperationName(op) << " dimensions=[";
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

void expectEveryDenseValueMaskUsesOrdainedPath(const std::vector<uint64_t>& dimensions) {
    ASSERT_FALSE(dimensions.empty());
    ASSERT_LT(dimensions.size(), 32U);

    const uint32_t rank = static_cast<uint32_t>(dimensions.size());
    const uint32_t mask_count = 1U << rank;
    for (CubReductionOp op : ALL_VALUE_OPERATIONS) {
        for (uint32_t mask = 1; mask < mask_count; ++mask) {
            const std::vector<uint32_t> axes = axesFromMask(rank, mask);
            SCOPED_TRACE(denseValueGateContext(op, dimensions, axes));
            const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(op, dimensions, axes);
            EXPECT_TRUE(isOrdainedDenseValuePath(geometry.path));
            EXPECT_NE(geometry.path, CubReductionPath::StridedFixedSegment);
        }
    }
}

}  // namespace

TEST(CubReductionDenseValueGate, EverySupportedOperationAndNonemptyMaskAcrossDenseRanksAvoidsStridedFallback) {
    // Exhaust every reduction mask for every public value operation. Increasing ranks naturally cover prefix, suffix,
    // middle, alternating R/K runs, leading/trailing retained runs, and many-run compositions. Keeping this list of
    // operations explicit makes additions/reorderings to CubReductionOp require a corresponding gate update.
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
        expectEveryDenseValueMaskUsesOrdainedPath(dimensions);
    }
}

TEST(CubReductionDenseValueGate, SingletonSeparatedDenseRanksAvoidStridedFallbackForEveryOperationAndMask) {
    // Retained and reduced singleton dimensions must not manufacture a fake irregular geometry. This also covers
    // reductions made entirely of singleton axes, for which transform/finalization still has to happen exactly once.
    expectEveryDenseValueMaskUsesOrdainedPath({2, 1, 3, 1, 5, 1, 2, 1, 3});
    expectEveryDenseValueMaskUsesOrdainedPath({1, 2, 1, 3, 1, 5, 1, 2, 1});
}

TEST(CubReductionDenseValueGate, DensePhysicalPermutationRemainsOnOrdainedPathForEveryValueOperation) {
    // This is the logical [K,I,J] view over physically dense [I,J,K] used by the permutation-aware tiled tests.
    // DENSE-VALUE-GATE must not accidentally classify a dense physical permutation as an arbitrary irregular view.
    const std::vector<uint64_t> dimensions{2, 2, 3};
    const std::vector<uint64_t> strides{1, 6, 2};
    const std::vector<uint32_t> axes{2};
    for (CubReductionOp op : ALL_VALUE_OPERATIONS) {
        SCOPED_TRACE(valueOperationName(op));
        const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(op, dimensions, strides, axes);
        EXPECT_EQ(geometry.path, CubReductionPath::TiledFixedSegment);
        EXPECT_TRUE(geometry.physical_layout_is_dense_permutation);
        EXPECT_TRUE(geometry.permutation_aware_tiled_geometry.has_value());
        EXPECT_NE(geometry.path, CubReductionPath::StridedFixedSegment);
    }
}

TEST(CubReductionDenseValueGate, GenuineIrregularViewsRemainOutsideTheDenseValueGateForEveryOperation) {
    // DENSE-VALUE-GATE closes ordinary dense value reductions only. A gapped view remains on the arbitrary-view
    // fallback until VIEW-1 replaces that machinery.
    const std::vector<uint64_t> dimensions{2, 3, 4};
    const std::vector<uint64_t> strides{20, 4, 1};
    const std::vector<uint32_t> axes{0, 2};
    for (CubReductionOp op : ALL_VALUE_OPERATIONS) {
        SCOPED_TRACE(valueOperationName(op));
        const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(op, dimensions, strides, axes);
        EXPECT_EQ(geometry.path, CubReductionPath::StridedFixedSegment);
        EXPECT_FALSE(geometry.dense_run_geometry.has_value());
    }
}
