#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "Utilities/Exceptions.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <optional>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;

namespace {

constexpr std::array<ReductionOp, 8> ALL_VALUE_OPERATIONS = {
    ReductionOp::Sum,
    ReductionOp::Min,
    ReductionOp::Max,
    ReductionOp::Product,
    ReductionOp::Mean,
    ReductionOp::L1Norm,
    ReductionOp::L2Norm,
    ReductionOp::SumSquares,
};

static_assert(static_cast<uint8_t>(ReductionOp::Sum) == 0);
static_assert(static_cast<uint8_t>(ReductionOp::Min) == 1);
static_assert(static_cast<uint8_t>(ReductionOp::Max) == 2);
static_assert(static_cast<uint8_t>(ReductionOp::Product) == 3);
static_assert(static_cast<uint8_t>(ReductionOp::Mean) == 4);
static_assert(static_cast<uint8_t>(ReductionOp::L1Norm) == 5);
static_assert(static_cast<uint8_t>(ReductionOp::L2Norm) == 6);
static_assert(static_cast<uint8_t>(ReductionOp::SumSquares) == 7);

[[nodiscard]] bool isOrdainedDenseValuePath(ReductionLayout path) {
    return path == ReductionLayout::SingleOutput || path == ReductionLayout::TrailingBlock
           || path == ReductionLayout::MiddleBlock || path == ReductionLayout::DisjointRuns;
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

[[nodiscard]] const char* valueOperationName(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
            return "Sum";
        case ReductionOp::Min:
            return "Min";
        case ReductionOp::Max:
            return "Max";
        case ReductionOp::Product:
            return "Product";
        case ReductionOp::Mean:
            return "Mean";
        case ReductionOp::L1Norm:
            return "L1Norm";
        case ReductionOp::L2Norm:
            return "L2Norm";
        case ReductionOp::SumSquares:
            return "SumSquares";
    }
    return "Unknown";
}

[[nodiscard]] std::string denseValueGateContext(ReductionOp op,
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
    for (ReductionOp op : ALL_VALUE_OPERATIONS) {
        for (uint32_t mask = 1; mask < mask_count; ++mask) {
            const std::vector<uint32_t> axes = axesFromMask(rank, mask);
            SCOPED_TRACE(denseValueGateContext(op, dimensions, axes));
            const ReductionGeometry structural_geometry = Reducer::analyzeGeometry(dimensions, axes);
            const ReductionGeometry geometry = Reducer::analyzeValueGeometry(op, dimensions, axes);
            EXPECT_EQ(geometry.path, structural_geometry.path);
            EXPECT_TRUE(isOrdainedDenseValuePath(structural_geometry.path));
            EXPECT_TRUE(isOrdainedDenseValuePath(geometry.path));
        }
    }
}

}  // namespace


TEST(ReducerDenseValueGate, StructuralDenseClassifierHasNoLegacyEscapeHatchAcrossAllMasksThroughRankTen) {
    // The dense gate is an operation-independent architecture invariant. Exercise the structural selector directly,
    // including more masks than the per-operation matrix above, so a future planner regression cannot silently turn a
    // canonical dense tensor back into the arbitrary-view family before value/ARG wrappers see it.
    for (uint32_t rank = 1; rank <= 10; ++rank) {
        std::vector<uint64_t> dimensions;
        dimensions.reserve(rank);
        for (uint32_t axis = 0; axis < rank; ++axis) {
            dimensions.push_back(axis % 3 == 0 ? 2U : axis % 3 == 1 ? 3U : 5U);
        }

        const uint32_t mask_count = 1U << rank;
        for (uint32_t mask = 1; mask < mask_count; ++mask) {
            const std::vector<uint32_t> axes = axesFromMask(rank, mask);
            SCOPED_TRACE(denseValueGateContext(ReductionOp::Sum, dimensions, axes));
            const ReductionGeometry geometry = Reducer::analyzeGeometry(dimensions, axes);
            EXPECT_TRUE(isOrdainedDenseValuePath(geometry.path));
        }
    }
}

TEST(ReducerDenseValueGate, EverySupportedOperationAndNonemptyMaskAcrossDenseRanksAvoidsStridedFallback) {
    // Exhaust every reduction mask for every public value operation. Increasing ranks naturally cover prefix, suffix,
    // middle, alternating R/K runs, leading/trailing retained runs, and many-run compositions. Keeping this list of
    // operations explicit makes additions/reorderings to ReductionOp require a corresponding gate update.
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

TEST(ReducerDenseValueGate, SingletonSeparatedDenseRanksAvoidStridedFallbackForEveryOperationAndMask) {
    // Retained and reduced singleton dimensions must not manufacture a fake irregular geometry. This also covers
    // reductions made entirely of singleton axes, for which transform/finalization still has to happen exactly once.
    expectEveryDenseValueMaskUsesOrdainedPath({2, 1, 3, 1, 5, 1, 2, 1, 3});
    expectEveryDenseValueMaskUsesOrdainedPath({1, 2, 1, 3, 1, 5, 1, 2, 1});
}

TEST(ReducerDenseValueGate, RotatedOutputPermutationIsRefusedAndSplitForEveryValueOperation) {
    // This is the logical [K,I,J] view over physically dense [I,J,K] used by the rotated-output tests. The dense value
    // gate must not classify a dense physical permutation as an arbitrary irregular view: the reducer refuses it as a
    // rotated output, which a caller runs in two parts (Reducer::splitRefusedView), and it has that split.
    const std::vector<uint64_t> dimensions{2, 2, 3};
    const std::vector<uint64_t> strides{1, 6, 2};
    const std::vector<uint32_t> axes{2};
    EXPECT_THROW((void)Reducer::analyzeGeometry(dimensions, strides, axes), NotImplementedException);
    for (ReductionOp op : ALL_VALUE_OPERATIONS) {
        SCOPED_TRACE(valueOperationName(op));
        EXPECT_THROW((void)Reducer::analyzeValueGeometry(op, dimensions, strides, axes), NotImplementedException);
    }
    const std::optional<ViewSplit> split = Reducer::splitRefusedView(dimensions, strides, axes);
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->kind, ViewSplit::Kind::RotatedOutput);
    EXPECT_EQ(split->dense_dimensions, (std::vector<uint64_t>{2, 3, 2}));
}

TEST(ReducerDenseValueGate, GenuineIrregularViewsRemainOutsideTheDenseValueGateForEveryOperation) {
    // Structural analysis rejects an arbitrary gapped view, and value planning rejects it for every operation.
    const std::vector<uint64_t> dimensions{2, 3, 4};
    const std::vector<uint64_t> strides{20, 4, 1};
    const std::vector<uint32_t> axes{0, 2};
    EXPECT_THROW((void)Reducer::analyzeGeometry(dimensions, strides, axes), NotImplementedException);
    for (ReductionOp op : ALL_VALUE_OPERATIONS) {
        SCOPED_TRACE(valueOperationName(op));
        EXPECT_THROW((void)Reducer::analyzeValueGeometry(op, dimensions, strides, axes),
                     NotImplementedException);
    }
}
