#include "Utilities/TensorOperations/Ragged/RaggedConv1dGeometry.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

TEST(RaggedConv1dGeometry, EffectiveKernelUsesDilationAndRejectsInvalidGeometry) {
    EXPECT_EQ(RaggedConv1dGeometry::valid(5, 1, 3).effectiveKernelWidth(), 13u);
    EXPECT_EQ(RaggedConv1dGeometry::valid(1, 7, 9).effectiveKernelWidth(), 1u);

    EXPECT_THROW((void)RaggedConv1dGeometry::valid(0), std::invalid_argument);
    EXPECT_THROW((void)RaggedConv1dGeometry::valid(3, 0, 1), std::invalid_argument);
    EXPECT_THROW((void)RaggedConv1dGeometry::valid(3, 1, 0), std::invalid_argument);

    RaggedConv1dGeometry malformed = RaggedConv1dGeometry::valid(3);
    malformed.explicit_left_padding = 1;
    EXPECT_THROW(malformed.validate(), std::invalid_argument);

    const uint64_t hugeKernel = std::numeric_limits<uint64_t>::max();
    EXPECT_THROW((void)RaggedConv1dGeometry::valid(hugeKernel, 1, 2), std::overflow_error);
}

TEST(RaggedConv1dGeometry, ValidUsesOrdinaryConvolutionFormulaIncludingShortAndEmptyRows) {
    const RaggedConv1dGeometry geometry = RaggedConv1dGeometry::valid(5, 2, 2);  // effective kernel 9
    EXPECT_EQ(geometry.resolvedPadding(20), (RaggedConv1dResolvedPadding{}));
    EXPECT_EQ(geometry.outputLength(0), 0u);
    EXPECT_EQ(geometry.outputLength(8), 0u);
    EXPECT_EQ(geometry.outputLength(9), 1u);
    EXPECT_EQ(geometry.outputLength(10), 1u);
    EXPECT_EQ(geometry.outputLength(11), 2u);
}

TEST(RaggedConv1dGeometry, CausalPaddingTracksEffectiveKernelAndStride) {
    const RaggedConv1dGeometry geometry = RaggedConv1dGeometry::causal(4, 3, 2);  // effective kernel 7
    EXPECT_EQ(geometry.resolvedPadding(0), (RaggedConv1dResolvedPadding{6, 0}));
    EXPECT_EQ(geometry.resolvedPadding(17), (RaggedConv1dResolvedPadding{6, 0}));
    EXPECT_EQ(geometry.outputLength(0), 0u);
    EXPECT_EQ(geometry.outputLength(1), 1u);
    EXPECT_EQ(geometry.outputLength(2), 1u);
    EXPECT_EQ(geometry.outputLength(3), 1u);
    EXPECT_EQ(geometry.outputLength(4), 2u);
}

TEST(RaggedConv1dGeometry, SameUpperResolvesPaddingPerRowAndPlacesOddPaddingOnRight) {
    const RaggedConv1dGeometry geometry = RaggedConv1dGeometry::sameUpper(4, 3, 2);  // effective kernel 7

    EXPECT_EQ(geometry.resolvedPadding(0), (RaggedConv1dResolvedPadding{0, 0}));
    EXPECT_EQ(geometry.outputLength(0), 0u);

    // L=5 -> ceil(5/3)=2, required=(2-1)*3+7=10, total pad=5 -> (2,3).
    EXPECT_EQ(geometry.resolvedPadding(5), (RaggedConv1dResolvedPadding{2, 3}));
    EXPECT_EQ(geometry.outputLength(5), 2u);

    // L=6 has the same output length but needs one less element of padding.
    EXPECT_EQ(geometry.resolvedPadding(6), (RaggedConv1dResolvedPadding{2, 2}));
    EXPECT_EQ(geometry.outputLength(6), 2u);

    // L=7 -> ceil(7/3)=3, required=13, total pad=6 -> (3,3).
    EXPECT_EQ(geometry.resolvedPadding(7), (RaggedConv1dResolvedPadding{3, 3}));
    EXPECT_EQ(geometry.outputLength(7), 3u);
}

TEST(RaggedConv1dGeometry, ExplicitPaddingCanProducePaddingOnlyOutputsFromEmptyRows) {
    const RaggedConv1dGeometry geometry = RaggedConv1dGeometry::explicitPadding(3, 4, 2, 2, 1);
    EXPECT_EQ(geometry.resolvedPadding(0), (RaggedConv1dResolvedPadding{4, 2}));
    EXPECT_EQ(geometry.outputLength(0), 2u);  // 1 + floor((6 - 3) / 2)
    EXPECT_EQ(geometry.outputLength(1), 3u);

    const RaggedConv1dGeometry notEnoughPadding = RaggedConv1dGeometry::explicitPadding(5, 1, 2);
    EXPECT_EQ(notEnoughPadding.outputLength(0), 0u);
}

TEST(RaggedConv1dGeometry, RowLengthPreservationClassificationIsSemantic) {
    EXPECT_TRUE(RaggedConv1dGeometry::causal(7, 1, 3).preservesRowLengths());
    EXPECT_TRUE(RaggedConv1dGeometry::sameUpper(6, 1, 4).preservesRowLengths());
    EXPECT_TRUE(RaggedConv1dGeometry::valid(1, 1, 9).preservesRowLengths());
    EXPECT_TRUE(RaggedConv1dGeometry::explicitPadding(5, 1, 3, 1, 1).preservesRowLengths());
    EXPECT_TRUE(RaggedConv1dGeometry::explicitPadding(3, 4, 0, 1, 2).preservesRowLengths());

    EXPECT_FALSE(RaggedConv1dGeometry::causal(7, 2, 3).preservesRowLengths());
    EXPECT_FALSE(RaggedConv1dGeometry::sameUpper(6, 2, 4).preservesRowLengths());
    EXPECT_FALSE(RaggedConv1dGeometry::valid(3).preservesRowLengths());
    EXPECT_FALSE(RaggedConv1dGeometry::explicitPadding(5, 1, 2).preservesRowLengths());
}

TEST(RaggedConv1dGeometry, DerivesExactOffsetsFromAuthoritativeHostPartition) {
    const std::vector<uint64_t> inputOffsets{0, 0, 1, 6, 12, 19};

    const RaggedConv1dGeometry same = RaggedConv1dGeometry::sameUpper(4, 3, 2);
    EXPECT_EQ(deriveRaggedConv1dOutputOffsets(inputOffsets, same),
              (std::vector<uint64_t>{0, 0, 1, 3, 5, 8}));

    const RaggedConv1dGeometry valid = RaggedConv1dGeometry::valid(4, 2, 2);  // effective kernel 7
    EXPECT_EQ(deriveRaggedConv1dOutputOffsets(inputOffsets, valid),
              (std::vector<uint64_t>{0, 0, 0, 0, 0, 1}));

    const RaggedConv1dGeometry explicitPadding = RaggedConv1dGeometry::explicitPadding(3, 4, 2, 2, 1);
    EXPECT_EQ(deriveRaggedConv1dOutputOffsets(inputOffsets, explicitPadding),
              (std::vector<uint64_t>{0, 2, 5, 10, 15, 21}));
}

TEST(RaggedConv1dGeometry, OffsetDerivationRejectsMalformedPartitionsAndOverflow) {
    const RaggedConv1dGeometry geometry = RaggedConv1dGeometry::valid(1);
    EXPECT_THROW((void)deriveRaggedConv1dOutputOffsets({}, geometry), std::invalid_argument);
    EXPECT_THROW((void)deriveRaggedConv1dOutputOffsets(std::vector<uint64_t>{1, 2}, geometry), std::invalid_argument);
    EXPECT_THROW((void)deriveRaggedConv1dOutputOffsets(std::vector<uint64_t>{0, 4, 3}, geometry), std::invalid_argument);

    const RaggedConv1dGeometry paddingOverflow =
        RaggedConv1dGeometry::explicitPadding(1, std::numeric_limits<uint64_t>::max(), 1);
    EXPECT_THROW((void)paddingOverflow.outputLength(0), std::overflow_error);
}

TEST(RaggedConv1dGeometry, CapacityDerivationIsConservativeAndOverflowChecked) {
    const RaggedConv1dGeometry same = RaggedConv1dGeometry::sameUpper(5, 3, 2);
    EXPECT_EQ(deriveRaggedConv1dOutputCapacity(8, 17, same),
              (RaggedConv1dOutputCapacity{.max_values_per_row = 6, .max_total_values = 48}));

    const RaggedConv1dGeometry explicitPadding = RaggedConv1dGeometry::explicitPadding(3, 4, 2, 2, 1);
    EXPECT_EQ(deriveRaggedConv1dOutputCapacity(4, 10, explicitPadding),
              (RaggedConv1dOutputCapacity{.max_values_per_row = 7, .max_total_values = 28}));

    EXPECT_EQ(deriveRaggedConv1dOutputCapacity(0, 10, explicitPadding),
              (RaggedConv1dOutputCapacity{.max_values_per_row = 7, .max_total_values = 0}));

    const RaggedConv1dGeometry identity = RaggedConv1dGeometry::valid(1);
    EXPECT_THROW((void)deriveRaggedConv1dOutputCapacity(std::numeric_limits<uint64_t>::max(), 2, identity),
                 std::overflow_error);
}

TEST(RaggedConv1dGeometry, ExtremeSameUpperExtentChecksOverflowInsteadOfWrapping) {
    const RaggedConv1dGeometry geometry = RaggedConv1dGeometry::sameUpper(2, 1, 2);  // effective kernel 3
    EXPECT_THROW((void)geometry.resolvedPadding(std::numeric_limits<uint64_t>::max()), std::overflow_error);
}
