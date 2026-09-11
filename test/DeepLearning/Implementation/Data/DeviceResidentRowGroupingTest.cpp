#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRowGrouping.h"

#include <gtest/gtest.h>

TEST(DeviceResidentRowGroupingTest, SmallAndMediumBatchesRetainPayloadSizedGrouping) {
    // The original measured small/medium-batch behavior remains unchanged.
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(2048, 1), 4u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(2048, 64), 4u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(2048, 256), 4u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(2048, 257), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(2048, 512), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(2048, 513), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(2048, 4096), 1u);
}

TEST(DeviceResidentRowGroupingTest, LargeBatchesIncreaseGroupingForNarrowRows) {
    // Batch-derived grouping grows at powers of two around the 512-CTA target.
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(4095, 32), 4u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(4096, 32), 8u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8191, 32), 8u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 32), 16u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16383, 32), 16u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 32), 32u);
}

TEST(DeviceResidentRowGroupingTest, Sub512RowsKeepOneKiBPayloadBudgetAtLargeBatches) {
    // Keep the previously measured narrow-row region unchanged. Exact 512-byte
    // rows have their own measured refinement below.
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(8192, 511), 1024u);
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(16384, 511), 1024u);

    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 32), 32u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 64), 16u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 128), 8u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 256), 4u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 257), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 511), 2u);
}

TEST(DeviceResidentRowGroupingTest, Exact512ByteRowsUseMeasuredFourRowGroupingAtLargeBatches) {
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(8191, 512), 1024u);
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(8192, 512), 2048u);
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(16384, 512), 2048u);

    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8191, 512), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 512), 4u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 512), 4u);

    // Neighboring widths deliberately remain on the general rule.
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 511), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 513), 2u);
}

TEST(DeviceResidentRowGroupingTest, Exact1024ByteRowsStartTwoRowGroupingAt4096Rows) {
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(4095, 1024), 1024u);
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(4096, 1024), 2048u);

    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(4095, 1024), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(4096, 1024), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 1024), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 1024), 4u);

    // The exact-width refinement must not pull neighboring rows forward.
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(4096, 1023), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(4096, 1025), 1u);
}

TEST(DeviceResidentRowGroupingTest, WideRowsIncreasePayloadBudgetAtMeasuredBatchBoundaries) {
    // General wider rows keep the 1 KiB budget through 8191 rows, use 2 KiB
    // at 8192, and 4 KiB at 16384. Exact 1024-byte rows are the measured
    // exception above: they opt into the 2 KiB budget from 4096 rows onward.
    // These are deliberately measured boundaries rather than an extrapolated
    // unbounded scaling rule.
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(8191, 513), 1024u);
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(8192, 513), 2048u);
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(16383, 513), 2048u);
    EXPECT_EQ(DeviceResidentRowGrouping::targetPayloadBytesPerCta(16384, 513), 4096u);

    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8191, 513), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 513), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16383, 513), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 513), 4u);

    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8191, 1024), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 1024), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16383, 1024), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 1024), 4u);
}

TEST(DeviceResidentRowGroupingTest, WideRowPayloadLimitStillWinsAsRowsGetWider) {
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 1025), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(8192, 2048), 1u);

    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 1025), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 2048), 2u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 2049), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 4096), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(16384, 4097), 1u);
}

TEST(DeviceResidentRowGroupingTest, GroupingIsClampedToImplementedSpecializations) {
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(1ull << 30, 1), 256u);
}

TEST(DeviceResidentRowGroupingTest, ZeroSizedDefensiveCasesUseOneRowPerCta) {
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(0, 64), 1u);
    EXPECT_EQ(DeviceResidentRowGrouping::selectRowsPerCta(1024, 0), 1u);
}
