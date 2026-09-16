#include "Utilities/LogicalWork.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace ThorImplementation {
namespace {

TEST(LogicalWorkCount, DefaultsToNoWork) {
    constexpr LogicalWorkCount work{};
    static_assert(work.empty());
    EXPECT_EQ(work.floatingPointOperations, 0U);
    EXPECT_EQ(work.bytes, 0U);
}

TEST(LogicalWorkCount, AddsIndependentFlopAndByteDimensions) {
    constexpr LogicalWorkCount lhs{.floatingPointOperations = 120, .bytes = 64};
    constexpr LogicalWorkCount rhs{.floatingPointOperations = 30, .bytes = 16};
    constexpr auto total = LogicalWorkCount::tryAdd(lhs, rhs);

    static_assert(total.has_value());
    static_assert(total->floatingPointOperations == 150);
    static_assert(total->bytes == 80);
    ASSERT_TRUE(total.has_value());
    EXPECT_EQ(total->floatingPointOperations, 150U);
    EXPECT_EQ(total->bytes, 80U);
}

TEST(LogicalWorkCount, AdditionOverflowIsNonThrowingAccountingFailure) {
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();

    constexpr auto flopOverflow = LogicalWorkCount::tryAdd(
        {.floatingPointOperations = max, .bytes = 4}, {.floatingPointOperations = 1, .bytes = 8});
    constexpr auto byteOverflow = LogicalWorkCount::tryAdd(
        {.floatingPointOperations = 4, .bytes = max}, {.floatingPointOperations = 8, .bytes = 1});

    static_assert(!flopOverflow.has_value());
    static_assert(!byteOverflow.has_value());
    EXPECT_FALSE(flopOverflow.has_value());
    EXPECT_FALSE(byteOverflow.has_value());
}

TEST(LogicalWorkCount, ScalesBothDimensionsWithCheckedArithmetic) {
    constexpr LogicalWorkCount perExample{.floatingPointOperations = 17, .bytes = 10};
    constexpr auto batch = LogicalWorkCount::tryScale(perExample, 8);

    static_assert(batch.has_value());
    static_assert(batch->floatingPointOperations == 136);
    static_assert(batch->bytes == 80);
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->floatingPointOperations, 136U);
    EXPECT_EQ(batch->bytes, 80U);
}

TEST(LogicalWorkCount, ScaleOverflowIsNonThrowingAccountingFailure) {
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();

    constexpr auto flopOverflow = LogicalWorkCount::tryScale(
        {.floatingPointOperations = max / 2 + 1, .bytes = 1}, 2);
    constexpr auto byteOverflow = LogicalWorkCount::tryScale(
        {.floatingPointOperations = 1, .bytes = max / 2 + 1}, 2);

    static_assert(!flopOverflow.has_value());
    static_assert(!byteOverflow.has_value());
    EXPECT_FALSE(flopOverflow.has_value());
    EXPECT_FALSE(byteOverflow.has_value());
}

TEST(LogicalWorkCount, EqualBranchEstimateIsDeterministicAndOverflowSafe) {
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();

    constexpr auto evenAverage = LogicalWorkCount::equalBranchEstimate(
        {.floatingPointOperations = 20, .bytes = 8}, {.floatingPointOperations = 40, .bytes = 12});
    constexpr auto roundedHalf = LogicalWorkCount::equalBranchEstimate(
        {.floatingPointOperations = 5, .bytes = 8}, {.floatingPointOperations = 6, .bytes = 9});
    constexpr auto maxAverage = LogicalWorkCount::equalBranchEstimate(
        {.floatingPointOperations = max, .bytes = max}, {.floatingPointOperations = max, .bytes = max});

    static_assert(evenAverage.floatingPointOperations == 30);
    static_assert(evenAverage.bytes == 10);
    static_assert(roundedHalf.floatingPointOperations == 6);
    static_assert(roundedHalf.bytes == 9);
    static_assert(maxAverage.floatingPointOperations == max);
    static_assert(maxAverage.bytes == max);

    EXPECT_EQ(evenAverage.floatingPointOperations, 30U);
    EXPECT_EQ(evenAverage.bytes, 10U);
    EXPECT_EQ(roundedHalf.floatingPointOperations, 6U);
    EXPECT_EQ(roundedHalf.bytes, 9U);
    EXPECT_EQ(maxAverage.floatingPointOperations, max);
    EXPECT_EQ(maxAverage.bytes, max);
}

TEST(LogicalWorkCount, LogicalTensorBytesUseSemanticPayloadExtent) {
    // The caller supplies active semantic elements rather than padded capacity.
    constexpr auto fp32Active = tryLogicalTensorByteCount(/*activeElements=*/37, /*elementSizeBytes=*/4);
    constexpr auto fp16Active = tryLogicalTensorByteCount(/*activeElements=*/37, /*elementSizeBytes=*/2);

    static_assert(fp32Active.has_value() && *fp32Active == 148);
    static_assert(fp16Active.has_value() && *fp16Active == 74);
    ASSERT_TRUE(fp32Active.has_value());
    ASSERT_TRUE(fp16Active.has_value());
    EXPECT_EQ(*fp32Active, 148U);
    EXPECT_EQ(*fp16Active, 74U);
}

TEST(LogicalWorkCount, LogicalTensorByteOverflowIsNonThrowingAccountingFailure) {
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    constexpr auto bytes = tryLogicalTensorByteCount(max, 2);

    static_assert(!bytes.has_value());
    EXPECT_FALSE(bytes.has_value());
}

}  // namespace
}  // namespace ThorImplementation
