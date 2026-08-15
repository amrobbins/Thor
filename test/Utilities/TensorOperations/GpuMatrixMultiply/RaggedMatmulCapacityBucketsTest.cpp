#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

TEST(RaggedMatmulCapacityBuckets, CapacitiesBelow16UseOnlyFullCapacity) {
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(1), (std::vector<uint64_t>{1}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(7), (std::vector<uint64_t>{7}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(8), (std::vector<uint64_t>{8}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(15), (std::vector<uint64_t>{15}));
}

TEST(RaggedMatmulCapacityBuckets, FullCapacityAndPowerOfTwoClassesStopAt8) {
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(16), (std::vector<uint64_t>{8, 16}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(17), (std::vector<uint64_t>{8, 17}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(18), (std::vector<uint64_t>{8, 18}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(33), (std::vector<uint64_t>{8, 16, 33}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(64), (std::vector<uint64_t>{8, 16, 32, 64}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(65), (std::vector<uint64_t>{8, 16, 32, 65}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(66), (std::vector<uint64_t>{8, 16, 32, 66}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(100), (std::vector<uint64_t>{8, 16, 32, 64, 100}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(128), (std::vector<uint64_t>{8, 16, 32, 64, 128}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(256), (std::vector<uint64_t>{8, 16, 32, 64, 128, 256}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(1000), (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 1000}));
}

TEST(RaggedMatmulCapacityBuckets, NearestHalfPowerUsesArithmeticDistanceAndBreaksTiesUpward) {
    // 23 / 2 = 11.5, which is closer to 8 than 16.
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(23), (std::vector<uint64_t>{8, 23}));

    // 24 / 2 = 12, exactly halfway between 8 and 16.  Prefer 16 so the
    // largest interval before the full-capacity kernel stays smaller.
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(24), (std::vector<uint64_t>{8, 16, 24}));

    // 25 / 2 = 12.5, which is closer to 16.
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(25), (std::vector<uint64_t>{8, 16, 25}));

    // Keep the same policy at a larger scale.
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(95), (std::vector<uint64_t>{8, 16, 32, 95}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(96), (std::vector<uint64_t>{8, 16, 32, 64, 96}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(97), (std::vector<uint64_t>{8, 16, 32, 64, 97}));
}

TEST(RaggedMatmulCapacityBuckets, PowerOfTwoFullCapacityIsNotDuplicated) {
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(512), (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(1024), (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 1024}));
}

TEST(RaggedMatmulCapacityBuckets, HandlesLargeUint64CapacitiesWithoutOverflow) {
    const uint64_t fullCapacity = std::numeric_limits<uint64_t>::max();
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(fullCapacity);

    ASSERT_FALSE(buckets.empty());
    EXPECT_EQ(buckets.front(), 8U);
    EXPECT_EQ(buckets.back(), fullCapacity);
    EXPECT_EQ(buckets[buckets.size() - 2], uint64_t{1} << 63);
}

TEST(RaggedMatmulCapacityBuckets, RejectsZeroFullCapacity) {
    EXPECT_THROW((void)makeRaggedMatmulCapacityBuckets(0), std::invalid_argument);
}

TEST(RaggedMatmulCapacityBuckets, SelectsSmallestCapacityThatContainsActiveRows) {
    const std::vector<uint64_t> buckets{8, 16, 32, 66};

    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(1, buckets), 8U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(8, buckets), 8U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(9, buckets), 16U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(16, buckets), 16U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(17, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(31, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(32, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(33, buckets), 66U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(65, buckets), 66U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(66, buckets), 66U);
}

TEST(RaggedMatmulCapacityBuckets, SelectionWorksAcrossMultiplePowerOfTwoClasses) {
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(1000);

    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(8, buckets), 8U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(9, buckets), 16U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(16, buckets), 16U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(17, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(32, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(33, buckets), 64U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(64, buckets), 64U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(65, buckets), 128U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(513, buckets), 1000U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(1000, buckets), 1000U);
}

TEST(RaggedMatmulCapacityBuckets, SelectionRejectsInvalidActiveCountsAndEmptyBuckets) {
    const std::vector<uint64_t> buckets{8, 16, 32, 66};
    const std::vector<uint64_t> noBuckets;

    EXPECT_THROW((void)chooseRaggedMatmulCapacityBucket(0, buckets), std::invalid_argument);
    EXPECT_THROW((void)chooseRaggedMatmulCapacityBucket(67, buckets), std::invalid_argument);
    EXPECT_THROW((void)chooseRaggedMatmulCapacityBucket(1, noBuckets), std::invalid_argument);
}
