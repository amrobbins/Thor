#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include "gtest/gtest.h"

#include <algorithm>
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

TEST(RaggedMatmulCapacityBuckets, UsesCanonicalPowerOfTwoBucketsThrough512) {
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(16), (std::vector<uint64_t>{8, 16}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(17), (std::vector<uint64_t>{8, 16, 17}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(18), (std::vector<uint64_t>{8, 16, 18}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(33), (std::vector<uint64_t>{8, 16, 32, 33}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(64), (std::vector<uint64_t>{8, 16, 32, 64}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(65), (std::vector<uint64_t>{8, 16, 32, 64, 65}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(66), (std::vector<uint64_t>{8, 16, 32, 64, 66}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(100), (std::vector<uint64_t>{8, 16, 32, 64, 100}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(128), (std::vector<uint64_t>{8, 16, 32, 64, 128}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(256), (std::vector<uint64_t>{8, 16, 32, 64, 128, 256}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(512),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512}));
}

TEST(RaggedMatmulCapacityBuckets, Includes768Between512And1024) {
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(767),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 767}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(768),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 768}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(769),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 768, 769}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(1000),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 768, 1000}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(1024),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 768, 1024}));
}

TEST(RaggedMatmulCapacityBuckets, UsesQuarterOctaveBucketsStartingAt1024) {
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(2048),
              (std::vector<uint64_t>{8,
                                     16,
                                     32,
                                     64,
                                     128,
                                     256,
                                     512,
                                     768,
                                     1024,
                                     1280,
                                     1536,
                                     1792,
                                     2048}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(4096),
              (std::vector<uint64_t>{8,
                                     16,
                                     32,
                                     64,
                                     128,
                                     256,
                                     512,
                                     768,
                                     1024,
                                     1280,
                                     1536,
                                     1792,
                                     2048,
                                     2560,
                                     3072,
                                     3584,
                                     4096}));
}

TEST(RaggedMatmulCapacityBuckets, AppendsExactFullCapacityBetweenCanonicalBuckets) {
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(95), (std::vector<uint64_t>{8, 16, 32, 64, 95}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(1100),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 768, 1024, 1100}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(1300),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1300}));
    EXPECT_EQ(makeRaggedMatmulCapacityBuckets(5000),
              (std::vector<uint64_t>{8,
                                     16,
                                     32,
                                     64,
                                     128,
                                     256,
                                     512,
                                     768,
                                     1024,
                                     1280,
                                     1536,
                                     1792,
                                     2048,
                                     2560,
                                     3072,
                                     3584,
                                     4096,
                                     5000}));
}

TEST(RaggedMatmulCapacityBuckets, QuarterOctavePolicyLimitsLargeCanonicalBucketGrowthTo25Percent) {
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(1U << 20);
    ASSERT_TRUE(std::is_sorted(buckets.begin(), buckets.end()));

    for (size_t i = 1; i < buckets.size(); ++i) {
        if (buckets[i - 1] < 1024)
            continue;
        EXPECT_LE(buckets[i] * 4, buckets[i - 1] * 5)
            << "bucket growth exceeded 25% between " << buckets[i - 1] << " and " << buckets[i];
    }
}

TEST(RaggedMatmulCapacityBuckets, HandlesLargeUint64CapacitiesWithoutOverflow) {
    const uint64_t fullCapacity = std::numeric_limits<uint64_t>::max();
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(fullCapacity);

    ASSERT_FALSE(buckets.empty());
    EXPECT_EQ(buckets.front(), 8U);
    EXPECT_EQ(buckets.back(), fullCapacity);
    EXPECT_TRUE(std::is_sorted(buckets.begin(), buckets.end()));
    ASSERT_GE(buckets.size(), 2U);
    EXPECT_EQ(buckets[buckets.size() - 2], 7U * (uint64_t{1} << 61));
}

TEST(RaggedMatmulCapacityBuckets, RmsNormMirrorsGemmsCanonicalBucketPolicy) {
    for (const uint64_t fullCapacity : std::vector<uint64_t>{1,
                                                            7,
                                                            8,
                                                            15,
                                                            16,
                                                            17,
                                                            66,
                                                            95,
                                                            96,
                                                            512,
                                                            767,
                                                            768,
                                                            769,
                                                            1000,
                                                            1024,
                                                            1100,
                                                            2048,
                                                            4096,
                                                            5000,
                                                            26208}) {
        EXPECT_EQ(makeRaggedRmsNormCapacityBuckets(fullCapacity), makeRaggedMatmulCapacityBuckets(fullCapacity))
            << "fullCapacity=" << fullCapacity;
    }
}

TEST(RaggedMatmulCapacityBuckets, RmsNormIncludes768AndQuarterOctaveBuckets) {
    EXPECT_EQ(makeRaggedRmsNormCapacityBuckets(1024),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 256, 512, 768, 1024}));
    EXPECT_EQ(makeRaggedRmsNormCapacityBuckets(2048),
              (std::vector<uint64_t>{8,
                                     16,
                                     32,
                                     64,
                                     128,
                                     256,
                                     512,
                                     768,
                                     1024,
                                     1280,
                                     1536,
                                     1792,
                                     2048}));
}

TEST(RaggedMatmulCapacityBuckets, RejectsZeroFullCapacity) {
    EXPECT_THROW((void)makeRaggedMatmulCapacityBuckets(0), std::invalid_argument);
    EXPECT_THROW((void)makeRaggedRmsNormCapacityBuckets(0), std::invalid_argument);
}

TEST(RaggedMatmulCapacityBuckets, SelectsSmallestCapacityThatContainsActiveRows) {
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(66);

    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(1, buckets), 8U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(8, buckets), 8U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(9, buckets), 16U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(16, buckets), 16U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(17, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(31, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(32, buckets), 32U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(33, buckets), 64U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(64, buckets), 64U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(65, buckets), 66U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(66, buckets), 66U);
}

TEST(RaggedMatmulCapacityBuckets, SelectionUsesFinerLargeCapacityClasses) {
    const std::vector<uint64_t> buckets = makeRaggedMatmulCapacityBuckets(4096);

    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(513, buckets), 768U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(769, buckets), 1024U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(1025, buckets), 1280U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(1281, buckets), 1536U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(1537, buckets), 1792U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(1793, buckets), 2048U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(2049, buckets), 2560U);
    EXPECT_EQ(chooseRaggedMatmulCapacityBucket(4096, buckets), 4096U);
}

TEST(RaggedMatmulCapacityBuckets, SelectionRejectsInvalidActiveCountsAndEmptyBuckets) {
    const std::vector<uint64_t> buckets{8, 16, 32, 66};
    const std::vector<uint64_t> noBuckets;

    EXPECT_THROW((void)chooseRaggedMatmulCapacityBucket(0, buckets), std::invalid_argument);
    EXPECT_THROW((void)chooseRaggedMatmulCapacityBucket(67, buckets), std::invalid_argument);
    EXPECT_THROW((void)chooseRaggedMatmulCapacityBucket(1, noBuckets), std::invalid_argument);
}
