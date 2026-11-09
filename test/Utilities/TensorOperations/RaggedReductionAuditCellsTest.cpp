// 1263: the host-side cell generator of thor_ragged_reduction_audit (benchmarks/RaggedReductionAuditCells.h): offsets
// are valid partitions in every distribution, distributions have the shapes they are named for, shapes hit their byte
// targets, 0.5 encodes exactly in every input format, and the closed-form verification catches a moved boundary.
#include "benchmarks/RaggedReductionAuditCells.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation::RaggedReductionAudit;

namespace {

void expectPartition(const std::vector<uint64_t>& offsets, uint64_t rows, uint64_t segments) {
    ASSERT_EQ(offsets.size(), segments + 1);
    EXPECT_EQ(offsets.front(), 0U);
    EXPECT_EQ(offsets.back(), rows);
    for (uint64_t i = 0; i < segments; ++i) ASSERT_LE(offsets[i], offsets[i + 1]) << "segment " << i;
}

}  // namespace

TEST(RaggedReductionAuditCells, EveryDistributionIsAValidPartition) {
    const uint64_t seed = 1263;
    for (Distribution d : DISTRIBUTIONS) {
        SCOPED_TRACE(distributionName(d));
        for (uint64_t trial = 0; trial < 40; ++trial) {
            const uint64_t h = mix(seed ^ trial);
            const uint64_t rows = 1 + h % 5000;
            const uint64_t segments = 1 + mix(h) % (trial % 2 ? rows : 64);
            expectPartition(makeOffsets(rows, segments, d, h), rows, segments);
        }
        expectPartition(makeOffsets(1, 1, d, seed), 1, 1);
        expectPartition(makeOffsets(7, 7, d, seed), 7, 7);
    }
}

TEST(RaggedReductionAuditCells, DistributionsHaveTheirShapes) {
    const uint64_t rows = 100000, segments = 1000, seed = 1263;

    const OffsetStats uniform = offsetStats(makeOffsets(rows, segments, Distribution::Uniform, seed));
    EXPECT_EQ(uniform.max_length, 100U);
    EXPECT_EQ(uniform.empty_segments, 0U);

    const OffsetStats giant = offsetStats(makeOffsets(rows, segments, Distribution::OneGiant, seed));
    EXPECT_EQ(giant.max_length, rows);
    EXPECT_EQ(giant.empty_segments, segments - 1);

    const OffsetStats half = offsetStats(makeOffsets(rows, segments, Distribution::HalfEmpty, seed));
    EXPECT_GT(half.empty_segments, 400U);
    EXPECT_LT(half.empty_segments, 600U);

    const OffsetStats random = offsetStats(makeOffsets(rows, segments, Distribution::Random, seed));
    EXPECT_LT(random.empty_segments, 20U);
    EXPECT_GT(random.max_length, 300U);  // exponential: the longest of 1000 is about 7.5x the mean
    EXPECT_LT(random.max_length, 2000U);

    // Pareto, shape 1.1: the longest segment is far above the mean, more than the exponential's.
    const OffsetStats pareto = offsetStats(makeOffsets(rows, segments, Distribution::Pareto, seed));
    EXPECT_GT(pareto.max_length, random.max_length);
    EXPECT_GT(pareto.max_length, 1000U);

    // Deterministic in the seed, and the seed matters.
    EXPECT_EQ(makeOffsets(rows, segments, Distribution::Pareto, seed),
              makeOffsets(rows, segments, Distribution::Pareto, seed));
    EXPECT_NE(makeOffsets(rows, segments, Distribution::Pareto, seed),
              makeOffsets(rows, segments, Distribution::Pareto, seed + 1));
}

TEST(RaggedReductionAuditCells, NamesRoundTrip) {
    for (Distribution d : DISTRIBUTIONS) EXPECT_EQ(distributionFromName(distributionName(d)), d);
    EXPECT_THROW(static_cast<void>(distributionFromName("zipf")), std::invalid_argument);
}

TEST(RaggedReductionAuditCells, ShapesMoveAboutTheirTargetBytes) {
    for (uint64_t shape = 0; shape < 200; ++shape) {
        const ShapeDraw draw = drawShape(1263, shape, 6, 14, 2, 2, 1ULL << 20, 256ULL << 20, 1ULL << 22);
        EXPECT_LT(draw.op, 6U);
        EXPECT_LT(draw.d, 14U);
        EXPECT_GE(draw.target_moved_bytes, 1ULL << 20);
        EXPECT_LE(draw.target_moved_bytes, 256ULL << 20);
        EXPECT_GE(draw.mean_length, 1.0);
        EXPECT_LE(draw.mean_length, double(1ULL << 22) * 1.000001);
        for (uint64_t d : {1ULL, 3ULL, 256ULL, 5000ULL}) {
            for (uint64_t value_bytes : {1ULL, 4ULL}) {
                const Shape s = shapeFor(draw.target_moved_bytes, draw.mean_length, d, value_bytes, 4, 4);
                ASSERT_GE(s.segments, 1U);
                ASSERT_LE(s.segments, s.rows);
                const double moved = double(movedBytes(s.rows, s.segments, d, value_bytes, 4, 4));
                // Within one row's bytes plus the rounding of the segment count (one segment's offset and outputs).
                const double slack = double(d * value_bytes) + 2.0 * double(4 + d * 4) + 8.0;
                EXPECT_NEAR(moved, double(draw.target_moved_bytes), slack) << "shape " << shape << " d " << d;
            }
        }
    }
    // All-length-1: the mean length is 1, so every segment has one row.
    const Shape ones = shapeFor(12ULL << 20, 1.0, 1, 4, 4, 4);
    EXPECT_EQ(ones.segments, ones.rows);
    EXPECT_EQ(movedBytes(ones.rows, ones.segments, 1, 4, 4, 4), 12 * ones.rows + 4);
}

TEST(RaggedReductionAuditCells, OneHalfEncodesAndDecodesExactlyInEveryFormat) {
    for (StorageFormat format : {StorageFormat::FP32, StorageFormat::FP16, StorageFormat::BF16, StorageFormat::FP8_E4M3,
                                 StorageFormat::FP8_E5M2}) {
        const std::vector<uint8_t> half = elementBytesOfOneHalf(format);
        ASSERT_EQ(half.size(), elementBytes(format));
        EXPECT_EQ(decodeElement(format, half.data()), 0.5f) << static_cast<int>(format);
    }
    const auto decode = [](StorageFormat format, std::vector<uint8_t> bytes) { return decodeElement(format, bytes.data()); };
    // Other values, specials and subnormals.
    EXPECT_EQ(decode(StorageFormat::FP32, {0x00, 0x00, 0x80, 0x3F}), 1.0f);
    EXPECT_EQ(decode(StorageFormat::FP32, {0x30, 0x30, 0x30, 0x30}), std::ldexp(float(0xB03030), -54));
    EXPECT_EQ(decode(StorageFormat::BF16, {0x30, 0x30}), std::ldexp(1.375f, -31));
    EXPECT_EQ(decode(StorageFormat::FP16, {0x30, 0x30}), 0.130859375f);  // 2^-3 * (1 + 48 / 1024)
    EXPECT_EQ(decode(StorageFormat::FP16, {0x00, 0x00}), 0.0f);
    EXPECT_EQ(decode(StorageFormat::FP16, {0x01, 0x00}), std::ldexp(1.0f, -24));
    EXPECT_EQ(decode(StorageFormat::FP16, {0x00, 0xFC}), -std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isnan(decode(StorageFormat::FP8_E4M3, {0x7F})));
    EXPECT_EQ(decode(StorageFormat::FP8_E4M3, {0x7E}), 448.0f);
    EXPECT_EQ(decode(StorageFormat::FP8_E4M3, {0x01}), std::ldexp(1.0f, -9));
    EXPECT_EQ(decode(StorageFormat::FP8_E5M2, {0x7C}), std::numeric_limits<float>::infinity());
    EXPECT_EQ(decode(StorageFormat::FP8_E5M2, {0x01}), std::ldexp(1.0f, -16));
    EXPECT_EQ(decode(StorageFormat::FP8_E5M2, {0xC0}), -2.0f);
}

TEST(RaggedReductionAuditCells, VerificationAcceptsTheClosedFormAndCatchesAMovedBoundary) {
    const std::vector<uint64_t> offsets{0, 3, 3, 7, 8};
    const uint64_t d = 2;
    const float v = 0.5f, inf = std::numeric_limits<float>::infinity();

    const std::vector<float> sums{1.5f, 1.5f, 0.0f, 0.0f, 2.0f, 2.0f, 0.5f, 0.5f};
    EXPECT_TRUE(verifyValueOutput(Op::Sum, offsets, d, v, sums, 1e-6).ok());
    EXPECT_TRUE(verifyValueOutput(Op::Mean, offsets, d, v, {v, v, 0, 0, v, v, v, v}, 1e-6).ok());
    EXPECT_TRUE(verifyValueOutput(Op::Min, offsets, d, v, {v, v, inf, inf, v, v, v, v}, 0).ok());
    EXPECT_TRUE(verifyValueOutput(Op::Max, offsets, d, v, {v, v, -inf, -inf, v, v, v, v}, 0).ok());

    // A boundary one row late: segment 0 has 4 rows, segment 2 has 3.
    const Verification moved =
        verifyValueOutput(Op::Sum, offsets, d, v, {2.0f, 2.0f, 0.0f, 0.0f, 1.5f, 1.5f, 0.5f, 0.5f}, 1e-3);
    EXPECT_EQ(moved.mismatches, 4U);
    EXPECT_NE(moved.first.find("segment 0"), std::string::npos);
    // An empty segment must be exactly empty.
    EXPECT_FALSE(verifyValueOutput(Op::Min, offsets, d, v, {v, v, v, inf, v, v, v, v}, 0).ok());
    EXPECT_FALSE(verifyValueOutput(Op::Sum, offsets, d, v, {1.5f, 1.5f}, 1e-6).ok());  // wrong size
    // Below EXACT_SUM_ROWS a sum must be exact whatever the tolerance; above it the tolerance applies.
    EXPECT_FALSE(verifyValueOutput(Op::Sum, offsets, d, v, {1.5f, 1.5000001f, 0, 0, 2, 2, 0.5f, 0.5f}, 1e-3).ok());
    const std::vector<uint64_t> long_segment{0, EXACT_SUM_ROWS + 2};
    const float long_sum = 0.5f * float(EXACT_SUM_ROWS + 2);
    EXPECT_TRUE(verifyValueOutput(Op::Sum, long_segment, 1, v, {long_sum * 1.0005f}, 1e-3).ok());
    EXPECT_FALSE(verifyValueOutput(Op::Sum, long_segment, 1, v, {0.5f * long_sum}, 1e-3).ok());  // far off
    // 1267: the census's 29.6M-row pareto segment: FP32 rounds its sum (14806426.5) to 14806426, within the tolerance.
    const std::vector<uint64_t> census_segment{0, 29612853};
    EXPECT_TRUE(verifyValueOutput(Op::Sum, census_segment, 1, v, {14806426.0f}, 1e-3).ok());

    const uint64_t sentinel = std::numeric_limits<uint32_t>::max();
    EXPECT_TRUE(verifyArgOutput(offsets, d, {0, 1, sentinel, sentinel, 6, 7, 14, 15}, sentinel).ok());
    const Verification late = verifyArgOutput(offsets, d, {0, 1, sentinel, sentinel, 8, 9, 14, 15}, sentinel);
    EXPECT_EQ(late.mismatches, 2U);
    EXPECT_NE(late.first.find("segment 2"), std::string::npos);
}
