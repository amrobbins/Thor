// ARG-A1: exhaustive checks of the FP8 ARG key helpers (Utilities/TensorOperations/Cub/Fp8ArgKey.h) against the FP32
// ARG candidate rules, on the host and on the GPU.

#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"
#include "test/Utilities/TensorOperations/Fp8ArgKeyDevice.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using namespace ThorImplementation::CubReductionInternal;
namespace Reference = ThorImplementation::Fp8ArgReference;
namespace Device = ThorImplementation::Fp8ArgKeyDevice;

namespace {

struct Mode {
    bool maximum;
    bool e5m2;
};
constexpr std::array<Mode, 4> MODES{{{false, false}, {true, false}, {false, true}, {true, true}}};

// Key and negative-zero bit of one byte, placed at `byte` of an otherwise zero word.
uint32_t candidateOf(uint32_t value, uint32_t byte, uint32_t row, Fp8ArgFold fold) {
    return fp8ArgLaneCandidate(fp8ArgKeys4(value << (8u * byte), fold), byte, row);
}

// Specials-heavy FP8 bytes: NaNs, +-0, +-inf (E5M2), extremes and subnormals, then uniform bytes.
uint8_t sampleByte(std::mt19937& rng) {
    static constexpr std::array<uint8_t, 14> SPECIALS{
        {0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x7C, 0xFC, 0x7D, 0xFD, 0x01, 0x81, 0x08, 0x88}};
    const uint32_t pick = rng() % 4u;
    return pick == 0 ? SPECIALS[rng() % SPECIALS.size()] : static_cast<uint8_t>(rng());
}

Reference::Candidate referenceColumn(const std::vector<uint8_t>& input, uint32_t rows, uint32_t cols, uint32_t col,
                                     Mode mode) {
    Reference::Candidate best{0, Reference::decode(input[col], mode.e5m2)};
    for (uint32_t r = 1; r < rows; ++r)
        best = Reference::combine(best, {r, Reference::decode(input[static_cast<size_t>(r) * cols + col], mode.e5m2)},
                                  mode.maximum);
    return best;
}

}  // namespace

// Every ordered pair of encodings, both formats, both operations, at every byte position and in both row orders:
// the lane-candidate minimum picks the same index as the FP32 reference and rebuilds the same value (sign of zero
// included).
TEST(Fp8ArgKey, EveryPairMatchesFp32ArgReference) {
    for (const Mode mode : MODES) {
        const Fp8ArgFold fold = fp8ArgFold(mode.maximum, mode.e5m2);
        for (uint32_t a = 0; a < 256; ++a) {
            for (uint32_t b = 0; b < 256; ++b) {
                for (const auto& [row_a, row_b] : {std::pair<uint32_t, uint32_t>{3, 7}, {7, 3}}) {
                    const uint32_t byte = (a ^ b ^ row_a) & 3u;
                    const uint32_t winner = fp8ArgMin(candidateOf(a, byte, row_a, fold), candidateOf(b, byte, row_b, fold));
                    const Reference::Candidate expected = Reference::combine(
                        {row_a, Reference::decode(a, mode.e5m2)}, {row_b, Reference::decode(b, mode.e5m2)}, mode.maximum);
                    ASSERT_EQ(expected.index, (winner >> 1) & (FP8_ARG_MAX_LANE_ROWS - 1u))
                        << "a=" << a << " b=" << b << " max=" << mode.maximum << " e5m2=" << mode.e5m2;
                    const uint32_t value = fp8ArgValueByte(winner >> 24, winner & 1u, fold);
                    ASSERT_TRUE(Reference::sameValue(expected.value, Reference::decode(value, mode.e5m2)))
                        << "a=" << a << " b=" << b << " rebuilt=" << value;
                }
            }
        }
    }
}

TEST(Fp8ArgKey, ValueByteRoundTripsEveryNonNanEncoding) {
    for (const Mode mode : MODES) {
        const Fp8ArgFold fold = fp8ArgFold(mode.maximum, mode.e5m2);
        for (uint32_t value = 0; value < 256; ++value) {
            const uint32_t c = candidateOf(value, value & 3u, 0, fold);
            const uint32_t rebuilt = fp8ArgValueByte(c >> 24, c & 1u, fold);
            if (std::isnan(Reference::decode(value, mode.e5m2))) {
                EXPECT_EQ(FP8_ARG_NAN_KEY, c >> 24) << value;
                EXPECT_TRUE(std::isnan(Reference::decode(rebuilt, mode.e5m2))) << value;
            } else {
                EXPECT_EQ(value, rebuilt) << value;
                EXPECT_NE(FP8_ARG_NAN_KEY, c >> 24) << value;
                EXPECT_NE(0xFFu, c >> 24) << value;  // key 0xFF never occurs: the lane identity is strict
            }
        }
    }
}

// The packed four-byte transform equals the per-byte transform at every position, whatever the other bytes hold
// (no carry or borrow crosses a byte).
TEST(Fp8ArgKey, PackedKeysMatchBytewise) {
    std::mt19937 rng(1104);
    for (const Mode mode : MODES) {
        const Fp8ArgFold fold = fp8ArgFold(mode.maximum, mode.e5m2);
        for (uint32_t trial = 0; trial < 4096; ++trial) {
            std::array<uint32_t, 4> bytes{};
            for (uint32_t& byte : bytes) byte = sampleByte(rng);
            bytes[trial & 3u] = trial >> 4;  // every value at every position
            const uint32_t word = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
            const Fp8ArgKeys4 packed = fp8ArgKeys4(word, fold);
            for (uint32_t i = 0; i < 4; ++i) {
                const Fp8ArgKeys4 single = fp8ArgKeys4(bytes[i] << (8u * i), fold);
                ASSERT_EQ((single.keys >> (8u * i)) & 0xFFu, (packed.keys >> (8u * i)) & 0xFFu) << word;
                ASSERT_EQ((single.negative_zero >> (8u * i)) & 0xFFu, (packed.negative_zero >> (8u * i)) & 0xFFu)
                    << word;
                ASSERT_EQ(fp8ArgLaneCandidate(single, i, 5), fp8ArgLaneCandidate(packed, i, 5)) << word;
            }
        }
    }
}

// 1124: four rows of one component in one word (byte i = row first_row + i * stride) reduce to the same candidate as
// the per-byte lane candidates, ties (lowest row) and the sign of zero included.
TEST(Fp8ArgKey, RowsCandidate4MatchesBytewise) {
    std::mt19937 rng(1124);
    static constexpr std::array<uint8_t, 4> TIES{{0x38, 0xB8, 0x00, 0x80}};
    for (const Mode mode : MODES) {
        const Fp8ArgFold fold = fp8ArgFold(mode.maximum, mode.e5m2);
        for (uint32_t trial = 0; trial < 65536; ++trial) {
            std::array<uint32_t, 4> bytes{};
            for (uint32_t& byte : bytes) byte = (trial & 1u) ? TIES[rng() % TIES.size()] : sampleByte(rng);
            const uint32_t word = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
            const uint32_t stride = 1u + (rng() % 300u);
            const uint32_t first = rng() % (FP8_ARG_MAX_LANE_ROWS - 3u * stride);
            const Fp8ArgKeys4 keys = fp8ArgKeys4(word, fold);
            uint32_t expected = FP8_ARG_LANE_IDENTITY;
            for (uint32_t i = 0; i < 4; ++i)
                expected = fp8ArgMin(expected, fp8ArgLaneCandidate(keys, i, first + i * stride));
            ASSERT_EQ(expected, fp8ArgRowsCandidate4(keys, first, stride)) << word << " stride=" << stride;
        }
    }
}

TEST(Fp8ArgKey, IdentityAndWideningLimits) {
    const Fp8ArgFold fold = fp8ArgFold(false, false);
    const uint32_t last_row = FP8_ARG_MAX_LANE_ROWS - 1u;
    // The largest real lane candidate (largest key, last lane row, -0 bit) is still below the identity.
    EXPECT_LT(candidateOf(0x7E, 0, last_row, fold) | 1u, FP8_ARG_LANE_IDENTITY);
    EXPECT_EQ(FP8_ARG_WIDE_IDENTITY, fp8ArgWiden(FP8_ARG_LANE_IDENTITY, 12345));
    const uint64_t wide = fp8ArgWiden(candidateOf(0x80, 2, last_row, fold), 0xFFFFFFFFu - last_row);
    EXPECT_EQ(0xFFFFFFFFu, fp8ArgWideIndex(wide));
    EXPECT_EQ(0x80u, fp8ArgWideKey(wide));
    EXPECT_EQ(1u, fp8ArgWideNegativeZero(wide));
    EXPECT_LT(wide, FP8_ARG_WIDE_IDENTITY);
    // Widened order: key first, then index (a later lane can still win on key).
    EXPECT_LT(fp8ArgWiden(candidateOf(0x01, 0, 9, fold), 1000), fp8ArgWiden(candidateOf(0x02, 0, 0, fold), 0));
    EXPECT_LT(fp8ArgWiden(candidateOf(0x02, 0, 9, fold), 0), fp8ArgWiden(candidateOf(0x02, 0, 0, fold), 1000));
    // 1132: 55-bit widened indices. Row bases past 2^32 keep their full index, and the largest real candidate
    // (largest key, largest index, -0 bit) stays below the identity.
    const uint64_t past32 = fp8ArgWiden(candidateOf(0x80, 1, 5, fold), (uint64_t{1} << 32) + 7);
    EXPECT_EQ((uint64_t{1} << 32) + 12, fp8ArgWideIndex(past32));
    EXPECT_EQ(0x80u, fp8ArgWideKey(past32));
    EXPECT_EQ(1u, fp8ArgWideNegativeZero(past32));
    const uint64_t top = fp8ArgWiden(candidateOf(0x7E, 0, last_row, fold) | 1u, FP8_ARG_MAX_WIDE_INDEX - last_row);
    EXPECT_EQ(FP8_ARG_MAX_WIDE_INDEX, fp8ArgWideIndex(top));
    EXPECT_LT(top, FP8_ARG_WIDE_IDENTITY);
    // Key still dominates a far larger index; index then orders equal keys.
    EXPECT_LT(fp8ArgWiden(candidateOf(0x01, 0, 0, fold), FP8_ARG_MAX_WIDE_INDEX - last_row),
              fp8ArgWiden(candidateOf(0x02, 0, 0, fold), 0));
    EXPECT_LT(fp8ArgWiden(candidateOf(0x02, 0, 0, fold), (uint64_t{1} << 40)),
              fp8ArgWiden(candidateOf(0x02, 0, 0, fold), (uint64_t{1} << 40) + 1));
}

// Columns split into lanes with row bases, widened and merged: host and GPU agree with the sequential reference.
TEST(Fp8ArgKey, LaneSplitColumnsMatchReferenceOnHostAndDevice) {
    std::mt19937 rng(42);
    constexpr uint32_t rows = 777;
    constexpr uint32_t cols = 515;
    std::vector<uint8_t> input(static_cast<size_t>(rows) * cols);
    for (uint8_t& byte : input) byte = sampleByte(rng);
    // Long runs of one value force index tie-breaks across lane boundaries.
    for (uint32_t col = 0; col < 64; ++col)
        for (uint32_t r = 0; r < rows; ++r) input[static_cast<size_t>(r) * cols + col] = col < 32 ? 0x00 : 0x80;
    for (const Mode mode : MODES) {
        const Fp8ArgFold fold = fp8ArgFold(mode.maximum, mode.e5m2);
        for (const uint32_t lanes : {1u, 2u, 5u, 64u}) {
            std::vector<uint32_t> device_index, device_value;
            Device::columns(input, rows, cols, lanes, fold, device_index, device_value);
            const uint32_t chunk = (rows + lanes - 1u) / lanes;
            for (uint32_t col = 0; col < cols; ++col) {
                uint64_t best = FP8_ARG_WIDE_IDENTITY;
                for (uint32_t base = 0; base < rows; base += chunk) {
                    uint32_t lane = FP8_ARG_LANE_IDENTITY;
                    for (uint32_t r = base; r < std::min(rows, base + chunk); ++r)
                        lane = fp8ArgMin(lane, candidateOf(input[static_cast<size_t>(r) * cols + col], col & 3u,
                                                           r - base, fold));
                    best = fp8ArgMin(best, fp8ArgWiden(lane, base));
                }
                const Reference::Candidate expected = referenceColumn(input, rows, cols, col, mode);
                const uint32_t value = fp8ArgValueByte(fp8ArgWideKey(best), fp8ArgWideNegativeZero(best), fold);
                ASSERT_EQ(expected.index, fp8ArgWideIndex(best)) << "col=" << col << " lanes=" << lanes;
                ASSERT_TRUE(Reference::sameValue(expected.value, Reference::decode(value, mode.e5m2))) << col;
                ASSERT_EQ(fp8ArgWideIndex(best), device_index[col]) << "device col=" << col << " lanes=" << lanes;
                ASSERT_EQ(value, device_value[col]) << "device col=" << col << " lanes=" << lanes;
            }
        }
    }
}

// The GPU computes the same packed keys as the host for every byte at every position.
TEST(Fp8ArgKey, DeviceKeysMatchHost) {
    std::mt19937 rng(7);
    std::vector<uint32_t> words;
    for (uint32_t i = 0; i < 1024; ++i) words.push_back((rng() & ~(0xFFu << (8u * (i & 3u)))) | ((i >> 2) << (8u * (i & 3u))));
    for (uint32_t i = 0; i < 4096; ++i) words.push_back(rng());
    for (const Mode mode : MODES) {
        const Fp8ArgFold fold = fp8ArgFold(mode.maximum, mode.e5m2);
        std::vector<uint32_t> keys, negative_zero;
        Device::keys(words, fold, keys, negative_zero);
        for (size_t i = 0; i < words.size(); ++i) {
            const Fp8ArgKeys4 host = fp8ArgKeys4(words[i], fold);
            ASSERT_EQ(host.keys, keys[i]) << words[i];
            ASSERT_EQ(host.negative_zero, negative_zero[i]) << words[i];
        }
    }
}
