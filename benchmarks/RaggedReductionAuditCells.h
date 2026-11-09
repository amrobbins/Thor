#pragma once
// 1263: the cells of thor_ragged_reduction_audit (benchmarks/RaggedReductionAudit.cpp), host only and free of CUDA so
// test/Utilities/TensorOperations/RaggedReductionAuditCellsTest.cpp can check them without a GPU.
//
// A shape is (op, D, offset dtype, capacity ratio, active rows, segments). Every shape runs under every distribution of
// segment lengths, so a report can compare distributions at a fixed (rows, segments, D, dtype, op): the distribution
// spread that plans/Thor_Ragged_Reduction_Plan_2026-11-06.md gates on.
//
// Distributions (all deterministic in the seed, platform independent: splitmix64 and IEEE double only):
//   uniform     equal lengths (rows / segments, the remainder spread one row each over the first segments)
//   random      exponential weights: lengths vary around the mean, few empty segments unless the mean is below one
//   pareto      Pareto weights, shape 1.1: a heavy tail, where a few segments hold most rows
//   half_empty  about half the segments empty (each segment by a coin), the rest equal
//   one_giant   one segment holds every row, every other segment is empty
// Weights become offsets by cumulative rounding (offsets[i] = round(rows * W_i / W_B)), which is monotone and ends at
// exactly `rows`. All-length-1 and few-segment cases come from the shape's mean length, not from a distribution.
//
// Verification: the audit's input ring holds 0.5 in every element (in the input dtype), so each segment's output is
// known in closed form and exact: SUM len * 0.5, MEAN, MIN and MAX 0.5, ARG the segment's first packed index (the lowest
// tied index), and the empty values (0, 0, +inf, -inf, the sentinel). That checks every segment boundary of every cell;
// it does not check arithmetic on varied data (the unit tests do).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ThorImplementation::RaggedReductionAudit {

inline uint64_t mix(uint64_t x) {  // splitmix64 finalizer, as thor_dense_production_audit
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
inline double unit(uint64_t h) { return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0); }  // [0, 1)
inline double openUnit(uint64_t h) {  // (0, 1]
    return static_cast<double>((h >> 11) + 1) * (1.0 / 9007199254740992.0);
}
inline uint64_t logUniform(uint64_t h, uint64_t low, uint64_t high) {
    if (high <= low) return low;
    const double l = std::log(double(low)), u = std::log(double(high));
    return std::clamp<uint64_t>(static_cast<uint64_t>(std::llround(std::exp(l + (u - l) * unit(h)))), low, high);
}

enum class Distribution : uint8_t { Uniform, Random, Pareto, HalfEmpty, OneGiant };
inline constexpr Distribution DISTRIBUTIONS[] = {Distribution::Uniform, Distribution::Random, Distribution::Pareto,
                                                 Distribution::HalfEmpty, Distribution::OneGiant};

inline const char* distributionName(Distribution d) {
    switch (d) {
        case Distribution::Uniform: return "uniform";
        case Distribution::Random: return "random";
        case Distribution::Pareto: return "pareto";
        case Distribution::HalfEmpty: return "half_empty";
        case Distribution::OneGiant: return "one_giant";
    }
    return "unknown";
}

inline Distribution distributionFromName(std::string_view name) {
    for (Distribution d : DISTRIBUTIONS)
        if (name == distributionName(d)) return d;
    throw std::invalid_argument("--distributions takes uniform, random, pareto, half_empty and one_giant");
}

// The segment weight of segment i (>= 0). HalfEmpty and OneGiant need the index of a segment that is never empty.
inline double segmentWeight(Distribution d, uint64_t seed, uint64_t i, uint64_t keep) {
    const uint64_t h = mix(seed ^ mix(i + 1));
    switch (d) {
        case Distribution::Uniform: return 1.0;
        case Distribution::Random: return -std::log(openUnit(h));
        case Distribution::Pareto: return std::pow(openUnit(h), -1.0 / 1.1);
        case Distribution::HalfEmpty: return (i == keep || (h & 1) != 0) ? 1.0 : 0.0;
        case Distribution::OneGiant: return i == keep ? 1.0 : 0.0;
    }
    return 1.0;
}

// Offsets [segments + 1] partitioning `rows` rows among `segments` segments: offsets[0] = 0, offsets[segments] = rows,
// nondecreasing.
inline std::vector<uint64_t> makeOffsets(uint64_t rows, uint64_t segments, Distribution d, uint64_t seed) {
    if (segments == 0) throw std::invalid_argument("makeOffsets needs at least one segment");
    std::vector<uint64_t> offsets(segments + 1, 0);
    if (d == Distribution::Uniform) {
        for (uint64_t i = 0; i <= segments; ++i)
            offsets[i] = static_cast<uint64_t>(static_cast<unsigned __int128>(rows) * i / segments);
        return offsets;
    }
    const uint64_t keep = mix(seed ^ 0x6B65657020736567ULL) % segments;
    long double total = 0;
    for (uint64_t i = 0; i < segments; ++i) total += segmentWeight(d, seed, i, keep);
    long double prefix = 0;
    for (uint64_t i = 0; i < segments; ++i) {
        prefix += segmentWeight(d, seed, i, keep);
        const long double exact = static_cast<long double>(rows) * prefix / total;
        const uint64_t rounded = std::min<uint64_t>(rows, static_cast<uint64_t>(std::floor(exact + 0.5L)));
        offsets[i + 1] = std::max(offsets[i], rounded);
    }
    offsets[segments] = rows;
    return offsets;
}

struct OffsetStats {
    uint64_t max_length = 0;
    uint64_t empty_segments = 0;
};

inline OffsetStats offsetStats(const std::vector<uint64_t>& offsets) {
    OffsetStats s;
    for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        const uint64_t length = offsets[i + 1] - offsets[i];
        s.max_length = std::max(s.max_length, length);
        s.empty_segments += length == 0;
    }
    return s;
}

// Bytes a ragged reduction must move: the active input, the offsets and the outputs.
inline uint64_t movedBytes(uint64_t rows, uint64_t segments, uint64_t d, uint64_t value_bytes, uint64_t offset_bytes,
                           uint64_t output_bytes) {
    return rows * d * value_bytes + (segments + 1) * offset_bytes + segments * d * output_bytes;
}

struct ShapeDraw {
    size_t op = 0, d = 0, offset_dtype = 0, capacity = 0;  // indices into the settings' lists
    uint64_t target_moved_bytes = 0;
    double mean_length = 1.0;
};

// The draw of shape `shape_id`: list indices, a log-uniform moved-bytes target in [min_bytes, max_bytes], and a
// log-uniform mean segment length in [1, max_mean_length].
inline ShapeDraw drawShape(uint64_t seed, uint64_t shape_id, size_t ops, size_t ds, size_t offset_dtypes,
                           size_t capacities, uint64_t min_bytes, uint64_t max_bytes, uint64_t max_mean_length) {
    const uint64_t h = mix(seed ^ mix(shape_id * 0x100 + 0x52));
    ShapeDraw s;
    s.op = mix(h ^ 1) % ops;
    s.d = mix(h ^ 2) % ds;
    s.offset_dtype = mix(h ^ 3) % offset_dtypes;
    s.capacity = mix(h ^ 4) % capacities;
    s.target_moved_bytes = logUniform(mix(h ^ 5), min_bytes, max_bytes);
    s.mean_length = std::exp(std::log(double(std::max<uint64_t>(1, max_mean_length))) * unit(mix(h ^ 6)));
    return s;
}

struct Shape {
    uint64_t rows = 1;
    uint64_t segments = 1;
};

// Active rows and segments that move about `target_moved_bytes` at `mean_length` rows per segment.
inline Shape shapeFor(uint64_t target_moved_bytes, double mean_length, uint64_t d, uint64_t value_bytes,
                      uint64_t offset_bytes, uint64_t output_bytes) {
    const double per_row = double(d * value_bytes) + double(offset_bytes + d * output_bytes) / mean_length;
    Shape s;
    s.rows = std::max<uint64_t>(1, static_cast<uint64_t>(double(target_moved_bytes) / per_row));
    s.segments = std::clamp<uint64_t>(static_cast<uint64_t>(std::llround(double(s.rows) / mean_length)), 1, s.rows);
    return s;
}

// The storage formats of the audit's dtypes. The audit fills its input ring with one element, 0.5 in the input dtype
// (elementBytesOfOneHalf), so every partial sum of a segment shorter than 2^25 rows is exact in FP32 in any order.
enum class StorageFormat : uint8_t { FP32, FP16, BF16, FP8_E4M3, FP8_E5M2 };

inline uint64_t elementBytes(StorageFormat format) {
    switch (format) {
        case StorageFormat::FP32: return 4;
        case StorageFormat::FP16:
        case StorageFormat::BF16: return 2;
        default: return 1;
    }
}

// The little-endian bytes of 0.5 in `format`.
inline std::vector<uint8_t> elementBytesOfOneHalf(StorageFormat format) {
    switch (format) {
        case StorageFormat::FP32: return {0x00, 0x00, 0x00, 0x3F};  // 0x3F000000
        case StorageFormat::FP16: return {0x00, 0x38};              // 0x3800
        case StorageFormat::BF16: return {0x00, 0x3F};              // 0x3F00
        case StorageFormat::FP8_E4M3: return {0x30};                // 0.0110.000: 2^(6 - 7)
        case StorageFormat::FP8_E5M2: return {0x38};                // 0.01110.00: 2^(14 - 15)
    }
    return {};
}

// One element of `format` from its little-endian bytes, widened to float.
inline float decodeElement(StorageFormat format, const uint8_t* bytes) {
    switch (format) {
        case StorageFormat::FP32: {
            uint32_t bits = 0;
            for (int i = 3; i >= 0; --i) bits = (bits << 8) | bytes[i];
            float f;
            std::memcpy(&f, &bits, sizeof f);
            return f;
        }
        case StorageFormat::BF16: {
            const uint32_t bits = (uint32_t(bytes[1]) << 24) | (uint32_t(bytes[0]) << 16);
            float f;
            std::memcpy(&f, &bits, sizeof f);
            return f;
        }
        case StorageFormat::FP16: {
            const uint32_t h = (uint32_t(bytes[1]) << 8) | bytes[0];
            const int sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1F, mantissa = h & 0x3FF;
            float magnitude;
            if (exponent == 0) magnitude = std::ldexp(float(mantissa), -24);
            else if (exponent == 31) magnitude = mantissa ? std::numeric_limits<float>::quiet_NaN()
                                                          : std::numeric_limits<float>::infinity();
            else magnitude = std::ldexp(float(1024 + mantissa), exponent - 25);
            return sign ? -magnitude : magnitude;
        }
        case StorageFormat::FP8_E4M3: {  // bias 7; S.1111.111 is NaN, no infinities
            const uint8_t byte = bytes[0];
            const int sign = byte >> 7, exponent = (byte >> 3) & 0xF, mantissa = byte & 7;
            float magnitude;
            if (exponent == 15 && mantissa == 7) magnitude = std::numeric_limits<float>::quiet_NaN();
            else if (exponent == 0) magnitude = std::ldexp(float(mantissa), -9);
            else magnitude = std::ldexp(float(8 + mantissa), exponent - 10);
            return sign ? -magnitude : magnitude;
        }
        case StorageFormat::FP8_E5M2: {  // bias 15, IEEE-like specials
            const uint8_t byte = bytes[0];
            const int sign = byte >> 7, exponent = (byte >> 2) & 0x1F, mantissa = byte & 3;
            float magnitude;
            if (exponent == 31) magnitude = mantissa ? std::numeric_limits<float>::quiet_NaN()
                                                     : std::numeric_limits<float>::infinity();
            else if (exponent == 0) magnitude = std::ldexp(float(mantissa), -16);
            else magnitude = std::ldexp(float(4 + mantissa), exponent - 17);
            return sign ? -magnitude : magnitude;
        }
    }
    return 0.0f;
}

enum class Op : uint8_t { Sum, Mean, Min, Max, ArgMin, ArgMax };

struct Verification {
    uint64_t checked = 0;
    uint64_t mismatches = 0;
    std::string first;  // the first mismatch, for the failure line
    [[nodiscard]] bool ok() const { return mismatches == 0; }
};

// Segments up to this many rows of 0.5 have exact FP32 sums in any summation order: every partial sum is a multiple of
// 0.5 of at most 2^23, which FP32's 24-bit significand holds. (1267: was 2^25, so a 29.6M-row segment, sum
// 14806426.5, was required to be exact; FP32 rounds it to 14806426.)
inline constexpr uint64_t EXACT_SUM_ROWS = uint64_t{1} << 24;

// VALUE output [segments, d] of an input whose every element is `value` (0.5 in the audit). Every output must be exact,
// except SUM and MEAN of a segment longer than EXACT_SUM_ROWS, which accept a relative error of `relative_tolerance`
// (a backend that sums such a segment serially in FP32 stagnates far beyond any tolerance, which is worth reporting).
inline Verification verifyValueOutput(Op op, const std::vector<uint64_t>& offsets, uint64_t d, float value,
                                      const std::vector<float>& got, double relative_tolerance) {
    Verification v;
    const uint64_t segments = offsets.size() - 1;
    if (got.size() != segments * d) {
        v.mismatches = 1;
        v.first = "output has " + std::to_string(got.size()) + " values, expected " + std::to_string(segments * d);
        return v;
    }
    for (uint64_t s = 0; s < segments; ++s) {
        const uint64_t length = offsets[s + 1] - offsets[s];
        double expected = 0;
        bool exact = true;
        switch (op) {
            case Op::Sum:
                expected = double(length) * double(value);
                exact = length <= EXACT_SUM_ROWS;
                break;
            case Op::Mean:
                expected = length == 0 ? 0.0 : double(value);
                exact = length <= EXACT_SUM_ROWS;
                break;
            case Op::Min:
                expected = length == 0 ? double(std::numeric_limits<float>::infinity()) : double(value);
                break;
            case Op::Max:
                expected = length == 0 ? -double(std::numeric_limits<float>::infinity()) : double(value);
                break;
            default:
                throw std::invalid_argument("verifyValueOutput takes a VALUE op");
        }
        for (uint64_t c = 0; c < d; ++c) {
            const double actual = got[s * d + c];
            ++v.checked;
            const bool match = exact ? actual == expected
                                     : std::abs(actual - expected) <= relative_tolerance * std::abs(expected);
            if (!match) {
                if (v.mismatches++ == 0) {
                    std::ostringstream text;
                    text.precision(9);
                    text << "segment " << s << " component " << c << " length " << length << ": got " << actual
                         << ", expected " << expected;
                    v.first = text.str();
                }
            }
        }
    }
    return v;
}

// ARG index output [segments, d] of a constant input: every element ties, so the winner is the segment's first packed
// index begin * d + component; an empty segment holds `sentinel`.
inline Verification verifyArgOutput(const std::vector<uint64_t>& offsets, uint64_t d, const std::vector<uint64_t>& got,
                                    uint64_t sentinel) {
    Verification v;
    const uint64_t segments = offsets.size() - 1;
    if (got.size() != segments * d) {
        v.mismatches = 1;
        v.first = "output has " + std::to_string(got.size()) + " indices, expected " + std::to_string(segments * d);
        return v;
    }
    for (uint64_t s = 0; s < segments; ++s) {
        const bool empty = offsets[s + 1] == offsets[s];
        for (uint64_t c = 0; c < d; ++c) {
            const uint64_t expected = empty ? sentinel : offsets[s] * d + c;
            ++v.checked;
            if (got[s * d + c] != expected && v.mismatches++ == 0) {
                v.first = "segment " + std::to_string(s) + " component " + std::to_string(c) + ": got " +
                          std::to_string(got[s * d + c]) + ", expected " + std::to_string(expected);
            }
        }
    }
    return v;
}

}  // namespace ThorImplementation::RaggedReductionAudit
