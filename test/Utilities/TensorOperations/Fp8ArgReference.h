#pragma once

// Host reference for FP8 ARG (ARG-A1 tests and benchmarks/Fp8ArgKeyMicrobenchmark.cu): independent FP8 decoding and
// the FP32 candidate rules of ArgMinimumCandidateFp32 / ArgMaximumCandidateFp32 (CubArgReductionOperation.cuh):
// NaNs propagate, the lowest index wins ties, including NaN-NaN and -0/+0 ties.

#include <cmath>
#include <cstdint>
#include <limits>

namespace ThorImplementation::Fp8ArgReference {

inline float decodeE4M3(uint32_t byte) {
    const bool negative = (byte & 0x80u) != 0;
    const uint32_t exponent = (byte >> 3) & 0xFu;
    const uint32_t mantissa = byte & 0x7u;
    float magnitude;
    if (exponent == 0xFu && mantissa == 0x7u) return std::numeric_limits<float>::quiet_NaN();
    if (exponent == 0) magnitude = std::ldexp(static_cast<float>(mantissa), -9);
    else magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, static_cast<int>(exponent) - 7);
    return negative ? -magnitude : magnitude;
}

inline float decodeE5M2(uint32_t byte) {
    const bool negative = (byte & 0x80u) != 0;
    const uint32_t exponent = (byte >> 2) & 0x1Fu;
    const uint32_t mantissa = byte & 0x3u;
    float magnitude;
    if (exponent == 0x1Fu) {
        if (mantissa != 0) return std::numeric_limits<float>::quiet_NaN();
        magnitude = std::numeric_limits<float>::infinity();
    } else if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -16);
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 4.0f, static_cast<int>(exponent) - 15);
    }
    return negative ? -magnitude : magnitude;
}

inline float decode(uint32_t byte, bool e5m2) { return e5m2 ? decodeE5M2(byte) : decodeE4M3(byte); }

struct Candidate {
    uint64_t index;
    float value;
};

// Mirrors ArgMinimumCandidateFp32 (maximum = false) and ArgMaximumCandidateFp32 (maximum = true).
inline Candidate combine(const Candidate& lhs, const Candidate& rhs, bool maximum) {
    const bool lhs_nan = std::isnan(lhs.value);
    const bool rhs_nan = std::isnan(rhs.value);
    if (lhs_nan || rhs_nan) {
        if (lhs_nan && rhs_nan) return lhs.index <= rhs.index ? lhs : rhs;
        return lhs_nan ? lhs : rhs;
    }
    if (maximum ? lhs.value > rhs.value : lhs.value < rhs.value) return lhs;
    if (maximum ? rhs.value > lhs.value : rhs.value < lhs.value) return rhs;
    return lhs.index <= rhs.index ? lhs : rhs;
}

// Same float, including the sign of zero; any NaN matches any NaN.
inline bool sameValue(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return a == b && std::signbit(a) == std::signbit(b);
}

}  // namespace ThorImplementation::Fp8ArgReference
