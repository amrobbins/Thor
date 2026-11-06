#pragma once

// 1237: FP16 / BF16 ARG keys, the 2-byte counterpart of Fp8ArgKey.h (plans/Thor_FP16_BF16_Dense_Reduction_Plan,
// decision 2): argmin / argmax on raw 16-bit patterns without decoding.
//
// Every 16-bit element maps to a 16-bit key whose unsigned order is the value order, after the runtime fold (argmax =
// argmin of the sign-flipped value, so both operations share one kernel):
//   positive x: key = x ^ 0x8000      (0x8000 .. ; key 0x8000 is +-0)
//   negative x: key = x ^ 0xFFFF      (0x0001 .. 0x7FFF)
//   -0:         key = 0x8000, the +0 key (they compare equal; the sign survives in a separate bit)
//   NaN:        key = 0x0000          (NaN wins and propagates; no ordinary value has key 0)
// FP16 and BF16 differ only in the NaN threshold, one runtime per-half bias, so one kernel covers both. Key 0xFFFF
// never occurs (x = 0x7FFF is NaN in both formats), so an all-ones candidate is a strict identity, and a half-word
// given key 0xFFFF never wins against an element.
//
// The candidate is one 64-bit word (decision 2):
//   bits 63..48 key | bits 47..1 row-local index | bit 0 negative zero
// The smallest candidate is the smallest key, then the lowest index: exactly Thor's ARG contract (NaNs propagate, the
// lowest index wins ties, including ties between NaNs and between -0 and +0).
// Pack's per-thread lane candidate is 32 bits (key << 16 | row-local index << 1 | negative zero; rows of at most 16
// elements) and widens to the 64-bit word.
//
// f16ArgValueBits() rebuilds the winning 16-bit pattern from (key, negative-zero bit), exactly for every non-NaN
// element; a NaN winner comes back as 0xFFFF, a NaN in both formats.
//
// Host-only translation units (tests, selectors) can include this header without CUDA.

#include <cstdint>

#if defined(__CUDACC__)
#define THOR_F16_ARG_HD __host__ __device__ __forceinline__
#else
#define THOR_F16_ARG_HD inline
#endif

namespace ThorImplementation::CubReductionInternal {

struct F16ArgFold {
    uint32_t xor_mask;  // 0 for argmin, 0x80008000 for argmax
    uint32_t nan_bias;  // per half: 0x7FFF minus the largest non-NaN magnitude (FP16 0x7C00 -> 0x03FF, BF16 0x7F80
                        // -> 0x007F)
};

[[nodiscard]] THOR_F16_ARG_HD constexpr F16ArgFold f16ArgFold(bool maximum, bool bf16) {
    return {maximum ? 0x80008000u : 0u, bf16 ? 0x007F007Fu : 0x03FF03FFu};
}

struct F16ArgKeys2 {
    uint32_t keys;           // one key per half-word of the input word
    uint32_t negative_zero;  // 0x8000 in each half whose folded value is -0, 0 elsewhere
};

inline constexpr uint32_t F16_ARG_LANE_IDENTITY = 0xFFFFFFFFu;
inline constexpr uint64_t F16_ARG_WIDE_IDENTITY = ~uint64_t{0};
inline constexpr uint32_t F16_ARG_WIDE_INDEX_BITS = 47;
inline constexpr uint64_t F16_ARG_MAX_WIDE_INDEX = (uint64_t{1} << F16_ARG_WIDE_INDEX_BITS) - 1;
inline constexpr uint32_t F16_ARG_LANE_INDEX_BITS = 15;  // Pack: a row-local index of a row of <= 16 elements

// Keys for the two half-words of one input word: integer SIMD within one register (no borrow or carry crosses a
// half).
THOR_F16_ARG_HD F16ArgKeys2 f16ArgKeys2(uint32_t word, F16ArgFold fold) {
    const uint32_t x = word ^ fold.xor_mask;
    const uint32_t sign = x & 0x80008000u;
    // sign - (sign >> 15) is 0x7FFF in negative halves: negative halves get x ^ 0xFFFF.
    uint32_t keys = x ^ 0x80008000u ^ (sign - (sign >> 15));
    // Exact per-half zero test of x ^ 0x8000: 0x8000 where x is -0.
    const uint32_t t = x ^ 0x80008000u;
    const uint32_t negative_zero = ~(((t & 0x7FFF7FFFu) + 0x7FFF7FFFu) | t | 0x7FFF7FFFu);
    keys += negative_zero >> 15;  // -0 key 0x7FFF -> 0x8000
    // Magnitude above the format's largest non-NaN encoding sets bit 15 of (magnitude + bias): NaN -> key 0.
    const uint32_t nan = ((x & 0x7FFF7FFFu) + fold.nan_bias) & 0x80008000u;
    keys &= ~(nan | (nan - (nan >> 15)));
    return {keys, negative_zero};
}

// The 64-bit candidate of half `half` (0 low, 1 high) of a key word at row-local index `index`.
THOR_F16_ARG_HD uint64_t f16ArgWideCandidate(F16ArgKeys2 keys, uint32_t half, uint64_t index) {
    const uint32_t shift = 16u * half;
    return (static_cast<uint64_t>((keys.keys >> shift) & 0xFFFFu) << (F16_ARG_WIDE_INDEX_BITS + 1)) | (index << 1) |
           ((keys.negative_zero >> (shift + 15u)) & 1u);
}

// Pack's 32-bit lane candidate of half `half` at row-local index `index` (< 2^15).
THOR_F16_ARG_HD uint32_t f16ArgLaneCandidate(F16ArgKeys2 keys, uint32_t half, uint32_t index) {
    const uint32_t shift = 16u * half;
    return (((keys.keys >> shift) & 0xFFFFu) << 16) | (index << 1) | ((keys.negative_zero >> (shift + 15u)) & 1u);
}

THOR_F16_ARG_HD uint64_t f16ArgWidenLane(uint32_t lane) {
    const uint64_t key = lane >> 16;
    const uint64_t wide = (key << (F16_ARG_WIDE_INDEX_BITS + 1)) | (lane & 0xFFFFu);
    return lane == F16_ARG_LANE_IDENTITY ? F16_ARG_WIDE_IDENTITY : wide;
}

THOR_F16_ARG_HD uint64_t f16ArgWideIndex(uint64_t wide) { return (wide >> 1) & F16_ARG_MAX_WIDE_INDEX; }
THOR_F16_ARG_HD uint32_t f16ArgWideKey(uint64_t wide) {
    return static_cast<uint32_t>(wide >> (F16_ARG_WIDE_INDEX_BITS + 1)) & 0xFFFFu;
}
THOR_F16_ARG_HD uint32_t f16ArgWideNegativeZero(uint64_t wide) { return static_cast<uint32_t>(wide) & 1u; }

// The original 16-bit pattern for a winning (key, negative-zero bit). Exact for every non-NaN element; key 0 gives
// 0xFFFF (NaN in both formats).
THOR_F16_ARG_HD uint32_t f16ArgValueBits(uint32_t key, uint32_t negative_zero, F16ArgFold fold) {
    const uint32_t x = (key >= 0x8000u ? key ^ 0x8000u : key ^ 0xFFFFu) | (negative_zero << 15);
    return (x ^ fold.xor_mask) & 0xFFFFu;
}

}  // namespace ThorImplementation::CubReductionInternal
