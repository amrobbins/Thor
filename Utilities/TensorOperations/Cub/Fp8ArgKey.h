#pragma once

// FP8 ARG keys (ARG-A1): argmin/argmax on raw FP8 bytes without decoding.
//
// Every FP8 byte maps to an 8-bit key whose unsigned order is the FP8 value order, after the runtime fold
// (argmax = argmin of the sign-flipped value, so both operations share one kernel):
//   positive x: key = x ^ 0x80        (0x80 .. 0xFE; key 0x80 is +-0)
//   negative x: key = x ^ 0xFF        (0x01 .. 0x7F)
//   -0:         key = 0x80, the +0 key (they compare equal; the sign survives in a separate bit)
//   NaN:        key = 0x00            (NaN wins and propagates; no ordinary value has key 0)
// The NaN set and -0 depend on the format only through one runtime per-byte bias, so one kernel per geometry also
// covers E4M3 and E5M2. Key 0xFF never occurs (x = 0x7F is NaN in both formats), so 0xFFFFFFFF is a strict identity.
//
// Per-component lane state is one uint32 candidate, compared with one unsigned min:
//   bits 31..24 key | bits 23..1 lane-local row | bit 0 negative zero
// The smallest candidate is the smallest key, then the lowest row: exactly Thor's ARG contract (NaNs propagate, the
// lowest index wins ties, including ties between NaNs and between -0 and +0). The low bit never decides a comparison
// because rows within one lane are distinct. Kernels must keep a lane below FP8_ARG_MAX_LANE_ROWS rows.
//
// Across lanes the candidate widens to uint64 with a 55-bit index (1132; 32-bit before, which capped staged ARG at
// R <= 2^32 - 1):
//   bits 63..56 key | bits 55..1 index | bit 0 negative zero
// and the same unsigned min merges lanes, warps and CTAs deterministically.
//
// fp8ArgValueByte() rebuilds the winning FP8 byte from (key, negative-zero bit), exactly for every non-NaN byte; a
// NaN winner comes back as the NaN encoding 0xFF or 0x7F.
//
// Host-only translation units (tests, selectors) can include this header without CUDA.

#include <cstdint>

#if defined(__CUDACC__)
#define THOR_FP8_ARG_HD __host__ __device__ __forceinline__
#else
#define THOR_FP8_ARG_HD inline
#endif

namespace ThorImplementation::CubReductionInternal {

struct Fp8ArgFold {
    uint32_t xor_mask;  // 0 for argmin, 0x80808080 for argmax
    uint32_t nan_bias;  // per byte: 0x7F minus the largest non-NaN magnitude (E4M3 0x7E -> 1, E5M2 0x7C -> 3)
};

// Host and device (a constexpr host-only function called from a kernel would need --expt-relaxed-constexpr).
[[nodiscard]] THOR_FP8_ARG_HD constexpr Fp8ArgFold fp8ArgFold(bool maximum, bool e5m2) {
    return {maximum ? 0x80808080u : 0u, e5m2 ? 0x03030303u : 0x01010101u};
}

struct Fp8ArgKeys4 {
    uint32_t keys;           // one key per byte of the packed input word
    uint32_t negative_zero;  // 0x80 in each byte whose folded value is -0, 0 elsewhere
};

inline constexpr uint32_t FP8_ARG_ROW_BITS = 23;
inline constexpr uint32_t FP8_ARG_MAX_LANE_ROWS = 1u << FP8_ARG_ROW_BITS;
inline constexpr uint32_t FP8_ARG_LANE_IDENTITY = 0xFFFFFFFFu;
inline constexpr uint64_t FP8_ARG_WIDE_IDENTITY = ~uint64_t{0};
inline constexpr uint32_t FP8_ARG_WIDE_INDEX_BITS = 55;
// Widened indices (shard row base + lane-local row) must stay below this.
inline constexpr uint64_t FP8_ARG_MAX_WIDE_INDEX = (uint64_t{1} << FP8_ARG_WIDE_INDEX_BITS) - 1;
inline constexpr uint32_t FP8_ARG_NAN_KEY = 0;

// Keys for four packed FP8 bytes: integer SIMD within one register (no per-byte video intrinsics, no decode).
THOR_FP8_ARG_HD Fp8ArgKeys4 fp8ArgKeys4(uint32_t packed, Fp8ArgFold fold) {
    const uint32_t x = packed ^ fold.xor_mask;
    const uint32_t sign = x & 0x80808080u;
    // sign - (sign >> 7) is 0x7F in negative bytes (no borrow crosses a byte): negative bytes get x ^ 0xFF.
    uint32_t keys = x ^ 0x80808080u ^ (sign - (sign >> 7));
    // Exact per-byte zero test of x ^ 0x80 (no carry crosses a byte): 0x80 where x is -0.
    const uint32_t t = x ^ 0x80808080u;
    const uint32_t negative_zero = ~(((t & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | t | 0x7F7F7F7Fu);
    keys += negative_zero >> 7;  // -0 key 0x7F -> 0x80
    // Magnitude above the format's largest non-NaN encoding sets bit 7 of (magnitude + bias): NaN -> key 0.
    const uint32_t nan = ((x & 0x7F7F7F7Fu) + fold.nan_bias) & 0x80808080u;
    keys &= ~(nan | (nan - (nan >> 7)));
    return {keys, negative_zero};
}

// Lane candidate for byte `byte` (0..3, compile-time in kernels) of a key word at lane-local row `row`.
THOR_FP8_ARG_HD uint32_t fp8ArgLaneCandidate(Fp8ArgKeys4 keys, uint32_t byte, uint32_t row) {
    const uint32_t shift = 8u * byte;
    return (((keys.keys >> shift) & 0xFFu) << 24) | (row << 1) | ((keys.negative_zero >> (shift + 7u)) & 1u);
}

THOR_FP8_ARG_HD uint32_t fp8ArgMin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// __byte_perm (PRMT): result byte n is byte (s >> 4n) & 7 of {x, y} (x bytes 0-3, y bytes 4-7).
THOR_FP8_ARG_HD uint32_t fp8ArgBytePerm(uint32_t x, uint32_t y, uint32_t s) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(x, y, s);
#else
    const uint64_t source = (static_cast<uint64_t>(y) << 32) | x;
    uint32_t result = 0;
    for (uint32_t n = 0; n < 4u; ++n) result |= static_cast<uint32_t>((source >> (8u * ((s >> (4u * n)) & 7u))) & 0xFFu) << (8u * n);
    return result;
#endif
}

// ARG-A3h (1124): the winning lane candidate among the four bytes of one key word when they are four rows of the SAME
// component (byte j = row first_row + j * row_stride): min over j of fp8ArgLaneCandidate(keys, j, row_j), with the
// rows resolved after the minimum. Byte j becomes v_j = key_j << 24 | tag_j << 16 | (don't care), tag_j = 2j |
// negative_zero_j, one PRMT each; tags are distinct, so the minimum is decided by (key, j) and the low bytes never
// matter. The winner's row and -0 bit come back out of its tag. Equal keys keep the lowest row (lowest j), exactly as
// the per-byte candidate minimum. OneWindow c1/c2 SASS (1123) spent ~26 instructions per word on per-byte candidates.
THOR_FP8_ARG_HD uint32_t fp8ArgRowsCandidate4(Fp8ArgKeys4 keys, uint32_t first_row, uint32_t row_stride) {
    const uint32_t tags = 0x06040200u | (keys.negative_zero >> 7);
    const uint32_t v0 = fp8ArgBytePerm(keys.keys, tags, 0x0400u);
    const uint32_t v1 = fp8ArgBytePerm(keys.keys, tags, 0x1511u);
    const uint32_t v2 = fp8ArgBytePerm(keys.keys, tags, 0x2622u);
    const uint32_t v3 = fp8ArgBytePerm(keys.keys, tags, 0x3733u);
    const uint32_t best = fp8ArgMin(fp8ArgMin(v0, v1), fp8ArgMin(v2, v3));
    const uint32_t row = first_row + ((best >> 17) & 3u) * row_stride;
    return (best & 0xFF000000u) | (row << 1) | ((best >> 16) & 1u);
}
THOR_FP8_ARG_HD uint64_t fp8ArgMin(uint64_t a, uint64_t b) { return a < b ? a : b; }

// Widens a lane candidate whose rows are row_base + lane-local row. The identity stays the identity.
THOR_FP8_ARG_HD uint64_t fp8ArgWiden(uint32_t lane, uint64_t row_base) {
    const uint64_t key = lane >> 24;
    const uint64_t index = row_base + ((lane >> 1) & (FP8_ARG_MAX_LANE_ROWS - 1u));
    const uint64_t wide = (key << (FP8_ARG_WIDE_INDEX_BITS + 1)) | (index << 1) | (lane & 1u);
    return lane == FP8_ARG_LANE_IDENTITY ? FP8_ARG_WIDE_IDENTITY : wide;
}

THOR_FP8_ARG_HD uint64_t fp8ArgWideIndex(uint64_t wide) { return (wide >> 1) & FP8_ARG_MAX_WIDE_INDEX; }
THOR_FP8_ARG_HD uint32_t fp8ArgWideKey(uint64_t wide) {
    return static_cast<uint32_t>(wide >> (FP8_ARG_WIDE_INDEX_BITS + 1)) & 0xFFu;
}
THOR_FP8_ARG_HD uint32_t fp8ArgWideNegativeZero(uint64_t wide) { return static_cast<uint32_t>(wide) & 1u; }

// The original FP8 byte for a winning (key, negative-zero bit). Exact for every non-NaN byte; key 0 gives NaN.
THOR_FP8_ARG_HD uint32_t fp8ArgValueByte(uint32_t key, uint32_t negative_zero, Fp8ArgFold fold) {
    const uint32_t x = (key >= 0x80u ? key ^ 0x80u : key ^ 0xFFu) | (negative_zero << 7);
    return (x ^ fold.xor_mask) & 0xFFu;
}

}  // namespace ThorImplementation::CubReductionInternal
