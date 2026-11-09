#pragma once

// 1298 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged ARG boundary records and index stores,
// shared by the ragged ARG main kernels (RaggedKRArgKernels.cuh), the ARG finisher (CubRaggedFinishArg.cu) and the host
// tests. Host / device, no CUDA needed on the host.
//
// A ragged ARG candidate is one 64-bit word whose unsigned order is Thor's ARG contract: the dense key word of the input
// dtype in the top bits, then the winner's GLOBAL row (its row in the input tensor), then the negative-zero bit
//   FP8:  bits 63..56 key | bits 55..1 row | bit 0 negative zero   (Fp8ArgKey.h's wide candidate, the index a row)
//   1311, FP16 / BF16: bits 63..48 key | bits 47..1 row | bit 0 0 (F16ArgKey.h's layout)
// so the smallest word is the smallest key (NaN first), then the lowest row: ties go to the lowest index whichever CTA
// holds them, and the fold order never matters (unsigned min is commutative and associative). The output index is the
// packed `row * D + component`; a segment without rows gets the sentinel of the index dtype (all ones). The negative
// zero bit never decides a comparison (rows are distinct) and the index output does not use it.
//
// The records, as the VALUE records (RaggedKRWalk.h, raggedRecordLayout) with 8-byte values: each CTA's key (its head
// segment or NO_SEGMENT), then component c's head words for CTAs 0 .. grid - 1, then the tail words the same way.
//
// 1313, FP32: a 32-bit key and a global row do not fit one 64-bit word. Inside the main kernel FP32's words carry the
// CTA-local row instead (bits 63..32 key | bits 31..1 local row | bit 0 0; a CTA's rows stay below 2^31, as its events
// do), which orders a CTA's candidates exactly as the global rows would; the stores add the CTA's first row. Its
// records are pairs (raggedArgPairRecordLayout: a 32-bit key and a 64-bit global row per CTA and component), finished
// by the pair finisher (CubRaggedFinishArgFp32.cu) with the (key, row) minimum.
#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"
#include "Utilities/TensorOperations/Cub/RaggedKRWalk.h"
#include "Utilities/TensorOperations/Cub/RaggedMergePath.h"

#include <cstdint>

namespace ThorImplementation::CubReductionInternal::RaggedMergePath {

inline constexpr uint64_t RAGGED_ARG_IDENTITY = ~uint64_t{0};
// FP8: the row field of the 64-bit word (Fp8ArgKey.h's FP8_ARG_WIDE_INDEX_BITS); rows must stay below 2^55 - 1.
inline constexpr uint32_t RAGGED_ARG_FP8_ROW_BITS = FP8_ARG_WIDE_INDEX_BITS;
// 1311: FP16 / BF16: bits 63..48 key | 47..1 row | 0 (F16ArgKey.h's layout); rows below 2^47 - 1.
inline constexpr uint32_t RAGGED_ARG_F16_ROW_BITS = 47;
// 1313: FP32's in-kernel words: bits 63..32 key | 31..1 CTA-local row | 0.
inline constexpr uint32_t RAGGED_ARG_FP32_LOCAL_ROW_BITS = 31;
// 1313: the pair records' identity: key 0xFFFFFFFF (no FP32 key has it) and row all ones.
inline constexpr uint32_t RAGGED_ARG_PAIR_KEY_IDENTITY = 0xFFFFFFFFU;
inline constexpr uint64_t RAGGED_ARG_PAIR_ROW_IDENTITY = ~uint64_t{0};

struct RaggedArgRecordLayout {
    uint64_t keys = 0, heads = 0, tails = 0, bytes = 0;  // byte offsets of the three arrays, and the total
};

THOR_RAGGED_HD RaggedArgRecordLayout raggedArgRecordLayout(uint32_t grid, uint32_t components) {
    RaggedArgRecordLayout layout;
    const uint64_t words = (uint64_t{8} * grid * components + 15U) & ~uint64_t{15};
    layout.keys = 0;
    layout.heads = (uint64_t{8} * grid + 15U) & ~uint64_t{15};
    layout.tails = layout.heads + words;
    layout.bytes = layout.tails + words;
    return layout;
}

// 1313: the pair records (FP32): each CTA's key (8 bytes), then per component the head keys (4 bytes), the head rows
// (8 bytes, global), the tail keys and the tail rows, CTA c of component k at k * stride + c. The stride is the grid
// rounded up to 4, so each array starts 16-byte aligned and the pair finisher reads four CTAs per vector load (the
// padding entries are never written and never read as records).
struct RaggedArgPairRecordLayout {
    uint64_t keys = 0, head_keys = 0, head_rows = 0, tail_keys = 0, tail_rows = 0, bytes = 0;
    uint32_t stride = 0;
};

THOR_RAGGED_HD RaggedArgPairRecordLayout raggedArgPairRecordLayout(uint32_t grid, uint32_t components) {
    RaggedArgPairRecordLayout layout;
    layout.stride = (grid + 3U) & ~3U;
    const uint64_t entries = uint64_t{layout.stride} * components;
    layout.keys = 0;
    layout.head_keys = uint64_t{8} * layout.stride;
    layout.head_rows = layout.head_keys + 4 * entries;
    layout.tail_keys = layout.head_rows + 8 * entries;
    layout.tail_rows = layout.tail_keys + 4 * entries;
    layout.bytes = layout.tail_rows + 8 * entries;
    return layout;
}

// 1313: a pair's index: the sentinel for the identity, else row * D + component.
THOR_RAGGED_HD uint64_t raggedArgPairOutputIndex(uint64_t row, uint64_t components, uint64_t component,
                                                 uint32_t index_bytes) {
    if (row == RAGGED_ARG_PAIR_ROW_IDENTITY) return index_bytes == 8 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFU};
    return row * components + component;
}

// The row of a candidate word whose row field is `row_bits` wide above the negative-zero bit.
THOR_RAGGED_HD uint64_t raggedArgWordRow(uint64_t word, uint32_t row_bits) {
    return (word >> 1) & ((uint64_t{1} << row_bits) - 1U);
}

// The stored index of a candidate word: the sentinel (all ones in the index width) for the identity, else
// row * D + component. index_bytes: 4 (UINT32) or 8 (UINT64).
THOR_RAGGED_HD uint64_t raggedArgOutputIndex(uint64_t word, uint32_t row_bits, uint64_t components, uint64_t component,
                                             uint32_t index_bytes) {
    if (word == RAGGED_ARG_IDENTITY) return index_bytes == 8 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFU};
    return raggedArgWordRow(word, row_bits) * components + component;
}

// 1313: the stored index of an FP32 in-kernel word (CTA-local row) of the CTA whose first row is `row_base`.
THOR_RAGGED_HD uint64_t raggedArgLocalOutputIndex(uint64_t word, uint64_t row_base, uint64_t components,
                                                  uint64_t component, uint32_t index_bytes) {
    if (word == RAGGED_ARG_IDENTITY) return index_bytes == 8 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFU};
    return (row_base + raggedArgWordRow(word, RAGGED_ARG_FP32_LOCAL_ROW_BITS)) * components + component;
}

// 1313: an FP32 in-kernel word as its pair record's row (global; the identity stays the identity) and key.
THOR_RAGGED_HD uint64_t raggedArgLocalWordGlobalRow(uint64_t word, uint64_t row_base) {
    return word == RAGGED_ARG_IDENTITY ? RAGGED_ARG_PAIR_ROW_IDENTITY
                                       : row_base + raggedArgWordRow(word, RAGGED_ARG_FP32_LOCAL_ROW_BITS);
}
THOR_RAGGED_HD uint32_t raggedArgLocalWordKey(uint64_t word) { return static_cast<uint32_t>(word >> 32); }

THOR_RAGGED_HD void raggedArgStoreIndex(void* output, uint32_t index_bytes, uint64_t at, uint64_t index) {
    if (index_bytes == 8) {
        static_cast<uint64_t*>(output)[at] = index;
    } else {
        static_cast<uint32_t*>(output)[at] = static_cast<uint32_t>(index);
    }
}

// 1301: the ragged FP8 ARG keys of four packed bytes after the argmax fold: 0x80 + magnitude for a positive byte, 0x80 -
// magnitude for a negative one (so -0 takes +0's key and no separate negative-zero bit is needed), 0 for NaN. The same
// order as Fp8ArgKey.h's fp8ArgKeys4 (whose negative keys sit one lower), in about 13 integer instructions against its
// 17: no carry or borrow crosses a byte (0x80 + m <= 0xFF, 0x80 - m >= 1 for any non-NaN magnitude m <= 0x7E), and the
// key 0xFF never occurs (the kernels mark rows outside a tile with it).
THOR_RAGGED_HD uint32_t raggedFp8ArgKeys(uint32_t packed, Fp8ArgFold fold) {
    const uint32_t x = packed ^ fold.xor_mask;
    const uint32_t magnitude = x & 0x7F7F7F7FU;
    const uint32_t sign = x & 0x80808080U;
    const uint32_t negative = sign - (sign >> 7);  // 0x7F in negative bytes
    uint32_t keys = 0x80808080U + (magnitude & ~negative) - (magnitude & negative);
    const uint32_t nan = (magnitude + fold.nan_bias) & 0x80808080U;
    keys &= ~(nan | (nan - (nan >> 7)));
    return keys;
}

// 1301: the minimum 32-bit tile candidate of 16 rows with no segment end among them: keys[w] byte b is row 4w + b, whose
// row field is first_field + 4w + b (RaggedKRArgKernels.cuh: key << 24 | field << 1). One byte permute per row puts
// its key on top of its row tag (4w + b, from a constant word), fifteen minimums pick the smallest (key, row), and the
// winner's row field is decoded once. Rows marked with key 0xFF lose to every other row.
THOR_RAGGED_HD uint32_t raggedFp8ArgFold16(const uint32_t (&keys)[4], uint32_t first_field) {
    uint32_t best = 0xFFFFFFFFU;
THOR_RAGGED_UNROLL
    for (uint32_t w = 0; w < 4; ++w) {
        const uint32_t tags = 0x03020100U + 0x04040404U * w;
THOR_RAGGED_UNROLL
        for (uint32_t b = 0; b < 4; ++b) {
            const uint32_t v = fp8ArgBytePerm(keys[w], tags, (b << 12) | ((4U + b) << 8) | ((4U + b) << 4) | (4U + b));
            best = v < best ? v : best;
        }
    }
    return (best & 0xFF000000U) | ((first_field + ((best >> 16) & 0xFFU)) << 1);
}

// 1311: the ragged FP16 / BF16 ARG keys of two packed halves after the argmax fold (fold.xor_mask 0x80008000 for
// argmax; fold.nan_bias per half, F16ArgKey.h's f16ArgFold: FP16 0x03FF, BF16 0x007F): 0x8000 + magnitude for a
// positive half, 0x8000 - magnitude for a negative one (-0 on +0's key), 0 for NaN; raggedFp8ArgKeys' form per half.
// The same order as f16ArgKeys2; no carry or borrow crosses a half (0x8000 + m <= 0xFF80, 0x8000 - m >= 0x80 for any
// non-NaN magnitude, m + bias <= 0x83FE), and the key 0xFFFF never occurs (the kernels' outside-the-tile marker).
THOR_RAGGED_HD uint32_t raggedF16ArgKeys(uint32_t packed, Fp8ArgFold fold) {
    const uint32_t x = packed ^ fold.xor_mask;
    const uint32_t magnitude = x & 0x7FFF7FFFU;
    const uint32_t sign = x & 0x80008000U;
    const uint32_t negative = sign - (sign >> 15);  // 0x7FFF in negative halves
    uint32_t keys = 0x80008000U + (magnitude & ~negative) - (magnitude & negative);
    const uint32_t nan = (magnitude + fold.nan_bias) & 0x80008000U;
    keys &= ~(nan | (nan - (nan >> 15)));
    return keys;
}

// 1311: raggedFp8ArgFold16's FP16 / BF16 form: keys[w] half h is row 2w + h of 8, the candidate key << 16 | field << 1
// with field first_field + 2w + h. One byte permute per row (its key's two bytes over its row tag), seven minimums, one
// row decode. Rows marked with key 0xFFFF lose to every other row.
THOR_RAGGED_HD uint32_t raggedF16ArgFold8(const uint32_t (&keys)[4], uint32_t first_field) {
    uint32_t best = 0xFFFFFFFFU;
THOR_RAGGED_UNROLL
    for (uint32_t w = 0; w < 4; ++w) {
        const uint32_t tags = 0x0100U + 0x0202U * w;
THOR_RAGGED_UNROLL
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t v = fp8ArgBytePerm(keys[w], tags,
                                              ((2U * h + 1U) << 12) | ((2U * h) << 8) | ((4U + h) << 4) | (4U + h));
            best = v < best ? v : best;
        }
    }
    return (best & 0xFFFF0000U) | ((first_field + ((best >> 8) & 0xFFU)) << 1);
}

// 1313: the ragged FP32 ARG key of one element after the argmax fold (xor_mask 0x80000000 for argmax): 0x80000000 +
// magnitude for a positive value, 0x80000000 - magnitude for a negative one (so -0 takes +0's key, as Fp32ArgKey.cuh
// gives it, without a separate negative-zero bit), 0 for NaN. The same order as fp32ArgKey (whose negative keys, ~x,
// sit one lower): the smallest non-NaN key is -inf's 0x00800000, the largest +inf's 0xFF800000, so 0xFFFFFFFF never
// occurs (the kernels mark rows outside a tile with it).
THOR_RAGGED_HD uint32_t raggedFp32ArgKey(uint32_t bits, uint32_t xor_mask) {
    const uint32_t x = bits ^ xor_mask;
    const uint32_t magnitude = x & 0x7FFFFFFFU;
    const uint32_t negative = 0U - (x >> 31);  // all ones for a negative value
    const uint32_t key = 0x80000000U + ((magnitude ^ negative) - negative);
    return magnitude > 0x7F800000U ? 0U : key;
}

// 1313: the minimum 64-bit tile candidate of a lane's 4 FP32 rows with no segment end among them: row k's candidate is
// key << 32 | (first_field + k) << 1, so the smallest key wins and the lowest row among equal keys. A tree of two
// levels (strict comparisons, the lower row kept on ties), one candidate built. Rows marked with key 0xFFFFFFFF lose to
// every other row.
THOR_RAGGED_HD uint64_t raggedFp32ArgFold4(const uint32_t (&keys)[4], uint32_t first_field) {
    const bool high01 = keys[1] < keys[0], high23 = keys[3] < keys[2];
    const uint32_t best01 = high01 ? keys[1] : keys[0], best23 = high23 ? keys[3] : keys[2];
    const bool high = best23 < best01;
    const uint32_t best = high ? best23 : best01;
    const uint32_t at = high ? 2U + (high23 ? 1U : 0U) : (high01 ? 1U : 0U);
    return (uint64_t{best} << 32) | ((first_field + at) << 1);
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedMergePath
