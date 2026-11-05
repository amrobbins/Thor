#pragma once

// FP32 argmin / argmax candidates (1181 for R / KR, shared with RK since 1198; plans/Thor_FP32_Dense_Reduction_Plan).
// The candidate is two words: a 32-bit ordered key and a 64-bit tail = index << 1 | negative zero, compared key first,
// then tail. The key of the folded value x (argmax = argmin of -x, so both share the kernels):
//   positive x: key = x | 0x80000000        (+0 -> 0x80000000, +inf -> 0xFF800000)
//   negative x: key = ~x                    (-inf -> 0x007FFFFF; -0 is given the +0 key, its sign in the tail)
//   NaN:        key = 0                     (NaN wins and propagates; no ordinary value has key 0)
// so unsigned order is value order, and key 0xFFFFFFFF never occurs (a strict identity). Within one lane the indices
// only increase, so a strictly smaller key replaces the lane's best (ties keep the lower index); across lanes, warps,
// CTAs and shards the full (key, tail) minimum merges deterministically: Thor's ARG contract (NaNs propagate, the
// lowest index wins ties, NaN-NaN and -0/+0 ties included).
//
// 1198: staged RK partials pack to one 64-bit word, key << 32 | shard-local row << 1 | negative zero (shards of at
// most 2^31 - 1 rows; launchers check), widened by the continuation with the shard's first row.

#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cstdint>

namespace ThorImplementation::CubReductionInternal {

inline constexpr uint32_t FP32_ARG_KEY_IDENTITY = 0xFFFFFFFFU;
inline constexpr uint64_t FP32_ARG_TAIL_IDENTITY = ~uint64_t{0};

struct Fp32ArgCandidate {
    uint32_t key;
    uint64_t tail;
};

__device__ __forceinline__ Fp32ArgCandidate fp32ArgIdentity() {
    return {FP32_ARG_KEY_IDENTITY, FP32_ARG_TAIL_IDENTITY};
}

__device__ __forceinline__ Fp32ArgCandidate fp32ArgMin(const Fp32ArgCandidate& a, const Fp32ArgCandidate& b) {
    return (b.key < a.key || (b.key == a.key && b.tail < a.tail)) ? b : a;
}

// Key and negative-zero bit of one FP32 word after the argmax sign fold.
__device__ __forceinline__ uint32_t fp32ArgKey(uint32_t bits, uint32_t xor_mask, uint32_t& negative_zero) {
    const uint32_t x = bits ^ xor_mask;
    negative_zero = x == 0x80000000U ? 1U : 0U;
    const uint32_t ordered = (x & 0x80000000U) != 0 ? ~x : (x | 0x80000000U);
    const uint32_t key = negative_zero != 0 ? 0x80000000U : ordered;
    return (x & 0x7fffffffU) > 0x7f800000U ? 0U : key;
}

// The winning FP32 value: exact for every non-NaN value (the sign of zero included); key 0 gives a NaN.
__device__ __forceinline__ float fp32ArgValue(uint32_t key, uint64_t tail, uint32_t xor_mask) {
    const uint32_t x = (key >= 0x80000000U ? key ^ 0x80000000U : ~key) | (static_cast<uint32_t>(tail & 1U) << 31);
    return __uint_as_float(x ^ xor_mask);
}

struct Fp32ArgOutputs {
    void* value;  // null: no value output
    DataType value_dtype;
    void* index;  // null: no index output
    DataType index_dtype;
    uint64_t domain_stride;  // index = row-local index * domain_stride
    uint32_t xor_mask;       // 0 for argmin, 0x80000000 for argmax
};

__device__ __forceinline__ void storeFp32ArgResult(const Fp32ArgOutputs& outputs, uint64_t output,
                                                   const Fp32ArgCandidate& best) {
    if (outputs.value != nullptr) {
        storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output,
                                fp32ArgValue(best.key, best.tail, outputs.xor_mask));
    }
    if (outputs.index != nullptr) {
        const uint64_t index = (best.tail >> 1) * outputs.domain_stride;
        if (outputs.index_dtype == DataType::UINT32) {
            static_cast<uint32_t*>(outputs.index)[output] = static_cast<uint32_t>(index);
        } else {
            static_cast<uint64_t*>(outputs.index)[output] = index;
        }
    }
}


__device__ __forceinline__ Fp32ArgCandidate fp32ArgShuffleXor(const Fp32ArgCandidate& value, int offset) {
    Fp32ArgCandidate other;
    other.key = __shfl_xor_sync(0xffffffffU, value.key, offset);
    other.tail = __shfl_xor_sync(0xffffffffU, static_cast<unsigned long long>(value.tail), offset);
    return other;
}

// ---- 1198: packed staged RK partials.
inline constexpr uint64_t FP32_ARG_PACKED_IDENTITY = ~uint64_t{0};
inline constexpr uint64_t FP32_ARG_MAX_SHARD_ROWS = (uint64_t{1} << 31) - 1;

__device__ __forceinline__ uint64_t fp32ArgPack(const Fp32ArgCandidate& candidate) {
    return (static_cast<uint64_t>(candidate.key) << 32) | (candidate.tail & 0xFFFFFFFFULL);
}

// The candidate of a packed shard partial whose shard begins at row_begin. An identity partial (an empty shard) stays
// the identity.
__device__ __forceinline__ Fp32ArgCandidate fp32ArgWiden(uint64_t packed, uint64_t row_begin) {
    if (packed == FP32_ARG_PACKED_IDENTITY) return fp32ArgIdentity();
    const uint64_t low = packed & 0xFFFFFFFFULL;
    return {static_cast<uint32_t>(packed >> 32), (((low >> 1) + row_begin) << 1) | (low & 1U)};
}

// ---- 1208: carried-index (intermediate) FP32 ARG passes (plans/Thor_FP32_Dense_Reduction_Plan, phase 8). The input is
// FP32 values with a carried index (UINT32 / UINT64) per element; an element's composed original-domain index is
// carried + row * row_stride (row_stride: the current run's original-domain stride, 0 for an intermediate run), and its
// tail is composed << 1 | negative zero. Composed indices are not monotone in the row, so every merge is the full
// (key, tail) minimum. Staged partials are wide (user decision 2026-11-03): a uint32 key array, then (256-byte
// aligned) a uint64 tail array, [outer, shards, K] each; the indices are final, so the continuation only merges.
struct Fp32ArgWidePartials {
    uint32_t* keys;
    uint64_t* tails;
};

// Keeps the lesser of `best` and (key, tail) (the full minimum).
__device__ __forceinline__ void fp32ArgKeepMin(uint32_t& best_key, uint64_t& best_tail, uint32_t key, uint64_t tail) {
    if (key < best_key || (key == best_key && tail < best_tail)) {
        best_key = key;
        best_tail = tail;
    }
}

}  // namespace ThorImplementation::CubReductionInternal
