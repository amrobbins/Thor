#pragma once

// 1298 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged KR ARG main kernel, FP8 input at
// D = 1 (argmin and argmax, E4M3 and E5M2 in one kernel); 1300: D = 2..15 (raggedKRArgComponentsKernel, below).
// 1311: both kernels are templates on the key format (RaggedArgKeyFp8, RaggedArgKeyF16 for FP16 and BF16 in one
// kernel: D = 1 and D 2..7), instantiated by CubRaggedKRArgFp8.cu and CubRaggedKRArgF16.cu. Below, "FP8" and its
// byte keys stand for the format: an F16 candidate is key << 16 | field << 1, its marker key 0xFFFF, 8 rows per lane.
// 1313: RaggedArgKeyFp32 (CubRaggedKRArgFp32.cu, D 1 to 3): a 32-bit key leaves no room for a row field in 32 bits, so
// its tile candidates (the traits' Lane) are 64-bit, key << 32 | field << 1, its marker key 0xFFFFFFFF, 4 rows per lane;
// and its 64-bit words hold the CTA-local row (RaggedArg.h), so its stores add the CTA's first row and its records are
// pairs with global rows (the pair finisher, CubRaggedFinishArgFp32.cu). The host launcher is CubRaggedKRArg.cu and the
// finishers CubRaggedFinishArg.cu (64-bit words) and CubRaggedFinishArgFp32.cu (pairs).
//
// The walk is the D = 1 VALUE kernel's (raggedKRValueKernel, RaggedKRKernels.cuh, with its history): one warp per CTA,
// the merge-path range, the ends ring and its prefetch, tiles of one aligned 512-byte span (a lane owns one 16-byte
// chunk, 16 rows), the bitmap of the tile's ends, the kinds and ranks of the ends, one lane per end storing in segment
// order, the boundary records. Only the values differ:
//   - a row's candidate is its FP8 ARG key (Fp8ArgKey.h: no decode; argmax runs as argmin of the sign-flipped bytes,
//     and the NaN threshold is a runtime per-byte bias, so one kernel serves both operations and both formats) with
//     its row: inside a tile a 32-bit candidate, key << 24 | (span row + 1) << 1 | negative zero (span rows < 512);
//     across tiles and CTAs the 64-bit word of RaggedArg.h with the global row, so every comparison is key first, then
//     the lowest index: Thor's ARG contract, whatever CTA or tile a tie falls in;
//   - the fold and the scan are an unsigned minimum (where no end falls, a lane's 16 rows in one fold), the identity
//     all ones;
//   - 1301: the keys are RaggedArg.h's raggedFp8ArgKeys (0x80 +- magnitude: -0 on +0's key without a negative-zero bit;
//     the same order as fp8ArgKeys4 in fewer instructions), and the fold where no end falls is raggedFp8ArgFold16 (one
//     byte permute per row onto a constant row tag, fifteen minimums, one row decode; 1299-1300: four
//     fp8ArgRowsCandidate4). The negative-zero bit of the candidates is now always 0;
//   - 1299: everything inside a tile is 32-bit. The open run arriving from the tiles before enters as the token
//     key << 24 (row field 0): its rows all come before the tile's, so it wins its key's ties, as its lower index
//     would. The token seeds the warp scan, the value slots are 32-bit, and a winner becomes the 64-bit word (the
//     carry itself for field 0, else the tile's base plus its span row) only where it leaves the tile: each end's
//     store, the head record and the carry. (1298 widened every emitted winner to 64 bits inside the fold, a
//     predicated 64-bit sequence per row, and joined the carry in 64 bits: 0.75-0.80 of VALUE MIN / MAX on tiles with
//     ends);
//   - rows outside the tile (the span's edges) get the key 0xFF, which no FP8 byte has (0x7F is NaN in both formats),
//     so they lose to every row of the tile; a run of only such rows becomes the identity;
//   - the outputs are indices (UINT32 or UINT64 at run time): row * D + component, or the sentinel for a segment
//     without rows (RaggedArg.h).
#include "Utilities/TensorOperations/Cub/RaggedArg.h"
#include "Utilities/TensorOperations/Cub/RaggedKRKernels.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

// The ARG main kernel's parameters.
struct RaggedKRArgParams {
    const uint8_t* bytes;  // the input's bytes, from its 16-byte aligned base
    uint32_t head;         // row 0's byte offset from `bytes`
    RaggedMergePath::Offsets offsets;
    uint64_t segments;
    uint64_t capacity_rows;
    RaggedMergePath::PathWeights weights;
    Fp8ArgFold fold;       // argmax: the sign flip of every element; the format's NaN bias (1311: f16ArgFold's for F16;
                           // 1313: FP32 the sign flip only)
    void* output;          // segments x D indices
    uint32_t index_bytes;  // 4 (UINT32) or 8 (UINT64)
    uint8_t* workspace;    // raggedArgRecordLayout(grid, components)
    uint32_t components;   // D (1300: 1 to 15)
    uint32_t workers;      // 1300: raggedComponentWorkers(components) (the components kernel)
    uint32_t ends_cap;     // 1300: raggedComponentEndsCap(components, RAGGED_KR_RING)
};

// The key format of a ragged ARG kernel (the first template argument, so the kernel is a template like the VALUE
// kernels and the SASS tool reads its dtype from it). 1298: FP8.
//
// 1311: the format's traits (both ARG kernels are templates on them): element bytes, rows per lane (one 16-byte chunk),
// elements per 32-bit word, key bits, the 64-bit word's row bits, and the per-word helpers: the keys of a word, the
// fold of a lane's rows where no end falls, and the 32-bit candidate of element `in_word` of a key word with row field
// `field2` (field << 1): key << (32 - KeyBits) | field << 1, one byte permute.
// 1313: also the tile candidate's type (Lane: 32-bit, or 64-bit for FP32), whether the 64-bit words hold CTA-local rows
// (LocalRows: FP32) and the stored index of a word (outputIndex: row_base is the CTA's first row, used by local rows).
struct RaggedArgKeyFp8 {
    static constexpr uint32_t RowBits = RaggedMergePath::RAGGED_ARG_FP8_ROW_BITS;
    static constexpr uint32_t Bytes = 1, Rows = 16, PerWord = 4, KeyBits = 8;
    static constexpr bool LocalRows = false;
    using Storage = uint8_t;
    using Lane = uint32_t;
    __device__ __forceinline__ static uint32_t keys(uint32_t word, Fp8ArgFold fold) {
        return RaggedMergePath::raggedFp8ArgKeys(word, fold);
    }
    __device__ __forceinline__ static uint32_t fold(const uint32_t (&keys)[4], uint32_t first_field) {
        return RaggedMergePath::raggedFp8ArgFold16(keys, first_field);
    }
    __device__ __forceinline__ static uint32_t candidate(uint32_t key_word, uint32_t in_word, uint32_t field2) {
        return __byte_perm(key_word, field2, (in_word << 12) | 0x0654U);
    }
    __device__ __forceinline__ static uint64_t outputIndex(uint64_t word, uint64_t /*row_base*/, uint64_t components,
                                                           uint64_t component, uint32_t index_bytes) {
        return RaggedMergePath::raggedArgOutputIndex(word, RowBits, components, component, index_bytes);
    }
};

// 1311: FP16 and BF16 (one kernel for both: the NaN bias is the fold's, f16ArgFold's values in Fp8ArgFold's fields).
struct RaggedArgKeyF16 {
    static constexpr uint32_t RowBits = RaggedMergePath::RAGGED_ARG_F16_ROW_BITS;
    static constexpr uint32_t Bytes = 2, Rows = 8, PerWord = 2, KeyBits = 16;
    static constexpr bool LocalRows = false;
    using Storage = uint16_t;
    using Lane = uint32_t;
    __device__ __forceinline__ static uint32_t keys(uint32_t word, Fp8ArgFold fold) {
        return RaggedMergePath::raggedF16ArgKeys(word, fold);
    }
    __device__ __forceinline__ static uint32_t fold(const uint32_t (&keys)[4], uint32_t first_field) {
        return RaggedMergePath::raggedF16ArgFold8(keys, first_field);
    }
    __device__ __forceinline__ static uint32_t candidate(uint32_t key_word, uint32_t in_word, uint32_t field2) {
        return __byte_perm(key_word, field2, ((2U * in_word + 1U) << 12) | ((2U * in_word) << 8) | 0x54U);
    }
    __device__ __forceinline__ static uint64_t outputIndex(uint64_t word, uint64_t /*row_base*/, uint64_t components,
                                                           uint64_t component, uint32_t index_bytes) {
        return RaggedMergePath::raggedArgOutputIndex(word, RowBits, components, component, index_bytes);
    }
};

// 1313: FP32 (one kernel for argmin and argmax: the fold's xor_mask; no NaN bias). A key per element (PerWord 1), 4 rows
// per lane, 64-bit tile candidates key << 32 | field << 1 (field < 2^31), and in-kernel words with CTA-local rows
// (RowBits 31: RaggedArg.h's RAGGED_ARG_FP32_LOCAL_ROW_BITS).
struct RaggedArgKeyFp32 {
    static constexpr uint32_t RowBits = RaggedMergePath::RAGGED_ARG_FP32_LOCAL_ROW_BITS;
    static constexpr uint32_t Bytes = 4, Rows = 4, PerWord = 1, KeyBits = 32;
    static constexpr bool LocalRows = true;
    using Storage = uint32_t;
    using Lane = uint64_t;
    __device__ __forceinline__ static uint32_t keys(uint32_t word, Fp8ArgFold fold) {
        return RaggedMergePath::raggedFp32ArgKey(word, fold.xor_mask);
    }
    __device__ __forceinline__ static uint64_t fold(const uint32_t (&keys)[4], uint32_t first_field) {
        return RaggedMergePath::raggedFp32ArgFold4(keys, first_field);
    }
    __device__ __forceinline__ static uint64_t candidate(uint32_t key_word, uint32_t /*in_word*/, uint32_t field2) {
        return (uint64_t{key_word} << 32) | field2;
    }
    __device__ __forceinline__ static uint64_t outputIndex(uint64_t word, uint64_t row_base, uint64_t components,
                                                           uint64_t component, uint32_t index_bytes) {
        return RaggedMergePath::raggedArgLocalOutputIndex(word, row_base, components, component, index_bytes);
    }
};

// 1313: a CTA's pair records (FP32): its key, and component `component`'s head and tail words with global rows.
__device__ __forceinline__ void raggedArgStorePairRecords(uint8_t* workspace, uint32_t grid, uint32_t components,
                                                          uint32_t component, uint64_t head_segment, uint64_t head_word,
                                                          uint64_t tail_word, uint64_t row_base) {
    using namespace RaggedMergePath;
    const RaggedArgPairRecordLayout layout = raggedArgPairRecordLayout(grid, components);
    if (component == 0) reinterpret_cast<uint64_t*>(workspace + layout.keys)[blockIdx.x] = head_segment;
    const uint64_t i = uint64_t{component} * layout.stride + blockIdx.x;
    reinterpret_cast<uint32_t*>(workspace + layout.head_keys)[i] = raggedArgLocalWordKey(head_word);
    reinterpret_cast<uint64_t*>(workspace + layout.head_rows)[i] = raggedArgLocalWordGlobalRow(head_word, row_base);
    reinterpret_cast<uint32_t*>(workspace + layout.tail_keys)[i] = raggedArgLocalWordKey(tail_word);
    reinterpret_cast<uint64_t*>(workspace + layout.tail_rows)[i] = raggedArgLocalWordGlobalRow(tail_word, row_base);
}

// Warp-wide exclusive segmented scan of `mine` seeded with `seed` (raggedWarpExclusiveScan's form for any value type):
// returns seed then every earlier lane's element, `total` seed then every lane's. Call from all 32 lanes.
template <typename T, typename Combine>
__device__ __forceinline__ RaggedMergePath::Carry<T> raggedWarpExclusiveScanOf(const RaggedMergePath::Carry<T>& mine,
                                                                              const RaggedMergePath::Carry<T>& seed,
                                                                              T identity, const Combine& combine,
                                                                              RaggedMergePath::Carry<T>& total) {
    using RaggedMergePath::Carry;
    const uint32_t lane = threadIdx.x & 31U;
    Carry<T> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const int reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, offset);
        const T value = __shfl_up_sync(0xffffffffU, inclusive.value, offset);
        if (lane >= offset) inclusive = RaggedMergePath::raggedCarryCombine(Carry<T>{reset != 0, value}, inclusive, combine);
    }
    const int previous_reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, 1);
    const T previous_value = __shfl_up_sync(0xffffffffU, inclusive.value, 1);
    total = RaggedMergePath::raggedCarryCombine(seed,
                                                Carry<T>{__shfl_sync(0xffffffffU, inclusive.reset ? 1 : 0, 31) != 0,
                                                         __shfl_sync(0xffffffffU, inclusive.value, 31)},
                                                combine);
    return RaggedMergePath::raggedCarryCombine(
        seed, lane == 0 ? Carry<T>{false, identity} : Carry<T>{previous_reset != 0, previous_value}, combine);
}

template <typename Key>
__global__ __launch_bounds__(RAGGED_KR_LANES, RAGGED_KR_MIN_CTAS_PER_SM) void raggedKRArgKernel(
    const RaggedKRArgParams p) {
    using namespace RaggedMergePath;
    constexpr uint32_t Bytes = Key::Bytes;
    constexpr uint32_t Rows = Key::Rows;  // rows per lane per tile: one 16-byte chunk
    constexpr uint32_t PerWord = Key::PerWord;
    using Lane = typename Key::Lane;  // 1313: the tile candidate (32-bit; FP32 64-bit)
    constexpr uint32_t KeyShift = 8 * sizeof(Lane) - Key::KeyBits;
    constexpr uint32_t KeyMask = 0xFFFFFFFFU >> (32 - Key::KeyBits);
    constexpr uint32_t FieldMask = (1U << (KeyShift - 1)) - 1U;
    constexpr uint32_t LaneMask = (1U << Rows) - 1U;
    static_assert(Rows * Bytes == 16 && Rows == 4 * PerWord);
    constexpr uint32_t Lanes = RAGGED_KR_LANES;
    constexpr uint32_t TileRows = Lanes * Rows;
    constexpr uint32_t Ring = RAGGED_KR_RING;
    constexpr uint32_t Mask = Ring - 1;
    constexpr uint32_t EndsPerLane = Ring / Lanes;
    constexpr uint32_t BitWords = TileRows / 32;
    constexpr Lane IdentityLane = ~Lane{0};
    constexpr uint64_t Identity64 = RAGGED_ARG_IDENTITY;
    static_assert(TileRows < FieldMask && Identity64 == FP8_ARG_WIDE_IDENTITY);
    __shared__ uint32_t s_ends[Ring];
    __shared__ Lane s_values[Ring];  // the tile's segment winners (1299: 32-bit; 1313: FP32 64-bit), by their bit's rank
    __shared__ __align__(16) uint8_t s_rows[2][16 * Lanes];
    __shared__ uint32_t s_bits[BitWords];

    const auto lane_min = [](Lane a, Lane b) { return a < b ? a : b; };
    const uint32_t lane = threadIdx.x;
    const RaggedSearchRound0 round0 = raggedSearchRound0Load(p.offsets, p.segments, lane);
    const Problem problem = makeProblem(p.offsets, p.segments, p.capacity_rows);
    const PathWeights weights = raggedEffectiveWeights(problem, p.weights, gridDim.x);
    const uint64_t cost = pathCost(problem, weights);

    const SearchedCoordinate mine = raggedHalfWarpSearchCost(
        problem, weights, ctaDiagonal(cost, gridDim.x, blockIdx.x + (lane >> 4)), lane, round0);
    const auto from_lane = [&](uint32_t source) {
        return SearchedCoordinate{Coordinate{__shfl_sync(0xffffffffU, mine.at.segment, source),
                                             __shfl_sync(0xffffffffU, mine.at.row, source)},
                                  __shfl_sync(0xffffffffU, mine.begin, source),
                                  __shfl_sync(0xffffffffU, mine.end, source)};
    };
    const SearchedCoordinate first = from_lane(0), last = from_lane(16);
    const Coordinate cta_begin = first.at, cta_end = last.at;
    const CtaRole role = raggedCtaRoleFromBounds(problem.segments, first, last);
    const uint64_t head_segment = role.head, tail_segment = role.tail;
    const bool has_head = head_segment != NO_SEGMENT;
    uint64_t head_value = Identity64;  // lane 0's

    const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
    const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
    const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
    const uint32_t end_diagonal = seg_count + row_count;
    // Element of relative row r: elem_base + r from p.bytes (the launcher keeps p.head a multiple of Bytes).
    const uint64_t elem_base = p.head / Bytes + row_base;
    const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
    const uint64_t first_end = seg_count > 0 ? first.end : 0;
    const bool first_has_rows = first_begin < first_end;

    const auto ring = [&](uint32_t segment) -> uint32_t { return s_ends[segment & Mask]; };
    const auto end_rel = [&](uint32_t segment) -> uint32_t {
        return static_cast<uint32_t>(problem.end(seg_base + segment) - row_base);
    };
    const auto fetch_rows = [&](uint32_t y, uint32_t buffer) {
        if (y >= row_count) return;
        const uint64_t origin = (elem_base + y) & ~uint64_t{Rows - 1};
        const uint32_t shift = static_cast<uint32_t>((elem_base + y) - origin);
        const uint32_t limit = y + TileRows - shift < row_count ? y + TileRows - shift : row_count;
        if (Rows * lane < shift + (limit - y))
            raggedCpAsync<16>(&s_rows[buffer][16 * lane], p.bytes + origin * Bytes + 16 * lane);
    };
    const auto load_ends = [&](uint32_t first_index, uint32_t count) {
        uint32_t ends[EndsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < count && i < count) ends[k] = end_rel(first_index + i);
        }
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < count && i < count) s_ends[(first_index + i) & Mask] = ends[k];
        }
    };

    uint32_t x0 = 0, y0 = 0, held = 0, buffer = 0, last_end = 0;
    {
        const uint32_t count = raggedEndsToLoad(0, Ring, seg_count);
        fetch_rows(0, 0);
        raggedCpAsyncCommit();
        load_ends(0, count);
        held = count;
        raggedCpAsyncWaitAll();
    }
    __syncwarp();

    uint64_t carry = Identity64;  // the open segment's winner so far (global rows)
    while (x0 + y0 < end_diagonal) {
        // 1. Ends: top up when the prefetch fell behind.
        if (x0 + held < seg_count && held <= Ring / 2) {
            const uint32_t count = raggedEndsToLoad(held, Ring, seg_count - (x0 + held));
            load_ends(x0 + held, count);
            __syncwarp();
            held += count;
        }
        const bool all_loaded = x0 + held == seg_count;

        // 2. The tile: one aligned span of rows, never past the last loaded end.
        const uint32_t shift = static_cast<uint32_t>((elem_base + y0) & (Rows - 1));
        const uint32_t row_limit = y0 + TileRows - shift < row_count ? y0 + TileRows - shift : row_count;
        uint32_t ends_within = 0;
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            if (k * Lanes >= held || ends_within < k * Lanes) break;
            const uint32_t i = lane + k * Lanes;
            ends_within += __popc(__ballot_sync(0xffffffffU, i < held && ring(x0 + i) <= row_limit));
        }
        const uint32_t last_loaded = held > 0 ? ring(x0 + held - 1) : 0;
        const Coordinate32 tile_end = raggedTileEnd(x0, held, all_loaded, last_loaded, row_limit, ends_within);
        const uint32_t n = tile_end.segment - x0, y1 = tile_end.row;

        // 3. Fetch the next tile's span and the ends after the loaded ones.
        fetch_rows(y1, buffer ^ 1U);
        raggedCpAsyncCommit();
        const uint32_t prefetch_first = x0 + held;
        const uint32_t prefetch_count = raggedEndsToLoad(held - n, Ring, seg_count - prefetch_first);
        uint32_t prefetched[EndsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < prefetch_count && i < prefetch_count) prefetched[k] = end_rel(prefetch_first + i);
        }

        // 4. The bits, with round 0's kinds and slots kept for step 6.
        if (lane < BitWords) s_bits[lane] = 0;
        __syncwarp();
        uint32_t facts0 = raggedPackEndFacts(RaggedEndKind::Empty, 0);
        uint32_t round_previous = last_end;
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            if (k * Lanes >= n) break;
            const uint32_t i = lane + k * Lanes;
            const bool active = i < n;
            const uint32_t e = active ? ring(x0 + i) : 0;
            const uint32_t below = __shfl_up_sync(0xffffffffU, e, 1);
            const uint32_t previous = lane == 0 ? round_previous : below;
            round_previous = __shfl_sync(0xffffffffU, e, Lanes - 1);
            const RaggedEndKind kind = raggedEndKindAfter(x0 + i == 0, first_has_rows, e, previous, y0);
            const bool ranked = active && kind == RaggedEndKind::Ranked;
            const uint32_t ballot = __ballot_sync(0xffffffffU, ranked);
            if (ranked) {
                const uint32_t j = shift + (e - 1 - y0);
                atomicOr(&s_bits[j >> 5], 1U << (j & 31U));
            }
            if (k == 0) facts0 = raggedPackEndFacts(kind, raggedRankedSlot(0, ballot, lane));
        }
        __syncwarp();

        // 5. This lane's 16 rows: their keys (four words), each ended segment's winner to its rank as the lane folds
        // (a minimum of each word's four rows where no end falls), the warp scan seeded with the carry's token, and
        // each lane's first winner completed with the run arriving at it. Span row j is global row tile_base + j
        // (modulo 2^64: the span may start before row 0 of the tensor; its rows there are outside the tile), and its
        // candidate's row field is j + 1 (field 0 is the carry's token).
        const uint32_t rows_in_tile = y1 - y0;
        const uint32_t lane_first = Rows * lane;
        const bool lane_whole = lane_first >= shift && lane_first + Rows <= shift + rows_in_tile;
        const uint64_t tile_base = row_base + y0 - shift;
        // 1313: the words' row base: the tile's global base, or its CTA-local one (FP32; modulo 2^64 as tile_base).
        const uint64_t word_base = Key::LocalRows ? uint64_t{y0} - shift : tile_base;
        const uint32_t bits = s_bits[lane_first >> 5] >> (lane_first & 31U) & LaneMask;
        uint32_t rank_base = __popc(bits);
#pragma unroll
        for (uint32_t offset = 1; offset < Lanes; offset <<= 1) {
            const uint32_t up = __shfl_up_sync(0xffffffffU, rank_base, offset);
            if (lane >= offset) rank_base += up;
        }
        rank_base -= __popc(bits);
        Lane acc;  // the lane's run since its first row or its last end (span rows)
        {
            const uint4 words = *reinterpret_cast<const uint4*>(&s_rows[buffer][16 * lane]);
            // 1301: the lean keys (raggedFp8ArgKeys; 1298-1300: fp8ArgKeys4 with its negative-zero bit).
            uint32_t keys[4] = {Key::keys(words.x, p.fold), Key::keys(words.y, p.fold), Key::keys(words.z, p.fold),
                                Key::keys(words.w, p.fold)};
            if (!lane_whole) {  // the first and last lanes of a tile hold rows outside it: key 0xFF
#pragma unroll
                for (uint32_t k = 0; k < Rows; ++k) {
                    const uint32_t local = lane_first + k;
                    if (local < shift || local >= shift + rows_in_tile)
                        keys[k / PerWord] |= KeyMask << (Key::KeyBits * (k % PerWord));
                }
            }
            if (!__any_sync(0xffffffffU, bits != 0)) {
                acc = Key::fold(keys, lane_first + 1);  // 1301 (1299-1300: four fp8ArgRowsCandidate4)
            } else {
                // 1299: row k's candidate in two instructions, its row field and one byte permute putting its key byte
                // on top (no negative-zero bit: it never decides a comparison and the index does not use it).
                const uint32_t field_base = 2U * (lane_first + 1U);
                const auto value = [&](uint32_t k) -> Lane {
                    return Key::candidate(keys[k / PerWord], k % PerWord, field_base + 2U * k);
                };
                // An emitted winner holds the row of its end's bit, a row of the tile, so never the marker key.
                acc = raggedLaneEmit<Rows>(bits, IdentityLane, IdentityLane, lane_min, value,
                                           [&](uint32_t k, Lane winner) {
                                               s_values[rank_base + raggedBitsBelow(bits, k)] = winner;
                                           });
            }
            if ((acc >> KeyShift) == KeyMask) acc = IdentityLane;  // only rows outside the tile
        }
        const bool reset_at_start = n > 0 && ring(x0) == y0;  // an end before the tile's rows closes the open run
        const uint64_t carry_in = reset_at_start ? Identity64 : carry;
        // The carry's token: its key with row field 0, below every row of the tile.
        const Lane token = carry_in == Identity64
                               ? IdentityLane
                               : (static_cast<Lane>(carry_in >> (Key::RowBits + 1)) & KeyMask) << KeyShift;
        // A tile winner as the 64-bit word: the carry itself for field 0, else row tile_base + field - 1.
        const auto widen = [&](Lane winner) -> uint64_t {  // 1313: from word_base (FP32: CTA-local rows)
            if (winner == IdentityLane) return Identity64;
            const uint32_t field = static_cast<uint32_t>((winner >> 1) & FieldMask);
            if (field == 0) return carry_in;
            return (uint64_t{winner >> KeyShift} << (Key::RowBits + 1)) | ((word_base + field - 1U) << 1) |
                   (winner & 1U);
        };
        Carry<Lane> total;
        const Carry<Lane> before = raggedWarpExclusiveScanOf(Carry<Lane>{bits != 0, acc}, Carry<Lane>{false, token},
                                                             IdentityLane, lane_min, total);
        if (bits != 0) s_values[rank_base] = lane_min(before.value, s_values[rank_base]);
        const uint64_t next_carry = widen(total.value);
        __syncwarp();

        // 6. One lane per end: its winner's index (the sentinel without rows), in segment order; the head to its record.
        uint32_t ranked_earlier = 0;
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            if (k * Lanes >= n) break;
            const uint32_t i = lane + k * Lanes;
            const bool active = i < n;
            RaggedEndKind kind;
            uint32_t slot;
            if (k == 0) {
                kind = raggedPackedKind(facts0);
                slot = raggedPackedSlot(facts0);
                if (n > Lanes)
                    ranked_earlier = __popc(__ballot_sync(0xffffffffU, active && kind == RaggedEndKind::Ranked));
            } else {
                const uint32_t e = active ? ring(x0 + i) : 0;
                const uint32_t previous = ring(x0 + i - 1);  // i >= 32
                kind = raggedEndKindAfter(false, first_has_rows, e, previous, y0);
                const uint32_t ballot = __ballot_sync(0xffffffffU, active && kind == RaggedEndKind::Ranked);
                slot = raggedRankedSlot(ranked_earlier, ballot, lane);
                ranked_earlier += __popc(ballot);
            }
            if (active) {
                const uint64_t winner = kind == RaggedEndKind::Ranked
                                            ? widen(s_values[slot])
                                            : (kind == RaggedEndKind::CarryIn ? carry : Identity64);
                if (x0 + i == 0 && has_head) {
                    head_value = winner;
                } else {
                    raggedArgStoreIndex(p.output, p.index_bytes, seg_base + x0 + i,
                                        Key::outputIndex(winner, row_base, 1, 0, p.index_bytes));
                }
            }
        }

        // 7. Advance.
        if (n > 0) last_end = ring(x0 + n - 1);
        carry = next_carry;
        __syncwarp();
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < prefetch_count && i < prefetch_count) s_ends[(prefetch_first + i) & Mask] = prefetched[k];
        }
        held += prefetch_count;
        held -= n;
        x0 = tile_end.segment;
        y0 = y1;
        buffer ^= 1U;
        raggedCpAsyncWaitAll();
        __syncwarp();
    }

    if (lane == 0) {
        if constexpr (Key::LocalRows) {  // 1313: FP32's pair records
            raggedArgStorePairRecords(p.workspace, gridDim.x, 1, 0, head_segment, head_value,
                                      tail_segment != NO_SEGMENT ? carry : Identity64, row_base);
        } else {
            const RaggedArgRecordLayout layout = raggedArgRecordLayout(gridDim.x, 1);
            reinterpret_cast<uint64_t*>(p.workspace + layout.keys)[blockIdx.x] = head_segment;
            reinterpret_cast<uint64_t*>(p.workspace + layout.heads)[blockIdx.x] = head_value;
            reinterpret_cast<uint64_t*>(p.workspace + layout.tails)[blockIdx.x] =
                tail_segment != NO_SEGMENT ? carry : Identity64;
        }
    }
}

// 1300: the ARG twin of raggedWarpExclusiveScanAcrossWorkers (RaggedReductionDevice.cuh) for any value type: each
// component's elements scanned across the `workers` workers of `components` lanes each, seeded with `seed`.
template <typename T, typename Combine>
__device__ __forceinline__ RaggedMergePath::Carry<T> raggedWarpExclusiveScanAcrossWorkersOf(
    const RaggedMergePath::Carry<T>& mine, const RaggedMergePath::Carry<T>& seed, T identity, const Combine& combine,
    uint32_t components, uint32_t workers, uint32_t worker, uint32_t component, RaggedMergePath::Carry<T>& total) {
    using RaggedMergePath::Carry;
    using RaggedMergePath::raggedCarryCombine;
    Carry<T> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 16; offset <<= 1) {
        if (offset >= workers) break;
        const uint32_t delta = offset * components;
        const int reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, delta);
        const T value = __shfl_up_sync(0xffffffffU, inclusive.value, delta);
        if (worker >= offset) inclusive = raggedCarryCombine(Carry<T>{reset != 0, value}, inclusive, combine);
    }
    const int previous_reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, components);
    const T previous_value = __shfl_up_sync(0xffffffffU, inclusive.value, components);
    const Carry<T> worker_exclusive =
        worker == 0 ? Carry<T>{false, identity} : Carry<T>{previous_reset != 0, previous_value};
    const uint32_t last_lane = (workers - 1) * components + component;
    const int last_reset = __shfl_sync(0xffffffffU, inclusive.reset ? 1 : 0, last_lane);
    const T last_value = __shfl_sync(0xffffffffU, inclusive.value, last_lane);
    total = raggedCarryCombine(seed, Carry<T>{last_reset != 0, last_value}, combine);
    return raggedCarryCombine(seed, worker_exclusive, combine);
}

// 1300: FP8 KR ARG at D = 2..15 (rows under 16 bytes), the ARG twin of raggedKRComponentsKernel (RaggedKRKernels.cuh,
// with its walk: lane = worker * D + component, worker w folds component `component` of the tile's rows
// 16w .. 16w + 15, the tile's ends' kinds and slots kept in s_facts, `workers` ends per store round). The values are
// raggedKRArgKernel's (1299 form): a 32-bit candidate per row, key << 24 | (tile row + 1) << 1 | negative zero, the
// carry entering as its key's token (field 0), the scan across workers seeded with it, 32-bit value slots
// (slot * D + component), and the 64-bit word only at the stores, the head record and the carry. Four rows of one
// component make one key word (three byte permutes, as the VALUE kernel builds its decode word).
template <typename Key>
__global__ __launch_bounds__(RAGGED_KR_LANES, RAGGED_KR_MIN_CTAS_PER_SM) void raggedKRArgComponentsKernel(
    const RaggedKRArgParams p) {
    using namespace RaggedMergePath;
    constexpr uint32_t Lanes = RAGGED_KR_LANES;
    constexpr uint32_t Bytes = Key::Bytes;
    constexpr uint32_t Rows = Key::Rows;  // rows per worker per tile: 16 bytes of each component
    constexpr uint32_t PerWord = Key::PerWord;
    using Lane = typename Key::Lane;  // 1313: the tile candidate (32-bit; FP32 64-bit)
    constexpr uint32_t KeyShift = 8 * sizeof(Lane) - Key::KeyBits;
    constexpr uint32_t KeyMask = 0xFFFFFFFFU >> (32 - Key::KeyBits);
    constexpr uint32_t FieldMask = (1U << (KeyShift - 1)) - 1U;
    constexpr uint32_t LaneMask = (1U << Rows) - 1U;
    using Storage = typename Key::Storage;
    static_assert(Rows * Bytes == 16 && Rows == 4 * PerWord);
    constexpr uint32_t MaxTileRows = Rows * (Lanes / 2);
    constexpr uint32_t SpanBytes = 16 * (Lanes + 1);
    constexpr uint32_t Ring = RAGGED_KR_RING;
    constexpr uint32_t Mask = Ring - 1;
    constexpr uint32_t EndsPerLane = Ring / Lanes;
    constexpr uint32_t BitWords = MaxTileRows / 32;
    constexpr uint32_t Slots = RAGGED_KR_COMPONENT_VALUE_SLOTS;
    constexpr Lane IdentityLane = ~Lane{0};
    constexpr uint64_t Identity64 = RAGGED_ARG_IDENTITY;
    static_assert(MaxTileRows < FieldMask && Identity64 == FP8_ARG_WIDE_IDENTITY);
    __shared__ uint32_t s_ends[Ring];
    __shared__ Lane s_values[Slots];  // the tile's winners: slot * D + component
    __shared__ uint16_t s_facts[Ring];
    __shared__ __align__(16) uint8_t s_rows[2][SpanBytes];
    __shared__ uint32_t s_bits[BitWords];

    const auto lane_min = [](Lane a, Lane b) { return a < b ? a : b; };
    const uint32_t lane = threadIdx.x;
    const uint32_t D = p.components, workers = p.workers;
    const uint32_t worker = lane / D;
    const uint32_t component = lane - worker * D;
    const bool lane_active = worker < workers;
    const uint32_t tile_rows = Rows * workers;
    const RaggedSearchRound0 round0 = raggedSearchRound0Load(p.offsets, p.segments, lane);
    const Problem problem = makeProblem(p.offsets, p.segments, p.capacity_rows);
    const PathWeights weights = raggedEffectiveWeights(problem, p.weights, gridDim.x);
    const uint64_t cost = pathCost(problem, weights);

    const SearchedCoordinate mine = raggedHalfWarpSearchCost(
        problem, weights, ctaDiagonal(cost, gridDim.x, blockIdx.x + (lane >> 4)), lane, round0);
    const auto from_lane = [&](uint32_t source) {
        return SearchedCoordinate{Coordinate{__shfl_sync(0xffffffffU, mine.at.segment, source),
                                             __shfl_sync(0xffffffffU, mine.at.row, source)},
                                  __shfl_sync(0xffffffffU, mine.begin, source),
                                  __shfl_sync(0xffffffffU, mine.end, source)};
    };
    const SearchedCoordinate first = from_lane(0), last = from_lane(16);
    const Coordinate cta_begin = first.at, cta_end = last.at;
    const CtaRole role = raggedCtaRoleFromBounds(problem.segments, first, last);
    const uint64_t head_segment = role.head, tail_segment = role.tail;
    const bool has_head = head_segment != NO_SEGMENT;
    uint64_t head_value = Identity64;  // the worker-0 lanes', one per component

    const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
    const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
    const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
    const uint32_t end_diagonal = seg_count + row_count;
    const uint32_t row_bytes = D * Bytes;
    const uint64_t byte_base = p.head + row_base * row_bytes;
    const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
    const uint64_t first_end = seg_count > 0 ? first.end : 0;
    const bool first_has_rows = first_begin < first_end;

    const auto ring = [&](uint32_t segment) -> uint32_t { return s_ends[segment & Mask]; };
    const auto end_rel = [&](uint32_t segment) -> uint32_t {
        return static_cast<uint32_t>(problem.end(seg_base + segment) - row_base);
    };
    const auto fetch_rows = [&](uint32_t y, uint32_t buffer) {
        if (y >= row_count) return;
        const uint64_t start = byte_base + uint64_t{y} * row_bytes;
        const uint64_t origin = start & ~uint64_t{15};
        const uint32_t shift = static_cast<uint32_t>(start - origin);
        const uint32_t limit = y + tile_rows < row_count ? y + tile_rows : row_count;
        const uint32_t bytes = shift + (limit - y) * row_bytes;
        if (16 * lane < bytes) raggedCpAsync<16>(&s_rows[buffer][16 * lane], p.bytes + origin + 16 * lane);
        if (lane == 0 && 16 * Lanes < bytes)
            raggedCpAsync<16>(&s_rows[buffer][16 * Lanes], p.bytes + origin + 16 * Lanes);
    };
    const auto load_ends = [&](uint32_t first_end_index, uint32_t count) {
        uint32_t ends[EndsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < count && i < count) ends[k] = end_rel(first_end_index + i);
        }
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < count && i < count) s_ends[(first_end_index + i) & Mask] = ends[k];
        }
    };

    uint32_t x0 = 0, y0 = 0, held = 0, buffer = 0, last_end = 0;
    {
        const uint32_t count = raggedEndsToLoad(0, Ring, seg_count);
        fetch_rows(0, 0);
        raggedCpAsyncCommit();
        load_ends(0, count);
        held = count;
        raggedCpAsyncWaitAll();
    }
    __syncwarp();

    uint64_t carry = Identity64;  // the open segment's winner so far, this lane's component (global rows)
    while (x0 + y0 < end_diagonal) {
        // 1. Ends: top up when the prefetch fell behind.
        if (x0 + held < seg_count && held <= Ring / 2) {
            const uint32_t count = raggedEndsToLoad(held, Ring, seg_count - (x0 + held));
            load_ends(x0 + held, count);
            __syncwarp();
            held += count;
        }
        const bool all_loaded = x0 + held == seg_count;

        // 2. The tile: Rows x workers rows, never past the last loaded end, at most ends_cap ends.
        const uint32_t row_limit = y0 + tile_rows < row_count ? y0 + tile_rows : row_count;
        uint32_t ends_within = 0;
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            if (k * Lanes >= held || ends_within < k * Lanes) break;
            const uint32_t i = lane + k * Lanes;
            ends_within += __popc(__ballot_sync(0xffffffffU, i < held && ring(x0 + i) <= row_limit));
        }
        const RaggedTileEnds capped = raggedCapTileEnds(held, all_loaded, ends_within, p.ends_cap);
        const uint32_t last_loaded = capped.held > 0 ? ring(x0 + capped.held - 1) : 0;
        const Coordinate32 tile_end =
            raggedTileEnd(x0, capped.held, capped.all_loaded, last_loaded, row_limit, capped.ends_within);
        const uint32_t n = tile_end.segment - x0, y1 = tile_end.row;

        // 3. Prefetch the next tile's bytes and the ends after the loaded ones.
        fetch_rows(y1, buffer ^ 1U);
        raggedCpAsyncCommit();
        const uint32_t prefetch_first = x0 + held;
        const uint32_t prefetch_count = raggedEndsToLoad(held - n, Ring, seg_count - prefetch_first);
        uint32_t prefetched[EndsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < prefetch_count && i < prefetch_count) prefetched[k] = end_rel(prefetch_first + i);
        }

        // 4. The bits, and every end's kind and slot (s_facts), in rounds of 32 ends.
        if (lane < BitWords) s_bits[lane] = 0;
        __syncwarp();
        uint32_t round_previous = last_end;
        uint32_t ranked_earlier = 0;
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            if (k * Lanes >= n) break;
            const uint32_t i = lane + k * Lanes;
            const bool active = i < n;
            const uint32_t e = active ? ring(x0 + i) : 0;
            const uint32_t below = __shfl_up_sync(0xffffffffU, e, 1);
            const uint32_t previous = lane == 0 ? round_previous : below;
            round_previous = __shfl_sync(0xffffffffU, e, Lanes - 1);
            const RaggedEndKind kind = raggedEndKindAfter(x0 + i == 0, first_has_rows, e, previous, y0);
            const bool ranked = active && kind == RaggedEndKind::Ranked;
            const uint32_t ballot = __ballot_sync(0xffffffffU, ranked);
            if (ranked) {
                const uint32_t j = e - 1 - y0;
                atomicOr(&s_bits[j >> 5], 1U << (j & 31U));
            }
            if (active)
                s_facts[i] = static_cast<uint16_t>(raggedPackEndFacts16(kind, raggedRankedSlot(ranked_earlier, ballot, lane)));
            ranked_earlier += __popc(ballot);
        }
        __syncwarp();

        // 5. This lane's component of its worker's 16 rows: four key words (each from four rows' bytes), each ended
        // segment's winner to its slot as the lane folds, the scan across workers seeded with the carry's token, and
        // each worker's first winner completed with the run arriving at it. Tile row r is global row
        // row_base + y0 + r, its candidate's row field r + 1 (field 0 is the token).
        const uint32_t rows_in_tile = y1 - y0;
        const uint32_t shift = static_cast<uint32_t>((byte_base + uint64_t{y0} * row_bytes) & 15U);
        const uint32_t first_row = Rows * worker;
        const uint32_t rows_valid =
            !lane_active || first_row >= rows_in_tile ? 0 : (rows_in_tile - first_row < Rows ? rows_in_tile - first_row : Rows);
        const uint32_t bits = lane_active ? (s_bits[first_row >> 5] >> (first_row & 31U)) & LaneMask : 0;
        uint32_t rank_inclusive = component == 0 ? static_cast<uint32_t>(__popc(bits)) : 0;
#pragma unroll
        for (uint32_t offset = 1; offset < Lanes; offset <<= 1) {
            const uint32_t up = __shfl_up_sync(0xffffffffU, rank_inclusive, offset);
            if (lane >= offset) rank_inclusive += up;
        }
        const uint32_t rank_base = __shfl_sync(0xffffffffU, rank_inclusive, worker * D) - __popc(bits);
        const uint64_t tile_base = row_base + y0;
        const uint64_t word_base = Key::LocalRows ? uint64_t{y0} : tile_base;  // 1313: FP32's CTA-local rows
        // Elements of Bytes bytes (the launcher keeps the input aligned to its element, so shift, row_bytes and the
        // component offset are multiples of Bytes), PerWord per key word.
        const Storage* element = reinterpret_cast<const Storage*>(
            &s_rows[buffer][lane_active ? shift + first_row * row_bytes + component * Bytes : 0]);
        uint32_t keys[4];
#pragma unroll
        for (uint32_t g = 0; g < 4; ++g) {
            uint32_t word;
            if constexpr (Bytes == 1) {
                const uint32_t b0 = element[0], b1 = element[D], b2 = element[2 * D], b3 = element[3 * D];
                word = __byte_perm(__byte_perm(b0, b1, 0x0040U), __byte_perm(b2, b3, 0x0040U), 0x5410U);
            } else if constexpr (Bytes == 2) {
                word = __byte_perm(uint32_t{element[0]}, uint32_t{element[D]}, 0x5410U);
            } else {  // 1313: FP32, one element per key word
                word = element[0];
            }
            element += PerWord * D;
            keys[g] = Key::keys(word, p.fold);  // 1301
        }
        if (__any_sync(0xffffffffU, lane_active && rows_valid < Rows)) {  // warp-uniform: the tile's edge workers
#pragma unroll
            for (uint32_t k = 0; k < Rows; ++k) {
                if (k >= rows_valid) keys[k / PerWord] |= KeyMask << (Key::KeyBits * (k % PerWord));
            }
        }
        Lane acc;
        if (!__any_sync(0xffffffffU, bits != 0)) {
            acc = Key::fold(keys, first_row + 1);  // 1301
        } else {
            const uint32_t field_base = 2U * (first_row + 1U);
            const auto value = [&](uint32_t k) -> Lane {
                return Key::candidate(keys[k / PerWord], k % PerWord, field_base + 2U * k);
            };
            acc = raggedLaneEmit<Rows>(bits, IdentityLane, IdentityLane, lane_min, value, [&](uint32_t k, Lane winner) {
                s_values[(rank_base + raggedBitsBelow(bits, k)) * D + component] = winner;
            });
        }
        if ((acc >> KeyShift) == KeyMask) acc = IdentityLane;  // only rows outside the tile
        const bool reset_at_start = n > 0 && ring(x0) == y0;
        const uint64_t carry_in = reset_at_start ? Identity64 : carry;
        const Lane token = carry_in == Identity64
                               ? IdentityLane
                               : (static_cast<Lane>(carry_in >> (Key::RowBits + 1)) & KeyMask) << KeyShift;
        const auto widen = [&](Lane winner) -> uint64_t {  // 1313: from word_base (FP32: CTA-local rows)
            if (winner == IdentityLane) return Identity64;
            const uint32_t field = static_cast<uint32_t>((winner >> 1) & FieldMask);
            if (field == 0) return carry_in;
            return (uint64_t{winner >> KeyShift} << (Key::RowBits + 1)) | ((word_base + field - 1U) << 1) |
                   (winner & 1U);
        };
        Carry<Lane> total;
        const Carry<Lane> before = raggedWarpExclusiveScanAcrossWorkersOf(
            Carry<Lane>{bits != 0, acc}, Carry<Lane>{false, token}, IdentityLane, lane_min, D, workers, worker,
            component, total);
        if (bits != 0) {
            const uint32_t slot = rank_base * D + component;
            s_values[slot] = lane_min(before.value, s_values[slot]);
        }
        const uint64_t next_carry = widen(total.value);
        __syncwarp();

        // 6. `workers` ends per round, one lane per component: the winner's index (the head to its record).
        for (uint32_t round_first = 0; round_first < n; round_first += workers) {
            const uint32_t i = round_first + worker;
            if (lane_active && i < n) {
                const uint32_t facts = s_facts[i];
                const RaggedEndKind kind = raggedPacked16Kind(facts);
                const uint64_t winner = kind == RaggedEndKind::Ranked
                                            ? widen(s_values[raggedPacked16Slot(facts) * D + component])
                                            : (kind == RaggedEndKind::CarryIn ? carry : Identity64);
                if (x0 + i == 0 && has_head) {
                    head_value = winner;  // the worker-0 lanes
                } else {
                    raggedArgStoreIndex(p.output, p.index_bytes, (seg_base + x0 + i) * D + component,
                                        Key::outputIndex(winner, row_base, D, component, p.index_bytes));
                }
            }
        }

        // 7. Advance.
        if (n > 0) last_end = ring(x0 + n - 1);
        carry = next_carry;
        __syncwarp();
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < prefetch_count && i < prefetch_count) s_ends[(prefetch_first + i) & Mask] = prefetched[k];
        }
        held += prefetch_count;
        held -= n;
        x0 = tile_end.segment;
        y0 = y1;
        buffer ^= 1U;
        raggedCpAsyncWaitAll();
        __syncwarp();
    }

    if constexpr (Key::LocalRows) {  // 1313: FP32's pair records
        if (lane < D)  // worker 0: component `lane`
            raggedArgStorePairRecords(p.workspace, gridDim.x, D, component, head_segment, head_value,
                                      tail_segment != NO_SEGMENT ? carry : Identity64, row_base);
    } else {
        const RaggedArgRecordLayout layout = raggedArgRecordLayout(gridDim.x, D);
        if (lane == 0) reinterpret_cast<uint64_t*>(p.workspace + layout.keys)[blockIdx.x] = head_segment;
        if (lane < D) {  // worker 0: component `lane`
            const uint64_t i = uint64_t{component} * gridDim.x + blockIdx.x;
            reinterpret_cast<uint64_t*>(p.workspace + layout.heads)[i] = head_value;
            reinterpret_cast<uint64_t*>(p.workspace + layout.tails)[i] = tail_segment != NO_SEGMENT ? carry : Identity64;
        }
    }
}

// The FP8 object's exports (CubRaggedKRArgFp8.cu) to the launcher (CubRaggedKRArg.cu). 1300: D = 1 runs
// raggedKRArgKernel, D = 2..15 raggedKRArgComponentsKernel (their residency differs).
void raggedKRArgLaunchMainFp8(uint32_t grid, const RaggedKRArgParams& params, cudaStream_t stream);
[[nodiscard]] int raggedKRArgResidentFp8(uint32_t components);
// 1311: the FP16 / BF16 object's (CubRaggedKRArgF16.cu), D 1 to 7.
void raggedKRArgLaunchMainF16(uint32_t grid, const RaggedKRArgParams& params, cudaStream_t stream);
[[nodiscard]] int raggedKRArgResidentF16(uint32_t components);
// 1313: the FP32 object's (CubRaggedKRArgFp32.cu), D 1 to 3; its records are pairs (raggedArgPairRecordLayout).
void raggedKRArgLaunchMainFp32(uint32_t grid, const RaggedKRArgParams& params, cudaStream_t stream);
[[nodiscard]] int raggedKRArgResidentFp32(uint32_t components);

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
