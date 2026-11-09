#pragma once

// 1284 (phase 3b): the retained ragged KR VALUE main kernels for every input dtype, moved from
// CubRaggedKRFp8.cu (1265-1283) and templated on the input element: one 16-byte chunk per lane (D = 1) or per worker
// (D >= 2) and tile, Rows = 16 / element bytes rows, a word of 4 / element bytes elements per decode. Instantiated by
// CubRaggedKRFp8.cu (E4M3, E5M2), CubRaggedKRF16.cu (FP16, BF16) and CubRaggedKRFp32.cu (FP32), one object per dtype
// group so they build in parallel; CubRaggedKR.cu holds the host launcher. The FP8 kernels' history:
//
// 1265: retained ragged KR VALUE for original FP8 input at D = 1 (plans/Thor_Ragged_Reduction_Plan_2026-11-06.md,
// phase 2). One persistent CTA per merge-path range (RaggedMergePath.h) walks its range in tiles (RaggedKRWalk.h):
// the segment ends in a shared-memory ring, the tile's rows staged in shared memory (any input address), one search
// per thread, a block-wide segmented scan, and the boundary records the finisher (CubRaggedFinish.cu) completes. MIN
// and MAX share the minimum kernel through the FP8 sign fold (P5-F); MEAN is the additive kernel with a per-segment
// divide. Kernels: 2 formats x {additive, minimum} x 8 / 16 items per thread.
//
// 1268 (the first sweep: 0.55 TB/s median, a 1.12x / 1.37x p50 / p90 spread):
//   - the CTA split weighs rows and ends by the bytes they move (RaggedMergePath::PathWeights);
//   - 32-bit coordinates relative to the CTA, a power-of-two ends ring (a mask, no 64-bit division), runs of rows
//     between ends decoded four at a time from shared memory;
//   - the ends load in bulk (one round trip), and the next tile's rows (cp.async into the other of two buffers) and
//     ends (registers, into the ring's free slots) are fetched while the current tile computes.
//
// 1270 (the 1268 profile: compute-bound at 36-150 thread instructions per row, segment ends serialized across lanes,
// 100 registers capping i16 at 2 CTAs per SM): one warp per CTA (about 24 per SM, the finisher takes up to 4096), and
// row-major tiles: each lane folds one aligned chunk of Rows rows (1271: 16 or 32, in pieces of 16; the ends loops skip
// their unused rounds; the ring holds 128 ends), a bitmap marks the tile's segment ends, a warp scan
// carries the open run, values go to slots ranked by their bit, and one lane per end finalizes and stores in segment
// order (RaggedKRWalk.h). No __syncthreads, no per-thread search, no per-event branches. Kernels: 2 formats x
// {additive, minimum} x 16 / 32 rows per lane.
//
// 1272 (the 1271 sweep: 0.94 TB/s median; a cost fit put about 15 us of fixed time on every cell, the rows at 1.48 TB/s
// marginal, and an end at 17-21 row-equivalents at 32 rows per lane against its 8-12 bytes of weight):
//   - start-up: the search's last round reads each candidate's begin and end, so the role and the first segment come
//     from it (two dependent round trips fewer); the finisher reads every record before any store (it read each head
//     behind the previous store);
//   - one pass over the rows: values go to their ranks as the lane folds (the ranks scan first), no second decode;
//   - the span copies read consecutive chunks, placed piece-major in shared memory (1271's ncu: every sector read
//     twice, 2-way bank conflicts on the 32-row lanes' reads);
//   - an end's split weight adds its work beyond its bytes (RaggedKRShape::end_extra).
//
// 1273 (the 1272 sweep: 1.035 TB/s median, i16 best in 264 of 300 cells; nsys on 1 MiB cells: main kernel 4.3-5.2 us,
// finisher 7.5-7.7 us, no gap between them):
//   - one 16-byte record per CTA (RaggedKRRecord: head segment, head value, tail value), stored by lane 0 at the end;
//     the finisher reads them coalesced (CubRaggedFinish.cu);
//   - one fold path per warp and piece (a warp with an end anywhere in the piece emits in every lane);
//   - the span copies write consecutive shared slots (1272's order conflicted at 32 rows per lane);
//   - the end's extra weight defaults to 4 (the 1272 fit).
//
// 1274 (the 1273 sweep: 1.14 TB/s median, 1.44 at >= 128 MiB; nsys on 1 MiB cells: main kernel 3.8-4.7 us, finisher
// 2.0 us at 1360 CTAs and 3.9 at 4080): the search starts from [0, segments), so its first round's offsets are read
// with the row count (raggedSearchRound0Load); the finisher scans once (CubRaggedFinish.cu).
//
// 1275 (the 1274 sweep: 1.20 TB/s median, 0.77 of copy; the 1273 profile: about 580 warp instructions per 512-row tile
// with ends against 380 without): each end's kind and value slot are found once, in the bits pass (step 4), and kept
// for the store pass (step 6), which recomputed them from the ring and found each slot by two shuffles from its bit's
// lane; the slot is now the count of Ranked ends before the end (a ballot), and the end before an end comes from the
// lane below (a shuffle, not a shared read). The head is the CTA's first end (a 32-bit test, not a 64-bit compare).
//
// 1280 (phase 3: FP8 at D = 2..15, rows under 16 bytes; plans/Thor_Ragged_Reduction_Plan_2026-11-06.md):
//   - raggedKRFp8ComponentsKernel, one kernel for every D >= 2 (D a runtime operand): a lane is one component of one
//     worker (RaggedKRWalk.h), a worker folds 16 consecutive rows, so the tile's ends, bitmap, ranks and kinds are the
//     D = 1 kernel's per row and only the values are D wide. Kernels: 2 formats x {additive, minimum}.
//   - the boundary records are three arrays (raggedRecordLayout: keys, head values, tail values), and the finisher runs
//     one CTA per component;
//   - the 32-rows-per-lane D = 1 kernels are gone (they won 3 cells from 1273 on): 2 formats x {additive, minimum}.
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"
#include "Utilities/TensorOperations/Cub/RaggedReductionDevice.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

using RaggedMergePath::Carry;
using RaggedMergePath::Coordinate;
using RaggedMergePath::Coordinate32;
using RaggedMergePath::CtaRole;
using RaggedMergePath::Problem;

inline void raggedKRCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Ragged KR: ") + what + ": " + cudaGetErrorString(status));
    }
}

// The main kernels' parameters (one struct for every dtype; 1265-1283: RaggedKRFp8Params).
struct RaggedKRParams {
    const uint8_t* bytes;  // the input's bytes, from its 16-byte aligned base
    uint32_t head;         // row 0's byte offset from `bytes` (1284: a multiple of the element's bytes)
    RaggedMergePath::Offsets offsets;
    uint64_t segments;
    uint64_t capacity_rows;
    RaggedMergePath::PathWeights weights;  // requested (the kernel falls back to unit weights for tiny problems)
    Fp8ExtremumFold fold;                  // MAX: the sign flip of every element in a word; otherwise none
    RaggedFinalize finalize;
    void* output;
    DataType output_dtype;
    uint8_t* workspace;  // 1280: raggedRecordLayout(grid, components) (1273-1279: one 16-byte record per CTA)
    uint32_t components;  // 1280: D (the components kernel; 1 for the D = 1 kernel)
    uint32_t workers;     // raggedComponentWorkers(components)
    uint32_t ends_cap;    // raggedComponentEndsCap(components, RAGGED_KR_RING)
};

// 1284: the input element: its bytes, rows per 16-byte chunk, elements per 32-bit word, and its storage word.
template <typename InputT>
struct RaggedKRInput;
template <>
struct RaggedKRInput<__half> {
    static constexpr uint32_t Bytes = 2;
    using Storage = uint16_t;
    static constexpr uint32_t Rows = 16 / Bytes, PerWord = 4 / Bytes;
};
template <>
struct RaggedKRInput<__nv_bfloat16> {
    static constexpr uint32_t Bytes = 2;
    using Storage = uint16_t;
    static constexpr uint32_t Rows = 16 / Bytes, PerWord = 4 / Bytes;
};
template <>
struct RaggedKRInput<float> {
    static constexpr uint32_t Bytes = 4;
    using Storage = uint32_t;
    static constexpr uint32_t Rows = 16 / Bytes, PerWord = 4 / Bytes;
};
#if THOR_CUB_ENABLE_FP8_TYPES
template <>
struct RaggedKRInput<__nv_fp8_e4m3> {
    static constexpr uint32_t Bytes = 1;
    using Storage = uint8_t;
    static constexpr uint32_t Rows = 16 / Bytes, PerWord = 4 / Bytes;
};
template <>
struct RaggedKRInput<__nv_fp8_e5m2> {
    static constexpr uint32_t Bytes = 1;
    using Storage = uint8_t;
    static constexpr uint32_t Rows = 16 / Bytes, PerWord = 4 / Bytes;
};
#endif

// 1284: one 32-bit word (already sign-folded for MAX) into its PerWord FP32 values, exactly, in element order.
template <typename InputT>
__device__ __forceinline__ void raggedKRDecodeWord(uint32_t word, float* out) {
    if constexpr (std::is_same_v<InputT, float>) {
        out[0] = __uint_as_float(word);
    } else if constexpr (std::is_same_v<InputT, __half>) {
        out[0] = __half2float(__ushort_as_half(static_cast<unsigned short>(word & 0xffffU)));
        out[1] = __half2float(__ushort_as_half(static_cast<unsigned short>(word >> 16)));
    } else if constexpr (std::is_same_v<InputT, __nv_bfloat16>) {
        out[0] = __uint_as_float(word << 16);
        out[1] = __uint_as_float(word & 0xffff0000U);
    } else {
#if THOR_CUB_ENABLE_FP8_TYPES
        const float4 d = decodeFp8x4ToFp32<InputT>(word);
        out[0] = d.x;
        out[1] = d.y;
        out[2] = d.z;
        out[3] = d.w;
#endif
    }
}

// cp.async of one lane's chunk (8 or 16 bytes, aligned on both sides).
template <uint32_t Bytes>
__device__ __forceinline__ void raggedCpAsync(void* shared, const void* global) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared));
    if constexpr (Bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global) : "memory");
    } else {
        static_assert(Bytes == 8);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(address), "l"(global) : "memory");
    }
}

__device__ __forceinline__ void raggedCpAsyncCommit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

__device__ __forceinline__ void raggedCpAsyncWaitAll() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// 1270: one warp per CTA (RaggedKRWalk.h has the tile logic and the host emulation's shared helpers). 1284: D = 1 for
// every dtype; a lane owns one 16-byte chunk of the tile's span, Rows = 16 / element bytes rows (16 FP8, 8 FP16 / BF16,
// 4 FP32), and everything below counts elements where the FP8 kernel counted bytes.
template <typename InputT, int FixedOp>
__global__ __launch_bounds__(RAGGED_KR_LANES, RAGGED_KR_MIN_CTAS_PER_SM) void raggedKRValueKernel(
    const RaggedKRParams p) {
    using namespace RaggedMergePath;
    constexpr uint32_t Bytes = RaggedKRInput<InputT>::Bytes;
    constexpr uint32_t Rows = RaggedKRInput<InputT>::Rows;     // elements per lane per tile
    constexpr uint32_t PerWord = RaggedKRInput<InputT>::PerWord;
    constexpr uint32_t Lanes = RAGGED_KR_LANES;
    constexpr uint32_t TileRows = Lanes * Rows;  // one aligned span of 512 bytes per tile
    constexpr uint32_t Ring = RAGGED_KR_RING;    // ends slots (and value slots: a tile's ends are loaded ones)
    constexpr uint32_t Mask = Ring - 1;
    constexpr uint32_t EndsPerLane = Ring / Lanes;
    constexpr uint32_t BitWords = TileRows / 32;
    constexpr uint32_t LaneMask = Rows >= 32 ? 0xffffffffU : ((1U << (Rows & 31U)) - 1U);
    static_assert(FixedOp == 0 || FixedOp == 2);
    static_assert(Rows == 4 || Rows == 8 || Rows == 16);
    static_assert((Ring & Mask) == 0 && Ring % Lanes == 0);
    __shared__ uint32_t s_ends[Ring];  // segment ends relative to the CTA's first row
    __shared__ float s_values[Ring];   // the tile's segment values, by the rank of their last row's bit
    __shared__ __align__(16) uint8_t s_rows[2][16 * Lanes];  // the current and the next tile's spans (lane l: chunk l)
    __shared__ uint32_t s_bits[BitWords];                    // bit j: a segment ends after the span's row j

    const Fp8TypedReductionOp<FixedOp> op{};
    const float identity = op.identity();
    const uint32_t lane = threadIdx.x;
    // 1274: the search's first round reads its offsets together with the row count's (raggedSearchRound0Load).
    const RaggedSearchRound0 round0 = raggedSearchRound0Load(p.offsets, p.segments, lane);
    const Problem problem = makeProblem(p.offsets, p.segments, p.capacity_rows);
    const PathWeights weights = raggedEffectiveWeights(problem, p.weights, gridDim.x);
    const uint64_t cost = pathCost(problem, weights);

    // The CTA's range: lanes 0-15 find its first coordinate, lanes 16-31 its last, each with its segment's begin and
    // end (1272), so the role and the first segment's rows need no further reads.
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
    const bool has_head = head_segment != NO_SEGMENT;  // the head, when there is one, is the CTA's first end
    float head_value = identity;  // lane 0's

    // Everything below is relative to the CTA's first coordinate, in 32 bits.
    const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
    const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
    const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
    const uint32_t end_diagonal = seg_count + row_count;
    // 1284: element of relative row r: elem_base + r from p.bytes (the launcher keeps p.head a multiple of Bytes).
    const uint64_t elem_base = p.head / Bytes + row_base;
    // The CTA's first segment may have begun in an earlier CTA: its rows and count come from the offsets.
    const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
    const uint64_t first_end = seg_count > 0 ? first.end : 0;
    const bool first_has_rows = first_begin < first_end;
    const uint64_t first_count = first_end - first_begin;

    const auto ring = [&](uint32_t segment) -> uint32_t { return s_ends[segment & Mask]; };
    const auto end_rel = [&](uint32_t segment) -> uint32_t {
        return static_cast<uint32_t>(problem.end(seg_base + segment) - row_base);
    };
    // The span of tile rows from relative row y into s_rows[buffer]: lane l copies the span's chunk l when it holds a
    // needed row (1272-1283 placed 16 or 32 rows per lane piece-major; one chunk per lane needs no placement). 1272:
    // cp.async's 16-byte requests reach L2 one by one in any order (half-sector use and a 50% L2 hit in the 1271 and
    // 1272 profiles alike).
    const auto fetch_rows = [&](uint32_t y, uint32_t buffer) {
        if (y >= row_count) return;
        const uint64_t origin = (elem_base + y) & ~uint64_t{Rows - 1};  // 16-byte aligned: Rows x Bytes = 16
        const uint32_t shift = static_cast<uint32_t>((elem_base + y) - origin);
        const uint32_t limit = y + TileRows - shift < row_count ? y + TileRows - shift : row_count;
        // 1284: one 16-byte chunk per lane (1271-1283: Rows / 16 chunks, piece-major).
        if (Rows * lane < shift + (limit - y))
            raggedCpAsync<16>(&s_rows[buffer][16 * lane], p.bytes + origin * Bytes + 16 * lane);
    };
    // Bulk ends load: [first, first + count) into the ring (count at most Ring), one round trip. 1271: every loop over
    // ends skips its unused rounds with a warp-uniform test (count is the same in every lane), so a tile with a few
    // ends pays for one round, not Ring / 32.
    const auto load_ends = [&](uint32_t first, uint32_t count) {
        uint32_t ends[EndsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < count && i < count) ends[k] = end_rel(first + i);
        }
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < count && i < count) s_ends[(first + i) & Mask] = ends[k];
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

    float carry = identity;  // the open segment's running value
    while (x0 + y0 < end_diagonal) {
        // 1. Ends: top up when the prefetch fell behind (half a ring or less loaded).
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
        uint32_t ends_within = 0;  // the loaded ends are sorted: the ones within the span are a prefix
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            if (k * Lanes >= held || ends_within < k * Lanes) break;
            const uint32_t i = lane + k * Lanes;
            ends_within += __popc(__ballot_sync(0xffffffffU, i < held && ring(x0 + i) <= row_limit));
        }
        const uint32_t last_loaded = held > 0 ? ring(x0 + held - 1) : 0;
        const Coordinate32 tile_end = raggedTileEnd(x0, held, all_loaded, last_loaded, row_limit, ends_within);
        const uint32_t n = tile_end.segment - x0, y1 = tile_end.row;

        // 3. Fetch the next tile's span (cp.async, the other buffer) and the ends after the loaded ones (registers).
        fetch_rows(y1, buffer ^ 1U);
        raggedCpAsyncCommit();
        // The ends this tile consumes free their slots once it is done, so the prefetch fills the ring for the next.
        const uint32_t prefetch_first = x0 + held;
        const uint32_t prefetch_count = raggedEndsToLoad(held - n, Ring, seg_count - prefetch_first);
        uint32_t prefetched[EndsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            const uint32_t i = lane + k * Lanes;
            if (k * Lanes < prefetch_count && i < prefetch_count) prefetched[k] = end_rel(prefetch_first + i);
        }

        // 4. The bits: every end whose segment has rows in the tile marks its last row. 1275: each end's kind and value
        // slot are found here, once: round 0's (every end of a tile with at most 32) stay in one register for step 6,
        // which 1270-1274 recomputed from the ring. The end before an end is the lane below's, by shuffle.
        if (lane < BitWords) s_bits[lane] = 0;
        __syncwarp();
        uint32_t facts0 = raggedPackEndFacts(RaggedEndKind::Empty, 0);
        uint32_t round_previous = last_end;  // the end before this round's lane 0
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

        // 5. This lane's rows: its 16-byte chunk (one shared load), Rows elements. 1272: one pass. The ranks come
        // first (a scan of the bit counts), so the fold writes each ended segment's value to its rank as it goes (a
        // tree where no end falls); the warp scan then carries the open runs, and each lane with an end completes its
        // first value with the run arriving at it. 1271 folded, scanned, then decoded and folded the ended pieces
        // again. 1273: one fold path per warp (a warp with an end in any lane runs the emitting fold in every lane).
        const uint32_t rows_in_tile = y1 - y0;
        const uint32_t lane_first = Rows * lane;
        const bool lane_whole = lane_first >= shift && lane_first + Rows <= shift + rows_in_tile;
        const uint32_t bits = (s_bits[lane_first >> 5] >> (lane_first & 31U)) & LaneMask;
        uint32_t rank_base = __popc(bits);
#pragma unroll
        for (uint32_t offset = 1; offset < Lanes; offset <<= 1) {
            const uint32_t up = __shfl_up_sync(0xffffffffU, rank_base, offset);
            if (lane >= offset) rank_base += up;
        }
        rank_base -= __popc(bits);  // exclusive
        float acc;                  // the lane's run since its first row or its last end
        {
            const uint4 words = *reinterpret_cast<const uint4*>(&s_rows[buffer][16 * lane]);
            const uint32_t word[4] = {words.x, words.y, words.z, words.w};
            float v[Rows];
#pragma unroll
            for (uint32_t w = 0; w < 4; ++w)
                raggedKRDecodeWord<InputT>(foldFp8ExtremumBytes<FixedOp == 2>(word[w], p.fold), &v[PerWord * w]);
            if (!lane_whole) {  // the first and last lanes of a tile hold rows outside it
#pragma unroll
                for (uint32_t k = 0; k < Rows; ++k) {
                    const uint32_t local = lane_first + k;
                    if (local < shift || local >= shift + rows_in_tile) v[k] = identity;
                }
            }
            const auto value = [&](uint32_t k) -> float { return v[k]; };
            if (!__any_sync(0xffffffffU, bits != 0)) {
                acc = raggedTreeFold<Rows, float>(op, value);
            } else {
                acc = raggedLaneEmit<Rows>(bits, identity, identity, op, value, [&](uint32_t k, float segment_value) {
                    s_values[rank_base + raggedBitsBelow(bits, k)] = segment_value;
                });
            }
        }
        const bool reset_at_start = n > 0 && ring(x0) == y0;  // an end before the tile's rows closes the open run
        Carry<float> total;
        const Carry<float> before = raggedWarpExclusiveScan(
            Carry<float>{bits != 0, acc}, Carry<float>{false, reset_at_start ? identity : carry}, identity, op, total);
        if (bits != 0) s_values[rank_base] = raggedLaneFirstValue(before.value, s_values[rank_base], op);
        __syncwarp();

        // 6. One lane per end: its value, finalized and stored in segment order (the head goes to its record). 1275:
        // round 0 takes its kind and slot from step 4; later rounds (more than 32 ends in the tile) recompute them, the
        // slot after the Ranked ends of the rounds before. The ring is read again only for MEAN's row counts.
        uint32_t ranked_earlier = 0;  // Ranked ends in the rounds before this one (warp-uniform)
#pragma unroll
        for (uint32_t k = 0; k < EndsPerLane; ++k) {
            if (k * Lanes >= n) break;
            const uint32_t i = lane + k * Lanes;
            const bool active = i < n;
            uint32_t e = 0, previous = 0;
            if (k > 0 || p.finalize.mean) {  // warp-uniform
                e = active ? ring(x0 + i) : 0;
                if (k == 0) {
                    const uint32_t below = __shfl_up_sync(0xffffffffU, e, 1);
                    previous = lane == 0 ? last_end : below;
                } else {
                    previous = ring(x0 + i - 1);  // i >= 32
                }
            }
            RaggedEndKind kind;
            uint32_t slot;
            if (k == 0) {
                kind = raggedPackedKind(facts0);
                slot = raggedPackedSlot(facts0);
                if (n > Lanes)  // warp-uniform: later rounds' slots follow round 0's Ranked ends
                    ranked_earlier = __popc(__ballot_sync(0xffffffffU, active && kind == RaggedEndKind::Ranked));
            } else {
                kind = raggedEndKindAfter(false, first_has_rows, e, previous, y0);
                const uint32_t ballot = __ballot_sync(0xffffffffU, active && kind == RaggedEndKind::Ranked);
                slot = raggedRankedSlot(ranked_earlier, ballot, lane);
                ranked_earlier += __popc(ballot);
            }
            if (active) {
                const float segment_value = kind == RaggedEndKind::Ranked
                                                ? s_values[slot]
                                                : (kind == RaggedEndKind::CarryIn ? carry : identity);
                if (x0 + i == 0 && has_head) {
                    head_value = segment_value;  // the head is the CTA's first end, so only lane 0
                } else {
                    const uint64_t count = x0 + i == 0 ? first_count : e - previous;
                    raggedStoreValue(p.output, p.output_dtype, seg_base + x0 + i, segment_value, p.finalize,
                                     p.finalize.mean ? count : 0);
                }
            }
        }

        // 7. Advance; the prefetched ends go to the slots this tile freed, once every lane is done reading them.
        if (n > 0) last_end = ring(x0 + n - 1);
        carry = total.value;
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

    if (lane == 0) {  // 1280: the record arrays (raggedRecordLayout)
        raggedStoreRecordKey(p.workspace, blockIdx.x, head_segment);
        raggedStoreRecordValues(p.workspace, raggedRecordLayout(gridDim.x, 1), gridDim.x, blockIdx.x, 0, head_value,
                                tail_segment != RaggedMergePath::NO_SEGMENT ? carry : identity);
    }
}

// 1280: FP8 at D = 2..15 (rows of D bytes, under 16). 1284: every dtype (rows of D elements, under 16 bytes; a worker's
// Rows = 16 / element bytes rows per tile). One warp per CTA, the D = 1 kernel's walk (the CTA split, the
// ends ring and its prefetch, the tile end, the bitmap, the kinds and ranks of the ends) with lanes regrouped: lane =
// worker * D + component (RaggedKRWalk.h, raggedComponentWorkers), worker w folds component `component` of the tile's
// rows 16w .. 16w + 15. So:
//   - a tile is 16 x workers rows (at most 512 bytes), staged as the 16-byte chunks covering its bytes (any input
//     address: `shift` is the tile's misalignment), one chunk per lane (lane 0 also the 33rd);
//   - a lane reads its 16 elements by runtime address (D is a runtime operand), four rows' bytes per FP8x4 decode;
//   - the ranks count each worker's bits once (its component-0 lane), the segmented scan runs across workers per
//     component (raggedWarpExclusiveScanAcrossWorkers), and a value slot holds D values (slot * D + component), so a
//     tile consumes at most raggedComponentEndsCap(D) ends (raggedCapTileEnds);
//   - the bits pass keeps every end's kind and slot in shared memory (s_facts), and the store pass takes `workers`
//     ends per round, one lane per component: each round stores workers x D consecutive outputs.
template <typename InputT, int FixedOp>
__global__ __launch_bounds__(RAGGED_KR_LANES, RAGGED_KR_MIN_CTAS_PER_SM) void raggedKRComponentsKernel(
    const RaggedKRParams p) {
    using namespace RaggedMergePath;
    constexpr uint32_t Lanes = RAGGED_KR_LANES;
    constexpr uint32_t Bytes = RaggedKRInput<InputT>::Bytes;
    constexpr uint32_t Rows = RaggedKRInput<InputT>::Rows;  // per worker per tile: 16 bytes of each component
    constexpr uint32_t PerWord = RaggedKRInput<InputT>::PerWord;
    constexpr uint32_t WorkerMask = (1U << Rows) - 1U;
    using Storage = typename RaggedKRInput<InputT>::Storage;
    constexpr uint32_t MaxTileRows = Rows * (Lanes / 2);  // D >= 2: at most 16 workers
    constexpr uint32_t SpanBytes = 16 * (Lanes + 1);      // a tile's at most 512 bytes and their misalignment
    constexpr uint32_t Ring = RAGGED_KR_RING;
    constexpr uint32_t Mask = Ring - 1;
    constexpr uint32_t EndsPerLane = Ring / Lanes;
    constexpr uint32_t BitWords = MaxTileRows / 32;
    constexpr uint32_t Slots = RAGGED_KR_COMPONENT_VALUE_SLOTS;
    static_assert(FixedOp == 0 || FixedOp == 2);
    static_assert(SpanBytes == 16 * 33 && Rows * Bytes == 16);
    __shared__ uint32_t s_ends[Ring];
    __shared__ float s_values[Slots];   // the tile's segment values: slot * D + component
    __shared__ uint16_t s_facts[Ring];  // each end's kind and slot (raggedPackEndFacts16), by its index in the tile
    __shared__ __align__(16) uint8_t s_rows[2][SpanBytes];
    __shared__ uint32_t s_bits[BitWords];  // bit j: a segment ends after the tile's row j

    const Fp8TypedReductionOp<FixedOp> op{};
    const float identity = op.identity();
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
    float head_value = identity;  // the worker-0 lanes', one per component

    const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
    const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
    const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
    const uint32_t end_diagonal = seg_count + row_count;
    const uint32_t row_bytes = D * Bytes;
    const uint64_t byte_base = p.head + row_base * row_bytes;  // byte of relative row r: byte_base + r x row_bytes
    const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
    const uint64_t first_end = seg_count > 0 ? first.end : 0;
    const bool first_has_rows = first_begin < first_end;
    const uint64_t first_count = first_end - first_begin;

    const auto ring = [&](uint32_t segment) -> uint32_t { return s_ends[segment & Mask]; };
    const auto end_rel = [&](uint32_t segment) -> uint32_t {
        return static_cast<uint32_t>(problem.end(seg_base + segment) - row_base);
    };
    // The bytes of the tile from relative row y, as the 16-byte chunks covering them, in order: lane l copies chunk l,
    // lane 0 also chunk 32 (a tile's bytes plus its misalignment are at most 527).
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

    float carry = identity;  // the open segment's running value, this lane's component
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

        // 5. This lane's component of its worker's Rows rows: the ranks (each worker counted once, at its component-0
        // lane), then one pass writing each ended segment's value to its slot, the scan across workers, and each
        // worker's first value completed with the run arriving at it.
        const uint32_t rows_in_tile = y1 - y0;
        const uint32_t shift = static_cast<uint32_t>((byte_base + uint64_t{y0} * row_bytes) & 15U);
        const uint32_t first_row = Rows * worker;
        const uint32_t rows_valid =
            !lane_active || first_row >= rows_in_tile ? 0 : (rows_in_tile - first_row < Rows ? rows_in_tile - first_row : Rows);
        const uint32_t bits = lane_active ? (s_bits[first_row >> 5] >> (first_row & 31U)) & WorkerMask : 0;
        uint32_t rank_inclusive = component == 0 ? static_cast<uint32_t>(__popc(bits)) : 0;
#pragma unroll
        for (uint32_t offset = 1; offset < Lanes; offset <<= 1) {
            const uint32_t up = __shfl_up_sync(0xffffffffU, rank_inclusive, offset);
            if (lane >= offset) rank_inclusive += up;
        }
        const uint32_t rank_base = __shfl_sync(0xffffffffU, rank_inclusive, worker * D) - __popc(bits);
        // 1281 (the 1280 profile: about 650 warp instructions per tile against the D = 1 kernel's 320, 180 of them
        // in this read: each element's address rebuilt by a chain of multiplies, and each predicated load compiled to
        // its own branch): unpredicated loads at a running offset (one add per element), four bytes packed by three
        // byte permutes, and the rows outside the tile masked after the decode, only in warps that have such rows. An
        // active lane's Rows reads stay inside the span buffer whatever the tile's rows (its last byte is at most
        // 15 + 16 x workers x D - 1 <= 526); idle lanes read from byte 0 and are never used.
        // 1284: elements of Bytes bytes (natural alignment: the launcher keeps the input aligned to its element, and
        // shift, row_bytes and the component offset are multiples of Bytes), PerWord per decoded word.
        const Storage* element = reinterpret_cast<const Storage*>(
            &s_rows[buffer][lane_active ? shift + first_row * row_bytes + component * Bytes : 0]);
        float v[Rows];
#pragma unroll
        for (uint32_t g = 0; g < Rows / PerWord; ++g) {
            uint32_t word;
            if constexpr (Bytes == 1) {
                const uint32_t b0 = element[0], b1 = element[D], b2 = element[2 * D], b3 = element[3 * D];
                word = __byte_perm(__byte_perm(b0, b1, 0x0040U), __byte_perm(b2, b3, 0x0040U), 0x5410U);
            } else if constexpr (Bytes == 2) {
                word = __byte_perm(uint32_t{element[0]}, uint32_t{element[D]}, 0x5410U);
            } else {
                word = element[0];
            }
            element += PerWord * D;
            raggedKRDecodeWord<InputT>(foldFp8ExtremumBytes<FixedOp == 2>(word, p.fold), &v[PerWord * g]);
        }
        if (__any_sync(0xffffffffU, lane_active && rows_valid < Rows)) {  // warp-uniform: the tile's edge workers
#pragma unroll
            for (uint32_t k = 0; k < Rows; ++k) {
                if (k >= rows_valid) v[k] = identity;
            }
        }
        const auto value = [&](uint32_t k) -> float { return v[k]; };
        float acc;
        if (!__any_sync(0xffffffffU, bits != 0)) {
            acc = raggedTreeFold<Rows, float>(op, value);
        } else {
            acc = raggedLaneEmit<Rows>(bits, identity, identity, op, value, [&](uint32_t k, float segment_value) {
                s_values[(rank_base + raggedBitsBelow(bits, k)) * D + component] = segment_value;
            });
        }
        const bool reset_at_start = n > 0 && ring(x0) == y0;
        Carry<float> total;
        const Carry<float> before = raggedWarpExclusiveScanAcrossWorkers(
            Carry<float>{bits != 0, acc}, Carry<float>{false, reset_at_start ? identity : carry}, identity, op, D,
            workers, worker, component, total);
        if (bits != 0) {
            const uint32_t slot = rank_base * D + component;
            s_values[slot] = raggedLaneFirstValue(before.value, s_values[slot], op);
        }
        __syncwarp();

        // 6. `workers` ends per round, one lane per component: the value, finalized and stored (the head to its
        // record).
        for (uint32_t round_first = 0; round_first < n; round_first += workers) {
            const uint32_t i = round_first + worker;
            if (lane_active && i < n) {
                const uint32_t facts = s_facts[i];
                const RaggedEndKind kind = raggedPacked16Kind(facts);
                const float segment_value =
                    kind == RaggedEndKind::Ranked ? s_values[raggedPacked16Slot(facts) * D + component]
                                                  : (kind == RaggedEndKind::CarryIn ? carry : identity);
                if (x0 + i == 0 && has_head) {
                    head_value = segment_value;  // the head is the CTA's first end: the worker-0 lanes
                } else {
                    uint64_t count = 0;
                    if (p.finalize.mean) {
                        const uint32_t e = ring(x0 + i);
                        const uint32_t previous = i == 0 ? last_end : ring(x0 + i - 1);
                        count = x0 + i == 0 ? first_count : e - previous;
                    }
                    raggedStoreValue(p.output, p.output_dtype, (seg_base + x0 + i) * D + component, segment_value,
                                     p.finalize, count);
                }
            }
        }

        // 7. Advance.
        if (n > 0) last_end = ring(x0 + n - 1);
        carry = total.value;
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

    const RaggedRecordLayout layout = raggedRecordLayout(gridDim.x, D);
    if (lane == 0) raggedStoreRecordKey(p.workspace, blockIdx.x, head_segment);
    if (lane < D) {  // worker 0: component `lane`
        raggedStoreRecordValues(p.workspace, layout, gridDim.x, blockIdx.x, component, head_value,
                                tail_segment != NO_SEGMENT ? carry : identity);
    }
}

// 1280: D = 1 runs the one-chunk-per-lane kernel, D >= 2 the components kernel. 1284: for every dtype.
template <typename InputT, int FixedOp>
void raggedKRLaunchMain(uint32_t grid, const RaggedKRParams& params, cudaStream_t stream) {
    if (params.components == 1) {
        raggedKRValueKernel<InputT, FixedOp><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else {
        raggedKRComponentsKernel<InputT, FixedOp><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    }
    raggedKRCheck(cudaGetLastError(), "main kernel launch");
}

// 1284: what each dtype object (CubRaggedKRFp8.cu, CubRaggedKRF16.cu, CubRaggedKRFp32.cu) exports to the launcher
// (CubRaggedKR.cu): the main kernel's launch (fixed_op 0: additive, 2: minimum) and its resident CTAs per SM (they
// depend on D = 1 against D >= 2, not on the format or the operation).
template <typename InputT>
void raggedKRLaunchMainFor(int fixed_op, uint32_t grid, const RaggedKRParams& params, cudaStream_t stream) {
    if (fixed_op == 0) {
        raggedKRLaunchMain<InputT, 0>(grid, params, stream);
    } else {
        raggedKRLaunchMain<InputT, 2>(grid, params, stream);
    }
}

template <typename InputT>
int raggedKRResidentFor(uint32_t components) {
    int resident = 0;
    if (components == 1) {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, raggedKRValueKernel<InputT, 0>,
                                                                    RAGGED_KR_LANES, 0),
                      "occupancy");
    } else {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, raggedKRComponentsKernel<InputT, 0>,
                                                                    RAGGED_KR_LANES, 0),
                      "occupancy");
    }
    return resident;
}

// 1284: the dtype objects' exports. fixed_op 0 (additive) or 2 (minimum); `input` picks the format within the object.
// Without FP8 types the FP8 object's functions throw std::invalid_argument.
void raggedKRLaunchMainFp8(DataType input, int fixed_op, uint32_t grid, const RaggedKRParams& params,
                           cudaStream_t stream);
void raggedKRLaunchMainF16(DataType input, int fixed_op, uint32_t grid, const RaggedKRParams& params,
                           cudaStream_t stream);
void raggedKRLaunchMainFp32(int fixed_op, uint32_t grid, const RaggedKRParams& params, cudaStream_t stream);
[[nodiscard]] int raggedKRResidentFp8(uint32_t components);
[[nodiscard]] int raggedKRResidentF16(uint32_t components);
[[nodiscard]] int raggedKRResidentFp32(uint32_t components);

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
