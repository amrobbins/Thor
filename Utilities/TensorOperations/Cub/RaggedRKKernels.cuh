#pragma once

// 1285 (phase 4, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged RK VALUE main kernel, rows of
// R = D x element bytes >= 16, templated on the input element as the KR kernels (RaggedKRKernels.cuh) and instantiated
// per dtype group (CubRaggedRKFp8.cu: E4M3, E5M2; 1295: CubRaggedRKF16.cu: FP16, BF16; CubRaggedRKFp32.cu; each with
// its narrow-rows object); the host launcher is CubRaggedRK.cu, the finisher
// CubRaggedFinish.cu. Geometry: RaggedRKWalk.h.
//
// One warp per CTA, persistent: CTA c takes K tile c mod ktiles of merge-path range c / ktiles. The walk over the
// range is the KR components kernel's per row (the ends ring and its prefetch, the tile end, the bitmap, each end's
// kind), with tiles of Rows rows per worker (the template's: RAGGED_RK_ROWS = 2, and 1288's RAGGED_RK_ROWS_NARROW = 4
// on narrow rows, chosen by raggedRKRowsPerWorker; the Rows = 2 kernels are 1286's). Values:
//   - a lane decodes its 16-byte chunk of each of its worker's rows from the staged tile (any alignment: five word
//     loads and four funnel shifts, or one 16-byte load when the input base and R are 16-byte aligned) and folds them,
//     cut at the worker's ends;
//   - a segment that ends inside a worker after its first end there is complete in the lane, and its 16 / element
//     bytes outputs go straight to global memory (the worker's P lanes write the row's outputs contiguously): no value
//     slots and no cap on a tile's ends;
//   - the scan across workers carries each lane's values; each worker's first value then takes the run arriving at it;
//   - ends without rows in the tile (an empty segment: the identity; a segment whose rows all came before the tile:
//     the run) are stored by a pass of W ends per round.
// The boundary records are raggedRecordLayout(ranges, D): the range's key (K tile 0's CTA), and each column's head
// and tail values (each K tile's CTA its own columns).
#include "Utilities/TensorOperations/Cub/RaggedKRKernels.cuh"
#include "Utilities/TensorOperations/Cub/RaggedRKWalk.h"

namespace ThorImplementation::CubReductionInternal::RaggedKR {

// 16 resident warps per SM: a lane holds its run, its first value and the carry, 16 / element bytes floats each.
inline constexpr uint32_t RAGGED_RK_MIN_CTAS_PER_SM = 16;

// 1286: one lane's finalized outputs of an end, `valid` consecutive values (at most V) from output element `index`,
// in vector stores where the address allows (the 1285 profile: the 16 scalar 4-byte stores per lane, 64 bytes apart
// across the lanes, were 77-80% of the L2 sectors in short-segment cells and held L2 at 71-74% busy). Each value is
// converted exactly as storeFp32AsRuntimeDType converts it.
//   FP32: 16-byte stores from the first 16-byte boundary (scalars before and after it);
//   FP16 / BF16: 16-byte stores when aligned, else 4-byte pairs when the index is even, else scalars;
//   FP8: one 16-byte store when aligned, else scalars.
template <uint32_t V, uint32_t Head>
__device__ __forceinline__ void raggedRKStoreFp32From(float* out, const float (&v)[V]) {
    static_assert(Head < 4 && V % 4 == 0);
    if constexpr (Head > 0) {  // 1286a: no loop at Head 0 (nvcc's #186-D, an error under --Werror)
#pragma unroll
        for (uint32_t k = 0; k < Head; ++k) out[k] = v[k];
    }
#pragma unroll
    for (uint32_t k = Head; k + 4 <= V; k += 4)
        *reinterpret_cast<float4*>(out + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
    if constexpr (Head > 0) {  // the 4 - Head scalars after the last 16-byte store
#pragma unroll
        for (uint32_t k = V - 4 + Head; k < V; ++k) out[k] = v[k];
    }
}

template <uint32_t V>
__device__ __forceinline__ void raggedRKStoreChunk(void* output, DataType output_dtype, uint64_t index,
                                                   const float (&v)[V], uint32_t valid) {
    if (valid == V) {
        if (output_dtype == DataType::FP32) {
            float* out = static_cast<float*>(output) + index;
            const uint32_t misaligned = static_cast<uint32_t>((reinterpret_cast<uintptr_t>(out) >> 2) & 3U);
            if ((reinterpret_cast<uintptr_t>(out) & 3U) == 0) {
                switch ((4U - misaligned) & 3U) {  // the scalars before the first 16-byte boundary
                    case 0: raggedRKStoreFp32From<V, 0>(out, v); return;
                    case 1: raggedRKStoreFp32From<V, 1>(out, v); return;
                    case 2: raggedRKStoreFp32From<V, 2>(out, v); return;
                    default: raggedRKStoreFp32From<V, 3>(out, v); return;
                }
            }
        } else if (output_dtype == DataType::FP16 || output_dtype == DataType::BF16) {
            uint16_t* out = static_cast<uint16_t*>(output) + index;
            uint32_t pairs[V / 2];
#pragma unroll
            for (uint32_t k = 0; k < V / 2; ++k) {
                uint32_t lo, hi;
                if (output_dtype == DataType::FP16) {
                    lo = __half_as_ushort(__float2half_rn(v[2 * k]));
                    hi = __half_as_ushort(__float2half_rn(v[2 * k + 1]));
                } else {
                    lo = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * k]));
                    hi = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * k + 1]));
                }
                pairs[k] = lo | (hi << 16);
            }
            if ((reinterpret_cast<uintptr_t>(out) & 15U) == 0 && V % 8 == 0) {
#pragma unroll
                for (uint32_t k = 0; k < V / 2; k += 4)
                    *reinterpret_cast<uint4*>(out + 2 * k) = make_uint4(pairs[k], pairs[k + 1], pairs[k + 2], pairs[k + 3]);
                return;
            }
            if ((reinterpret_cast<uintptr_t>(out) & 3U) == 0) {
#pragma unroll
                for (uint32_t k = 0; k < V / 2; ++k) reinterpret_cast<uint32_t*>(out)[k] = pairs[k];
                return;
            }
#pragma unroll
            for (uint32_t k = 0; k < V / 2; ++k) {
                out[2 * k] = static_cast<uint16_t>(pairs[k] & 0xffffU);
                out[2 * k + 1] = static_cast<uint16_t>(pairs[k] >> 16);
            }
            return;
        }
#if THOR_CUB_ENABLE_FP8_TYPES
        else if ((output_dtype == DataType::FP8_E4M3 || output_dtype == DataType::FP8_E5M2) && V == 16) {
            uint8_t* out = static_cast<uint8_t*>(output) + index;
            if ((reinterpret_cast<uintptr_t>(out) & 15U) == 0) {
                uint32_t words[4] = {0, 0, 0, 0};
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) {
                    const uint32_t byte =
                        output_dtype == DataType::FP8_E4M3 ? ThorLowPrecision::toFp8E4M3Satfinite(v[k]).__x
                                                           : ThorLowPrecision::toFp8E5M2Satfinite(v[k]).__x;
                    words[k / 4] |= byte << (8 * (k % 4));
                }
                *reinterpret_cast<uint4*>(out) = make_uint4(words[0], words[1], words[2], words[3]);
                return;
            }
        }
#endif
    }
#pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        if (k < valid) storeFp32AsRuntimeDType(output, output_dtype, index + k, v[k]);
    }
}

struct RaggedRKParams {
    const uint8_t* bytes;  // the input's bytes, from its 16-byte aligned base
    uint32_t head;         // row 0's byte offset from `bytes` (0 to 15, a multiple of the element's bytes)
    RaggedMergePath::Offsets offsets;
    uint64_t segments;
    uint64_t capacity_rows;
    RaggedMergePath::PathWeights weights;
    Fp8ExtremumFold fold;
    RaggedFinalize finalize;
    void* output;
    DataType output_dtype;
    uint8_t* workspace;  // raggedRecordLayout(ranges, components)
    uint64_t components;  // D
    uint32_t row_bytes;   // R = D x element bytes
    uint32_t ranges;      // path ranges; gridDim.x = ranges x ktiles
    uint32_t ktiles;
    uint32_t tile_bytes;  // T
    uint32_t chunks;      // P
    uint32_t workers;     // W
#if THOR_RAGGED_RK_PROBE
    unsigned long long* probe;  // 1290: RAGGED_RK_PROBE_WORDS per CTA, or null
#endif
};

#if THOR_RAGGED_RK_PROBE
__device__ __forceinline__ unsigned long long raggedRKGlobalTimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// clock64 read after `after` is available: `after` is first stored to a scratch word (the store issues only once its
// value has arrived, and a warp issues in order), then the clock is read.
__device__ __forceinline__ unsigned long long raggedRKClockAfter(uint32_t after, unsigned long long* scratch) {
    unsigned long long c;
    asm volatile("st.volatile.global.u32 [%2], %1;\n\tmov.u64 %0, %%clock64;"
                 : "=l"(c)
                 : "r"(after), "l"(scratch)
                 : "memory");
    return c;
}
#endif

template <typename InputT, int FixedOp, uint32_t Rows>
__global__ __launch_bounds__(RAGGED_KR_LANES, RAGGED_RK_MIN_CTAS_PER_SM) void raggedRKValueKernel(
    const RaggedRKParams p) {
    using namespace RaggedMergePath;
    constexpr uint32_t Lanes = RAGGED_KR_LANES;
    constexpr uint32_t Bytes = RaggedKRInput<InputT>::Bytes;
    constexpr uint32_t PerWord = RaggedKRInput<InputT>::PerWord;
    constexpr uint32_t V = RAGGED_RK_CHUNK_BYTES / Bytes;  // values per lane per row
    constexpr uint32_t SpanBytes = raggedRKSpanBytesFor(Rows);
    constexpr uint32_t ContiguousCopies = (raggedRKStagedChunksFor(Rows) + Lanes - 1) / Lanes;  // 3, or 5 at Rows 4
    constexpr uint32_t MaxTileRows = Rows * Lanes;
    constexpr uint32_t Ring = RAGGED_KR_RING;
    constexpr uint32_t Mask = Ring - 1;
    constexpr uint32_t EndsPerLane = Ring / Lanes;
    constexpr uint32_t BitWords = MaxTileRows / 32;
    constexpr uint32_t RowMask = (1U << Rows) - 1U;
    static_assert(FixedOp == 0 || FixedOp == 2);
    static_assert((Rows == RAGGED_RK_ROWS || Rows == RAGGED_RK_ROWS_NARROW) && MaxTileRows <= 255 && Ring <= 255 &&
                  SpanBytes % 16 == 0 && 32 % Rows == 0);
    __shared__ uint32_t s_ends[Ring];
    __shared__ uint8_t s_kind[Ring];             // each end's kind (RaggedEndKind), by its index in the tile
    __shared__ uint8_t s_row_end[MaxTileRows];   // the Ranked end after each tile row, by its index in the tile
    __shared__ uint32_t s_bits[BitWords];        // bit j: a segment with rows in the tile ends after tile row j
    __shared__ __align__(16) uint32_t s_rows[2][SpanBytes / 4];

    const Fp8TypedReductionOp<FixedOp> op{};
    const float identity = op.identity();
    const uint32_t lane = threadIdx.x;
    const RaggedRKGeometry geometry{p.ktiles, p.tile_bytes, p.chunks, p.workers, p.ktiles == 1};
    const uint32_t P = p.chunks, W = p.workers, R = p.row_bytes;
    const uint32_t worker = lane / P;
    const uint32_t chunk = lane - worker * P;
    const bool lane_active = worker < W;
    const uint32_t ktile = blockIdx.x % p.ktiles;
    const uint32_t range = blockIdx.x / p.ktiles;
    const uint32_t k0 = ktile * p.tile_bytes;
    const uint32_t tile_k = raggedRKTileBytesOf(geometry, R, ktile);
    const uint32_t chunk_bytes =
        lane_active && RAGGED_RK_CHUNK_BYTES * chunk < tile_k
            ? (tile_k - RAGGED_RK_CHUNK_BYTES * chunk < RAGGED_RK_CHUNK_BYTES ? tile_k - RAGGED_RK_CHUNK_BYTES * chunk
                                                                              : RAGGED_RK_CHUNK_BYTES)
            : 0;
    const uint32_t chunk_values = chunk_bytes / Bytes;
    const uint64_t column0 = (k0 + RAGGED_RK_CHUNK_BYTES * chunk) / Bytes;  // the lane's first output column
    const bool aligned = ((p.head | R) & 15U) == 0;  // every chunk read is one aligned 16-byte load
    const uint32_t tile_rows = Rows * W;  // raggedRKTileRows(geometry): the launcher picks Rows by raggedRKRowsPerWorker
    const uint32_t slot_bytes = raggedRKSlotBytes(geometry);

#if THOR_RAGGED_RK_PROBE
    unsigned long long probe_t0 = 0, probe_c0 = 0, probe_c1 = 0, probe_c2 = 0, probe_c3 = 0;
    uint32_t probe_tiles = 0;
    unsigned long long* const probe_scratch =
        p.probe != nullptr ? p.probe + uint64_t{blockIdx.x} * RAGGED_RK_PROBE_WORDS + 7 : nullptr;
    if (p.probe != nullptr && lane == 0) {
        probe_t0 = raggedRKGlobalTimer();
        probe_c0 = raggedRKClockAfter(0, probe_scratch);
    }
#endif
    const RaggedSearchRound0 round0 = raggedSearchRound0Load(p.offsets, p.segments, lane);
    const Problem problem = makeProblem(p.offsets, p.segments, p.capacity_rows);
    const PathWeights weights = raggedEffectiveWeights(problem, p.weights, p.ranges);
    const uint64_t cost = pathCost(problem, weights);
    const SearchedCoordinate mine =
        raggedHalfWarpSearchCost(problem, weights, ctaDiagonal(cost, p.ranges, range + (lane >> 4)), lane, round0);
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

    const uint64_t row_base = cta_begin.row, seg_base = cta_begin.segment;
    const uint32_t seg_count = static_cast<uint32_t>(cta_end.segment - seg_base);
    const uint32_t row_count = static_cast<uint32_t>(cta_end.row - row_base);
    const uint32_t end_diagonal = seg_count + row_count;
    const uint64_t byte_base = p.head + row_base * R + k0;  // byte of relative row r's K tile: byte_base + r x R
    const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
    const uint64_t first_end = seg_count > 0 ? first.end : 0;
    const bool first_has_rows = first_begin < first_end;
    const uint64_t first_count = first_end - first_begin;
#if THOR_RAGGED_RK_PROBE
    if (p.probe != nullptr && lane == 0) probe_c1 = raggedRKClockAfter(end_diagonal ^ static_cast<uint32_t>(first_end), probe_scratch);
#endif
    const RaggedRecordLayout layout = raggedRecordLayout(p.ranges, static_cast<uint32_t>(p.components));
    float* record_heads = reinterpret_cast<float*>(p.workspace + layout.heads);
    float* record_tails = reinterpret_cast<float*>(p.workspace + layout.tails);

    const auto ring = [&](uint32_t segment) -> uint32_t { return s_ends[segment & Mask]; };
    const auto end_rel = [&](uint32_t segment) -> uint32_t {
        return static_cast<uint32_t>(problem.end(seg_base + segment) - row_base);
    };
    // K-tiled copies: task t = lane + 32 q is chunk t mod (P + 1) of tile row t / (P + 1) (at most 96 tasks).
    uint32_t task_row[3], task_chunk[3];
#pragma unroll
    for (uint32_t q = 0; q < 3; ++q) {
        const uint32_t t = lane + Lanes * q;
        task_row[q] = t / (P + 1);
        task_chunk[q] = t - task_row[q] * (P + 1);
    }
    // The tile's bytes from relative row y into s_rows[buffer], as the 16-byte chunks covering them.
    const auto fetch_rows = [&](uint32_t y, uint32_t buffer) {
        if (y >= row_count) return;
        const uint32_t rows = (y + tile_rows < row_count ? y + tile_rows : row_count) - y;
        uint8_t* staged = reinterpret_cast<uint8_t*>(s_rows[buffer]);
        if (geometry.contiguous) {
            const uint64_t start = byte_base + uint64_t{y} * R;
            const uint64_t origin = start & ~uint64_t{15};
            const uint32_t bytes = static_cast<uint32_t>(start - origin) + rows * R;  // <= 1039 (Rows 4: 2063)
#pragma unroll
            for (uint32_t q = 0; q < ContiguousCopies; ++q) {
                const uint32_t c = lane + Lanes * q;
                if (16 * c < bytes) raggedCpAsync<16>(staged + 16 * c, p.bytes + origin + 16 * c);
            }
        } else {
#pragma unroll
            for (uint32_t q = 0; q < 3; ++q) {
                const uint32_t j = task_row[q], c = task_chunk[q];
                if (j < rows) {
                    const uint64_t start = byte_base + uint64_t{y + j} * R;
                    const uint64_t origin = start & ~uint64_t{15};
                    const uint32_t bytes = static_cast<uint32_t>(start - origin) + tile_k;
                    if (16 * c < bytes) raggedCpAsync<16>(staged + j * slot_bytes + 16 * c, p.bytes + origin + 16 * c);
                }
            }
        }
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
#if THOR_RAGGED_RK_PROBE
    if (p.probe != nullptr && lane == 0) probe_c2 = raggedRKClockAfter(s_ends[0] ^ s_rows[0][0], probe_scratch);
#endif

    float carry[V];  // the open segment's running value, this lane's columns (the same in every worker)
#pragma unroll
    for (uint32_t k = 0; k < V; ++k) carry[k] = identity;

    // End i of the tile, complete in this lane: its outputs (or, for the CTA's head, its record's head values).
    const auto store_end = [&](uint32_t i, const float (&values)[V]) {
        if (x0 + i == 0 && has_head) {
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) {
                if (k < chunk_values)
                    record_heads[raggedRecordValueIndex(p.ranges, range, static_cast<uint32_t>(column0 + k))] =
                        values[k];
            }
            return;
        }
        uint64_t count = 0;
        if (p.finalize.mean) {
            const uint32_t e = ring(x0 + i);
            const uint32_t previous = i == 0 ? last_end : ring(x0 + i - 1);
            count = x0 + i == 0 ? first_count : e - previous;
        }
        const uint64_t output_base = (seg_base + x0 + i) * p.components + column0;
        // 1286: finalized as raggedStoreValue (MEAN's divide, then MAX's sign), stored as one chunk.
        float finalized[V];
#pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            const float value = p.finalize.mean ? raggedMeanOutOfLine(values[k], count) : values[k];
            finalized[k] = value * p.finalize.sign;
        }
        raggedRKStoreChunk<V>(p.output, p.output_dtype, output_base, finalized, chunk_values);
    };

    while (x0 + y0 < end_diagonal) {
        // 1. Ends: top up when the prefetch fell behind.
        if (x0 + held < seg_count && held <= Ring / 2) {
            const uint32_t count = raggedEndsToLoad(held, Ring, seg_count - (x0 + held));
            load_ends(x0 + held, count);
            __syncwarp();
            held += count;
        }
        const bool all_loaded = x0 + held == seg_count;

        // 2. The tile: Rows x W rows, never past the last loaded end.
        const uint32_t row_limit = y0 + tile_rows < row_count ? y0 + tile_rows : row_count;
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

        // 4. The bits, each end's kind, and each Ranked end's row.
        if (lane < BitWords) s_bits[lane] = 0;
        __syncwarp();
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
            if (active) {
                s_kind[i] = static_cast<uint8_t>(kind);
                if (kind == RaggedEndKind::Ranked) {
                    const uint32_t j = e - 1 - y0;
                    atomicOr(&s_bits[j >> 5], 1U << (j & 31U));
                    s_row_end[j] = static_cast<uint8_t>(i);
                }
            }
        }
        __syncwarp();

        // 5. Ends without rows in the tile, W per round: the run before the tile (carry-in) or the identity (empty).
        for (uint32_t round_first = 0; round_first < n; round_first += W) {
            const uint32_t i = round_first + worker;
            if (lane_active && i < n) {
                const RaggedEndKind kind = static_cast<RaggedEndKind>(s_kind[i]);
                if (kind != RaggedEndKind::Ranked) {
                    float values[V];
#pragma unroll
                    for (uint32_t k = 0; k < V; ++k) values[k] = kind == RaggedEndKind::CarryIn ? carry[k] : identity;
                    store_end(i, values);
                }
            }
        }

        // 6. This lane's chunk of its worker's rows, folded and cut at the worker's ends.
        const uint32_t rows_in_tile = y1 - y0;
        const uint32_t tile_shift = static_cast<uint32_t>((byte_base + uint64_t{y0} * R) & 15U);
        const uint32_t first_row = Rows * worker;
        const uint32_t bits = lane_active ? (s_bits[first_row >> 5] >> (first_row & 31U)) & RowMask : 0;
        const bool masked = __any_sync(0xffffffffU, lane_active && (first_row + Rows > rows_in_tile ||
                                                                     chunk_values < V));  // warp-uniform
        const uint8_t* staged = reinterpret_cast<const uint8_t*>(s_rows[buffer]);
        const auto decode_row = [&](uint32_t i, float (&v)[V]) {
            const uint32_t j = first_row + i;
            const uint32_t offset = lane_active ? raggedRKStagedOffset(geometry, R, tile_shift, j, chunk) : 0;
            uint32_t word[4];
            if (aligned) {
                const uint4 w = *reinterpret_cast<const uint4*>(staged + offset);
                word[0] = w.x;
                word[1] = w.y;
                word[2] = w.z;
                word[3] = w.w;
            } else {
                const uint32_t* words = reinterpret_cast<const uint32_t*>(staged) + (offset >> 2);
                const uint32_t shift = (offset & 3U) * 8U;
                const uint32_t w0 = words[0], w1 = words[1], w2 = words[2], w3 = words[3], w4 = words[4];
                word[0] = __funnelshift_r(w0, w1, shift);
                word[1] = __funnelshift_r(w1, w2, shift);
                word[2] = __funnelshift_r(w2, w3, shift);
                word[3] = __funnelshift_r(w3, w4, shift);
            }
#pragma unroll
            for (uint32_t w = 0; w < 4; ++w)
                raggedKRDecodeWord<InputT>(foldFp8ExtremumBytes<FixedOp == 2>(word[w], p.fold), &v[PerWord * w]);
            if (masked) {
                const bool row_valid = j < rows_in_tile;
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) {
                    if (!row_valid || k >= chunk_values) v[k] = identity;
                }
            }
        };
        float acc[V];
        float first_value[V];
        uint32_t first_end_row = Rows;  // Rows: no end in the worker's rows
        if constexpr (Rows == RAGGED_RK_ROWS) {  // the 1286 kernels
            if (!__any_sync(0xffffffffU, bits != 0)) {
                float v[V];
                decode_row(0, acc);
#pragma unroll
                for (uint32_t i = 1; i < Rows; ++i) {
                    decode_row(i, v);
#pragma unroll
                    for (uint32_t k = 0; k < V; ++k) acc[k] = op(acc[k], v[k]);
                }
            } else {
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) acc[k] = identity;
#pragma unroll
                for (uint32_t i = 0; i < Rows; ++i) {
                    float v[V];
                    decode_row(i, v);
#pragma unroll
                    for (uint32_t k = 0; k < V; ++k) acc[k] = op(acc[k], v[k]);
                    if ((bits >> i) & 1U) {
                        if (first_end_row == Rows) {
                            first_end_row = i;
#pragma unroll
                            for (uint32_t k = 0; k < V; ++k) first_value[k] = acc[k];
                        } else {
                            store_end(s_row_end[first_row + i], acc);
                        }
#pragma unroll
                        for (uint32_t k = 0; k < V; ++k) acc[k] = identity;
                    }
                }
            }
        } else {
            // 1289: the narrow kernels fold the rows in pairs, a loop over the pairs (1287's form: 1288's full unroll
            // over 4 rows was 2% slower on D 16 / 17 and 40% larger, its ended path inlining store_end per row).
            if (!__any_sync(0xffffffffU, bits != 0)) {
                float v[V];
                decode_row(0, acc);
                decode_row(1, v);
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) acc[k] = op(acc[k], v[k]);
#pragma unroll 1
                for (uint32_t base = 2; base < Rows; base += 2) {
#pragma unroll
                    for (uint32_t h = 0; h < 2; ++h) {
                        decode_row(base + h, v);
#pragma unroll
                        for (uint32_t k = 0; k < V; ++k) acc[k] = op(acc[k], v[k]);
                    }
                }
            } else {
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) acc[k] = identity;
#pragma unroll 1
                for (uint32_t base = 0; base < Rows; base += 2) {
#pragma unroll
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint32_t i = base + h;
                        float v[V];
                        decode_row(i, v);
#pragma unroll
                        for (uint32_t k = 0; k < V; ++k) acc[k] = op(acc[k], v[k]);
                        if ((bits >> i) & 1U) {
                            if (first_end_row == Rows) {
                                first_end_row = i;
#pragma unroll
                                for (uint32_t k = 0; k < V; ++k) first_value[k] = acc[k];
                            } else {
                                store_end(s_row_end[first_row + i], acc);
                            }
#pragma unroll
                            for (uint32_t k = 0; k < V; ++k) acc[k] = identity;
                        }
                    }
                }
            }
        }

        // 7. The scan across workers (in place: acc becomes each worker's inclusive run, then the carry), each
        // worker's first value completed with the run arriving at it.
        const bool reset_at_start = n > 0 && ring(x0) == y0;  // an end before the tile's rows closes the open run
        bool inclusive_reset = bits != 0;
#pragma unroll
        for (uint32_t offset = 1; offset < Lanes; offset <<= 1) {
            if (offset >= W) break;  // warp-uniform
            const uint32_t delta = offset * P;
            const bool up_reset = __shfl_up_sync(0xffffffffU, inclusive_reset ? 1 : 0, delta) != 0;
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) {
                const float up = __shfl_up_sync(0xffffffffU, acc[k], delta);
                if (worker >= offset && !inclusive_reset) acc[k] = op(up, acc[k]);
            }
            if (worker >= offset) inclusive_reset = inclusive_reset || up_reset;
        }
        const bool previous_reset = __shfl_up_sync(0xffffffffU, inclusive_reset ? 1 : 0, P) != 0;
        const uint32_t last_lane = (W - 1) * P + chunk;
        const bool last_reset = __shfl_sync(0xffffffffU, inclusive_reset ? 1 : 0, last_lane) != 0;
#pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            const float seed = reset_at_start ? identity : carry[k];
            const float previous = __shfl_up_sync(0xffffffffU, acc[k], P);
            // The run arriving at this worker: the seed, then the workers before it (from their last reset).
            const float arriving =
                worker == 0 ? seed : (previous_reset ? previous : op(seed, previous));
            if (first_end_row < Rows) first_value[k] = op(arriving, first_value[k]);
            const float last_value = __shfl_sync(0xffffffffU, acc[k], last_lane);
            carry[k] = last_reset ? last_value : op(seed, last_value);
        }
        if (first_end_row < Rows) store_end(s_row_end[first_row + first_end_row], first_value);
        __syncwarp();

        // 8. Advance.
        if (n > 0) last_end = ring(x0 + n - 1);
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
#if THOR_RAGGED_RK_PROBE
        ++probe_tiles;
#endif
    }
#if THOR_RAGGED_RK_PROBE
    if (p.probe != nullptr && lane == 0) probe_c3 = raggedRKClockAfter(x0 + y0, probe_scratch);
#endif

    if (lane == 0 && ktile == 0) raggedStoreRecordKey(p.workspace, range, head_segment);
    if (worker == 0) {
#pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            if (k < chunk_values)
                record_tails[raggedRecordValueIndex(p.ranges, range, static_cast<uint32_t>(column0 + k))] =
                    tail_segment != NO_SEGMENT ? carry[k] : identity;
        }
    }
#if THOR_RAGGED_RK_PROBE
    if (p.probe != nullptr && lane == 0) {
        const unsigned long long c4 = raggedRKClockAfter(0, probe_scratch), t4 = raggedRKGlobalTimer();
        uint32_t smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        unsigned long long* r = p.probe + uint64_t{blockIdx.x} * RAGGED_RK_PROBE_WORDS;
        r[0] = probe_t0;
        r[1] = probe_c0;
        r[2] = probe_c1;
        r[3] = probe_c2;
        r[4] = probe_c3;
        r[5] = c4;
        r[6] = t4;
        r[7] = (static_cast<unsigned long long>(smid) << 32) | probe_tiles;
    }
#endif
}

template <typename InputT, int FixedOp, uint32_t Rows>
void raggedRKLaunchMain(uint32_t grid, const RaggedRKParams& params, cudaStream_t stream) {
    raggedRKValueKernel<InputT, FixedOp, Rows><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    raggedKRCheck(cudaGetLastError(), "RK main kernel launch");
}

template <typename InputT, uint32_t Rows>
void raggedRKLaunchMainFor(int fixed_op, uint32_t grid, const RaggedRKParams& params, cudaStream_t stream) {
    if (fixed_op == 0) {
        raggedRKLaunchMain<InputT, 0, Rows>(grid, params, stream);
    } else {
        raggedRKLaunchMain<InputT, 2, Rows>(grid, params, stream);
    }
}

template <typename InputT, uint32_t Rows>
int raggedRKResidentFor() {
    int resident = 0;
    raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, raggedRKValueKernel<InputT, 0, Rows>,
                                                                RAGGED_KR_LANES, 0),
                  "RK occupancy");
    return resident;
}

// 1285: the dtype objects' exports (CubRaggedRKFp8.cu: E4M3, E5M2, Rows 2; 1288: CubRaggedRKFp8Narrow.cu, Rows 4, in
// its own object so the two compile in parallel). `rows` is raggedRKRowsPerWorker's. Without FP8 types the FP8 ones
// throw. 1295: the same pair for FP16 / BF16 (CubRaggedRKF16.cu, CubRaggedRKF16Narrow.cu) and FP32
// (CubRaggedRKFp32.cu, CubRaggedRKFp32Narrow.cu); a lane's 16-byte chunk is 8 or 4 values of a row.
void raggedRKLaunchMainFp8(DataType input, int fixed_op, uint32_t rows, uint32_t grid, const RaggedRKParams& params,
                           cudaStream_t stream);
void raggedRKLaunchMainFp8Narrow(DataType input, int fixed_op, uint32_t grid, const RaggedRKParams& params,
                                 cudaStream_t stream);
[[nodiscard]] int raggedRKResidentFp8();        // the smaller of the two kernels' residency
[[nodiscard]] int raggedRKResidentFp8Narrow();
void raggedRKLaunchMainF16(DataType input, int fixed_op, uint32_t rows, uint32_t grid, const RaggedRKParams& params,
                           cudaStream_t stream);
void raggedRKLaunchMainF16Narrow(DataType input, int fixed_op, uint32_t grid, const RaggedRKParams& params,
                                 cudaStream_t stream);
[[nodiscard]] int raggedRKResidentF16();
[[nodiscard]] int raggedRKResidentF16Narrow();
void raggedRKLaunchMainFp32(int fixed_op, uint32_t rows, uint32_t grid, const RaggedRKParams& params,
                            cudaStream_t stream);
void raggedRKLaunchMainFp32Narrow(int fixed_op, uint32_t grid, const RaggedRKParams& params, cudaStream_t stream);
[[nodiscard]] int raggedRKResidentFp32();
[[nodiscard]] int raggedRKResidentFp32Narrow();

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
