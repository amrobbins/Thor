#pragma once

// 1314 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged RK ARG main kernel, rows of
// R = D x element bytes >= 16, FP8 first (CubRaggedRKArgFp8.cu: argmin and argmax, E4M3 and E5M2 in one kernel per rows
// per worker; 1320: CubRaggedRKArgF16.cu, FP16 and BF16 in one kernel per rows per worker, RaggedArgKeyF16: 8 columns
// a lane, candidates key << 16 | field << 1, the marker key 0xFFFF, 47-bit record rows; 1321: CubRaggedRKArgFp32.cu,
// FP32, RaggedArgKeyFp32: 4 columns a lane, 64-bit candidates key << 32 | field << 1, the marker key 0xFFFFFFFF, and
// pair records). The host launcher is CubRaggedRKArg.cu; the finisher CubRaggedFinishArg.cu (D up to 15) and
// CubRaggedFinishArgWide.cu (wider D), 1321: for FP32's pairs CubRaggedFinishArgFp32.cu (D up to 15) and
// CubRaggedFinishArgPairWide.cu (wider D).
//
// The walk and the geometry are the RK VALUE kernel's (raggedRKValueKernel, RaggedRKKernels.cuh, with its history):
// one warp per CTA, persistent, CTA c takes K tile c mod ktiles of merge-path range c / ktiles; the ends ring and its
// prefetch, the tile end, the bitmap, each end's kind and each Ranked end's row; a lane owns one 16-byte chunk of each
// of its worker's Rows rows (any alignment), so V = 16 / element bytes columns; ends without rows in the tile go
// through a pass of W per round; a segment that ends inside a worker after its first end there is complete in the lane
// and stored at once; the scan across workers carries each column; each worker's first winner takes the run arriving
// at it. Only the values differ, as the KR ARG kernels' (RaggedKRArgKernels.cuh, with their history):
//   - a column's candidate in a tile is 32-bit, its ARG key on top and (tile row + 1) << 1 below (the key format's
//     traits: RaggedArgKeyFp8's lean keys, one byte permute per candidate); field 0 is the token of the run arriving
//     from the tiles before (its key alone: those rows come first, so it wins its key's ties);
//   - the fold and the scan are an unsigned minimum, the identity all ones; rows past the tile's end take the marker key
//     (0xFF), and a run of only such rows becomes the identity;
//   - 1315: the carry between tiles is, per column, the open run's token (its winner's key with row field 0, or the
//     identity) and its winner's CTA-local row (1314: the 64-bit word with a global row, rebuilt into a token and
//     widened back every tile, about 25 instructions a column a tile). A tile winner's CTA-local row is the carry's for
//     field 0, else y0 + field - 1; an index is the lane's base (row_base x D + its first column) + k + local row x D;
//     the 64-bit words of RaggedArg.h (global rows) are built only for the records (raggedArgRecordLayout(ranges, D):
//     8-byte head and tail words per column and range);
//   - 1316: no open run is the token NoRun (the marker key, field 0), not the identity: it loses to every real row, so
//     every winner with the marker key (NoRun, the identity, a marker row) means no rows (the sentinel, the identity
//     word). The carry then moves on without a branch: the token is the warp total & NoRun, the row is set where the
//     total's field is not 0; and a reset at the tile's start sets the tokens once (1315 selected the seed in two loops
//     and moved the carry through two branches, about 17 instructions a column a tile in the profile, against 1 for
//     VALUE's minimum);
//   - 1321: FP32 (Key::LocalRows, 64-bit lanes): a 32-bit key and a row do not fit one 64-bit word, so the records are
//     pairs (RaggedArg.h's raggedArgPairRecordLayout(ranges, D): a 32-bit key and a 64-bit global row per column and
//     range, as 1313's FP32 KR ARG); the rest of the kernel is the same expressions at 64-bit lanes, and the FP8 / F16
//     records' code is unchanged (its own constexpr branch);
//   - the outputs are indices row * D + column (UINT32 or UINT64 at run time), the sentinel for a segment without rows,
//     in 16-byte stores where the address allows.
// Columns past the K tile's bytes (a lane's partial last chunk) are never stored, so they are not masked.
#include "Utilities/TensorOperations/Cub/RaggedKRArgKernels.cuh"
#include "Utilities/TensorOperations/Cub/RaggedRKKernels.cuh"

namespace ThorImplementation::CubReductionInternal::RaggedKR {

struct RaggedRKArgParams {
    const uint8_t* bytes;  // the input's bytes, from its 16-byte aligned base
    uint32_t head;         // row 0's byte offset from `bytes` (0 to 15, a multiple of the element's bytes)
    RaggedMergePath::Offsets offsets;
    uint64_t segments;
    uint64_t capacity_rows;
    RaggedMergePath::PathWeights weights;
    Fp8ArgFold fold;       // argmax: the sign flip of every element; the format's NaN bias
    void* output;          // segments x D indices
    uint32_t index_bytes;  // 4 (UINT32) or 8 (UINT64)
    uint8_t* workspace;    // raggedArgRecordLayout(ranges, components); 1321 FP32: raggedArgPairRecordLayout
    uint64_t components;   // D
    uint32_t row_bytes;    // R = D x element bytes
    uint32_t ranges;       // path ranges; gridDim.x = ranges x ktiles
    uint32_t ktiles;
    uint32_t tile_bytes;  // T
    uint32_t chunks;      // P
    uint32_t workers;     // W
};

// 1317: V UINT32 indices from `out`, 16-byte stores from the first 16-byte boundary, Head scalars before it and
// 4 - Head after the last 16-byte store (raggedRKStoreFp32From's form, RK VALUE's 1286 FP32 stores).
template <uint32_t V, uint32_t Head>
__device__ __forceinline__ void raggedRKArgStoreU32From(uint32_t* out, const uint64_t (&v)[V]) {
    static_assert(Head < 4 && V % 4 == 0);
    if constexpr (Head > 0) {  // no loop at Head 0 (nvcc's #186-D, 1286a)
#pragma unroll
        for (uint32_t k = 0; k < Head; ++k) out[k] = static_cast<uint32_t>(v[k]);
    }
#pragma unroll
    for (uint32_t k = Head; k + 4 <= V; k += 4)
        *reinterpret_cast<uint4*>(out + k) = make_uint4(static_cast<uint32_t>(v[k]), static_cast<uint32_t>(v[k + 1]),
                                                        static_cast<uint32_t>(v[k + 2]), static_cast<uint32_t>(v[k + 3]));
    if constexpr (Head > 0) {
#pragma unroll
        for (uint32_t k = V - 4 + Head; k < V; ++k) out[k] = static_cast<uint32_t>(v[k]);
    }
}

// One lane's indices of an end: `valid` (at most V) consecutive ones from output element `index`. With all V valid:
// UINT32 in 16-byte stores from the first 16-byte boundary at any 4-byte alignment (1317; 1314 only when 16-byte
// aligned, which rows of D x 4 bytes rarely are at odd D: the 1315 profile's shape 95 wrote 16 scalar stores a lane,
// about 7.7x the ideal L2 sectors), UINT64 in 16-byte stores when 16-byte aligned; else one at a time.
template <uint32_t V>
__device__ __forceinline__ void raggedRKArgStoreIndices(void* output, uint32_t index_bytes, uint64_t index,
                                                        const uint64_t (&v)[V], uint32_t valid) {
    if (valid == V) {
        if (index_bytes == 4) {
            uint32_t* out = static_cast<uint32_t*>(output) + index;
            if constexpr (V % 4 == 0) {
                if ((reinterpret_cast<uintptr_t>(out) & 3U) == 0) {
                    const uint32_t misaligned = static_cast<uint32_t>((reinterpret_cast<uintptr_t>(out) >> 2) & 3U);
                    switch ((4U - misaligned) & 3U) {  // the scalars before the first 16-byte boundary
                        case 0: raggedRKArgStoreU32From<V, 0>(out, v); return;
                        case 1: raggedRKArgStoreU32From<V, 1>(out, v); return;
                        case 2: raggedRKArgStoreU32From<V, 2>(out, v); return;
                        default: raggedRKArgStoreU32From<V, 3>(out, v); return;
                    }
                }
            }
        } else {
            unsigned long long* out = static_cast<unsigned long long*>(output) + index;
            if constexpr (V % 2 == 0) {
                if ((reinterpret_cast<uintptr_t>(out) & 15U) == 0) {
#pragma unroll
                    for (uint32_t k = 0; k < V; k += 2)
                        *reinterpret_cast<ulonglong2*>(out + k) = make_ulonglong2(v[k], v[k + 1]);
                    return;
                }
            }
        }
    }
#pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        if (k < valid) RaggedMergePath::raggedArgStoreIndex(output, index_bytes, index + k, v[k]);
    }
}

template <typename Key, uint32_t Rows>
__global__ __launch_bounds__(RAGGED_KR_LANES, RAGGED_RK_MIN_CTAS_PER_SM) void raggedRKArgKernel(
    const RaggedRKArgParams p) {
    using namespace RaggedMergePath;
    using Lane = typename Key::Lane;
    // 32-bit tile candidates and global-row words (FP8, F16), or 1321's 64-bit candidates and pair records (FP32).
    static_assert(Key::LocalRows == (sizeof(Lane) == 8) && (sizeof(Lane) == 4 || sizeof(Lane) == 8));
    constexpr uint32_t Lanes = RAGGED_KR_LANES;
    constexpr uint32_t Bytes = Key::Bytes;
    constexpr uint32_t PerWord = Key::PerWord;
    constexpr uint32_t V = RAGGED_RK_CHUNK_BYTES / Bytes;  // columns per lane
    constexpr uint32_t KeyShift = 8 * sizeof(Lane) - Key::KeyBits;
    constexpr uint32_t KeyMask = 0xFFFFFFFFU >> (32 - Key::KeyBits);
    constexpr uint32_t FieldMask = (1U << (KeyShift - 1)) - 1U;
    constexpr Lane IdentityLane = ~Lane{0};
    // 1316: no open run (the marker key with field 0). Every lane value at or above it has the marker key: no real row.
    constexpr Lane NoRun = Lane{KeyMask} << KeyShift;
    constexpr uint64_t Identity64 = RAGGED_ARG_IDENTITY;
    constexpr uint32_t SpanBytes = raggedRKSpanBytesFor(Rows);
    constexpr uint32_t ContiguousCopies = (raggedRKStagedChunksFor(Rows) + Lanes - 1) / Lanes;
    constexpr uint32_t MaxTileRows = Rows * Lanes;
    constexpr uint32_t Ring = RAGGED_KR_RING;
    constexpr uint32_t Mask = Ring - 1;
    constexpr uint32_t EndsPerLane = Ring / Lanes;
    constexpr uint32_t BitWords = MaxTileRows / 32;
    constexpr uint32_t RowMask = (1U << Rows) - 1U;
    static_assert((Rows == RAGGED_RK_ROWS || Rows == RAGGED_RK_ROWS_NARROW) && MaxTileRows <= 255 && Ring <= 255 &&
                  SpanBytes % 16 == 0 && 32 % Rows == 0 && V == 4 * PerWord && MaxTileRows < FieldMask);
    __shared__ uint32_t s_ends[Ring];
    __shared__ uint8_t s_kind[Ring];
    __shared__ uint8_t s_row_end[MaxTileRows];
    __shared__ uint32_t s_bits[BitWords];
    __shared__ __align__(16) uint32_t s_rows[2][SpanBytes / 4];

    const auto lane_min = [](Lane a, Lane b) { return a < b ? a : b; };
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
    const uint64_t column0 = (k0 + RAGGED_RK_CHUNK_BYTES * chunk) / Bytes;  // the lane's first column
    const bool aligned = ((p.head | R) & 15U) == 0;
    const uint32_t tile_rows = Rows * W;
    const uint32_t slot_bytes = raggedRKSlotBytes(geometry);

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
    const uint64_t byte_base = p.head + row_base * R + k0;
    const uint64_t first_begin = seg_count > 0 ? first.begin : 0;
    const uint64_t first_end = seg_count > 0 ? first.end : 0;
    const bool first_has_rows = first_begin < first_end;
    const RaggedArgRecordLayout layout = raggedArgRecordLayout(p.ranges, static_cast<uint32_t>(p.components));
    uint64_t* record_heads = reinterpret_cast<uint64_t*>(p.workspace + layout.heads);
    uint64_t* record_tails = reinterpret_cast<uint64_t*>(p.workspace + layout.tails);

    const auto ring = [&](uint32_t segment) -> uint32_t { return s_ends[segment & Mask]; };
    const auto end_rel = [&](uint32_t segment) -> uint32_t {
        return static_cast<uint32_t>(problem.end(seg_base + segment) - row_base);
    };
    uint32_t task_row[3], task_chunk[3];
#pragma unroll
    for (uint32_t q = 0; q < 3; ++q) {
        const uint32_t t = lane + Lanes * q;
        task_row[q] = t / (P + 1);
        task_chunk[q] = t - task_row[q] * (P + 1);
    }
    const auto fetch_rows = [&](uint32_t y, uint32_t buffer) {
        if (y >= row_count) return;
        const uint32_t rows = (y + tile_rows < row_count ? y + tile_rows : row_count) - y;
        uint8_t* staged = reinterpret_cast<uint8_t*>(s_rows[buffer]);
        if (geometry.contiguous) {
            const uint64_t start = byte_base + uint64_t{y} * R;
            const uint64_t origin = start & ~uint64_t{15};
            const uint32_t bytes = static_cast<uint32_t>(start - origin) + rows * R;
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

    uint32_t x0 = 0, held = 0, buffer = 0, last_end = 0;
    {
        const uint32_t count = raggedEndsToLoad(0, Ring, seg_count);
        fetch_rows(0, 0);
        raggedCpAsyncCommit();
        load_ends(0, count);
        held = count;
        raggedCpAsyncWaitAll();
    }
    __syncwarp();

    // 1315: the open segment's winner so far, this lane's columns (the same in every worker): its token (key << KeyShift
    // with row field 0, or 1316's NoRun) and its CTA-local row.
    Lane carry_token[V];
    uint32_t carry_row[V];
#pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        carry_token[k] = NoRun;
        carry_row[k] = 0;
    }
    const uint32_t D32 = static_cast<uint32_t>(p.components);  // D <= 2^30 (the finisher's limit)
    const uint64_t lane_index_base = row_base * p.components + column0;  // the index of local row 0, column 0
    const uint64_t sentinel = p.index_bytes == 8 ? ~uint64_t{0} : uint64_t{0xFFFFFFFFU};
    uint32_t y0 = 0;  // the tile's first CTA-local row (the walk's)
    // A real tile winner's CTA-local row: the carry's for field 0 (the token; never after a reset at the tile's start,
    // whose seed is NoRun), else tile row field - 1.
    const auto local_row = [&](uint32_t k, Lane winner) -> uint32_t {
        const uint32_t field = (winner >> 1) & FieldMask;
        return field == 0 ? carry_row[k] : y0 + field - 1U;
    };
    const auto index_of = [&](uint32_t k, Lane winner) -> uint64_t {
        if (winner >= NoRun) return sentinel;  // the marker key: no rows
        return lane_index_base + k + static_cast<uint64_t>(local_row(k, winner)) * D32;
    };
    const auto word_of = [&](uint32_t k, Lane winner) -> uint64_t {  // the 64-bit record word (global row)
        if (winner >= NoRun) return Identity64;
        return (uint64_t{winner >> KeyShift} << (Key::RowBits + 1)) | ((row_base + local_row(k, winner)) << 1);
    };
    // 1321: FP32's pair records (raggedArgPairRecordLayout(ranges, D)): column k's head or tail pair of this range, the
    // winner's key and its global row, or the pair identity for no rows.
    const RaggedArgPairRecordLayout pair_layout =
        raggedArgPairRecordLayout(p.ranges, static_cast<uint32_t>(p.components));
    const auto store_pair = [&](uint64_t keys_at, uint64_t rows_at, uint32_t k, Lane winner) {
        const uint64_t i = (column0 + k) * pair_layout.stride + range;
        const bool none = winner >= NoRun;
        reinterpret_cast<uint32_t*>(p.workspace + keys_at)[i] =
            none ? RAGGED_ARG_PAIR_KEY_IDENTITY : static_cast<uint32_t>(winner >> KeyShift);
        reinterpret_cast<uint64_t*>(p.workspace + rows_at)[i] =
            none ? RAGGED_ARG_PAIR_ROW_IDENTITY : row_base + local_row(k, winner);
    };

    // End i of the tile, complete in this lane, its column k's tile winner winner(k): its indices (or, for the CTA's
    // head, its record's head words).
    const auto store_end = [&](uint32_t i, const auto& winner) {
        if (x0 + i == 0 && has_head) {
            if constexpr (Key::LocalRows) {  // 1321: FP32's pair records
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) {
                    if (k < chunk_values) store_pair(pair_layout.head_keys, pair_layout.head_rows, k, winner(k));
                }
            } else {
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) {
                    if (k < chunk_values)
                        record_heads[raggedRecordValueIndex(p.ranges, range, static_cast<uint32_t>(column0 + k))] =
                            word_of(k, winner(k));
                }
            }
            return;
        }
        uint64_t indices[V];
#pragma unroll
        for (uint32_t k = 0; k < V; ++k) indices[k] = index_of(k, winner(k));
        raggedRKArgStoreIndices<V>(p.output, p.index_bytes, (seg_base + x0 + i) * p.components + column0, indices,
                                   chunk_values);
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
                    const bool carried = kind == RaggedEndKind::CarryIn;
                    store_end(i, [&](uint32_t k) { return carried ? carry_token[k] : IdentityLane; });
                }
            }
        }

        // 6. This lane's chunk of its worker's rows: their candidates, folded and cut at the worker's ends. Tile row j
        // is CTA-local row y0 + j, its candidates' row field j + 1 (field 0 is the carry's token).
        const uint32_t rows_in_tile = y1 - y0;
        const uint32_t tile_shift = static_cast<uint32_t>((byte_base + uint64_t{y0} * R) & 15U);
        const uint32_t first_row = Rows * worker;
        const uint32_t bits = lane_active ? (s_bits[first_row >> 5] >> (first_row & 31U)) & RowMask : 0;
        const bool masked = __any_sync(0xffffffffU, lane_active && first_row + Rows > rows_in_tile);  // warp-uniform
        const bool reset_at_start = n > 0 && ring(x0) == y0;  // an end before the tile's rows closes the open run
        const uint8_t* staged = reinterpret_cast<const uint8_t*>(s_rows[buffer]);
        // Row i's keys (four words of V / 4 columns each); a row past the tile's end takes the marker key.
        const auto keys_of_row = [&](uint32_t i, uint32_t (&keys)[4]) {
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
            for (uint32_t w = 0; w < 4; ++w) keys[w] = Key::keys(word[w], p.fold);
            if (masked && j >= rows_in_tile) {
#pragma unroll
                for (uint32_t w = 0; w < 4; ++w) keys[w] = 0xFFFFFFFFU;  // every element's key KeyMask
            }
        };
        const auto candidate = [&](const uint32_t (&keys)[4], uint32_t k, uint32_t field2) -> Lane {
            return Key::candidate(keys[k / PerWord], k % PerWord, field2);
        };
        const uint32_t field_base = 2U * (first_row + 1U);  // row i's field2: field_base + 2 i
        Lane acc[V];
        Lane first_value[V];
        uint32_t first_end_row = Rows;  // Rows: no end in the worker's rows
        // The ended path: row i folded into acc, then (at an end) the worker's first winner kept, a later one stored.
        const auto fold_row_ended = [&](uint32_t i) {
            uint32_t keys[4];
            keys_of_row(i, keys);
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) acc[k] = lane_min(acc[k], candidate(keys, k, field_base + 2U * i));
            if ((bits >> i) & 1U) {
                if (first_end_row == Rows) {
                    first_end_row = i;
#pragma unroll
                    for (uint32_t k = 0; k < V; ++k) first_value[k] = acc[k];
                } else {
                    store_end(s_row_end[first_row + i], [&](uint32_t k) { return acc[k]; });
                }
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) acc[k] = IdentityLane;
            }
        };
        if constexpr (Rows == RAGGED_RK_ROWS) {
            if (!__any_sync(0xffffffffU, bits != 0)) {
                uint32_t keys0[4], keys1[4];
                keys_of_row(0, keys0);
                keys_of_row(1, keys1);
#pragma unroll
                for (uint32_t k = 0; k < V; ++k)
                    acc[k] = lane_min(candidate(keys0, k, field_base), candidate(keys1, k, field_base + 2U));
            } else {
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) acc[k] = IdentityLane;
#pragma unroll
                for (uint32_t i = 0; i < Rows; ++i) fold_row_ended(i);
            }
        } else {
            // The narrow kernels: rows in pairs, a loop over the pairs (the VALUE kernel's 1289 form).
            if (!__any_sync(0xffffffffU, bits != 0)) {
                uint32_t keys0[4], keys1[4];
                keys_of_row(0, keys0);
                keys_of_row(1, keys1);
#pragma unroll
                for (uint32_t k = 0; k < V; ++k)
                    acc[k] = lane_min(candidate(keys0, k, field_base), candidate(keys1, k, field_base + 2U));
#pragma unroll 1
                for (uint32_t base = 2; base < Rows; base += 2) {
                    keys_of_row(base, keys0);
                    keys_of_row(base + 1, keys1);
#pragma unroll
                    for (uint32_t k = 0; k < V; ++k)
                        acc[k] = lane_min(acc[k], lane_min(candidate(keys0, k, field_base + 2U * base),
                                                           candidate(keys1, k, field_base + 2U * base + 2U)));
                }
            } else {
#pragma unroll
                for (uint32_t k = 0; k < V; ++k) acc[k] = IdentityLane;
#pragma unroll 1
                for (uint32_t base = 0; base < Rows; base += 2) {
                    fold_row_ended(base);
                    fold_row_ended(base + 1);
                }
            }
        }
        if (masked) {  // a run of only rows past the tile's end
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) {
                if ((acc[k] >> KeyShift) == KeyMask) acc[k] = IdentityLane;
            }
        }

        // 7. The scan across workers (in place: acc becomes each worker's inclusive run), seeded with each column's
        // token; each worker's first winner completed with the run arriving at it; the carry from the warp's total.
        bool inclusive_reset = bits != 0;
#pragma unroll
        for (uint32_t offset = 1; offset < Lanes; offset <<= 1) {
            if (offset >= W) break;  // warp-uniform
            const uint32_t delta = offset * P;
            const bool up_reset = __shfl_up_sync(0xffffffffU, inclusive_reset ? 1 : 0, delta) != 0;
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) {
                const Lane up = __shfl_up_sync(0xffffffffU, acc[k], delta);
                if (worker >= offset && !inclusive_reset) acc[k] = lane_min(up, acc[k]);
            }
            if (worker >= offset) inclusive_reset = inclusive_reset || up_reset;
        }
        const bool previous_reset = __shfl_up_sync(0xffffffffU, inclusive_reset ? 1 : 0, P) != 0;
        const uint32_t last_lane = (W - 1) * P + chunk;
        const bool last_reset = __shfl_sync(0xffffffffU, inclusive_reset ? 1 : 0, last_lane) != 0;
        // 1316: an end before the tile's rows closed the open run (its carry-in end was stored in step 5), so the tokens
        // are NoRun from here: set once (warp-uniform), and both loops below seed with the token as it stands.
        if (reset_at_start) {
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) carry_token[k] = NoRun;
        }
#pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            const Lane seed = carry_token[k];
            const Lane previous = __shfl_up_sync(0xffffffffU, acc[k], P);
            const Lane arriving = worker == 0 ? seed : (previous_reset ? previous : lane_min(seed, previous));
            if (first_end_row < Rows) first_value[k] = lane_min(arriving, first_value[k]);
        }
        // The first winner may be the token (field 0: the carry's row), so it is stored before the carry moves on.
        if (first_end_row < Rows)
            store_end(s_row_end[first_row + first_end_row], [&](uint32_t k) { return first_value[k]; });
        // The carry moves on, without a branch (1316): the token is the total's key with field 0 (the same token when
        // it won, NoRun for the identity or a marker), and the row moves where a tile row won (field not 0; for the
        // identity too, whose token NoRun never reads it).
#pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            const Lane last_value = __shfl_sync(0xffffffffU, acc[k], last_lane);
            const Lane total = last_reset ? last_value : lane_min(carry_token[k], last_value);
            const uint32_t field = (total >> 1) & FieldMask;
            carry_row[k] = field != 0 ? y0 + field - 1U : carry_row[k];
            carry_token[k] = total & NoRun;
        }
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
    }

    if (lane == 0 && ktile == 0) reinterpret_cast<uint64_t*>(p.workspace + layout.keys)[range] = head_segment;
    if (worker == 0) {
        if constexpr (Key::LocalRows) {  // 1321: FP32's pair records (their keys array is also at offset 0)
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) {
                if (k < chunk_values)
                    store_pair(pair_layout.tail_keys, pair_layout.tail_rows, k,
                               tail_segment != NO_SEGMENT ? carry_token[k] : NoRun);
            }
        } else {
#pragma unroll
            for (uint32_t k = 0; k < V; ++k) {
                if (k < chunk_values)
                    record_tails[raggedRecordValueIndex(p.ranges, range, static_cast<uint32_t>(column0 + k))] =
                        tail_segment != NO_SEGMENT ? word_of(k, carry_token[k]) : Identity64;
            }
        }
    }
}

template <typename Key, uint32_t Rows>
int raggedRKArgResidentFor() {
    int resident = 0;
    raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, raggedRKArgKernel<Key, Rows>,
                                                                RAGGED_KR_LANES, 0),
                  "RK ARG occupancy");
    return resident;
}

// 1314: the FP8 object's exports (CubRaggedRKArgFp8.cu): `rows` is raggedRKRowsPerWorker's (2, or 4 on narrow rows);
// the residency is the smaller of the two kernels'.
void raggedRKArgLaunchMainFp8(uint32_t rows, uint32_t grid, const RaggedRKArgParams& params, cudaStream_t stream);
[[nodiscard]] int raggedRKArgResidentFp8();
// 1320: the FP16 / BF16 object's (CubRaggedRKArgF16.cu), the same form.
void raggedRKArgLaunchMainF16(uint32_t rows, uint32_t grid, const RaggedRKArgParams& params, cudaStream_t stream);
[[nodiscard]] int raggedRKArgResidentF16();
// 1321: the FP32 object's (CubRaggedRKArgFp32.cu), the same form; its records are pairs.
void raggedRKArgLaunchMainFp32(uint32_t rows, uint32_t grid, const RaggedRKArgParams& params, cudaStream_t stream);
[[nodiscard]] int raggedRKArgResidentFp32();

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
