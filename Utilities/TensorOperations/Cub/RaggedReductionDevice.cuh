#pragma once

// 1265: device helpers shared by the ragged reduction translation units (RaggedKRKernels.cuh, CubRaggedFinish.cu):
// the warp-wide path search, the block-wide segmented scan, the boundary records and the finalizing store. Lean on
// purpose (no CUB or Thrust), so the ragged objects compile quickly.
#include "Utilities/TensorOperations/Cub/CubReductionScalarOps.cuh"
#include "Utilities/TensorOperations/Cub/RaggedKRWalk.h"
#include "Utilities/TensorOperations/Cub/RaggedMergePath.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"

#include <cstdint>

namespace ThorImplementation::CubReductionInternal::RaggedMergePath {

// The coordinate holding the events that start below `cost` under weights `w` (searchPathCost), found by one warp in
// rounds of 32 parallel probes (about log33 of the segment count rounds, each one dependent load) instead of a serial
// binary search's log2. With unit weights `cost` is a diagonal. Every lane returns it.
__device__ __forceinline__ Coordinate raggedWarpSearchCost(const Problem& problem, const PathWeights& w, uint64_t cost,
                                                           uint32_t lane) {
    // The answer is the first x in [lo, hi) whose end does not start below `cost`, or hi.
    const CostSearchRange range = raggedCostSearchRange(problem, w, cost);
    uint64_t lo = range.lo, hi = range.hi;
    const auto consumed = [&](uint64_t x) { return w.end * x + w.row * problem.end(x) < cost; };
    while (hi - lo > 32) {
        const uint64_t step = (hi - lo + 32) / 33;
        uint64_t probe = lo + step * (lane + 1);
        if (probe >= hi) probe = hi - 1;
        const uint32_t below = __ballot_sync(0xffffffffU, consumed(probe));
        const uint32_t count = __popc(below);  // probes are nondecreasing and `consumed` is monotone: a prefix
        const uint64_t first_false = __shfl_sync(0xffffffffU, probe, count < 32 ? count : 31);
        const uint64_t last_true = __shfl_sync(0xffffffffU, probe, count > 0 ? count - 1 : 0);
        if (count > 0) lo = last_true + 1;
        if (count < 32) hi = first_false;
    }
    const uint64_t x = lo + lane;
    const uint32_t below = __ballot_sync(0xffffffffU, x < hi && consumed(x));
    const uint64_t segment = lo + __popc(below);
    return Coordinate{segment, raggedRowsAtCost(problem, w, cost, segment)};
}

// 1270: the same search for two costs at once, one per half warp (lanes 0-15 search `cost` of their half, lanes 16-31
// theirs) in rounds of 16 probes: a one-warp CTA finds its first and last coordinates in about log17 of the segment
// count dependent loads instead of twice log33. Every lane returns its half's coordinate.
// 1272: the rounds narrow the range to 15 candidates, so the last round's 16 lanes read every candidate's begin and end
// (two loads each, one round trip): the coordinate's rows and its segment's bounds come from them, and the CTA's role
// and first segment need no further dependent reads (two round trips fewer before the first tile).
// 1274: the search starts from [0, segments) (1272: the cost's range, which needs the row count), so its first round's
// reads (raggedSearchRound0Load, the raw offsets at raggedSearchRound0Probe) can go out before the row count arrives:
// one dependent round trip fewer before the first tile.
struct RaggedSearchRound0 {
    uint64_t raw_begin = 0;  // offsets[x] (x <= segments)
    uint64_t raw_end = 0;    // offsets[x + 1] (x < segments)
};

__device__ __forceinline__ RaggedSearchRound0 raggedSearchRound0Load(const Offsets& offsets, uint64_t segments,
                                                                     uint32_t lane) {
    const uint64_t x = raggedSearchRound0Probe(segments, lane & 15U);
    RaggedSearchRound0 r;
    if (x <= segments) r.raw_begin = loadOffset(offsets, x);
    if (x < segments) r.raw_end = loadOffset(offsets, x + 1);
    return r;
}

__device__ __forceinline__ SearchedCoordinate raggedHalfWarpSearchCost(const Problem& problem, const PathWeights& w,
                                                                       uint64_t cost, uint32_t lane,
                                                                       const RaggedSearchRound0& round0) {
    const uint32_t half = lane >> 4, slot = lane & 15U, shift = 16U * half;
    const uint64_t segments = problem.segments;
    uint64_t lo = 0, hi = segments;
    const auto below = [&](uint64_t x, uint64_t end_x) { return w.end * x + w.row * end_x < cost; };
    // Narrows [lo, hi) by one round from each probe's end; the same in every half (`searching` per half).
    const auto narrow = [&](bool searching, uint64_t probe, bool consumed) {
        const uint32_t ballot = __ballot_sync(0xffffffffU, searching && consumed);
        const uint32_t count = __popc((ballot >> shift) & 0xffffU);  // a prefix of this half's probes
        const uint64_t first_false = __shfl_sync(0xffffffffU, probe, shift + (count < 16 ? count : 15));
        const uint64_t last_true = __shfl_sync(0xffffffffU, probe, shift + (count > 0 ? count - 1 : 0));
        if (searching) {
            if (count > 0) lo = last_true + 1;
            if (count < 16) hi = first_false;
        }
    };
    if (segments > 15) {  // the first round, from the preloaded offsets (uniform: the segments are the same everywhere)
        const uint64_t probe = raggedSearchRound0Probe(segments, slot);
        narrow(true, probe, below(probe, clampRows(round0.raw_end, problem.rows)));
    }
    while (__any_sync(0xffffffffU, hi - lo > 15)) {
        const bool searching = hi - lo > 15;
        const uint64_t step = (hi - lo + 16) / 17;
        uint64_t probe = lo + step * (slot + 1);
        if (probe >= hi) probe = hi - 1;
        narrow(searching, probe, searching && below(probe, problem.end(probe)));
    }
    // hi - lo <= 15: the answer is lo + count with count <= 15, one of this half's lanes. With at most 15 segments the
    // preloaded first round is this round (lo = 0, x = slot).
    const uint64_t x = lo + slot;
    uint64_t x_begin, x_end;
    if (segments <= 15) {
        x_begin = x <= segments ? clampRows(round0.raw_begin, problem.rows) : problem.rows;
        x_end = x < segments ? clampRows(round0.raw_end, problem.rows) : problem.rows;
    } else {
        x_begin = x <= segments ? problem.begin(x) : problem.rows;
        x_end = x < segments ? problem.end(x) : problem.rows;
    }
    const uint32_t ballot = __ballot_sync(0xffffffffU, x < hi && below(x, x_end));
    const uint32_t count = __popc((ballot >> shift) & 0xffffU);
    const uint64_t segment = lo + count;
    const uint64_t begin = __shfl_sync(0xffffffffU, x_begin, shift + count);
    const uint64_t end = __shfl_sync(0xffffffffU, x_end, shift + count);
    // raggedRowsAtCost: end(segment - 1) is the segment's begin (0 for segment 0), end(segment) its end.
    const uint64_t row = raggedRowsAtCostBounded(w, cost, segment, segment > 0 ? begin : 0, end);
    return SearchedCoordinate{Coordinate{segment, row}, begin, end};
}

template <uint32_t Threads>
struct ScanScratch {
    static constexpr uint32_t Warps = Threads / 32;
    float value[Warps];
    int reset[Warps];
};

// Block-wide exclusive segmented scan of `mine` (raggedCarryCombine), seeded with `seed`: returns seed then every
// earlier thread's element; `total` receives seed then every thread's. Contains __syncthreads; call from all threads.
template <uint32_t Threads, typename Combine>
__device__ __forceinline__ Carry<float> raggedBlockExclusiveScan(const Carry<float>& mine, const Carry<float>& seed,
                                                                 float identity, const Combine& combine,
                                                                 ScanScratch<Threads>& scratch, Carry<float>& total) {
    constexpr uint32_t Warps = Threads / 32;
    static_assert(Threads % 32 == 0 && Warps <= 32);
    const uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x >> 5;
    Carry<float> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const int reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, offset);
        const float value = __shfl_up_sync(0xffffffffU, inclusive.value, offset);
        if (lane >= offset) inclusive = raggedCarryCombine(Carry<float>{reset != 0, value}, inclusive, combine);
    }
    if (lane == 31) {
        scratch.reset[warp] = inclusive.reset ? 1 : 0;
        scratch.value[warp] = inclusive.value;
    }
    __syncthreads();
    if (warp == 0) {
        Carry<float> w = lane < Warps ? Carry<float>{scratch.reset[lane] != 0, scratch.value[lane]}
                                      : Carry<float>{false, identity};
#pragma unroll
        for (uint32_t offset = 1; offset < Warps; offset <<= 1) {
            const int reset = __shfl_up_sync(0xffffffffU, w.reset ? 1 : 0, offset);
            const float value = __shfl_up_sync(0xffffffffU, w.value, offset);
            if (lane >= offset) w = raggedCarryCombine(Carry<float>{reset != 0, value}, w, combine);
        }
        __syncwarp();
        if (lane < Warps) {
            scratch.reset[lane] = w.reset ? 1 : 0;
            scratch.value[lane] = w.value;
        }
    }
    __syncthreads();
    const int previous_reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, 1);
    const float previous_value = __shfl_up_sync(0xffffffffU, inclusive.value, 1);
    const Carry<float> lane_exclusive =
        lane == 0 ? Carry<float>{false, identity} : Carry<float>{previous_reset != 0, previous_value};
    const Carry<float> warp_exclusive = warp == 0 ? Carry<float>{false, identity}
                                                  : Carry<float>{scratch.reset[warp - 1] != 0, scratch.value[warp - 1]};
    total = raggedCarryCombine(seed, Carry<float>{scratch.reset[Warps - 1] != 0, scratch.value[Warps - 1]}, combine);
    const Carry<float> prefix =
        raggedCarryCombine(seed, raggedCarryCombine(warp_exclusive, lane_exclusive, combine), combine);
    __syncthreads();  // the scratch is reused by the next scan
    return prefix;
}

// 1270: warp-wide exclusive segmented scan of `mine` (raggedCarryCombine), seeded with `seed`: returns seed then every
// earlier lane's element; `total` receives seed then every lane's. Call from all 32 lanes.
template <typename Combine>
__device__ __forceinline__ Carry<float> raggedWarpExclusiveScan(const Carry<float>& mine, const Carry<float>& seed,
                                                                float identity, const Combine& combine,
                                                                Carry<float>& total) {
    const uint32_t lane = threadIdx.x & 31U;
    Carry<float> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const int reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, offset);
        const float value = __shfl_up_sync(0xffffffffU, inclusive.value, offset);
        if (lane >= offset) inclusive = raggedCarryCombine(Carry<float>{reset != 0, value}, inclusive, combine);
    }
    const int previous_reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, 1);
    const float previous_value = __shfl_up_sync(0xffffffffU, inclusive.value, 1);
    const Carry<float> lane_exclusive =
        lane == 0 ? Carry<float>{false, identity} : Carry<float>{previous_reset != 0, previous_value};
    const int last_reset = __shfl_sync(0xffffffffU, inclusive.reset ? 1 : 0, 31);
    const float last_value = __shfl_sync(0xffffffffU, inclusive.value, 31);
    total = raggedCarryCombine(seed, Carry<float>{last_reset != 0, last_value}, combine);
    return raggedCarryCombine(seed, lane_exclusive, combine);
}

// 1280: the same scan over `workers` workers of `components` lanes each (lane = worker * components + component, the
// components kernel): each component's elements are scanned across the workers, in worker order. `total` receives seed
// then every worker's element of the lane's component. Idle lanes (worker >= workers) get values they must not use.
// `workers` is warp-uniform. Call from all 32 lanes.
template <typename Combine>
__device__ __forceinline__ Carry<float> raggedWarpExclusiveScanAcrossWorkers(const Carry<float>& mine,
                                                                             const Carry<float>& seed, float identity,
                                                                             const Combine& combine, uint32_t components,
                                                                             uint32_t workers, uint32_t worker,
                                                                             uint32_t component, Carry<float>& total) {
    Carry<float> inclusive = mine;
    // 1281: unrolled over the at most 4 rounds (workers <= 16 at D >= 2), stopping at a warp-uniform test (1280's
    // runtime loop cost a reconvergence and its bookkeeping per round).
#pragma unroll
    for (uint32_t offset = 1; offset < 16; offset <<= 1) {
        if (offset >= workers) break;
        const uint32_t delta = offset * components;
        const int reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, delta);
        const float value = __shfl_up_sync(0xffffffffU, inclusive.value, delta);
        if (worker >= offset) inclusive = raggedCarryCombine(Carry<float>{reset != 0, value}, inclusive, combine);
    }
    const int previous_reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, components);
    const float previous_value = __shfl_up_sync(0xffffffffU, inclusive.value, components);
    const Carry<float> worker_exclusive =
        worker == 0 ? Carry<float>{false, identity} : Carry<float>{previous_reset != 0, previous_value};
    const uint32_t last_lane = (workers - 1) * components + component;  // < 32 for every lane's component
    const int last_reset = __shfl_sync(0xffffffffU, inclusive.reset ? 1 : 0, last_lane);
    const float last_value = __shfl_sync(0xffffffffU, inclusive.value, last_lane);
    total = raggedCarryCombine(seed, Carry<float>{last_reset != 0, last_value}, combine);
    return raggedCarryCombine(seed, worker_exclusive, combine);
}

// MEAN's divide, out of line once per translation unit: inline, nvcc copies the IEEE divide and its slow path into
// every store site and output dtype (the P5-A lesson).
inline __device__ __noinline__ float raggedMeanOutOfLine(float sum, uint64_t count) {
    return count == 0 ? 0.0f : sum / static_cast<float>(count);
}

// Finalizes one segment's FP32 aggregate (MEAN divides by its rows; MAX undoes the sign fold) and stores it.
__device__ __forceinline__ void raggedStoreValue(void* output, DataType output_dtype, uint64_t segment, float value,
                                                 const RaggedFinalize& finalize, uint64_t count) {
    if (finalize.mean) value = raggedMeanOutOfLine(value, count);
    storeFp32AsRuntimeDType(output, output_dtype, segment, value * finalize.sign);
}

// 1280: the ragged KR VALUE boundary records as three arrays (raggedRecordLayout, RaggedKRWalk.h; 1273-1279: one
// 16-byte record per CTA): lane 0 stores its CTA's key, and the lane of each component its head and tail values.
__device__ __forceinline__ void raggedStoreRecordKey(uint8_t* workspace, uint32_t cta, uint64_t key) {
    reinterpret_cast<uint64_t*>(workspace)[cta] = key;
}

__device__ __forceinline__ void raggedStoreRecordValues(uint8_t* workspace, const RaggedRecordLayout& layout,
                                                        uint32_t grid, uint32_t cta, uint32_t component,
                                                        float head_value, float tail_value) {
    const uint64_t i = raggedRecordValueIndex(grid, cta, component);
    reinterpret_cast<float*>(workspace + layout.heads)[i] = head_value;
    reinterpret_cast<float*>(workspace + layout.tails)[i] = tail_value;
}

// Boundary records (RaggedMergePath::boundaryLayout with one FP32 partial): a segment id, then the value at byte 16.
__device__ __forceinline__ void raggedWriteRecord(uint8_t* workspace, uint64_t record, uint64_t segment, float value) {
    *reinterpret_cast<uint64_t*>(workspace + record) = segment;
    *reinterpret_cast<float*>(workspace + record + 16) = value;
}

__device__ __forceinline__ void raggedWriteRecordSegment(uint8_t* workspace, uint64_t record, uint64_t segment) {
    *reinterpret_cast<uint64_t*>(workspace + record) = segment;
}

__device__ __forceinline__ void raggedWriteRecordValue(uint8_t* workspace, uint64_t record, float value) {
    *reinterpret_cast<float*>(workspace + record + 16) = value;
}

__device__ __forceinline__ uint64_t raggedReadRecordSegment(const uint8_t* workspace, uint64_t record) {
    return *reinterpret_cast<const uint64_t*>(workspace + record);
}

__device__ __forceinline__ float raggedReadRecordValue(const uint8_t* workspace, uint64_t record) {
    return *reinterpret_cast<const float*>(workspace + record + 16);
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedMergePath
