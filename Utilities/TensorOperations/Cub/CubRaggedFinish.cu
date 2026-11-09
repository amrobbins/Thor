// 1265: the ragged reductions' finishing kernel (plans/Thor_Ragged_Reduction_Plan_2026-11-06.md, design: finishing
// kernel). One CTA of 1024 threads reads the main kernel's boundary records (1273: one 16-byte record per CTA,
// RaggedKRWalk.h): a block-wide segmented scan over the CTAs' tails, in CTA order, restarting at each head's CTA,
// gives each head the run of its segment's rows in the CTAs before it, so a segment spanning every CTA costs no more
// than one spanning two. Fixed order, no atomics. Kernels: {additive, minimum}.
// 1280: the records are three arrays (raggedRecordLayout: keys, then each component's head values, then its tail
// values), and the finisher runs one CTA per component (D up to 15), each the D = 1 algorithm on its component.
// 1285 (ragged RK, D >= 16 / element bytes): any D. 1291: D above 15 takes the wide finisher
// (raggedFinishValueWideKernel: up to 16 warps per component, every load first, one warp scan per warp; 1294: the
// records transposed through shared memory above 128 records, read straight per lane up to 128); D up to 15 the block
// per component.
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"
#include "Utilities/TensorOperations/Cub/RaggedReductionDevice.cuh"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {
namespace {

using RaggedMergePath::Carry;
using RaggedMergePath::Problem;

constexpr uint32_t FINISH_THREADS = RAGGED_FINISH_THREADS;
constexpr uint32_t FINISH_ROUNDS = RAGGED_FINISH_MAX_GRID / FINISH_THREADS;

struct RaggedFinishParams {
    const uint8_t* workspace;  // the main kernel's records (raggedRecordLayout of grid and components)
    uint32_t grid;
    uint32_t components;  // D: one finisher CTA per component
    RaggedMergePath::Offsets offsets;
    uint64_t segments;
    uint64_t capacity_rows;
    RaggedFinalize finalize;
    void* output;
    DataType output_dtype;
    uint32_t warps_per_component;  // 1291: the wide finisher's (raggedFinishWideLaunch)
    uint32_t components_per_cta;
};

// 1273: thread t loads CTAs t, t + 1024, t + 2048 and t + 3072, so every record load is one 16-byte load per lane at
// consecutive addresses (64 KB in all for 4096 CTAs, coalesced). The 1272 profile put this kernel at 7.5-7.7 us of a
// 1 MiB cell's 13: two 32-byte records per CTA (256 KB) read at a 128-byte stride by one SM, latency-bound.
// 1274: one block scan, not four. The tails and head flags go to shared memory in CTA order; thread t folds CTAs
// 4t .. 4t + 3, one block-wide exclusive scan carries the runs between threads, and each thread writes the run arriving
// at each of its four CTAs back in place; the loading threads then finish their heads from it. (1273 scanned in four
// rounds of 1024 CTAs, about 0.5 us each: 3.9 us at 4080 CTAs against 2.0 us at 1360.)
template <int FixedOp>
__global__ __launch_bounds__(FINISH_THREADS) void raggedFinishValueKernel(const RaggedFinishParams p) {
    using namespace RaggedMergePath;
    static_assert(FixedOp == 0 || FixedOp == 2);
    static_assert(FINISH_ROUNDS == 4 && FINISH_THREADS * FINISH_ROUNDS == RAGGED_FINISH_MAX_GRID);
    __shared__ __align__(16) float s_run[RAGGED_FINISH_MAX_GRID];  // each CTA's tail, then the run arriving at it
    __shared__ __align__(4) uint8_t s_head[RAGGED_FINISH_MAX_GRID];  // 1 where the CTA has a head (the run restarts)
    __shared__ ScanScratch<FINISH_THREADS> s_scan;

    const Fp8TypedReductionOp<FixedOp> op{};
    const float identity = op.identity();
    const uint32_t tid = threadIdx.x;
    const uint32_t component = blockIdx.x;
    const RaggedRecordLayout layout = raggedRecordLayout(p.grid, p.components);
    const uint64_t* record_keys = reinterpret_cast<const uint64_t*>(p.workspace + layout.keys);
    const float* record_heads = reinterpret_cast<const float*>(p.workspace + layout.heads);
    const float* record_tails = reinterpret_cast<const float*>(p.workspace + layout.tails);

    // Every read first: the records, then MEAN's head bounds (the problem's row count loads with the records). CTAs at
    // or past the grid have no head and the identity tail.
    uint64_t key[FINISH_ROUNDS];
    float head_value[FINISH_ROUNDS];
#pragma unroll
    for (uint32_t r = 0; r < FINISH_ROUNDS; ++r) {
        const uint32_t c = tid + r * FINISH_THREADS;
        float tail = identity;
        key[r] = NO_SEGMENT;
        head_value[r] = identity;
        if (c < p.grid) {
            const uint64_t i = raggedRecordValueIndex(p.grid, c, component);
            key[r] = record_keys[c];
            head_value[r] = record_heads[i];
            tail = record_tails[i];
        }
        s_run[c] = tail;
        s_head[c] = raggedRecordHasHead(key[r]) ? 1 : 0;
    }
    const Problem problem = makeProblem(p.offsets, p.segments, p.capacity_rows);
    uint64_t count[FINISH_ROUNDS];
#pragma unroll
    for (uint32_t r = 0; r < FINISH_ROUNDS; ++r) {
        count[r] = p.finalize.mean && raggedRecordHasHead(key[r]) ? problem.end(key[r]) - problem.begin(key[r]) : 0;
    }
    __syncthreads();

    // Thread t's four CTAs in order (one 16-byte and one 4-byte shared read, conflict-free), their fold, the scan.
    const float4 tails = reinterpret_cast<const float4*>(s_run)[tid];
    const uint32_t heads = reinterpret_cast<const uint32_t*>(s_head)[tid];
    const float tail_of[4] = {tails.x, tails.y, tails.z, tails.w};
    Carry<float> mine{false, identity};
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        mine = raggedCarryCombine(mine, Carry<float>{((heads >> (8 * i)) & 0xffU) != 0, tail_of[i]}, op);
    }
    Carry<float> total;
    Carry<float> arriving = raggedBlockExclusiveScan<FINISH_THREADS>(mine, Carry<float>{false, identity}, identity, op,
                                                                     s_scan, total);
    float arriving_at[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        arriving_at[i] = arriving.value;
        arriving = raggedCarryCombine(arriving, Carry<float>{((heads >> (8 * i)) & 0xffU) != 0, tail_of[i]}, op);
    }
    reinterpret_cast<float4*>(s_run)[tid] = make_float4(arriving_at[0], arriving_at[1], arriving_at[2], arriving_at[3]);
    __syncthreads();

    // Each head: the run arriving at its CTA, then the head value.
#pragma unroll
    for (uint32_t r = 0; r < FINISH_ROUNDS; ++r) {
        const uint32_t c = tid + r * FINISH_THREADS;
        if (raggedRecordHasHead(key[r])) {
            raggedStoreValue(p.output, p.output_dtype, key[r] * p.components + component, op(s_run[c], head_value[r]),
                             p.finalize, count[r]);
        }
    }
}

// 1291: the wide finisher (D above 15, ragged RK; it replaces 1285's warp per component). Warp w of a component takes
// its records 256 w to 256 w + 255, n of them present; every load is issued first. Two forms (1294, on the grid:
// raggedFinishWideTransposes):
// - Transposed (grids above 128; 1292): as the block kernel, per warp: the records load coalesced (round r: 32
//   consecutive ones); the tails and head flags go to the warp's shared memory in CTA order; lane l folds records 8 l
//   to 8 l + 7, one warp scan carries the runs, and the lane writes the run arriving at each of its records back in
//   place (with its restart flag).
// - Direct (grids up to 128; 1293): lane l takes k = ceil(n / 32) <= 4 consecutive records straight from global
//   memory, folds them, one warp scan, and walks them giving each head its run; no shared-memory pass.
// Then the warps' totals through shared memory (each warp seeds its runs with its component's earlier warps') and
// each head stores. 1292 alone: 2.2-2.7 us up to 1024 components but 3.4-3.9 us at D 4096 / 5000 (a few dozen records
// per component paid the 256-record pass); 1293 alone fixed those (2.8-3.2) but cost 0.3-0.5 us on full warps (the
// lanes' strided loads). The same order as the block kernel: fold in CTA order, restart at each head.
template <int FixedOp, bool Transposed>
__global__ __launch_bounds__(RAGGED_FINISH_WIDE_MAX_THREADS) void raggedFinishValueWideKernel(const RaggedFinishParams p) {
    using namespace RaggedMergePath;
    static_assert(FixedOp == 0 || FixedOp == 2);
    constexpr uint32_t Rounds = RAGGED_FINISH_WIDE_ROUNDS;
    constexpr uint32_t Records = RAGGED_FINISH_WIDE_WARP_RECORDS;  // 32 x Rounds per warp
    constexpr uint32_t MaxWarps = RAGGED_FINISH_WIDE_MAX_THREADS / 32;
    static_assert(MaxWarps * Records == RAGGED_FINISH_MAX_GRID && Rounds == 8);
    // Direct: at most 4 records per lane (raggedFinishWideTransposes).
    constexpr uint32_t DirectRecords = RAGGED_FINISH_WIDE_DIRECT_MAX_GRID / 32;
    static_assert(DirectRecords * 32 == RAGGED_FINISH_WIDE_DIRECT_MAX_GRID && DirectRecords <= Rounds);
    constexpr uint32_t Slots = Transposed ? Rounds : DirectRecords;  // the records a lane holds
    constexpr uint32_t SharedRecords = Transposed ? Records : 4;     // the direct form keeps no records in shared
    __shared__ __align__(16) float s_run[MaxWarps][SharedRecords];   // tails in CTA order, then the run arriving at each
    __shared__ __align__(8) uint8_t s_flag[MaxWarps][SharedRecords]; // head flags, then the arriving run's restart flag
    __shared__ int s_reset[MaxWarps];
    __shared__ float s_value[MaxWarps];

    const Fp8TypedReductionOp<FixedOp> op{};
    const float identity = op.identity();
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t local = warp / p.warps_per_component;
    const uint32_t batch = warp - local * p.warps_per_component;
    const uint64_t component = uint64_t{blockIdx.x} * p.components_per_cta + local;
    const bool active = component < p.components;  // warp-uniform
    const RaggedRecordLayout layout = raggedRecordLayout(p.grid, p.components);
    const uint64_t* record_keys = reinterpret_cast<const uint64_t*>(p.workspace + layout.keys);
    const float* record_heads = reinterpret_cast<const float*>(p.workspace + layout.heads);
    const float* record_tails = reinterpret_cast<const float*>(p.workspace + layout.tails);
    const uint32_t base = batch * Records;
    const uint32_t present = active && base < p.grid ? (p.grid - base < Records ? p.grid - base : Records) : 0;
    const uint32_t k = (present + 31U) / 32U;  // warp-uniform: transposed, the rounds; direct, the records per lane
    // Slot r: transposed, record 32 r + lane of the warp's; direct, record k lane + r (consecutive per lane).
    const auto record_of = [&](uint32_t r) { return Transposed ? 32 * r + lane : k * lane + r; };

    uint64_t key[Slots];
    float tail[Slots], head_value[Slots];
#pragma unroll
    for (uint32_t r = 0; r < Slots; ++r) {
        const uint32_t j = record_of(r);
        key[r] = NO_SEGMENT;
        tail[r] = identity;
        head_value[r] = identity;
        if (r < k && j < present) {
            const uint64_t i = raggedRecordValueIndex(p.grid, base + j, static_cast<uint32_t>(component));
            key[r] = record_keys[base + j];
            tail[r] = record_tails[i];
            head_value[r] = record_heads[i];
        }
    }
    const Problem problem = makeProblem(p.offsets, p.segments, p.capacity_rows);
    uint64_t count[Slots];
#pragma unroll
    for (uint32_t r = 0; r < Slots; ++r) {
        count[r] = p.finalize.mean && raggedRecordHasHead(key[r]) ? problem.end(key[r]) - problem.begin(key[r]) : 0;
    }

    if constexpr (Transposed) {
#pragma unroll
        for (uint32_t r = 0; r < Rounds; ++r) {
            s_run[warp][32 * r + lane] = tail[r];
            s_flag[warp][32 * r + lane] = raggedRecordHasHead(key[r]) ? 1 : 0;
        }
        __syncwarp();

        // Lane l's 8 records in order (two 16-byte and one 8-byte shared read), their fold, the warp scan.
        const float4 t0 = reinterpret_cast<const float4*>(s_run[warp])[2 * lane];
        const float4 t1 = reinterpret_cast<const float4*>(s_run[warp])[2 * lane + 1];
        const uint2 f = reinterpret_cast<const uint2*>(s_flag[warp])[lane];
        const float tail_of[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
        const auto flag_of = [&](uint32_t i) { return (((i < 4 ? f.x : f.y) >> (8 * (i & 3U))) & 0xffU) != 0; };
        Carry<float> mine{false, identity};
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) mine = raggedCarryCombine(mine, Carry<float>{flag_of(i), tail_of[i]}, op);
        Carry<float> total;
        Carry<float> arriving = raggedWarpExclusiveScan(mine, Carry<float>{false, identity}, identity, op, total);
        float arriving_value[8];
        uint32_t arriving_reset[2] = {0, 0};
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            arriving_value[i] = arriving.value;
            arriving_reset[i / 4] |= (arriving.reset ? 1U : 0U) << (8 * (i & 3U));
            arriving = raggedCarryCombine(arriving, Carry<float>{flag_of(i), tail_of[i]}, op);
        }
        __syncwarp();
        reinterpret_cast<float4*>(s_run[warp])[2 * lane] =
            make_float4(arriving_value[0], arriving_value[1], arriving_value[2], arriving_value[3]);
        reinterpret_cast<float4*>(s_run[warp])[2 * lane + 1] =
            make_float4(arriving_value[4], arriving_value[5], arriving_value[6], arriving_value[7]);
        reinterpret_cast<uint2*>(s_flag[warp])[lane] = make_uint2(arriving_reset[0], arriving_reset[1]);
        if (lane == 0) {
            s_reset[warp] = total.reset ? 1 : 0;
            s_value[warp] = total.value;
        }
        __syncthreads();

        // The run arriving at this warp's records from the component's earlier warps, then each head.
        Carry<float> carry_in{false, identity};
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t w = local * p.warps_per_component + b;
            carry_in = raggedCarryCombine(carry_in, Carry<float>{s_reset[w] != 0, s_value[w]}, op);
        }
#pragma unroll
        for (uint32_t r = 0; r < Rounds; ++r) {
            if (raggedRecordHasHead(key[r])) {
                const uint32_t j = 32 * r + lane;
                const Carry<float> before =
                    raggedCarryCombine(carry_in, Carry<float>{s_flag[warp][j] != 0, s_run[warp][j]}, op);
                raggedStoreValue(p.output, p.output_dtype, key[r] * p.components + component,
                                 op(before.value, head_value[r]), p.finalize, count[r]);
            }
        }
    } else {
        Carry<float> mine{false, identity};
#pragma unroll
        for (uint32_t r = 0; r < Slots; ++r) {
            if (r < k) mine = raggedCarryCombine(mine, Carry<float>{raggedRecordHasHead(key[r]), tail[r]}, op);
        }
        Carry<float> total;
        const Carry<float> arriving = raggedWarpExclusiveScan(mine, Carry<float>{false, identity}, identity, op, total);
        if (lane == 0) {
            s_reset[warp] = total.reset ? 1 : 0;
            s_value[warp] = total.value;
        }
        __syncthreads();
        Carry<float> run{false, identity};  // the run arriving at this warp's records from the component's earlier warps
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t w = local * p.warps_per_component + b;
            run = raggedCarryCombine(run, Carry<float>{s_reset[w] != 0, s_value[w]}, op);
        }
        run = raggedCarryCombine(run, arriving, op);
#pragma unroll
        for (uint32_t r = 0; r < Slots; ++r) {
            if (r < k) {
                const bool head = raggedRecordHasHead(key[r]);
                if (head) {
                    raggedStoreValue(p.output, p.output_dtype, key[r] * p.components + component,
                                     op(run.value, head_value[r]), p.finalize, count[r]);
                }
                run = raggedCarryCombine(run, Carry<float>{head, tail[r]}, op);
            }
        }
    }
}

}  // namespace

void launchRaggedFinishValue(int fixed_op,
                             uint32_t grid,
                             uint32_t components,
                             const uint8_t* workspace,
                             RaggedMergePath::Offsets offsets,
                             uint64_t segments,
                             uint64_t capacity_rows,
                             RaggedFinalize finalize,
                             void* output,
                             DataType output_dtype,
                             cudaStream_t stream) {
    if (grid == 0 || grid > RAGGED_FINISH_MAX_GRID) {
        throw std::invalid_argument("launchRaggedFinishValue: the grid must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_GRID));
    }
    // 1285: any D (1265-1284: at most 15); the record arrays index components x grid in 64 bits.
    if (components == 0 || components > RAGGED_FINISH_MAX_COMPONENTS) {
        throw std::invalid_argument("launchRaggedFinishValue: the components must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_COMPONENTS));
    }
    if (fixed_op != 0 && fixed_op != 2) throw std::invalid_argument("launchRaggedFinishValue: fixed_op must be 0 or 2");
    const RaggedFinishWideLaunch wide = raggedFinishWideLaunch(grid);
    const RaggedFinishParams params{workspace, grid,     components, offsets,      segments,
                                    capacity_rows, finalize, output,     output_dtype, wide.warps_per_component,
                                    wide.components_per_cta};
    if (raggedFinishUsesWide(components)) {
        const uint32_t ctas = (components + wide.components_per_cta - 1) / wide.components_per_cta;
        const bool transposed = raggedFinishWideTransposes(grid);
        if (fixed_op == 0) {
            if (transposed) {
                raggedFinishValueWideKernel<0, true><<<ctas, wide.threads, 0, stream>>>(params);
            } else {
                raggedFinishValueWideKernel<0, false><<<ctas, wide.threads, 0, stream>>>(params);
            }
        } else if (transposed) {
            raggedFinishValueWideKernel<2, true><<<ctas, wide.threads, 0, stream>>>(params);
        } else {
            raggedFinishValueWideKernel<2, false><<<ctas, wide.threads, 0, stream>>>(params);
        }
    } else if (fixed_op == 0) {
        raggedFinishValueKernel<0><<<components, FINISH_THREADS, 0, stream>>>(params);
    } else {
        raggedFinishValueKernel<2><<<components, FINISH_THREADS, 0, stream>>>(params);
    }
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("launchRaggedFinishValue: ") + cudaGetErrorString(status));
    }
}

}  // namespace ThorImplementation::CubReductionInternal
