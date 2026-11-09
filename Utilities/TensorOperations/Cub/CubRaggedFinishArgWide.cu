// 1314 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the wide ragged ARG finishing kernel, for D above 15
// (ragged RK ARG), on the 64-bit candidate records (RaggedArg.h, raggedArgRecordLayout). The VALUE wide finisher's form
// (raggedFinishValueWideKernel, CubRaggedFinish.cu, with its history) with 64-bit words and the unsigned minimum: warp
// w of a component takes its records 256 w to 256 w + 255; every load is issued first; up to 128 records a lane reads
// its consecutive records straight from global memory (direct), above them the records load coalesced and transpose
// through the warp's shared memory (transposed); one warp scan per warp; the warps' totals through shared memory; each
// head stores the index of min(the run arriving at its CTA, its head word). The block finisher (CubRaggedFinishArg.cu)
// keeps D up to 15; launchRaggedFinishArg forwards wider D here. Its own object, so that kernel's SASS stays as it was.
#include "Utilities/TensorOperations/Cub/RaggedArg.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {
namespace {

using RaggedMergePath::Carry;

struct RaggedFinishArgWideParams {
    const uint8_t* workspace;  // raggedArgRecordLayout(grid, components)
    uint32_t grid;
    uint32_t components;
    uint32_t row_bits;     // the row field of the candidate words
    void* output;          // segments x D indices
    uint32_t index_bytes;  // 4 or 8
    uint32_t warps_per_component;
    uint32_t components_per_cta;
};

__device__ __forceinline__ Carry<uint64_t> argCombine(const Carry<uint64_t>& a, const Carry<uint64_t>& b) {
    return Carry<uint64_t>{a.reset || b.reset, b.reset ? b.value : (a.value < b.value ? a.value : b.value)};
}

// Warp-wide exclusive segmented scan of `mine` (no seed); `total` every lane's. Call from all 32 lanes.
__device__ __forceinline__ Carry<uint64_t> argWarpExclusiveScan(const Carry<uint64_t>& mine, Carry<uint64_t>& total) {
    constexpr uint64_t Identity = RaggedMergePath::RAGGED_ARG_IDENTITY;
    const uint32_t lane = threadIdx.x & 31U;
    Carry<uint64_t> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const int reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, offset);
        const uint64_t value = __shfl_up_sync(0xffffffffU, inclusive.value, offset);
        if (lane >= offset) inclusive = argCombine(Carry<uint64_t>{reset != 0, value}, inclusive);
    }
    total = Carry<uint64_t>{__shfl_sync(0xffffffffU, inclusive.reset ? 1 : 0, 31) != 0,
                            __shfl_sync(0xffffffffU, inclusive.value, 31)};
    const int previous_reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, 1);
    const uint64_t previous_value = __shfl_up_sync(0xffffffffU, inclusive.value, 1);
    return lane == 0 ? Carry<uint64_t>{false, Identity} : Carry<uint64_t>{previous_reset != 0, previous_value};
}

template <bool Transposed>
__global__ __launch_bounds__(RAGGED_FINISH_WIDE_MAX_THREADS) void raggedFinishArgWideKernel(
    const RaggedFinishArgWideParams p) {
    using namespace RaggedMergePath;
    constexpr uint64_t Identity = RAGGED_ARG_IDENTITY;
    constexpr uint32_t Rounds = RAGGED_FINISH_WIDE_ROUNDS;
    constexpr uint32_t Records = RAGGED_FINISH_WIDE_WARP_RECORDS;  // 32 x Rounds per warp
    constexpr uint32_t MaxWarps = RAGGED_FINISH_WIDE_MAX_THREADS / 32;
    static_assert(MaxWarps * Records == RAGGED_FINISH_MAX_GRID && Rounds == 8);
    constexpr uint32_t DirectRecords = RAGGED_FINISH_WIDE_DIRECT_MAX_GRID / 32;
    static_assert(DirectRecords * 32 == RAGGED_FINISH_WIDE_DIRECT_MAX_GRID && DirectRecords <= Rounds);
    constexpr uint32_t Slots = Transposed ? Rounds : DirectRecords;
    constexpr uint32_t SharedRecords = Transposed ? Records : 2;
    __shared__ __align__(16) uint64_t s_run[MaxWarps][SharedRecords];  // 32 KB transposed
    __shared__ __align__(8) uint8_t s_flag[MaxWarps][SharedRecords];
    __shared__ int s_reset[MaxWarps];
    __shared__ uint64_t s_value[MaxWarps];

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t local = warp / p.warps_per_component;
    const uint32_t batch = warp - local * p.warps_per_component;
    const uint64_t component = uint64_t{blockIdx.x} * p.components_per_cta + local;
    const bool active = component < p.components;  // warp-uniform
    const RaggedArgRecordLayout layout = raggedArgRecordLayout(p.grid, p.components);
    const uint64_t* record_keys = reinterpret_cast<const uint64_t*>(p.workspace + layout.keys);
    const uint64_t* record_heads = reinterpret_cast<const uint64_t*>(p.workspace + layout.heads);
    const uint64_t* record_tails = reinterpret_cast<const uint64_t*>(p.workspace + layout.tails);
    const uint32_t base = batch * Records;
    const uint32_t present = active && base < p.grid ? (p.grid - base < Records ? p.grid - base : Records) : 0;
    const uint32_t k = (present + 31U) / 32U;  // warp-uniform: transposed, the rounds; direct, the records per lane
    const auto record_of = [&](uint32_t r) { return Transposed ? 32 * r + lane : k * lane + r; };
    const auto store = [&](uint64_t segment, uint64_t winner) {
        raggedArgStoreIndex(p.output, p.index_bytes, segment * p.components + component,
                            raggedArgOutputIndex(winner, p.row_bits, p.components, component, p.index_bytes));
    };

    uint64_t key[Slots], tail[Slots], head_word[Slots];
#pragma unroll
    for (uint32_t r = 0; r < Slots; ++r) {
        const uint32_t j = record_of(r);
        key[r] = NO_SEGMENT;
        tail[r] = Identity;
        head_word[r] = Identity;
        if (r < k && j < present) {
            const uint64_t i = raggedRecordValueIndex(p.grid, base + j, static_cast<uint32_t>(component));
            key[r] = record_keys[base + j];
            tail[r] = record_tails[i];
            head_word[r] = record_heads[i];
        }
    }

    if constexpr (Transposed) {
#pragma unroll
        for (uint32_t r = 0; r < Rounds; ++r) {
            s_run[warp][32 * r + lane] = tail[r];
            s_flag[warp][32 * r + lane] = raggedRecordHasHead(key[r]) ? 1 : 0;
        }
        __syncwarp();

        // Lane l's 8 records in order (four 16-byte and one 8-byte shared read), their fold, the warp scan.
        const ulonglong2* runs = reinterpret_cast<const ulonglong2*>(s_run[warp]) + 4 * lane;
        const ulonglong2 t0 = runs[0], t1 = runs[1], t2 = runs[2], t3 = runs[3];
        const uint2 f = reinterpret_cast<const uint2*>(s_flag[warp])[lane];
        const uint64_t tail_of[8] = {t0.x, t0.y, t1.x, t1.y, t2.x, t2.y, t3.x, t3.y};
        const auto flag_of = [&](uint32_t i) { return (((i < 4 ? f.x : f.y) >> (8 * (i & 3U))) & 0xffU) != 0; };
        Carry<uint64_t> mine{false, Identity};
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) mine = argCombine(mine, Carry<uint64_t>{flag_of(i), tail_of[i]});
        Carry<uint64_t> total;
        Carry<uint64_t> arriving = argWarpExclusiveScan(mine, total);
        uint64_t arriving_value[8];
        uint32_t arriving_reset[2] = {0, 0};
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            arriving_value[i] = arriving.value;
            arriving_reset[i / 4] |= (arriving.reset ? 1U : 0U) << (8 * (i & 3U));
            arriving = argCombine(arriving, Carry<uint64_t>{flag_of(i), tail_of[i]});
        }
        __syncwarp();
        ulonglong2* out_runs = reinterpret_cast<ulonglong2*>(s_run[warp]) + 4 * lane;
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) out_runs[q] = make_ulonglong2(arriving_value[2 * q], arriving_value[2 * q + 1]);
        reinterpret_cast<uint2*>(s_flag[warp])[lane] = make_uint2(arriving_reset[0], arriving_reset[1]);
        if (lane == 0) {
            s_reset[warp] = total.reset ? 1 : 0;
            s_value[warp] = total.value;
        }
        __syncthreads();

        // The run arriving at this warp's records from the component's earlier warps, then each head.
        Carry<uint64_t> carry_in{false, Identity};
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t w = local * p.warps_per_component + b;
            carry_in = argCombine(carry_in, Carry<uint64_t>{s_reset[w] != 0, s_value[w]});
        }
#pragma unroll
        for (uint32_t r = 0; r < Rounds; ++r) {
            if (raggedRecordHasHead(key[r])) {
                const uint32_t j = 32 * r + lane;
                const Carry<uint64_t> before =
                    argCombine(carry_in, Carry<uint64_t>{s_flag[warp][j] != 0, s_run[warp][j]});
                store(key[r], before.value < head_word[r] ? before.value : head_word[r]);
            }
        }
    } else {
        Carry<uint64_t> mine{false, Identity};
#pragma unroll
        for (uint32_t r = 0; r < Slots; ++r) {
            if (r < k) mine = argCombine(mine, Carry<uint64_t>{raggedRecordHasHead(key[r]), tail[r]});
        }
        Carry<uint64_t> total;
        const Carry<uint64_t> arriving = argWarpExclusiveScan(mine, total);
        if (lane == 0) {
            s_reset[warp] = total.reset ? 1 : 0;
            s_value[warp] = total.value;
        }
        __syncthreads();
        Carry<uint64_t> run{false, Identity};
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t w = local * p.warps_per_component + b;
            run = argCombine(run, Carry<uint64_t>{s_reset[w] != 0, s_value[w]});
        }
        run = argCombine(run, arriving);
#pragma unroll
        for (uint32_t r = 0; r < Slots; ++r) {
            if (r < k) {
                const bool head = raggedRecordHasHead(key[r]);
                if (head) store(key[r], run.value < head_word[r] ? run.value : head_word[r]);
                run = argCombine(run, Carry<uint64_t>{head, tail[r]});
            }
        }
    }
}

}  // namespace

void launchRaggedFinishArgWide(uint32_t grid,
                               uint32_t components,
                               const uint8_t* workspace,
                               uint32_t row_bits,
                               void* output,
                               uint32_t index_bytes,
                               cudaStream_t stream) {
    if (grid == 0 || grid > RAGGED_FINISH_MAX_GRID)
        throw std::invalid_argument("launchRaggedFinishArgWide: the grid must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_GRID));
    if (components == 0 || components > RAGGED_FINISH_MAX_COMPONENTS)
        throw std::invalid_argument("launchRaggedFinishArgWide: the components must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_COMPONENTS));
    if (row_bits == 0 || row_bits > 62) throw std::invalid_argument("launchRaggedFinishArgWide: row_bits must be 1 to 62");
    if (index_bytes != 4 && index_bytes != 8)
        throw std::invalid_argument("launchRaggedFinishArgWide: index_bytes must be 4 or 8");
    const RaggedFinishWideLaunch wide = raggedFinishWideLaunch(grid);
    const RaggedFinishArgWideParams params{workspace,   grid,          components,
                                           row_bits,    output,        index_bytes,
                                           wide.warps_per_component, wide.components_per_cta};
    const uint32_t ctas = (components + wide.components_per_cta - 1) / wide.components_per_cta;
    if (raggedFinishWideTransposes(grid)) {
        raggedFinishArgWideKernel<true><<<ctas, wide.threads, 0, stream>>>(params);
    } else {
        raggedFinishArgWideKernel<false><<<ctas, wide.threads, 0, stream>>>(params);
    }
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launchRaggedFinishArgWide: ") + cudaGetErrorString(status));
}

}  // namespace ThorImplementation::CubReductionInternal
