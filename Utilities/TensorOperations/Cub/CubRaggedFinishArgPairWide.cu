// 1321 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the wide ragged ARG finishing kernel for pair
// records, for D above 15 (FP32 ragged RK ARG), on RaggedArg.h's raggedArgPairRecordLayout (a 32-bit key and a 64-bit
// global row per range and component). The wide ARG finisher's launch and order (CubRaggedFinishArgWide.cu, its direct
// form): warp w of a component takes its records 256 w to 256 w + 255, a lane its consecutive ones (up to 8) read
// straight from global memory, every load issued first; one warp scan per warp; the warps' totals through shared
// memory; each head stores the index of min(the run arriving at its range, its head pair). The minimum is on (key,
// row), so ties split across ranges go to the lowest row. There is no transposed form: staging the pairs through shared
// memory would take 16 warps x 256 x 13 bytes, more than a CTA's 48 KB of static shared memory.
//
// launchRaggedFinishArgPair (CubRaggedFinishArgFp32.cu, the block form, D up to 15) forwards wider D here; its own
// object, so the block form's SASS stays as it was.
#include "Utilities/TensorOperations/Cub/RaggedArg.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {
namespace {

using RaggedMergePath::Carry;

struct RaggedFinishArgPairWideParams {
    const uint8_t* workspace;  // raggedArgPairRecordLayout(grid, components)
    uint32_t grid;
    uint32_t components;
    void* output;          // segments x D indices
    uint32_t index_bytes;  // 4 or 8
    uint32_t warps_per_component;
    uint32_t components_per_cta;
};

struct ArgPair {
    uint32_t key;
    uint64_t row;
};

__device__ __forceinline__ ArgPair pairIdentity() {
    return ArgPair{RaggedMergePath::RAGGED_ARG_PAIR_KEY_IDENTITY, RaggedMergePath::RAGGED_ARG_PAIR_ROW_IDENTITY};
}

__device__ __forceinline__ ArgPair pairMin(const ArgPair& a, const ArgPair& b) {
    return (b.key < a.key || (b.key == a.key && b.row < a.row)) ? b : a;
}

__device__ __forceinline__ Carry<ArgPair> pairCombine(const Carry<ArgPair>& a, const Carry<ArgPair>& b) {
    return Carry<ArgPair>{a.reset || b.reset, b.reset ? b.value : pairMin(a.value, b.value)};
}

__device__ __forceinline__ Carry<ArgPair> pairShuffleUp(const Carry<ArgPair>& x, uint32_t offset) {
    return Carry<ArgPair>{__shfl_up_sync(0xffffffffU, x.reset ? 1 : 0, offset) != 0,
                          ArgPair{__shfl_up_sync(0xffffffffU, x.value.key, offset),
                                  static_cast<uint64_t>(__shfl_up_sync(
                                      0xffffffffU, static_cast<unsigned long long>(x.value.row), offset))}};
}

__device__ __forceinline__ Carry<ArgPair> pairShuffle(const Carry<ArgPair>& x, uint32_t source) {
    return Carry<ArgPair>{__shfl_sync(0xffffffffU, x.reset ? 1 : 0, source) != 0,
                          ArgPair{__shfl_sync(0xffffffffU, x.value.key, source),
                                  static_cast<uint64_t>(__shfl_sync(
                                      0xffffffffU, static_cast<unsigned long long>(x.value.row), source))}};
}

// Warp-wide exclusive segmented scan of `mine` (no seed); `total` every lane's. Call from all 32 lanes.
__device__ __forceinline__ Carry<ArgPair> pairWarpExclusiveScan(const Carry<ArgPair>& mine, Carry<ArgPair>& total) {
    const uint32_t lane = threadIdx.x & 31U;
    Carry<ArgPair> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const Carry<ArgPair> up = pairShuffleUp(inclusive, offset);
        if (lane >= offset) inclusive = pairCombine(up, inclusive);
    }
    total = pairShuffle(inclusive, 31);
    const Carry<ArgPair> previous = pairShuffleUp(inclusive, 1);
    return lane == 0 ? Carry<ArgPair>{false, pairIdentity()} : previous;
}

__global__ __launch_bounds__(RAGGED_FINISH_WIDE_MAX_THREADS) void raggedFinishArgPairWideKernel(
    const RaggedFinishArgPairWideParams p) {
    using namespace RaggedMergePath;
    constexpr uint32_t Slots = RAGGED_FINISH_WIDE_ROUNDS;
    constexpr uint32_t Records = RAGGED_FINISH_WIDE_WARP_RECORDS;  // 32 x Slots per warp
    constexpr uint32_t MaxWarps = RAGGED_FINISH_WIDE_MAX_THREADS / 32;
    static_assert(MaxWarps * Records == RAGGED_FINISH_MAX_GRID && Slots == 8);
    __shared__ int s_reset[MaxWarps];
    __shared__ uint32_t s_key[MaxWarps];
    __shared__ uint64_t s_row[MaxWarps];

    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t local = warp / p.warps_per_component;
    const uint32_t batch = warp - local * p.warps_per_component;
    const uint64_t component = uint64_t{blockIdx.x} * p.components_per_cta + local;
    const bool active = component < p.components;  // warp-uniform
    const RaggedArgPairRecordLayout layout = raggedArgPairRecordLayout(p.grid, p.components);
    const uint64_t* record_keys = reinterpret_cast<const uint64_t*>(p.workspace + layout.keys);
    const uint32_t* head_keys = reinterpret_cast<const uint32_t*>(p.workspace + layout.head_keys);
    const uint64_t* head_rows = reinterpret_cast<const uint64_t*>(p.workspace + layout.head_rows);
    const uint32_t* tail_keys = reinterpret_cast<const uint32_t*>(p.workspace + layout.tail_keys);
    const uint64_t* tail_rows = reinterpret_cast<const uint64_t*>(p.workspace + layout.tail_rows);
    const uint32_t base = batch * Records;
    const uint32_t present = active && base < p.grid ? (p.grid - base < Records ? p.grid - base : Records) : 0;
    const uint32_t k = (present + 31U) / 32U;  // warp-uniform: the records per lane
    const uint64_t column_at = component * layout.stride;

    uint64_t key[Slots];
    ArgPair head[Slots], tail[Slots];
#pragma unroll
    for (uint32_t r = 0; r < Slots; ++r) {
        const uint32_t j = k * lane + r;
        key[r] = NO_SEGMENT;
        head[r] = tail[r] = pairIdentity();
        if (r < k && j < present) {
            const uint64_t i = column_at + base + j;
            key[r] = record_keys[base + j];
            head[r] = ArgPair{head_keys[i], head_rows[i]};
            tail[r] = ArgPair{tail_keys[i], tail_rows[i]};
        }
    }

    Carry<ArgPair> mine{false, pairIdentity()};
#pragma unroll
    for (uint32_t r = 0; r < Slots; ++r) {
        if (r < k) mine = pairCombine(mine, Carry<ArgPair>{raggedRecordHasHead(key[r]), tail[r]});
    }
    Carry<ArgPair> total;
    const Carry<ArgPair> arriving = pairWarpExclusiveScan(mine, total);
    if (lane == 0) {
        s_reset[warp] = total.reset ? 1 : 0;
        s_key[warp] = total.value.key;
        s_row[warp] = total.value.row;
    }
    __syncthreads();
    Carry<ArgPair> run{false, pairIdentity()};
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t w = local * p.warps_per_component + b;
        run = pairCombine(run, Carry<ArgPair>{s_reset[w] != 0, ArgPair{s_key[w], s_row[w]}});
    }
    run = pairCombine(run, arriving);
#pragma unroll
    for (uint32_t r = 0; r < Slots; ++r) {
        if (r < k) {
            const bool has_head = raggedRecordHasHead(key[r]);
            if (has_head) {
                const ArgPair winner = pairMin(run.value, head[r]);
                raggedArgStoreIndex(p.output, p.index_bytes, key[r] * p.components + component,
                                    raggedArgPairOutputIndex(winner.row, p.components, component, p.index_bytes));
            }
            run = pairCombine(run, Carry<ArgPair>{has_head, tail[r]});
        }
    }
}

}  // namespace

void launchRaggedFinishArgPairWide(uint32_t grid,
                                   uint32_t components,
                                   const uint8_t* workspace,
                                   void* output,
                                   uint32_t index_bytes,
                                   cudaStream_t stream) {
    if (grid == 0 || grid > RAGGED_FINISH_MAX_GRID)
        throw std::invalid_argument("launchRaggedFinishArgPairWide: the grid must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_GRID));
    if (components == 0 || components > RAGGED_FINISH_MAX_COMPONENTS)
        throw std::invalid_argument("launchRaggedFinishArgPairWide: the components must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_COMPONENTS));
    if (index_bytes != 4 && index_bytes != 8)
        throw std::invalid_argument("launchRaggedFinishArgPairWide: index_bytes must be 4 or 8");
    const RaggedFinishWideLaunch wide = raggedFinishWideLaunch(grid);
    const RaggedFinishArgPairWideParams params{workspace, grid, components, output, index_bytes,
                                               wide.warps_per_component, wide.components_per_cta};
    const uint32_t ctas = (components + wide.components_per_cta - 1) / wide.components_per_cta;
    raggedFinishArgPairWideKernel<<<ctas, wide.threads, 0, stream>>>(params);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launchRaggedFinishArgPairWide: ") + cudaGetErrorString(status));
}

}  // namespace ThorImplementation::CubReductionInternal
