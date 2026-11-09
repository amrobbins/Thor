// 1298 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged ARG finishing kernel. As the VALUE block
// finisher (raggedFinishValueKernel, CubRaggedFinish.cu), one CTA of 1024 threads per component reads the main kernel's
// boundary records (RaggedArg.h: 64-bit candidate words with global rows) and runs a block-wide segmented scan over
// the CTAs' tails in CTA order, restarting at each head's CTA, so each head gets the winner of its segment's rows in the
// CTAs before it; the segment's winner is the minimum of that and the head word, stored as its index. The minimum is
// unsigned on (key, row), so ties split across CTAs go to the lowest row. Fixed order, no atomics. Its own object, so
// the VALUE finisher's SASS stays as it was. 1314: D above 15 (ragged RK ARG) goes to the wide finisher
// (CubRaggedFinishArgWide.cu).
#include "Utilities/TensorOperations/Cub/RaggedArg.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {
namespace {

using RaggedMergePath::Carry;

constexpr uint32_t FINISH_THREADS = RAGGED_FINISH_THREADS;
constexpr uint32_t FINISH_ROUNDS = RAGGED_FINISH_MAX_GRID / FINISH_THREADS;
constexpr uint32_t FINISH_WARPS = FINISH_THREADS / 32;

struct RaggedFinishArgParams {
    const uint8_t* workspace;  // raggedArgRecordLayout(grid, components)
    uint32_t grid;
    uint32_t components;
    uint32_t row_bits;     // the row field of the candidate words (RaggedArg.h)
    void* output;          // segments x D indices
    uint32_t index_bytes;  // 4 or 8
};

struct ArgScanScratch {
    uint64_t value[FINISH_WARPS];
    int reset[FINISH_WARPS];
};

__device__ __forceinline__ uint64_t argMin(uint64_t a, uint64_t b) { return a < b ? a : b; }

__device__ __forceinline__ Carry<uint64_t> argCombine(const Carry<uint64_t>& a, const Carry<uint64_t>& b) {
    return Carry<uint64_t>{a.reset || b.reset, b.reset ? b.value : argMin(a.value, b.value)};
}

// Block-wide exclusive segmented scan of `mine` (no seed): every earlier thread's element combined. Call from all
// threads (contains __syncthreads).
__device__ __forceinline__ Carry<uint64_t> argBlockExclusiveScan(const Carry<uint64_t>& mine, ArgScanScratch& scratch) {
    constexpr uint64_t Identity = RaggedMergePath::RAGGED_ARG_IDENTITY;
    const uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x >> 5;
    Carry<uint64_t> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const int reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, offset);
        const uint64_t value = __shfl_up_sync(0xffffffffU, inclusive.value, offset);
        if (lane >= offset) inclusive = argCombine(Carry<uint64_t>{reset != 0, value}, inclusive);
    }
    if (lane == 31) {
        scratch.reset[warp] = inclusive.reset ? 1 : 0;
        scratch.value[warp] = inclusive.value;
    }
    __syncthreads();
    if (warp == 0) {
        Carry<uint64_t> w = lane < FINISH_WARPS ? Carry<uint64_t>{scratch.reset[lane] != 0, scratch.value[lane]}
                                                : Carry<uint64_t>{false, Identity};
#pragma unroll
        for (uint32_t offset = 1; offset < FINISH_WARPS; offset <<= 1) {
            const int reset = __shfl_up_sync(0xffffffffU, w.reset ? 1 : 0, offset);
            const uint64_t value = __shfl_up_sync(0xffffffffU, w.value, offset);
            if (lane >= offset) w = argCombine(Carry<uint64_t>{reset != 0, value}, w);
        }
        __syncwarp();
        if (lane < FINISH_WARPS) {
            scratch.reset[lane] = w.reset ? 1 : 0;
            scratch.value[lane] = w.value;
        }
    }
    __syncthreads();
    const int previous_reset = __shfl_up_sync(0xffffffffU, inclusive.reset ? 1 : 0, 1);
    const uint64_t previous_value = __shfl_up_sync(0xffffffffU, inclusive.value, 1);
    const Carry<uint64_t> lane_exclusive =
        lane == 0 ? Carry<uint64_t>{false, Identity} : Carry<uint64_t>{previous_reset != 0, previous_value};
    const Carry<uint64_t> warp_exclusive =
        warp == 0 ? Carry<uint64_t>{false, Identity}
                  : Carry<uint64_t>{scratch.reset[warp - 1] != 0, scratch.value[warp - 1]};
    return argCombine(warp_exclusive, lane_exclusive);
}

// Thread t loads CTAs t + 1024 r (coalesced), every load first; the tails and head flags go to shared memory in CTA
// order, thread t folds CTAs 4t .. 4t + 3, one block scan, the run arriving at each CTA written back in place; then each
// loading thread finishes its heads.
__global__ __launch_bounds__(FINISH_THREADS) void raggedFinishArgKernel(const RaggedFinishArgParams p) {
    using namespace RaggedMergePath;
    static_assert(FINISH_ROUNDS == 4 && FINISH_THREADS * FINISH_ROUNDS == RAGGED_FINISH_MAX_GRID);
    __shared__ __align__(16) uint64_t s_run[RAGGED_FINISH_MAX_GRID];  // each CTA's tail, then the run arriving at it
    __shared__ __align__(4) uint8_t s_head[RAGGED_FINISH_MAX_GRID];
    __shared__ ArgScanScratch s_scan;

    const uint32_t tid = threadIdx.x;
    const uint32_t component = blockIdx.x;
    const RaggedArgRecordLayout layout = raggedArgRecordLayout(p.grid, p.components);
    const uint64_t* record_keys = reinterpret_cast<const uint64_t*>(p.workspace + layout.keys);
    const uint64_t* record_heads = reinterpret_cast<const uint64_t*>(p.workspace + layout.heads);
    const uint64_t* record_tails = reinterpret_cast<const uint64_t*>(p.workspace + layout.tails);

    uint64_t key[FINISH_ROUNDS], head_word[FINISH_ROUNDS];
#pragma unroll
    for (uint32_t r = 0; r < FINISH_ROUNDS; ++r) {
        const uint32_t c = tid + r * FINISH_THREADS;
        uint64_t tail = RAGGED_ARG_IDENTITY;
        key[r] = NO_SEGMENT;
        head_word[r] = RAGGED_ARG_IDENTITY;
        if (c < p.grid) {
            const uint64_t i = uint64_t{component} * p.grid + c;
            key[r] = record_keys[c];
            head_word[r] = record_heads[i];
            tail = record_tails[i];
        }
        s_run[c] = tail;
        s_head[c] = raggedRecordHasHead(key[r]) ? 1 : 0;
    }
    __syncthreads();

    const ulonglong2 t01 = reinterpret_cast<const ulonglong2*>(s_run)[2 * tid];
    const ulonglong2 t23 = reinterpret_cast<const ulonglong2*>(s_run)[2 * tid + 1];
    const uint32_t heads = reinterpret_cast<const uint32_t*>(s_head)[tid];
    const uint64_t tail_of[4] = {t01.x, t01.y, t23.x, t23.y};
    Carry<uint64_t> mine{false, RAGGED_ARG_IDENTITY};
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
        mine = argCombine(mine, Carry<uint64_t>{((heads >> (8 * i)) & 0xffU) != 0, tail_of[i]});
    Carry<uint64_t> arriving = argBlockExclusiveScan(mine, s_scan);
    uint64_t arriving_at[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        arriving_at[i] = arriving.value;
        arriving = argCombine(arriving, Carry<uint64_t>{((heads >> (8 * i)) & 0xffU) != 0, tail_of[i]});
    }
    reinterpret_cast<ulonglong2*>(s_run)[2 * tid] = make_ulonglong2(arriving_at[0], arriving_at[1]);
    reinterpret_cast<ulonglong2*>(s_run)[2 * tid + 1] = make_ulonglong2(arriving_at[2], arriving_at[3]);
    __syncthreads();

#pragma unroll
    for (uint32_t r = 0; r < FINISH_ROUNDS; ++r) {
        const uint32_t c = tid + r * FINISH_THREADS;
        if (raggedRecordHasHead(key[r])) {
            const uint64_t winner = argMin(s_run[c], head_word[r]);
            raggedArgStoreIndex(p.output, p.index_bytes, key[r] * p.components + component,
                                raggedArgOutputIndex(winner, p.row_bits, p.components, component, p.index_bytes));
        }
    }
}

}  // namespace

void launchRaggedFinishArg(uint32_t grid,
                           uint32_t components,
                           const uint8_t* workspace,
                           uint32_t row_bits,
                           void* output,
                           uint32_t index_bytes,
                           cudaStream_t stream) {
    if (grid == 0 || grid > RAGGED_FINISH_MAX_GRID)
        throw std::invalid_argument("launchRaggedFinishArg: the grid must be 1 to " + std::to_string(RAGGED_FINISH_MAX_GRID));
    // 1314: any D (1298-1313: at most 15); above 15 the wide finisher (CubRaggedFinishArgWide.cu, ragged RK ARG).
    if (components == 0 || components > RAGGED_FINISH_MAX_COMPONENTS)
        throw std::invalid_argument("launchRaggedFinishArg: the components must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_COMPONENTS));
    if (row_bits == 0 || row_bits > 62) throw std::invalid_argument("launchRaggedFinishArg: row_bits must be 1 to 62");
    if (index_bytes != 4 && index_bytes != 8) throw std::invalid_argument("launchRaggedFinishArg: index_bytes must be 4 or 8");
    if (raggedFinishUsesWide(components)) {
        launchRaggedFinishArgWide(grid, components, workspace, row_bits, output, index_bytes, stream);
        return;
    }
    const RaggedFinishArgParams params{workspace, grid, components, row_bits, output, index_bytes};
    raggedFinishArgKernel<<<components, FINISH_THREADS, 0, stream>>>(params);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("launchRaggedFinishArg: ") + cudaGetErrorString(status));
}

}  // namespace ThorImplementation::CubReductionInternal
