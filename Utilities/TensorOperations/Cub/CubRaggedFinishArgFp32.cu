// 1313 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the ragged ARG finishing kernel for pair records
// (FP32: a 32-bit key and a 64-bit global row per CTA and component, RaggedArg.h's raggedArgPairRecordLayout). The 64-bit
// word finisher's order (CubRaggedFinishArg.cu): one CTA of 1024 threads per component, a block-wide segmented scan
// over the CTAs' tails in CTA order restarting at each head's CTA, and each head's segment winner the minimum of the run
// arriving at its CTA and its head, stored as its index. The minimum is on (key, row), so ties split across CTAs go to
// the lowest row. Fixed order, no atomics.
//
// The pair records' stride is the grid rounded up to 4, so thread t reads CTAs 4t .. 4t + 3 of each array with vector
// loads (no staging in shared memory: the 64-bit finisher's transpose would need 48 KB for the pairs); the padding
// entries past the grid are masked, never used.
#include "Utilities/TensorOperations/Cub/RaggedArg.h"
#include "Utilities/TensorOperations/Cub/RaggedReduction.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {
namespace {

using RaggedMergePath::Carry;

constexpr uint32_t FINISH_THREADS = RAGGED_FINISH_THREADS;
constexpr uint32_t FINISH_WARPS = FINISH_THREADS / 32;

struct RaggedFinishArgPairParams {
    const uint8_t* workspace;  // raggedArgPairRecordLayout(grid, components)
    uint32_t grid;
    uint32_t components;
    void* output;          // segments x D indices
    uint32_t index_bytes;  // 4 or 8
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

struct PairScanScratch {
    uint64_t row[FINISH_WARPS];
    uint32_t key[FINISH_WARPS];
    int reset[FINISH_WARPS];
};

__device__ __forceinline__ Carry<ArgPair> pairShuffleUp(const Carry<ArgPair>& x, uint32_t offset) {
    return Carry<ArgPair>{__shfl_up_sync(0xffffffffU, x.reset ? 1 : 0, offset) != 0,
                          ArgPair{__shfl_up_sync(0xffffffffU, x.value.key, offset),
                                  static_cast<uint64_t>(__shfl_up_sync(
                                      0xffffffffU, static_cast<unsigned long long>(x.value.row), offset))}};
}

// Block-wide exclusive segmented scan of `mine` (no seed): every earlier thread's element combined. Call from all
// threads (contains __syncthreads).
__device__ __forceinline__ Carry<ArgPair> pairBlockExclusiveScan(const Carry<ArgPair>& mine, PairScanScratch& scratch) {
    const uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x >> 5;
    Carry<ArgPair> inclusive = mine;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const Carry<ArgPair> up = pairShuffleUp(inclusive, offset);
        if (lane >= offset) inclusive = pairCombine(up, inclusive);
    }
    if (lane == 31) {
        scratch.reset[warp] = inclusive.reset ? 1 : 0;
        scratch.key[warp] = inclusive.value.key;
        scratch.row[warp] = inclusive.value.row;
    }
    __syncthreads();
    if (warp == 0) {
        Carry<ArgPair> w = lane < FINISH_WARPS
                               ? Carry<ArgPair>{scratch.reset[lane] != 0, ArgPair{scratch.key[lane], scratch.row[lane]}}
                               : Carry<ArgPair>{false, pairIdentity()};
#pragma unroll
        for (uint32_t offset = 1; offset < FINISH_WARPS; offset <<= 1) {
            const Carry<ArgPair> up = pairShuffleUp(w, offset);
            if (lane >= offset) w = pairCombine(up, w);
        }
        __syncwarp();
        if (lane < FINISH_WARPS) {
            scratch.reset[lane] = w.reset ? 1 : 0;
            scratch.key[lane] = w.value.key;
            scratch.row[lane] = w.value.row;
        }
    }
    __syncthreads();
    const Carry<ArgPair> previous = pairShuffleUp(inclusive, 1);
    const Carry<ArgPair> lane_exclusive = lane == 0 ? Carry<ArgPair>{false, pairIdentity()} : previous;
    const Carry<ArgPair> warp_exclusive =
        warp == 0 ? Carry<ArgPair>{false, pairIdentity()}
                  : Carry<ArgPair>{scratch.reset[warp - 1] != 0, ArgPair{scratch.key[warp - 1], scratch.row[warp - 1]}};
    return pairCombine(warp_exclusive, lane_exclusive);
}

// Thread t owns CTAs 4t .. 4t + 3: every record load first (vector loads), its four tails folded, one block scan, then
// its heads finished with the run arriving at each.
__global__ __launch_bounds__(FINISH_THREADS) void raggedFinishArgPairKernel(const RaggedFinishArgPairParams p) {
    using namespace RaggedMergePath;
    static_assert(FINISH_THREADS * 4 == RAGGED_FINISH_MAX_GRID);
    __shared__ PairScanScratch s_scan;

    const uint32_t tid = threadIdx.x;
    const uint32_t component = blockIdx.x;
    const RaggedArgPairRecordLayout layout = raggedArgPairRecordLayout(p.grid, p.components);
    const uint32_t first = 4 * tid;
    const uint64_t at = uint64_t{component} * layout.stride + first;

    uint64_t key[4] = {NO_SEGMENT, NO_SEGMENT, NO_SEGMENT, NO_SEGMENT};
    ArgPair head[4], tail[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) head[i] = tail[i] = pairIdentity();
    if (first < p.grid) {
        const ulonglong2 k01 = reinterpret_cast<const ulonglong2*>(p.workspace + layout.keys)[first / 2];
        const ulonglong2 k23 = reinterpret_cast<const ulonglong2*>(p.workspace + layout.keys)[first / 2 + 1];
        const uint4 hk = reinterpret_cast<const uint4*>(p.workspace + layout.head_keys)[at / 4];
        const ulonglong2 hr01 = reinterpret_cast<const ulonglong2*>(p.workspace + layout.head_rows)[at / 2];
        const ulonglong2 hr23 = reinterpret_cast<const ulonglong2*>(p.workspace + layout.head_rows)[at / 2 + 1];
        const uint4 tk = reinterpret_cast<const uint4*>(p.workspace + layout.tail_keys)[at / 4];
        const ulonglong2 tr01 = reinterpret_cast<const ulonglong2*>(p.workspace + layout.tail_rows)[at / 2];
        const ulonglong2 tr23 = reinterpret_cast<const ulonglong2*>(p.workspace + layout.tail_rows)[at / 2 + 1];
        const uint64_t keys[4] = {k01.x, k01.y, k23.x, k23.y};
        const uint32_t head_keys[4] = {hk.x, hk.y, hk.z, hk.w}, tail_keys[4] = {tk.x, tk.y, tk.z, tk.w};
        const uint64_t head_rows[4] = {hr01.x, hr01.y, hr23.x, hr23.y}, tail_rows[4] = {tr01.x, tr01.y, tr23.x, tr23.y};
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            if (first + i < p.grid) {  // the padding past the grid is never written
                key[i] = keys[i];
                head[i] = ArgPair{head_keys[i], head_rows[i]};
                tail[i] = ArgPair{tail_keys[i], tail_rows[i]};
            }
        }
    }

    Carry<ArgPair> mine{false, pairIdentity()};
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) mine = pairCombine(mine, Carry<ArgPair>{raggedRecordHasHead(key[i]), tail[i]});
    Carry<ArgPair> arriving = pairBlockExclusiveScan(mine, s_scan);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        if (raggedRecordHasHead(key[i])) {
            const ArgPair winner = pairMin(arriving.value, head[i]);
            raggedArgStoreIndex(p.output, p.index_bytes, key[i] * p.components + component,
                                raggedArgPairOutputIndex(winner.row, p.components, component, p.index_bytes));
        }
        arriving = pairCombine(arriving, Carry<ArgPair>{raggedRecordHasHead(key[i]), tail[i]});
    }
}

}  // namespace

void launchRaggedFinishArgPair(uint32_t grid,
                               uint32_t components,
                               const uint8_t* workspace,
                               void* output,
                               uint32_t index_bytes,
                               cudaStream_t stream) {
    if (grid == 0 || grid > RAGGED_FINISH_MAX_GRID)
        throw std::invalid_argument("launchRaggedFinishArgPair: the grid must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_GRID));
    // 1321: any D (1313-1320: at most 15); above 15 the wide pair finisher (CubRaggedFinishArgPairWide.cu, FP32 ragged
    // RK ARG).
    if (components == 0 || components > RAGGED_FINISH_MAX_COMPONENTS)
        throw std::invalid_argument("launchRaggedFinishArgPair: the components must be 1 to " +
                                    std::to_string(RAGGED_FINISH_MAX_COMPONENTS));
    if (index_bytes != 4 && index_bytes != 8)
        throw std::invalid_argument("launchRaggedFinishArgPair: index_bytes must be 4 or 8");
    if (raggedFinishUsesWide(components)) {
        launchRaggedFinishArgPairWide(grid, components, workspace, output, index_bytes, stream);
        return;
    }
    const RaggedFinishArgPairParams params{workspace, grid, components, output, index_bytes};
    raggedFinishArgPairKernel<<<components, FINISH_THREADS, 0, stream>>>(params);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launchRaggedFinishArgPair: ") + cudaGetErrorString(status));
}

}  // namespace ThorImplementation::CubReductionInternal
