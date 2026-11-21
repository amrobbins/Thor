#pragma once

// The backward of a min or max scan as one kernel, reading x and dy and writing dx (ScanMinMaxBackward.h). What the
// host and the kernel share: the tensors' formats, the workspace layout and one launch's arguments. The kernel is
// ScanMinMaxBackwardKernel.cuh, instantiated in ScanMinMaxBackward.cu.

#include "Utilities/TensorOperations/Scan/SegmentedScanLaunch.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ThorImplementation::ScanMinMaxBackwardKernels {

// How a float tensor is stored. x, dy and dx each have one, chosen at launch: the kernel converts x and dy to FP32 as
// it loads them (exact for every format) and dx from FP32 as it stores it, so no tensor is cast before or after it.
enum class FloatFormat : uint32_t { Fp32 = 0, Fp16 = 1, Bf16 = 2, E4m3 = 3, E5m2 = 4 };

__host__ __device__ inline constexpr uint32_t formatBytes(FloatFormat format) {
    return format == FloatFormat::Fp32 ? 4U : (format == FloatFormat::Fp16 || format == FloatFormat::Bf16) ? 2U : 1U;
}

inline constexpr uint32_t kGradientItems = 8;   // elements per chunk: 32 bytes of FP32 in the kernel
inline constexpr uint32_t kGradientChunks = 1;  // chunks per thread (1497: two are no faster and take 66 to 72 registers)
// Blocks per SM the kernel is compiled for (its __launch_bounds__ minimum): 4 x 256 threads caps it at 64 registers.
// Without the cap, 1498's ragged kernels took 74 to 80 and an SM held 3 (plans/Thor_Scan_MinMax_Backward_Log_2026-11-20.md).
inline constexpr uint32_t kGradientMinBlocks = 4;

// Elements per tile.
inline constexpr uint64_t kGradientTile = static_cast<uint64_t>(SegmentedScanKernels::kThreads) * kGradientChunks * kGradientItems;

// The gradient sums the tiles publish, one 64-bit word each (an FP32 sum and the launch's epoch, as the scan's runs):
//   tile_sums   the sum from the tile's start to its first winner, or of the whole tile when it has no winner;
//   tile_tails  the sum for the tile's last winner up to the tile's end, when its run goes on past the tile;
//   group_sums, super_sums  the left fold of a group's tile_sums and of a supergroup's group_sums: the sum of a whole
//               group or supergroup of tiles with no winner (and unused otherwise).
struct GradientWorkspace {
    unsigned long long* tile_sums = nullptr;
    unsigned long long* tile_tails = nullptr;
    unsigned long long* group_sums = nullptr;
    unsigned long long* super_sums = nullptr;
};

// Workspace words for n elements in tiles of tile_elements: the keys' runs (two words each, as an arg scan's), then
// the gradient sums. Cleared once, when it is made.
struct WorkspaceCounts {
    uint64_t tiles = 0;
    uint64_t groups = 0;
    uint64_t supers = 0;
    uint64_t words() const { return 2 * (tiles + groups + supers) + 2 * tiles + groups + supers; }
};

inline WorkspaceCounts workspaceCounts(uint64_t n,
                                      uint64_t tile_elements,
                                      uint32_t group = SegmentedScanKernels::kGroup,
                                      uint32_t super = SegmentedScanKernels::kSuper) {
    WorkspaceCounts counts;
    counts.tiles = (n + tile_elements - 1) / tile_elements;
    counts.groups = (counts.tiles + group - 1) / group;
    counts.supers = (counts.groups + super - 1) / super;
    return counts;
}

struct BackwardLaunch {
    const void* x = nullptr;   // the scan's input
    const void* dy = nullptr;  // the gradient of the scan's output
    void* dx = nullptr;        // the gradient of the scan's input
    FloatFormat x_format = FloatFormat::Fp32;
    FloatFormat dy_format = FloatFormat::Fp32;
    FloatFormat dx_format = FloatFormat::Fp32;
    uint64_t n = 0;  // elements
    SegmentedScanKernels::Columns columns;  // dense rows
    const void* offsets = nullptr;          // ragged rows: rows + 1 offsets
    uint32_t rows = 0;
    SegmentedScanKernels::Workspace keys;  // num_tiles is the grid; the pointers are null when no tile publishes
    GradientWorkspace sums;
    bool min = false;        // a min scan's backward (a max scan's otherwise)
    bool exclusive = false;  // the scan was exclusive
};

// The workspace's words laid out in `memory` (counts.words() of them), for a launch that publishes.
inline void layoutWorkspace(unsigned long long* memory, const WorkspaceCounts& counts, BackwardLaunch& launch) {
    launch.keys.tile_runs = memory;
    launch.keys.group_runs = launch.keys.tile_runs + 2 * counts.tiles;
    launch.keys.super_runs = launch.keys.group_runs + 2 * counts.groups;
    launch.sums.tile_sums = launch.keys.super_runs + 2 * counts.supers;
    launch.sums.tile_tails = launch.sums.tile_sums + counts.tiles;
    launch.sums.group_sums = launch.sums.tile_tails + counts.tiles;
    launch.sums.super_sums = launch.sums.group_sums + counts.groups;
}

// One launch. offset_bytes: 0 for dense rows, 4 or 8 for ragged rows' UINT32 or UINT64 offsets.
void launch(const BackwardLaunch& args, bool reverse, uint32_t offset_bytes, cudaStream_t stream);

}  // namespace ThorImplementation::ScanMinMaxBackwardKernels
