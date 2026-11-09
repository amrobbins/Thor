#pragma once

// 1265: the retained ragged (offset-segmented) reduction kernels' host interface
// (plans/Thor_Ragged_Reduction_Plan_2026-11-06.md). Phase 2: ragged KR VALUE for original FP8 input at D = 1 and the
// finishing kernel (CubRaggedFinish.cu); 1280 (phase 3): FP8 at D = 2..15 (rows under 16 bytes) through the same
// launcher; 1284 (phase 3b): FP16, BF16 and FP32 input (rows under 16 bytes), one launcher for every dtype
// (CubRaggedKR.cu; the kernels: RaggedKRKernels.cuh, one object per dtype group); 1285 (phase 4): ragged RK VALUE for
// rows of 16 bytes and more, FP8 first (CubRaggedRK.cu; RaggedRKKernels.cuh, RaggedRKWalk.h), 1295 every dtype KR
// takes (one object pair per dtype group); 1298 (phase 5): ragged KR ARG, FP8 at D = 1 first (CubRaggedKRArg.cu). Not yet
// routed:
// CubSegmentedReduction still runs the legacy kernels; tests and the ragged audit call these launchers directly.
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/RaggedKRWalk.h"
#include "Utilities/TensorOperations/Cub/RaggedRKWalk.h"
#include "Utilities/TensorOperations/Cub/RaggedMergePath.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace ThorImplementation::CubReductionInternal {

// 1272: the finisher's block (1265-1271: 256 threads with up to 16 records each, read one after another behind
// each store; now 1024 threads with up to 4, every read issued before any store; 1273: CTA t + 1024 r in round r).
inline constexpr uint32_t RAGGED_FINISH_THREADS = 1024;
// 1270: the ragged KR main kernel is one warp per CTA, with a ring of RAGGED_KR_RING segment ends; the register budget
// is set for RAGGED_KR_MIN_CTAS_PER_SM resident CTAs (sm_120: 24 blocks per SM, 85 registers each).
inline constexpr uint32_t RAGGED_KR_LANES = 32;
inline constexpr uint32_t RAGGED_KR_RING = 128;  // 1271: 256 -> 128, so 32 rows per lane fit 24 CTAs per SM
inline constexpr uint32_t RAGGED_KR_MIN_CTAS_PER_SM = 24;
// The finisher is one CTA of RAGGED_FINISH_THREADS threads with at most 4 boundary records each.
inline constexpr uint32_t RAGGED_FINISH_MAX_GRID = RAGGED_FINISH_THREADS * 4;
// 1291: wide D (ragged RK, D above 15) at any grid: the wide finisher, a warp per 256 records of a component
// (RAGGED_FINISH_WIDE_ROUNDS of 32), so a component takes ceil(grid / 256) warps (1 to 16) and a CTA holds
// max(1, 8 / warps) components; every load is issued first. (1285-1290: a warp per component over at most 1024
// records, a lane's records read one after another, about 0.27 us each; above 1024 records the 1024-thread block per
// component, one CTA per SM: D 256 at 2720 ranges 4-5 us, D 1024 at 1360 ranges 12 us.)
inline constexpr uint32_t RAGGED_FINISH_WIDE_ROUNDS = 8;
inline constexpr uint32_t RAGGED_FINISH_WIDE_WARP_RECORDS = 32 * RAGGED_FINISH_WIDE_ROUNDS;
inline constexpr uint32_t RAGGED_FINISH_WIDE_MAX_THREADS = 32 * RAGGED_FINISH_MAX_GRID / RAGGED_FINISH_WIDE_WARP_RECORDS;
inline constexpr uint32_t RAGGED_FINISH_MAX_COMPONENTS = 1U << 30;
// 1294: the wide finisher's two forms. Up to 128 records (at most 4 per lane) a lane reads its consecutive records
// straight from global memory; above, the records load coalesced and transpose through the warp's shared memory.
inline constexpr uint32_t RAGGED_FINISH_WIDE_DIRECT_MAX_GRID = 128;
[[nodiscard]] inline bool raggedFinishWideTransposes(uint32_t grid) { return grid > RAGGED_FINISH_WIDE_DIRECT_MAX_GRID; }
[[nodiscard]] inline bool raggedFinishUsesWide(uint32_t components) {
    return components > RaggedMergePath::RAGGED_KR_MAX_COMPONENTS;
}
// The wide finisher's launch: warps per component, components per CTA, threads per CTA.
struct RaggedFinishWideLaunch {
    uint32_t warps_per_component = 1;
    uint32_t components_per_cta = 1;
    uint32_t threads = 32;
};
[[nodiscard]] inline RaggedFinishWideLaunch raggedFinishWideLaunch(uint32_t grid) {
    RaggedFinishWideLaunch l;
    l.warps_per_component = (grid + RAGGED_FINISH_WIDE_WARP_RECORDS - 1) / RAGGED_FINISH_WIDE_WARP_RECORDS;
    if (l.warps_per_component == 0) l.warps_per_component = 1;
    l.components_per_cta = l.warps_per_component <= 8 ? 8 / l.warps_per_component : 1;
    l.threads = 32 * l.warps_per_component * l.components_per_cta;
    return l;
}

// How a segment's FP32 aggregate becomes its output: MEAN divides by the segment's rows (0 when empty), and MAX, run
// as MIN of the sign-folded input, multiplies by -1.
struct RaggedFinalize {
    bool mean = false;
    float sign = 1.0f;
};

// A ragged KR launch shape: rows per lane per tile (1270: a tile is one aligned span of RAGGED_KR_LANES x items bytes)
// and resident CTAs per SM (0: as many as fit). The persistent grid is the SM count times that, at most
// RAGGED_FINISH_MAX_GRID. 1268: the CTA split weighs rows and ends by the bytes they move unless `weighted` is false
// (name suffix "_u"). 1272: plus `end_extra` for an end's work beyond its bytes, in row-equivalents (balanced CTAs need
// the work, not only the bytes; the 1272 cost fit: about 4 at 16 rows per lane, 2 at 32); -1 takes the default, 4
// (raggedKREndExtra). Weighted names carry the extra in force: "ragged_kr:i16_c0_e4".
struct RaggedKRShape {
    uint32_t items_per_thread = 16;  // FP8 rows per lane: 16 (1280; 1271-1279: 16 or 32); 1284: 16 bytes per lane
    uint32_t ctas_per_sm = 0;
    bool weighted = true;
    int32_t end_extra = -1;  // 0 to 255, or -1: the default
};

[[nodiscard]] bool raggedKRShapeLegal(const RaggedKRShape& shape);
[[nodiscard]] std::string raggedKRShapeName(const RaggedKRShape& shape);  // "ragged_kr:i16_c0_e12", "ragged_kr:i32_c0_u"
[[nodiscard]] uint32_t raggedKREndExtra(const RaggedKRShape& shape);   // the extra in force (0 when unweighted)

// 1284: an input dtype's element bytes for the ragged KR kernels: 1 (FP8), 2 (FP16, BF16), 4 (FP32), 0 otherwise. D
// is 1 to RaggedMergePath::raggedKRMaxComponents(bytes): rows under 16 bytes.
[[nodiscard]] uint32_t raggedKRInputBytes(DataType input_dtype);

// 1268: the path weights of a launch: a row its bytes (1280: D; 1284: D x the element's bytes), an end its offset plus
// its D output values' bytes (1272: plus raggedKREndExtra); unit weights when the shape is unweighted. Throws
// std::invalid_argument for an input dtype or D outside the kernels' domain (1265-1283: raggedKRFp8Weights).
[[nodiscard]] RaggedMergePath::PathWeights raggedKRWeights(const RaggedKRShape& shape, DataType input_dtype,
                                                           DataType offsets_dtype, DataType output_dtype,
                                                           uint32_t components = 1);

// The persistent grid of `shape` for an input of `input_dtype` and D `components` on `gpu`: SMs x min(ctas_per_sm,
// resident CTAs), at most RAGGED_FINISH_MAX_GRID. 1280: D = 1 and D >= 2 are different kernels, with their own
// residency; 1284: so is each dtype object (1265-1283: raggedKRFp8Grid).
[[nodiscard]] uint32_t raggedKRGrid(const RaggedKRShape& shape, DataType input_dtype, int gpu, uint32_t components = 1);

// Boundary workspace bytes of a VALUE launch on `grid` CTAs with D `components` (1280: raggedRecordLayout; 16 bytes
// per CTA at D = 1, as 1273's records).
[[nodiscard]] uint64_t raggedValueWorkspaceBytes(uint32_t grid, uint32_t components = 1);

// 1275: which kernels of a VALUE launch run. Only MainAndFinish computes the outputs; MainOnly and FinishOnly are for
// timing one kernel alone (the ragged audit's --ragged-kr-time): the main kernel leaves the heads unwritten, and the
// finisher alone reads whatever records the workspace holds.
enum class RaggedLaunchParts : uint8_t { MainAndFinish, MainOnly, FinishOnly };

// Ragged VALUE over input values [capacity] or [capacity, D] (1280: D 1 to 15) with UINT32 / UINT64 offsets
// [segments + 1]: output of segments x D values (any shape) in any floating dtype; SUM, MEAN, MIN or MAX. 1284: the input
// is FP8 (D 1 to 15), FP16 or BF16 (D 1 to 7) or FP32 (D 1 to 3), aligned to its element (1265-1283:
// launchRaggedKRFp8Value, FP8 only). `workspace` holds at least raggedValueWorkspaceBytes(grid, D) bytes (its contents
// on entry do not matter). Allocation-free; two launches on `stream` (the main kernel and the finisher). Throws
// std::invalid_argument for anything outside that domain.
void launchRaggedKRValue(CubReductionOp op,
                         const RaggedKRShape& shape,
                         uint32_t grid,
                         const Tensor& input,
                         const Tensor& offsets,
                         Tensor& output,
                         Tensor& workspace,
                         Stream& stream,
                         RaggedLaunchParts parts = RaggedLaunchParts::MainAndFinish);

// 1285: a ragged RK launch shape. tile_bytes: the K tile cap, 32 / 64 / 128 / 256 / 512, or 0 for the cap the input's
// size suggests (raggedRKAutoTileCap: the boundary records within 1/16 of the input's bytes); ctas_per_sm: resident
// CTAs per SM (0: as many as fit; the CTA budget is the SM count times that); the CTA split as RaggedKRShape.
// Names: "ragged_rk:ta_c0_e4" (auto cap), "ragged_rk:t128_c8_e4", "_u" unweighted.
struct RaggedRKShape {
    uint32_t tile_bytes = 0;
    uint32_t ctas_per_sm = 0;
    bool weighted = true;
    int32_t end_extra = -1;  // 0 to 255, or -1: the default (4, as KR)
};

[[nodiscard]] bool raggedRKShapeLegal(const RaggedRKShape& shape);
[[nodiscard]] std::string raggedRKShapeName(const RaggedRKShape& shape);
[[nodiscard]] uint32_t raggedRKEndExtra(const RaggedRKShape& shape);

// The CTA budget of `shape` for `input_dtype` on `gpu`: SMs x min(ctas_per_sm, resident CTAs of the RK kernel).
[[nodiscard]] uint32_t raggedRKCtas(const RaggedRKShape& shape, DataType input_dtype, int gpu);

// What a launch runs for an input of D `components` and `capacity_rows` rows on a CTA budget: the geometry (K tiles,
// tile bytes, chunks, workers), the path ranges (ctas / ktiles, 1 to RAGGED_FINISH_MAX_GRID), the grid (ranges x
// ktiles) and the workspace (raggedRecordLayout(ranges, D)). Throws std::invalid_argument outside the kernel's domain (rows
// under 16 bytes are KR's).
struct RaggedRKLaunch {
    RaggedMergePath::RaggedRKGeometry geometry;
    uint32_t tile_cap = 0;
    uint32_t ranges = 0;
    uint32_t grid = 0;
    uint64_t workspace_bytes = 0;
};
[[nodiscard]] RaggedRKLaunch raggedRKLaunch(const RaggedRKShape& shape, uint32_t ctas, DataType input_dtype,
                                            uint64_t components, uint64_t capacity_rows);

// The path weights of an RK launch: a row its D elements' bytes, an end its offset, D outputs and the extra.
[[nodiscard]] RaggedMergePath::PathWeights raggedRKWeights(const RaggedRKShape& shape, DataType input_dtype,
                                                           DataType offsets_dtype, DataType output_dtype,
                                                           uint64_t components);

// Ragged VALUE over input values [capacity, D...] with rows of 16 bytes or more (1285: FP8, D >= 16; 1295: FP16 / BF16
// D >= 8, FP32 D >= 4), UINT32 / UINT64 offsets [segments + 1]: output of segments x D values in any floating dtype;
// SUM, MEAN, MIN or MAX; any input address aligned to its element. `ctas` is the CTA budget (raggedRKCtas); `workspace` holds at least raggedRKLaunch(...).workspace_bytes.
// Allocation-free; two launches on `stream`. Throws std::invalid_argument outside that domain.
#if THOR_RAGGED_RK_PROBE
// 1290 (measurement only): later RK main launches write RAGGED_RK_PROBE_WORDS timestamps per CTA to `probe` (device
// memory of grid x 8 words), or nothing when null.
void raggedRKSetProbe(unsigned long long* probe);
#endif
void launchRaggedRKValue(CubReductionOp op,
                         const RaggedRKShape& shape,
                         uint32_t ctas,
                         const Tensor& input,
                         const Tensor& offsets,
                         Tensor& output,
                         Tensor& workspace,
                         Stream& stream,
                         RaggedLaunchParts parts = RaggedLaunchParts::MainAndFinish);

// 1314 (phase 5): ragged RK ARG (argmin / argmax) over rows of 16 bytes and more (CubRaggedRKArg.cu; the kernel
// RaggedRKArgKernels.cuh; the records RaggedArg.h's raggedArgRecordLayout(ranges, D); the finishers
// launchRaggedFinishArg). Whether it takes an input of `input_dtype` with D `components`: FP8 (E4M3, E5M2) at D >= 16,
// 1320: FP16 / BF16 at D >= 8, 1321: FP32 at D >= 4 (pair records, raggedArgPairRecordLayout; the pair finishers).
[[nodiscard]] bool raggedRKArgSupported(DataType input_dtype, uint64_t components);
// The CTA budget of `shape` for the RK ARG kernels: SMs x min(ctas_per_sm, their resident CTAs).
[[nodiscard]] uint32_t raggedRKArgCtas(const RaggedRKShape& shape, DataType input_dtype, int gpu);
// raggedRKLaunch for ARG: the auto K tile cap sized for 16-byte records per column and range (two 64-bit words; 1321
// FP32: 24-byte pairs), and the ARG records' workspace (raggedArgRecordLayout(ranges, D); FP32
// raggedArgPairRecordLayout(ranges, D)).
[[nodiscard]] RaggedRKLaunch raggedRKArgLaunch(const RaggedRKShape& shape, uint32_t ctas, DataType input_dtype,
                                               uint64_t components, uint64_t capacity_rows);
// The path weights of an RK ARG launch: a row its D elements' bytes, an end its offset, its D indices and the extra.
[[nodiscard]] RaggedMergePath::PathWeights raggedRKArgWeights(const RaggedRKShape& shape, DataType input_dtype,
                                                              DataType offsets_dtype, DataType index_dtype,
                                                              uint64_t components);
// Ragged ARGMIN / ARGMAX over input values [capacity, D...] with rows of 16 bytes or more (raggedRKArgSupported; any
// address aligned to the element), UINT32 / UINT64 offsets [segments + 1]: index output of segments x D UINT32 or
// UINT64 indices, each row * D + column (the row in the input tensor), NaNs propagating, the lowest index winning
// ties, the sentinel (all ones) for a segment without rows: launchRaggedKRArg's contract. A UINT32 index needs
// capacity x D <= 2^32 - 1. `ctas` is the CTA budget (raggedRKArgCtas); `workspace` holds at least
// raggedRKArgLaunch(...).workspace_bytes (its contents on entry do not matter). Allocation-free; two launches on
// `stream`. Throws std::invalid_argument outside that domain.
void launchRaggedRKArg(CubArgReductionOp op,
                       const RaggedRKShape& shape,
                       uint32_t ctas,
                       const Tensor& input,
                       const Tensor& offsets,
                       Tensor& index_output,
                       Tensor& workspace,
                       Stream& stream,
                       RaggedLaunchParts parts = RaggedLaunchParts::MainAndFinish);

// The finishing kernel (CubRaggedFinish.cu): one CTA combines each head with the tails before it of the same segment,
// in CTA order, and stores the finalized result. fixed_op: 0 (additive) or 2 (minimum, MAX folded). 1280: one CTA per
// component of D `components`; the records are raggedRecordLayout(grid, components). 1285: any D. 1291: above
// D = 15 the wide finisher (raggedFinishUsesWide, raggedFinishWideLaunch).
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
                             cudaStream_t stream);

// 1298 (phase 5): ragged ARG (argmin / argmax), KR first (CubRaggedKRArg.cu; the kernel RaggedKRArgKernels.cuh, the
// records and index stores RaggedArg.h, the finisher CubRaggedFinishArg.cu).
//
// Whether ragged KR ARG takes an input of `input_dtype` with D `components`: FP8 (E4M3, E5M2) at D = 1 (1298) to 15
// (1300: rows under 16 bytes); 1311: FP16 / BF16 at D 1 to 7; 1313: FP32 at D 1 to 3.
[[nodiscard]] bool raggedKRArgSupported(DataType input_dtype, uint64_t components);

// Boundary workspace bytes of an ARG launch for `input_dtype` on `grid` CTAs with D `components`: FP8 / FP16 / BF16
// raggedArgRecordLayout (an 8-byte key per CTA, then 8-byte head and tail words per CTA and component); 1313: FP32
// raggedArgPairRecordLayout (a key per CTA, then a 4-byte key and an 8-byte row per head and tail, CTA and component;
// the grid rounded up to 4). 1313: the input dtype is the first argument (the records differ by dtype).
[[nodiscard]] uint64_t raggedArgWorkspaceBytes(DataType input_dtype, uint32_t grid, uint32_t components = 1);

// The path weights of a KR ARG launch: a row its D elements' bytes, an end its offset, its D indices and the shape's
// extra (raggedKREndExtra). Throws std::invalid_argument outside raggedKRArgSupported or for an index dtype other than
// UINT32 / UINT64.
[[nodiscard]] RaggedMergePath::PathWeights raggedKRArgWeights(const RaggedKRShape& shape, DataType input_dtype,
                                                              DataType offsets_dtype, DataType index_dtype,
                                                              uint32_t components = 1);

// The persistent grid of `shape` for the KR ARG kernel: SMs x min(ctas_per_sm, its resident CTAs), at most
// RAGGED_FINISH_MAX_GRID.
[[nodiscard]] uint32_t raggedKRArgGrid(const RaggedKRShape& shape, DataType input_dtype, int gpu,
                                       uint32_t components = 1);

// Ragged ARGMIN / ARGMAX over input values [capacity] or [capacity, D] (FP8: 1298 D = 1, 1300 D 1 to 15; 1311 FP16 /
// BF16 D 1 to 7; 1313 FP32 D 1 to 3; aligned to the element) with
// UINT32 / UINT64 offsets [segments + 1]: index output of segments x D UINT32 or UINT64 indices, each the packed winner
// index row * D + component (the row in the input tensor), NaNs propagating, the lowest index winning ties, and the
// sentinel (all ones) for a segment without rows; Thor's ARG contract (CubSegmentedArgReduction), with values compared exactly. A UINT32
// index needs capacity x D <= 2^32 - 1. `workspace` holds at least raggedArgWorkspaceBytes(dtype, grid, D) bytes (its contents
// on entry do not matter). Allocation-free; two launches on `stream`. Throws std::invalid_argument outside that domain.
void launchRaggedKRArg(CubArgReductionOp op,
                       const RaggedKRShape& shape,
                       uint32_t grid,
                       const Tensor& input,
                       const Tensor& offsets,
                       Tensor& index_output,
                       Tensor& workspace,
                       Stream& stream,
                       RaggedLaunchParts parts = RaggedLaunchParts::MainAndFinish);

// The ARG finishing kernel (CubRaggedFinishArg.cu): one CTA per component (D up to 15) combines each head with the
// tails before it of the same segment by the unsigned minimum of the candidate words, and stores the winner's index.
// row_bits: the row field of the words (RaggedArg.h); index_bytes 4 or 8. 1314: any D; above 15 (raggedFinishUsesWide)
// it runs the wide ARG finisher (launchRaggedFinishArgWide, CubRaggedFinishArgWide.cu: the VALUE wide finisher's form).
void launchRaggedFinishArg(uint32_t grid,
                           uint32_t components,
                           const uint8_t* workspace,
                           uint32_t row_bits,
                           void* output,
                           uint32_t index_bytes,
                           cudaStream_t stream);

void launchRaggedFinishArgWide(uint32_t grid,
                               uint32_t components,
                               const uint8_t* workspace,
                               uint32_t row_bits,
                               void* output,
                               uint32_t index_bytes,
                               cudaStream_t stream);

// 1313: the ARG finishing kernel for pair records (CubRaggedFinishArgFp32.cu; FP32's raggedArgPairRecordLayout): the
// same order with the (key, row) minimum, one CTA per component (D up to 15). 1321: any D; above 15
// (raggedFinishUsesWide) it runs the wide pair finisher (launchRaggedFinishArgPairWide, CubRaggedFinishArgPairWide.cu).
void launchRaggedFinishArgPair(uint32_t grid,
                               uint32_t components,
                               const uint8_t* workspace,
                               void* output,
                               uint32_t index_bytes,
                               cudaStream_t stream);

void launchRaggedFinishArgPairWide(uint32_t grid,
                                   uint32_t components,
                                   const uint8_t* workspace,
                                   void* output,
                                   uint32_t index_bytes,
                                   cudaStream_t stream);

}  // namespace ThorImplementation::CubReductionInternal
