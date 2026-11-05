#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * Physical topology of one TiledRK reduction stage.
 *
 * This is deliberately independent of CubReductionStageRole/CubReductionStageSemantics. A mathematically Final pass
 * may still use Staged topology and defer its finalizer to the independently planned continuation, while a
 * mathematically First pass may use Complete topology and write its direct FP32 stage output without finalizing the
 * overall reduction operation.
 */
enum class TiledRKStageTopology : uint8_t {
    Complete = 0,
    Staged = 1,
};

struct KParallelTiledStagePlan {
    TiledRKStageTopology topology = TiledRKStageTopology::Staged;
    size_t packet_bytes = 0;
    uint32_t block_threads = 0;
    uint64_t shards_per_output = 0;
};

/**
 * Exact CUDA resource/occupancy metadata for one concrete kernel specialization.
 *
 * max_active_blocks_per_sm is reported by cudaOccupancyMaxActiveBlocksPerMultiprocessor() for the supplied CTA width
 * and dynamic shared-memory size. It therefore includes architectural, thread, register, and shared-memory residency
 * limits for that exact specialization.
 */
struct CubKernelOccupancyInfo {
    int registers_per_thread = 0;
    size_t static_shared_bytes = 0;
    size_t dynamic_shared_bytes = 0;
    int max_active_blocks_per_sm = 0;
    int max_active_warps_per_sm = 0;
};

/**
 * Launches the benchmark-proven cooperative y8/16-byte-packet first stage for a natural dense tiled reduction.
 * The stage applies only the logical input transform and associative SUM combine, emitting FP32
 * [outer,shards,inner] partials. Finalization and output scaling belong to the normally stamped remainder stage.
 */
void launchCooperativeShardedTiledFirstStage(const CubReductionStageSemantics& semantics,
                                             const Tensor& input,
                                             Tensor& fp32_partials,
                                             const CubReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards_per_output,
                                             Stream& stream);

/** Query exact CUDA occupancy metadata for the SUM/identity aligned R-cooperative staged specialization. */
[[nodiscard]] CubKernelOccupancyInfo queryCooperativeShardedTiledFirstStageOccupancy(DataType input_dtype,
                                                                                     uint64_t rows_per_shard);

/**
 * Launches one aligned K-parallel TiledRK stage. Packet width, CTA width, and R shard depth are planner geometry, not
 * distinct reducer families. Warps own independent K outputs, so the stage requires no inter-warp reduction.
 *
 * Staged topology requires shards_per_output > 1 and emits FP32 [outer,shards,inner] partials. It intentionally does
 * not apply semantics.finalize or output_scale; those belong to the independently stamped continuation. Complete
 * topology requires shards_per_output == 1 and writes the supplied stage output directly, applying the semantic
 * finalizer, runtime output scale, and output dtype conversion for this mathematical pass.
 *
 * packet_bytes is the naturally aligned per-lane input transaction width. FP16/BF16/FP32 KParallel uses 4/8/16 B
 * with block_threads in {32,64,128,256}. Every global lane transaction remains <=16 B.
 *
 * FP8 input is the retained AlignedK family only (Fp8RK::Fp8AlignedKPolicy): Staged P2/P8 x CTA32/64/128 and
 * Complete P2/CTA32, P8/CTA32 and P8/CTA64. Any other FP8 packet/CTA request is not compiled and throws
 * std::logic_error. Staged Identity SUM/PRODUCT/MIN/MAX share one operation-unified kernel per configuration;
 * Complete Identity uses joined P2/CTA32, in-branch P8/CTA32, and typed SUM/MEAN plus in-branch nonadditive
 * P8/CTA64. ABS/Square compositions use the typed packet kernel, minus the accepted pruned specializations.
 */
void launchKParallelTiledStage(const CubReductionStageSemantics& semantics,
                               const Tensor& input,
                               Tensor& stage_output,
                               const CubReductionGeometry& geometry,
                               const KParallelTiledStagePlan& plan,
                               float output_scale,
                               Stream& stream);

/** Query exact CUDA occupancy metadata for the physical production K-parallel kernel selected by plan.
 * For FP8 input only the retained AlignedK Identity configurations are compiled; others throw std::logic_error. */
[[nodiscard]] CubKernelOccupancyInfo queryKParallelTiledStageOccupancy(DataType input_dtype,
                                                                       const KParallelTiledStagePlan& plan);

/** Exact operation/output-aware occupancy of a retained AlignedK Identity Complete kernel.
 * In P8/CTA64, SUM/MEAN uses typed additive and PRODUCT/MIN/MAX uses unified
 * in-branch; neither the packet nor the CTA axis implies a compiled R depth. */
[[nodiscard]] CubKernelOccupancyInfo queryFp8AlignedKCompleteOccupancy(
    DataType input_dtype, DataType output_dtype, CubReductionStageCombine operation,
    const KParallelTiledStagePlan& plan);

/** Exact physical occupancy for both Identity and transformed AlignedK kernels. */
[[nodiscard]] CubKernelOccupancyInfo queryFp8AlignedKStageOccupancy(
    DataType input_dtype, DataType output_dtype, CubReductionStageInputTransform transform,
    CubReductionStageCombine operation, const KParallelTiledStagePlan& plan);

/** 1188: AlignedK on FP32 input (plans/Thor_FP32_Dense_Reduction_Plan, phase 5). A lane owns one 8- or 16-byte
 *  packet (2 or 4 components; K a multiple of that), or (1228) one 4-byte packet (one component, any K), and reduces
 *  its shard's rows in FP32; CTA 32 / 64 / 128.
 *  shards_per_output > 1 is Staged: unfinalized FP32 [outer, shards, K] partials at the balanced row split.
 *  shards_per_output == 1 is Complete: finalized, scaled, stored in the output dtype at the geometry's output strides.
 *  The input base is aligned to the packet. Semantics: SUM / MIN / MAX x {identity, |x|, x^2}, PRODUCT x identity. */
[[nodiscard]] bool fp32AlignedKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                        uint32_t packet_bytes, uint32_t block_threads, uint64_t shards_per_output);

void launchFp32AlignedKStage(const CubReductionStageSemantics& semantics,
                             const Tensor& input,
                             Tensor& stage_output,
                             const CubReductionGeometry& geometry,
                             uint32_t packet_bytes,
                             uint32_t block_threads,
                             uint64_t shards_per_output,
                             float output_scale,
                             Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32AlignedKOccupancy(uint32_t packet_bytes, uint32_t block_threads,
                                                                bool staged);

/** 1198: FP32 RK ARG (plans/Thor_FP32_Dense_Reduction_Plan, phase 6). Candidates are Fp32ArgKey.cuh's
 *  (key, index << 1 | negative zero): NaNs propagate, the lowest index wins ties (-0 / +0 included).
 *  The AlignedK twin: fp32AlignedKKernel's ownership (8- or 16-byte packets, CTA 32 / 64 / 128, balanced shard rows;
 *  no 4-byte packet: fp32AlignedKArgSupports rejects it).
 *  One shard writes the requested outputs; more write packed 64-bit partials [outer, shards, K] to `packed_scratch`
 *  (fp32ArgPackedScratchBytes) and finish them in the same call. value_output: any float storage (or null);
 *  index_output: UINT32 / UINT64 (or null), index = row * domain_stride. Rows per shard <= 2^31 - 1. */
[[nodiscard]] bool fp32AlignedKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint32_t packet_bytes, uint32_t block_threads, uint64_t shards_per_output);

[[nodiscard]] uint64_t fp32ArgPackedScratchBytes(uint64_t outer_size, uint64_t shards, uint64_t inner_size);

void launchFp32AlignedKArg(const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           uint32_t packet_bytes,
                           uint32_t block_threads,
                           uint64_t shards_per_output,
                           uint64_t* packed_scratch,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32AlignedKArgOccupancy(uint32_t packet_bytes, uint32_t block_threads,
                                                                   bool staged);

/** 1198: reduces packed FP32 ARG partials [outer, shards, K] over shards and writes the requested outputs. Shard s
 *  begins at s * uniform_rows_per_shard, or with 0 at the balanced split of R into `shards`. FP8's per-output kernel,
 *  or its shard-split kernel for few outputs with many shards (fp8ArgPackedContinuationSplitsShards). */
void launchFp32ArgPackedContinuation(const uint64_t* packed_partials,
                                     Tensor* value_output,
                                     Tensor* index_output,
                                     uint64_t outer_size,
                                     uint64_t reduction_size,
                                     uint64_t inner_size,
                                     uint64_t shards,
                                     uint64_t uniform_rows_per_shard,
                                     bool maximum,
                                     uint64_t domain_stride,
                                     Stream& stream);

/** 1208: carried-index (intermediate) FP32 ARG (plans/Thor_FP32_Dense_Reduction_Plan, phase 8). The
 *  input is FP32 values [outer, R, K] with a carried index (UINT32 / UINT64, the same shape) from earlier passes; an
 *  element's composed index is carried + row * row_stride (row_stride: the current run's original-domain stride, 0 for
 *  an intermediate run). Candidates compare (key, composed index) in full; the index output is the composed index.
 *  Staged passes write wide partials (a uint32 key and a uint64 tail per partial, fp32ArgWideScratchBytes, 256-byte
 *  aligned scratch) and finish them with launchFp32ArgWideContinuation in the same call. */
[[nodiscard]] uint64_t fp32ArgWideScratchBytes(uint64_t outer_size, uint64_t shards, uint64_t inner_size);

void launchFp32ArgWideContinuation(void* wide_scratch,
                                   Tensor* value_output,
                                   Tensor* index_output,
                                   uint64_t outer_size,
                                   uint64_t shards,
                                   uint64_t inner_size,
                                   bool maximum,
                                   Stream& stream);

/** 1208: the carried-index AlignedK ARG twin: launchFp32AlignedKArg's configurations (fp32AlignedKSupports), the carried
 *  index aligned like its packets (at most 16 bytes). */
void launchFp32AlignedKCarriedArg(const Tensor& input,
                                  const Tensor& carried_index,
                                  Tensor* value_output,
                                  Tensor* index_output,
                                  const CubReductionGeometry& geometry,
                                  uint32_t packet_bytes,
                                  uint32_t block_threads,
                                  uint64_t shards_per_output,
                                  void* wide_scratch,
                                  bool maximum,
                                  uint64_t row_stride,
                                  Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32AlignedKCarriedArgOccupancy(uint32_t packet_bytes,
                                                                          uint32_t block_threads, bool staged,
                                                                          DataType index_dtype);

/**
 * Launches exactly one rotated R-cooperative TiledRK stage. The stage applies the input transform and associative
 * combine for the nominated R shard geometry, emitting FP32 [outer,shards,inner] partials. It never applies the final
 * semantic finalizer/output scale and never launches or selects a successor reducer.
 */
void launchAwkwardAlignmentRotatedShardedFirstStage(const CubReductionStageSemantics& semantics,
                                                     const Tensor& input,
                                                     Tensor& fp32_partials,
                                                     const CubReductionGeometry& geometry,
                                                     uint64_t rows_per_shard,
                                                     uint64_t shards_per_output,
                                                     Stream& stream);

/** Query exact CUDA occupancy metadata for the SUM/identity rotated R-cooperative staged specialization.
 * FP16/BF16/FP32 only; original FP8 input uses the retained RuntimeK rotated first stage. */
[[nodiscard]] CubKernelOccupancyInfo queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(
    DataType input_dtype, uint64_t rows_per_shard);

/**
 * Retained RuntimeK rotated first stage: p16 packets, exact (512) or prefix-reserved (497) runtime K tiles,
 * and only the retained Fp8RuntimeKPolicy::rotated_row_depths. Emits FP32 [outer,shards,K] partials with no
 * K-tile merge; finalizer, scale and output conversion belong to the independently planned continuation.
 * Any other row depth is not compiled and fails with std::logic_error (occupancy: absent candidate).
 */
void launchFp8GeneralRParallelKTileRotatedCalibrationFirstStage(
    const Tensor& input,
    Tensor& fp32_partials,
    const CubReductionGeometry& geometry,
    bool exact_tile,
    uint64_t rows_per_shard,
    uint64_t shards_per_output,
    Stream& stream,
    CubReductionStageInputTransform transform = CubReductionStageInputTransform::Identity,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelKTileRotatedCalibrationOccupancy(
    DataType input_dtype, bool exact_tile, uint64_t rows_per_shard,
    CubReductionStageInputTransform transform = CubReductionStageInputTransform::Identity,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

/** 1190: the RuntimeK rotated first stage on FP32 input, PrefixReserved only (plans/Thor_FP32_Dense_Reduction_Plan,
 *  phase 5): p16 packets (4 components per lane), 4 cooperative warps (the fewest whose revisit stride,
 *  4 rows of K floats, is a whole number of packets), a logical K tile of 125 components (128 less a 0..3 prefix), and
 *  FP8's retained row depths (Fp8RuntimeKPolicy::rotated_row_depths). shards_per_output = ceil(R / rows) > 1. Emits
 *  unfinalized FP32 [outer, shards, K] partials; every combine x {identity, |x|, x^2}; the input is 4-byte aligned. */
inline constexpr uint32_t FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS = 125;

[[nodiscard]] bool fp32RuntimeKRotatedSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                               uint64_t rows_per_shard);

void launchFp32RuntimeKRotatedFirstStage(const Tensor& input,
                                         Tensor& fp32_partials,
                                         const CubReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards_per_output,
                                         Stream& stream,
                                         CubReductionStageInputTransform transform =
                                             CubReductionStageInputTransform::Identity,
                                         CubReductionStageCombine combine = CubReductionStageCombine::Sum);

/** 1200: the FP32 RK ARG rotated twin (plans/Thor_FP32_Dense_Reduction_Plan, phase 6): FP32 VALUE
 *  rotated's layout (16-byte packets, 4 warps, the 125-component tile; any 4-byte aligned input) with one ARG
 *  candidate per component and any shard depth up to 2^31 - 1 rows. rows_per_shard >= R is one Complete pass writing
 *  the requested outputs; otherwise packed partials go to `packed_scratch` (fp32ArgPackedScratchBytes with
 *  ceil(R / rows) shards) and launchFp32ArgPackedContinuation finishes them in the same call. Outputs as
 *  launchFp32AlignedKArg's. */
[[nodiscard]] bool fp32RuntimeKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint64_t rows_per_shard);

void launchFp32RuntimeKArg(const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           uint64_t rows_per_shard,
                           uint64_t* packed_scratch,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32RuntimeKArgOccupancy(bool staged);

/** 1210: the carried-index rotated ARG twin (phase 8; inputs, outputs and scratch as launchFp32AlignedKCarriedArg's):
 *  launchFp32RuntimeKArg's layout and any depth up to 2^32 - 1 rows (depth >= R: one Complete pass). Both the input
 *  and the carried index are 16-byte aligned. */
[[nodiscard]] bool fp32RuntimeKCarriedArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                                  uint64_t rows_per_shard);

void launchFp32RuntimeKCarriedArg(const Tensor& input,
                                  const Tensor& carried_index,
                                  Tensor* value_output,
                                  Tensor* index_output,
                                  const CubReductionGeometry& geometry,
                                  uint64_t rows_per_shard,
                                  void* wide_scratch,
                                  bool maximum,
                                  uint64_t row_stride,
                                  Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32RuntimeKCarriedArgOccupancy(bool staged, DataType index_dtype);

[[nodiscard]] CubKernelOccupancyInfo queryFp32RuntimeKRotatedOccupancy(
    uint64_t rows_per_shard,
    CubReductionStageInputTransform transform = CubReductionStageInputTransform::Identity,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

/**
 * Production FP16/BF16 narrow-flat R-cooperative stage for K<=32.
 * Emits unfinalized FP32 [outer,shard,K] partials in one physical pass.
 * Original FP8 RK VALUE inputs use the separate retained first-stage selector;
 * this generic family is not an FP8 compatibility fallback.
 */
void launchNarrowLowPrecisionFlatRCooperativeFirstStage(const CubReductionStageSemantics& semantics,
                                                         const Tensor& input,
                                                         Tensor& fp32_partials,
                                                         const CubReductionGeometry& geometry,
                                                         uint64_t rows_per_shard,
                                                         uint64_t shards_per_output,
                                                         Stream& stream);

/** Query exact CUDA occupancy metadata for the SUM/identity narrow-flat specialization. */
[[nodiscard]] CubKernelOccupancyInfo queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(
    DataType input_dtype, uint64_t rows_per_shard);

/** Retained DeepRSmallK FP8 P16 first stage, with typed SUM/PRODUCT/MIN/MAX,
 * typed Identity/ABS/Square, typed E4M3/E5M2, and the selected five depths.
 * Emits UNFINALIZED FP32 [outer, ceil(R/rows), K] partials. The caller must
 * independently plan an FP32 continuation with the same combine semantics.
 * This does not change the existing production planner or selector.
 */
void launchFp8DeepRSmallKRetainedFirstStage(
    const CubReductionStageSemantics& semantics, const Tensor& input, Tensor& fp32_partials,
    const CubReductionGeometry& geometry, uint64_t rows_per_shard, uint64_t shards_per_output,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8DeepRSmallKRetainedOccupancy(
    DataType input_dtype, uint64_t rows_per_shard, uint64_t inner_size,
    const CubReductionStageSemantics& semantics);

/** 1186: the DeepRSmallK first stage on FP32 input (plans/Thor_FP32_Dense_Reduction_Plan, phase 5).
 *  FP8's kernels at 4-byte elements: K <= 125 (a P16 window of 128 FP32, less a worst-case prefix), ownerships
 *  TinyK (K <= 4, <= 8) and one / two / four components per lane, FP8's five depths. Emits unfinalized FP32
 *  [outer, ceil(R / rows), K] partials. The input is 16-byte aligned. fp32DeepRSmallKSupports is the legality rule. */
[[nodiscard]] bool fp32DeepRSmallKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint64_t rows_per_shard);

void launchFp32DeepRSmallKRetainedFirstStage(
    const CubReductionStageSemantics& semantics, const Tensor& input, Tensor& fp32_partials,
    const CubReductionGeometry& geometry, uint64_t rows_per_shard, uint64_t shards_per_output,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32DeepRSmallKRetainedOccupancy(
    uint64_t rows_per_shard, uint64_t inner_size, const CubReductionStageSemantics& semantics);





/** Retained OneWindow Complete kernel. O1 supports W1/W2/W4/W8, while O2/O4/O8
 * partition a warp among outputs. Applies the runtime finalizer/scale and converts
 * to the final output dtype. This function name predates the retained cutover.
 * Since P5-J2 it runs the bucketed unit for (outputs per warp, packet bytes, K's
 * bucket): only packet_bytes 16 has units, and a request without one throws. */
void launchFp8GeneralRParallelRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream,
    const CubReductionStageSemantics& semantics = {},
    float output_scale = 1.0f);

[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

/** The OneWindow kernel by unit (P5-I bucketed; the only OneWindow kernel since P5-J2, which the selector reaches
 * through launchFp8GeneralRParallelRCooperativeCalibrationComplete). A unit is one compiled kernel shape: outputs per
 * warp (1/2/4/8), packet bytes (16), components-per-lane bucket (1/2/4/8/16) and group loop form.
 * fp8OneWindowRuntimeKUnits(K) lists the compiled units whose bucket holds K. warps_per_output is 1/2/4/8 for O1
 * and 1 otherwise. Output must hold outer x K elements (any dims). */
struct Fp8OneWindowRuntimeKUnit {
    uint32_t outputs_per_warp;
    uint32_t packet_bytes;
    uint32_t max_components_per_lane;
    bool load_ahead;
    constexpr bool operator==(const Fp8OneWindowRuntimeKUnit&) const = default;
};

[[nodiscard]] std::vector<Fp8OneWindowRuntimeKUnit> fp8OneWindowRuntimeKUnits(uint64_t inner_size);

void launchFp8OneWindowRuntimeKComplete(const Tensor& input,
                                        Tensor& output,
                                        const CubReductionGeometry& geometry,
                                        const Fp8OneWindowRuntimeKUnit& unit,
                                        uint32_t warps_per_output,
                                        Stream& stream,
                                        const CubReductionStageSemantics& semantics = {},
                                        float output_scale = 1.0f);

[[nodiscard]] CubKernelOccupancyInfo queryFp8OneWindowRuntimeKOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    const Fp8OneWindowRuntimeKUnit& unit,
    uint32_t warps_per_output,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

/** 1185: the OneWindow kernel on FP32 input (plans/Thor_FP32_Dense_Reduction_Plan, phase 5). The
 *  units are FP8's whose components-per-lane bucket FP32 reaches (<= 4: a 16-byte lane packet holds 4 FP32), so K <=
 *  125 / 61 / 29 / 13 at O1 / O2 / O4 / O8. fp32OneWindowRuntimeKUnits(K) lists the units whose bucket holds K. The
 *  input is 16-byte aligned and at most 2^32 - 1 bytes; MIN / MAX fold with the FP32 masks. */
[[nodiscard]] std::vector<Fp8OneWindowRuntimeKUnit> fp32OneWindowRuntimeKUnits(uint64_t inner_size);

void launchFp32OneWindowRuntimeKComplete(const Tensor& input,
                                         Tensor& output,
                                         const CubReductionGeometry& geometry,
                                         const Fp8OneWindowRuntimeKUnit& unit,
                                         uint32_t warps_per_output,
                                         Stream& stream,
                                         const CubReductionStageSemantics& semantics = {},
                                         float output_scale = 1.0f);

[[nodiscard]] CubKernelOccupancyInfo queryFp32OneWindowRuntimeKOccupancy(
    uint64_t inner_size,
    const Fp8OneWindowRuntimeKUnit& unit,
    uint32_t warps_per_output,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

/** 1203: the FP32 RK ARG DeepRSmallK twins (phase 6): FP32 VALUE DeepRSmallK's kernels and ownership
 *  (K <= 125: TinyK at K <= 4 / 8, then one / two / four components per lane; a retained depth with R > depth; a
 *  16-byte aligned input) with one ARG candidate per component. Packed partials [outer, ceil(R / depth), K] go to
 *  `packed_scratch` (fp32ArgPackedScratchBytes) and launchFp32ArgPackedContinuation finishes them in the same call.
 *  Outputs as launchFp32AlignedKArg's. */
[[nodiscard]] bool fp32DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                              uint64_t rows_per_shard);

void launchFp32DeepRSmallKArg(const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              uint64_t rows_per_shard,
                              uint64_t* packed_scratch,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32DeepRSmallKArgOccupancy(uint64_t inner_size);

/** 1213: the carried-index DeepRSmallK ARG twins (phase 8; inputs and outputs as launchFp32AlignedKCarriedArg's):
 *  launchFp32DeepRSmallKArg's kernels, ownership and depths (fp32DeepRSmallKSupports: a retained depth with R > depth),
 *  the input and the carried index 16-byte aligned. Wide partials go to `wide_scratch` (fp32ArgWideScratchBytes with
 *  ceil(R / depth) shards, 256-byte aligned) and launchFp32ArgWideContinuation finishes them in the same call. */
void launchFp32DeepRSmallKCarriedArg(const Tensor& input,
                                     const Tensor& carried_index,
                                     Tensor* value_output,
                                     Tensor* index_output,
                                     const CubReductionGeometry& geometry,
                                     uint64_t rows_per_shard,
                                     void* wide_scratch,
                                     bool maximum,
                                     uint64_t row_stride,
                                     Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32DeepRSmallKCarriedArgOccupancy(uint64_t inner_size,
                                                                             DataType index_dtype);

/** 1202: the FP32 RK ARG OneWindow twin (phase 6): launchFp32OneWindowRuntimeKComplete's units and
 *  ownership (O1 W1/W2/W4/W8, O2/O4/O8 W1; K <= 125 / 61 / 29 / 13) with one ARG candidate per owned component; Complete
 *  only, R <= 2^31 - 1, input at most 2^32 - 1 bytes and 16-byte aligned. Outputs as launchFp32AlignedKArg's. Not
 *  behind THOR_CUB_ENABLE_FP8_TYPES (its kernel is FP32-only). */
[[nodiscard]] bool fp32OneWindowArgSupports(uint64_t outer_size,
                                            uint64_t reduction_size,
                                            uint64_t inner_size,
                                            const Fp8OneWindowRuntimeKUnit& unit,
                                            uint32_t warps_per_output);

void launchFp32OneWindowArg(const Tensor& input,
                            Tensor* value_output,
                            Tensor* index_output,
                            const CubReductionGeometry& geometry,
                            const Fp8OneWindowRuntimeKUnit& unit,
                            uint32_t warps_per_output,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32OneWindowArgOccupancy(uint64_t inner_size,
                                                                    const Fp8OneWindowRuntimeKUnit& unit,
                                                                    uint32_t warps_per_output);

/** 1212: the carried-index OneWindow ARG twin (phase 8; inputs and outputs as launchFp32AlignedKCarriedArg's):
 *  launchFp32OneWindowArg's units and ownership; Complete only, R <= 2^32 - 1, at most 2^32 - 1 value bytes, the input
 *  and the carried index 16-byte aligned. */
[[nodiscard]] bool fp32OneWindowCarriedArgSupports(uint64_t outer_size,
                                                   uint64_t reduction_size,
                                                   uint64_t inner_size,
                                                   const Fp8OneWindowRuntimeKUnit& unit,
                                                   uint32_t warps_per_output);

void launchFp32OneWindowCarriedArg(const Tensor& input,
                                   const Tensor& carried_index,
                                   Tensor* value_output,
                                   Tensor* index_output,
                                   const CubReductionGeometry& geometry,
                                   const Fp8OneWindowRuntimeKUnit& unit,
                                   uint32_t warps_per_output,
                                   bool maximum,
                                   uint64_t row_stride,
                                   Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32OneWindowCarriedArgOccupancy(uint64_t inner_size,
                                                                           const Fp8OneWindowRuntimeKUnit& unit,
                                                                           uint32_t warps_per_output,
                                                                           DataType index_dtype);

/** ARG-A2: the OneWindow argmin/argmax twin of a OneWindow unit (one kernel per unit; argmin/argmax and E4M3/E5M2 are
 * runtime). Complete only: writes the winning value (FP8/FP16/BF16/FP32 storage, optional) and the index
 * row * domain_stride (UINT32/UINT64, optional) for each of outer x K outputs. NaN propagates; the lowest row wins
 * ties, -0 ties +0. fp8OneWindowArgSupports() is the legality rule shared by selection and launch
 * (R < 2^23, UINT32 element indexing, a compiled unit whose bucket holds K). */
[[nodiscard]] bool fp8OneWindowArgSupports(uint64_t outer_size,
                                           uint64_t reduction_size,
                                           uint64_t inner_size,
                                           const Fp8OneWindowRuntimeKUnit& unit,
                                           uint32_t warps_per_output);

/** ARG-A3: the RuntimeK PacketComplete argmin/argmax twin: VALUE's persistent [outer, K tile] work items with 1/2/4/8
 * warps per item over R, at most 256 components per tile (8 per lane; fp8PacketCompleteArgTileComponents keeps VALUE's
 * tile up to 256 unless a 256-component tile needs fewer key words per row). Complete: any supported value storage,
 * UINT32/UINT64 index. Requires a 16-byte aligned input base (as the VALUE kernel). */
[[nodiscard]] uint32_t fp8PacketCompleteArgTileComponents(uint64_t inner_size, uint32_t value_tile_components);
[[nodiscard]] bool fp8PacketCompleteArgSupports(uint64_t outer_size,
                                                uint64_t reduction_size,
                                                uint64_t inner_size,
                                                uint32_t warps_per_output,
                                                uint32_t tile_components);
void launchFp8PacketCompleteArg(const Tensor& input,
                                Tensor* value_output,
                                Tensor* index_output,
                                const CubReductionGeometry& geometry,
                                uint32_t warps_per_output,
                                uint32_t tile_components,
                                bool maximum,
                                uint64_t domain_stride,
                                Stream& stream);

/** ARG-A3: the AlignedK argmin/argmax twin: one thread owns one aligned 2- or 8-byte packet of components (K a multiple
 * of the packet; CTA 32/64/128 threads is a launch operand) over its shard's balanced rows. One shard is Complete (any
 * supported value storage, UINT32/UINT64 index); several write FP32 value + carried index partials. */
[[nodiscard]] bool fp8AlignedKArgSupports(uint64_t outer_size,
                                          uint64_t reduction_size,
                                          uint64_t inner_size,
                                          uint32_t packet_bytes,
                                          uint32_t cta_threads,
                                          uint64_t shards);
void launchFp8AlignedKArgPass(const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              uint32_t packet_bytes,
                              uint32_t cta_threads,
                              uint64_t shards,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream,
                              uint32_t* packed_partials = nullptr);

/** ARG-A3g (1116): packed staged partials. A staged twin with `packed_partials` writes one shard-relative lane
 * candidate per partial (key << 24 | shard row << 1 | negative zero, 4 bytes) instead of FP32 value + carried index
 * (8 bytes); launchFp8ArgPackedContinuation reduces them over shards into the requested outputs. Shard s begins at
 * s * uniform_rows_per_shard, or at VALUE's balanced split when uniform_rows_per_shard is 0 (AlignedK). */
void launchFp8DeepRSmallKArgFirstStagePacked(const Tensor& input,
                                             uint32_t* packed_partials,
                                             const CubReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards,
                                             bool maximum,
                                             Stream& stream);
/** 1135: the packed continuation splits each output's shards across a block (fp8ArgPackedContinuationSplitKernel)
 * when there are many shards and too few outputs for one thread per output to hide the shard loop's latency.
 * 1137 (1135 audit, against the per-output kernel): at 16-31 shards the split wins below 16 Ki outputs (0.5-0.7x)
 * and loses from there (1.2x at 16-64 Ki, 1.8x at 64-128 Ki); at >= 32 shards it wins below 64 Ki outputs
 * (0.5-0.9x) and loses above (1.5-1.7x). A layout must also fill a warp (columns x groups >= 32): K = 2 at 16 shards
 * left 24 of 32 lanes idle. */
inline constexpr uint64_t FP8_ARG_PACKED_SPLIT_MIN_SHARDS = 16;
inline constexpr uint64_t FP8_ARG_PACKED_SPLIT_MAX_OUTPUTS = uint64_t{1} << 14;
inline constexpr uint64_t FP8_ARG_PACKED_SPLIT_WIDE_MIN_SHARDS = 32;
inline constexpr uint64_t FP8_ARG_PACKED_SPLIT_WIDE_MAX_OUTPUTS = uint64_t{1} << 16;
[[nodiscard]] bool fp8ArgPackedContinuationSplitsShards(uint64_t outer_size, uint64_t inner_size, uint64_t shards);
/** Split layout: `columns` components (K below 32, else 32) x `groups` shard groups, `threads` = that rounded up to
 * whole warps (at most 1024). */
struct Fp8ArgPackedSplitLayout {
    uint32_t columns = 0;
    uint32_t groups = 0;
    uint32_t threads = 0;
};
[[nodiscard]] Fp8ArgPackedSplitLayout fp8ArgPackedSplitLayout(uint64_t inner_size, uint64_t shards);
void launchFp8ArgPackedContinuation(const uint32_t* packed_partials,
                                    Tensor* value_output,
                                    Tensor* index_output,
                                    DataType input_dtype,
                                    uint64_t outer_size,
                                    uint64_t reduction_size,
                                    uint64_t inner_size,
                                    uint64_t shards,
                                    uint64_t uniform_rows_per_shard,
                                    bool maximum,
                                    uint64_t domain_stride,
                                    Stream& stream);

/** ARG-A3: the RuntimeK rotated argmin/argmax twin, one rotated kernel (8-byte packets only since 1112; the logical K
 * tile is fp8RuntimeKArgTileComponents) at a runtime shard depth. One shard (rows_per_shard >= R) is a Complete pass writing the
 * requested outputs (value in any supported storage, index UINT32/UINT64); several shards write FP32 value + carried
 * index partials [outer, ceil(R / rows), K]. index = row * domain_stride. */
[[nodiscard]] uint32_t fp8RuntimeKArgTileComponents(uint32_t packet_bytes);
[[nodiscard]] bool fp8RuntimeKArgSupports(uint64_t outer_size,
                                          uint64_t reduction_size,
                                          uint64_t inner_size,
                                          uint64_t rows_per_shard,
                                          uint32_t packet_bytes);
void launchFp8RuntimeKArgPass(const Tensor& input,
                              Tensor* value_output,
                              Tensor* index_output,
                              const CubReductionGeometry& geometry,
                              uint64_t rows_per_shard,
                              uint32_t packet_bytes,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream,
                              uint32_t* packed_partials = nullptr);

/** ARG-A3: the DeepRSmallK argmin/argmax twin: staged first stage at a retained depth (Fp8DeepRSmallKPolicy) writing
 * FP32 value + carried index (UINT32/UINT64) partials [outer, ceil(R / rows), K]; index = row * domain_stride. */
[[nodiscard]] bool fp8DeepRSmallKArgSupports(uint64_t outer_size,
                                             uint64_t reduction_size,
                                             uint64_t inner_size,
                                             uint64_t rows_per_shard);

void launchFp8DeepRSmallKArgFirstStage(const Tensor& input,
                                       Tensor& value_partials,
                                       Tensor& index_partials,
                                       const CubReductionGeometry& geometry,
                                       uint64_t rows_per_shard,
                                       uint64_t shards,
                                       bool maximum,
                                       uint64_t domain_stride,
                                       Stream& stream);

void launchFp8OneWindowArgComplete(const Tensor& input,
                                   Tensor* value_output,
                                   Tensor* index_output,
                                   const CubReductionGeometry& geometry,
                                   const Fp8OneWindowRuntimeKUnit& unit,
                                   uint32_t warps_per_output,
                                   bool maximum,
                                   uint64_t domain_stride,
                                   Stream& stream);


/** Retained RuntimeK PacketP16 Complete kernel. K tiles and warp ownership are
 * runtime geometry; finalizer, scale, and output conversion occur on the completed result. */
void launchFp8GeneralRParallelKTileRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t tile_components,
    Stream& stream,
    const CubReductionStageSemantics& semantics = {},
    float output_scale = 1.0f);

[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelKTileRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    CubReductionStageInputTransform transform = CubReductionStageInputTransform::Identity,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

/** 1190: RuntimeK PacketComplete on FP32 input (plans/Thor_FP32_Dense_Reduction_Plan, phase 5). The
 *  FP8 kernel's topology at 4 components per p16 packet: persistent [outer, K tile] work items, W1/W2/W4/W8 warps per
 *  item over R, a K tile of at most 124 components (any per-row prefix of 0..3 still fits the 32-packet window). Writes
 *  the finalized, scaled output. The input base is 16-byte aligned. Every combine x {identity, |x|, x^2}. */
inline constexpr uint32_t FP32_PACKET_COMPLETE_MAX_TILE_COMPONENTS = 124;

[[nodiscard]] bool fp32PacketCompleteSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                              uint32_t warps_per_output, uint32_t tile_components);

void launchFp32PacketComplete(const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              uint32_t warps_per_output,
                              uint32_t tile_components,
                              Stream& stream,
                              const CubReductionStageSemantics& semantics = {},
                              float output_scale = 1.0f);

/** 1201: the FP32 RK ARG PacketComplete twin (phase 6): launchFp32PacketComplete's topology (W1/W2/W4/W8
 *  warps per [outer, K tile] item, tiles of 1..124 components, a 16-byte aligned input) with one ARG candidate per
 *  component; Complete only, R <= 2^31 - 1. Outputs as launchFp32AlignedKArg's. */
[[nodiscard]] bool fp32PacketCompleteArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                                 uint32_t warps_per_output, uint32_t tile_components);

void launchFp32PacketCompleteArg(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 uint32_t warps_per_output,
                                 uint32_t tile_components,
                                 bool maximum,
                                 uint64_t domain_stride,
                                 Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp32PacketCompleteArgOccupancy();

[[nodiscard]] CubKernelOccupancyInfo queryFp32PacketCompleteOccupancy(
    CubReductionStageInputTransform transform = CubReductionStageInputTransform::Identity,
    CubReductionStageCombine combine = CubReductionStageCombine::Sum);

size_t querySumReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchSumReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t querySumDivideReductionBytes(DataType input_dtype,
                                    const void* input,
                                    uint64_t input_elements,
                                    DataType output_dtype,
                                    void* output,
                                    const CubReductionGeometry& geometry,
                                    uint64_t divisor,
                                    float output_scale,
                                    const Stream& stream);
void launchSumDivideReduction(const Tensor& temp_storage,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              uint64_t divisor,
                              float output_scale,
                              Stream& stream);

size_t querySumSqrtReductionBytes(DataType input_dtype,
                                  const void* input,
                                  uint64_t input_elements,
                                  DataType output_dtype,
                                  void* output,
                                  const CubReductionGeometry& geometry,
                                  float output_scale,
                                  const Stream& stream);
void launchSumSqrtReduction(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& input,
                            Tensor& output,
                            const CubReductionGeometry& geometry,
                            float output_scale,
                            Stream& stream);

size_t queryProductReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchProductReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryMeanReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchMeanReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryL1NormReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchL1NormReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryL2NormReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchL2NormReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t querySumSquaresReductionBytes(DataType input_dtype,
                                     const void* input,
                                     uint64_t input_elements,
                                     DataType output_dtype,
                                     void* output,
                                     const CubReductionGeometry& geometry,
                                     float output_scale,
                                     const Stream& stream);
void launchSumSquaresReduction(const Tensor& temp_storage,
                               size_t temp_storage_bytes,
                               const Tensor& input,
                               Tensor& output,
                               const CubReductionGeometry& geometry,
                               float output_scale,
                               Stream& stream);

size_t queryMinReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchMinReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryMaxReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchMaxReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryOffsetSegmentedReductionBytes(CubReductionOp op,
                                          const Tensor& input,
                                          Tensor& output,
                                          const Tensor& segment_offsets,
                                          uint64_t num_items,
                                          uint64_t num_segments,
                                          const Stream& stream);
void launchOffsetSegmentedReduction(CubReductionOp op,
                                    const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    uint64_t num_items,
                                    uint64_t num_segments,
                                    Stream& stream);

size_t queryOffsetSegmentedArgReductionBytes(CubArgReductionOp op,
                                             const Tensor& input,
                                             Tensor& index_output,
                                             const Tensor& segment_offsets,
                                             uint64_t num_segments,
                                             const Stream& stream);
void launchOffsetSegmentedArgReduction(CubArgReductionOp op,
                                       const Tensor& temp_storage,
                                       size_t temp_storage_bytes,
                                       const Tensor& input,
                                       Tensor& index_output,
                                       const Tensor& segment_offsets,
                                       uint64_t num_segments,
                                       Stream& stream);

size_t queryArgMinReductionBytes(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
size_t queryArgMinReductionBytes(DataType input_dtype,
                                 std::optional<DataType> value_output_dtype,
                                 std::optional<DataType> index_output_dtype,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
void launchArgMinReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           Stream& stream);

size_t queryArgMaxReductionBytes(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
size_t queryArgMaxReductionBytes(DataType input_dtype,
                                 std::optional<DataType> value_output_dtype,
                                 std::optional<DataType> index_output_dtype,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
void launchArgMaxReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           Stream& stream);

// ARG-PLAN-2A direct-stage entry points. A null carried_index_input marks the cold root stage. Otherwise value_input
// is FP32 and carried_index_input contains the original-domain partial index selected by the preceding stage.
size_t queryComposedArgMinReductionStageBytes(const Tensor& value_input,
                                              const Tensor* carried_index_input,
                                              Tensor* value_output,
                                              Tensor* index_output,
                                              const CubReductionGeometry& geometry,
                                              uint64_t domain_stride,
                                              DataType carried_index_dtype,
                                              const Stream& stream);
size_t queryComposedArgMinReductionStageBytes(DataType value_input_dtype,
                                              bool has_carried_index_input,
                                              std::optional<DataType> value_output_dtype,
                                              std::optional<DataType> index_output_dtype,
                                              const CubReductionGeometry& geometry,
                                              uint64_t domain_stride,
                                              DataType carried_index_dtype,
                                              const Stream& stream);
void launchComposedArgMinReductionStage(const Tensor& temp_storage,
                                        size_t temp_storage_bytes,
                                        const Tensor& value_input,
                                        const Tensor* carried_index_input,
                                        Tensor* value_output,
                                        Tensor* index_output,
                                        const CubReductionGeometry& geometry,
                                        uint64_t domain_stride,
                                        DataType carried_index_dtype,
                                        Stream& stream);

size_t queryComposedArgMaxReductionStageBytes(const Tensor& value_input,
                                              const Tensor* carried_index_input,
                                              Tensor* value_output,
                                              Tensor* index_output,
                                              const CubReductionGeometry& geometry,
                                              uint64_t domain_stride,
                                              DataType carried_index_dtype,
                                              const Stream& stream);
size_t queryComposedArgMaxReductionStageBytes(DataType value_input_dtype,
                                              bool has_carried_index_input,
                                              std::optional<DataType> value_output_dtype,
                                              std::optional<DataType> index_output_dtype,
                                              const CubReductionGeometry& geometry,
                                              uint64_t domain_stride,
                                              DataType carried_index_dtype,
                                              const Stream& stream);
void launchComposedArgMaxReductionStage(const Tensor& temp_storage,
                                        size_t temp_storage_bytes,
                                        const Tensor& value_input,
                                        const Tensor* carried_index_input,
                                        Tensor* value_output,
                                        Tensor* index_output,
                                        const CubReductionGeometry& geometry,
                                        uint64_t domain_stride,
                                        DataType carried_index_dtype,
                                        Stream& stream);

/**
 * 1146-1148: retained FP8 R / KR VALUE pass over [shape.rows, shape.row_bytes] (Fp8KR/Fp8KRSegments.h) writing
 * `output` ([rows] elements, any FP8 Complete storage dtype; semantics finalize and output_scale apply once). Pack and
 * Group are one kernel; Split writes chunk partials to `scratch` (fp8KRScratchBytes(shape, false) bytes, 16-byte
 * aligned, no initialization) and a finishing kernel combines them in chunk order. Any input address works:
 * loads use the tensor's 4-byte word grid and may read up to 3 bytes on either side of a row (inside the allocation
 * and its trailing padding).
 */
/** 1158: the ranker site of a retained FP8 KR VALUE pass (Fp8KRSegments.h), shared by production and the sweep. */
[[nodiscard]] Fp8KR::Fp8KRRankerSite fp8KRValueRankerSite(const CubReductionStageSemantics& semantics,
                                                          DataType input_dtype);

/** 1147: semantics the retained FP8 KR VALUE kernels compile: SUM x {identity, |x|, x^2}, PRODUCT, MIN / MAX x any. */
[[nodiscard]] bool fp8KRValueSupports(const CubReductionStageSemantics& semantics);

void launchFp8KRValueSegments(const CubReductionStageSemantics& semantics,
                              const Tensor& input,
                              Tensor& output,
                              const Fp8KR::Fp8KRSegmentShape& shape,
                              void* scratch,
                              uint64_t scratch_bytes,
                              float output_scale,
                              Stream& stream,
                              Fp8KR::Fp8KRSplitStages stages = Fp8KR::Fp8KRSplitStages::Both);

/** 1179: the same retained R / KR VALUE pass on FP32 input (the FP8 Group / Pack / Block / Split kernels at one element
 *  per word). `shape` is in bytes (row_bytes = 4 * elements per row) and legal for element size 4
 *  (Fp8KR::retainedKRShapeLegalForElement: no vector Pack); the input must be 4-byte aligned. The FP32 KR
 *  selector picks the shape (plans/Thor_FP32_Dense_Reduction_Plan). */
void launchFp32KRValueSegments(const CubReductionStageSemantics& semantics,
                               const Tensor& input,
                               Tensor& output,
                               const Fp8KR::Fp8KRSegmentShape& shape,
                               void* scratch,
                               uint64_t scratch_bytes,
                               float output_scale,
                               Stream& stream,
                               Fp8KR::Fp8KRSplitStages stages = Fp8KR::Fp8KRSplitStages::Both);

/** 1181: the retained R / KR argmin/argmax pass on FP32 input (its own kernels: two-word candidates, a 32-bit ordered
 *  key and a 64-bit row-local index). `shape` is in bytes and legal for it (Fp8KR::fp32KRArgShapeLegal); the input is
 *  4-byte aligned; Split partials take fp8KRScratchBytes(shape, true) bytes. Index = row-local index * domain_stride.
 *  (plans/Thor_FP32_Dense_Reduction_Plan). */
void launchFp32KRArgSegments(const Tensor& input,
                             Tensor* value_output,
                             Tensor* index_output,
                             const Fp8KR::Fp8KRSegmentShape& shape,
                             void* scratch,
                             uint64_t scratch_bytes,
                             bool maximum,
                             uint64_t domain_stride,
                             Stream& stream,
                             Fp8KR::Fp8KRSplitStages stages = Fp8KR::Fp8KRSplitStages::Both);

/** 1214: wide Split scratch of the carried-index FP32 R / KR ARG pass (fp32ArgWideScratchBytes over [rows, chunks]);
 *  0 for the other families. */
[[nodiscard]] uint64_t fp32KRCarriedArgScratchBytes(const Fp8KR::Fp8KRSegmentShape& shape);

/** 1214: the carried-index (intermediate) FP32 R / KR ARG pass (phase 8): launchFp32KRArgSegments's
 *  families and shapes (Fp8KR::fp32KRArgShapeLegal) with a UINT32 / UINT64 carried index per element (the input's
 *  element count, naturally aligned). The index output holds the composed index carried + row-local position *
 *  row_stride. A Split pass writes wide partials to `scratch` (fp32KRCarriedArgScratchBytes, 256-byte aligned) and
 *  finishes them in the same call (unless PartialsOnly). */
void launchFp32KRCarriedArgSegments(const Tensor& input,
                                    const Tensor& carried_index,
                                    Tensor* value_output,
                                    Tensor* index_output,
                                    const Fp8KR::Fp8KRSegmentShape& shape,
                                    void* scratch,
                                    uint64_t scratch_bytes,
                                    bool maximum,
                                    uint64_t row_stride,
                                    Stream& stream,
                                    Fp8KR::Fp8KRSplitStages stages = Fp8KR::Fp8KRSplitStages::Both);

/** 1146-1148: retained FP8 R / KR argmin/argmax pass (same families; Split partials are 64-bit wide candidates in
 *  `scratch`, fp8KRScratchBytes(shape, true) bytes). Index = row-local index * domain_stride. */
void launchFp8KRArgSegments(const Tensor& input,
                            Tensor* value_output,
                            Tensor* index_output,
                            const Fp8KR::Fp8KRSegmentShape& shape,
                            void* scratch,
                            uint64_t scratch_bytes,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            Fp8KR::Fp8KRSplitStages stages = Fp8KR::Fp8KRSplitStages::Both);

}  // namespace ThorImplementation::CubReductionInternal
