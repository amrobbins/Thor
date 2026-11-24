#pragma once

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/TensorOperations/Reduction/KR/KRSegments.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StageSemantics.h"
#include "Utilities/TensorOperations/Reduction/Layout/ReductionGeometry.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation::ReductionInternal {

/**
 * Physical topology of one RK reduction stage.
 *
 * This is deliberately independent of ReductionStageRole/ReductionStageSemantics. A mathematically Final pass
 * may still use Staged topology and defer its finalizer to the independently planned continuation, while a
 * mathematically First pass may use Complete topology and write its direct FP32 stage output without finalizing the
 * overall reduction operation.
 */
enum class RKStageTopology : uint8_t {
    Complete = 0,
    Staged = 1,
};

struct Fp8AlignedKStagePlan {
    RKStageTopology topology = RKStageTopology::Staged;
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
struct KernelOccupancyInfo {
    int registers_per_thread = 0;
    size_t static_shared_bytes = 0;
    size_t dynamic_shared_bytes = 0;
    int max_active_blocks_per_sm = 0;
    int max_active_warps_per_sm = 0;
};

/**
 * Launches one FP8 AlignedK stage. Packet width, CTA width, and R shard depth are planner geometry, not distinct
 * reducer families. Warps own independent K outputs, so the stage requires no inter-warp reduction.
 *
 * Staged topology requires shards_per_output > 1 and emits FP32 [outer,shards,inner] partials. It intentionally does
 * not apply semantics.finalize or output_scale; those belong to the independently stamped continuation. Complete
 * topology requires shards_per_output == 1 and writes the supplied stage output directly, applying the semantic
 * finalizer, runtime output scale, and output dtype conversion for this mathematical pass.
 *
 * packet_bytes is the naturally aligned per-lane input transaction width; every global lane transaction remains
 * <=16 B.
 *
 * FP8 input only (FP16 / BF16 / FP32 use launchFp32AlignedKStage / launchF16AlignedKStage; any other input dtype throws
 * std::logic_error). The compiled AlignedK configurations (RK::Fp8::Fp8AlignedKPolicy): Staged P2/P8 x CTA32/64/128 and
 * Complete P2/CTA32, P8/CTA32 and P8/CTA64. Any other FP8 packet/CTA request is not compiled and throws
 * std::logic_error. Staged Identity SUM/PRODUCT/MIN/MAX share one operation-unified kernel per configuration; Complete
 * Identity uses joined P2/CTA32, in-branch P8/CTA32, and typed SUM/MEAN plus in-branch nonadditive P8/CTA64. ABS/Square
 * compositions use the typed packet kernel, minus the accepted pruned specializations.
 */
void launchFp8AlignedKStage(const ReductionStageSemantics& semantics,
                            const StridedSpan& input,
                            const StridedSpan& stage_output,
                            const ReductionGeometry& geometry,
                            const Fp8AlignedKStagePlan& plan,
                            float output_scale,
                            Stream& stream);

/** Query exact CUDA occupancy metadata for the FP8 AlignedK kernel selected by plan.
 * FP8 input only: the compiled AlignedK Identity configurations; anything else throws std::logic_error. */
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKIdentityOccupancy(DataType input_dtype,
                                                                    const Fp8AlignedKStagePlan& plan);

/** Exact operation/output-aware occupancy of an AlignedK Identity Complete kernel.
 * In P8/CTA64, SUM/MEAN uses typed additive and PRODUCT/MIN/MAX uses unified
 * in-branch; neither the packet nor the CTA axis implies a compiled R depth. */
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKCompleteOccupancy(
    DataType input_dtype, DataType output_dtype, ReductionStageCombine operation,
    const Fp8AlignedKStagePlan& plan);

/** Exact physical occupancy for both Identity and transformed AlignedK kernels. */
[[nodiscard]] KernelOccupancyInfo queryFp8AlignedKStageOccupancy(
    DataType input_dtype, DataType output_dtype, ReductionStageInputTransform transform,
    ReductionStageCombine operation, const Fp8AlignedKStagePlan& plan);

/** AlignedK on FP32 input. A lane owns one 8- or 16-byte packet (2 or 4 components; K a multiple of that), or one
 *  4-byte packet (one component, any K), and reduces its shard's rows in FP32; CTA 32 / 64 / 128.
 *  shards_per_output > 1 is Staged: unfinalized FP32 [outer, shards, K] partials at the balanced row split.
 *  shards_per_output == 1 is Complete: finalized, scaled, stored in the output dtype at the geometry's output strides.
 *  The input base is aligned to the packet. Semantics: SUM / MIN / MAX x {identity, |x|, x^2}, PRODUCT x identity. */
[[nodiscard]] bool fp32AlignedKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                        uint32_t packet_bytes, uint32_t block_threads, uint64_t shards_per_output);

void launchFp32AlignedKStage(const ReductionStageSemantics& semantics,
                             const StridedSpan& input,
                             const StridedSpan& stage_output,
                             const ReductionGeometry& geometry,
                             uint32_t packet_bytes,
                             uint32_t block_threads,
                             uint64_t shards_per_output,
                             float output_scale,
                             Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32AlignedKOccupancy(uint32_t packet_bytes, uint32_t block_threads,
                                                             bool staged);

/** The same AlignedK kernel on FP16 / BF16 input. A lane owns one 2-, 4-, 8- or 16-byte packet (1 / 2 / 4 / 8
 *  components; K a multiple of that), widens each component to FP32 and reduces exactly as on FP32 input; CTA
 *  32 / 64 / 128, Staged FP32 partials or Complete output, the input base aligned to the packet,
 *  launchFp32AlignedKStage's semantics. */
[[nodiscard]] bool f16AlignedKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                       uint32_t packet_bytes, uint32_t block_threads, uint64_t shards_per_output);

void launchF16AlignedKStage(const ReductionStageSemantics& semantics,
                            const StridedSpan& input,
                            const StridedSpan& stage_output,
                            const ReductionGeometry& geometry,
                            uint32_t packet_bytes,
                            uint32_t block_threads,
                            uint64_t shards_per_output,
                            float output_scale,
                            Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryF16AlignedKOccupancy(DataType input_dtype, uint32_t packet_bytes,
                                                            uint32_t block_threads, bool staged);

/** FP32 RK ARG. Candidates are Fp32ArgKey.cuh's (key, index << 1 | negative zero): NaNs propagate, the lowest index
 *  wins ties (-0 / +0 included).
 *  The AlignedK twin: alignedKValueKernel's ownership (8- or 16-byte packets, CTA 32 / 64 / 128, balanced shard rows;
 *  no 4-byte packet: fp32AlignedKArgSupports rejects it).
 *  One shard writes the requested outputs; more write packed 64-bit partials [outer, shards, K] to `packed_scratch`
 *  (fp32ArgPackedScratchBytes) and finish them in the same call. value_output: any float storage (or null);
 *  index_output: UINT32 / UINT64 (or null), index = row * domain_stride. Rows per shard <= 2^31 - 1. */
[[nodiscard]] bool fp32AlignedKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint32_t packet_bytes, uint32_t block_threads, uint64_t shards_per_output);

[[nodiscard]] uint64_t fp32ArgPackedScratchBytes(uint64_t outer_size, uint64_t shards, uint64_t inner_size);

void launchFp32AlignedKArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           uint32_t packet_bytes,
                           uint32_t block_threads,
                           uint64_t shards_per_output,
                           uint64_t* packed_scratch,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32AlignedKArgOccupancy(uint32_t packet_bytes, uint32_t block_threads,
                                                                bool staged);

/** The same AlignedK ARG twin on FP16 / BF16 input: each element widens to FP32 and takes the FP32 key (exact and
 *  order-preserving), so candidates, packed partials, the continuation and the outputs are launchFp32AlignedKArg's.
 *  4-, 8- or 16-byte packets (2, 4 or 8 components; K a multiple), CTA 32 / 64 / 128; the input base aligned to the
 *  packet. */
[[nodiscard]] bool f16AlignedKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                          uint32_t packet_bytes, uint32_t block_threads, uint64_t shards_per_output);

void launchF16AlignedKArg(const StridedSpan& input,
                          const StridedSpan* value_output,
                          const StridedSpan* index_output,
                          const ReductionGeometry& geometry,
                          uint32_t packet_bytes,
                          uint32_t block_threads,
                          uint64_t shards_per_output,
                          uint64_t* packed_scratch,
                          bool maximum,
                          uint64_t domain_stride,
                          Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryF16AlignedKArgOccupancy(DataType input_dtype, uint32_t packet_bytes,
                                                               uint32_t block_threads, bool staged);

/** Reduces packed FP32 ARG partials [outer, shards, K] over shards and writes the requested outputs. Shard s begins at
 *  s * uniform_rows_per_shard, or with 0 at the balanced split of R into `shards`. FP8's per-output kernel, or its
 *  shard-split kernel for few outputs with many shards (fp8ArgPackedContinuationSplitsShards). */
void launchFp32ArgPackedContinuation(const uint64_t* packed_partials,
                                     const StridedSpan* value_output,
                                     const StridedSpan* index_output,
                                     uint64_t outer_size,
                                     uint64_t reduction_size,
                                     uint64_t inner_size,
                                     uint64_t shards,
                                     uint64_t uniform_rows_per_shard,
                                     bool maximum,
                                     uint64_t domain_stride,
                                     Stream& stream);

/** Carried-index (intermediate) FP32 ARG. The input is FP32 values [outer, R, K] with a carried index (UINT32 / UINT64,
 *  the same shape) from earlier passes; an element's composed index is carried + row * row_stride (row_stride: the
 *  current run's original-domain stride, 0 for an intermediate run). Candidates compare (key, composed index) in full;
 *  the index output is the composed index.
 *  Staged passes write wide partials (a uint32 key and a uint64 tail per partial, fp32ArgWideScratchBytes, 256-byte
 *  aligned scratch) and finish them with launchFp32ArgWideContinuation in the same call. */
[[nodiscard]] uint64_t fp32ArgWideScratchBytes(uint64_t outer_size, uint64_t shards, uint64_t inner_size);

void launchFp32ArgWideContinuation(void* wide_scratch,
                                   const StridedSpan* value_output,
                                   const StridedSpan* index_output,
                                   uint64_t outer_size,
                                   uint64_t shards,
                                   uint64_t inner_size,
                                   bool maximum,
                                   Stream& stream);

/** The carried-index AlignedK ARG twin: launchFp32AlignedKArg's configurations (fp32AlignedKSupports), the carried
 *  index aligned like its packets (at most 16 bytes). */
void launchFp32AlignedKCarriedArg(const StridedSpan& input,
                                  const StridedSpan& carried_index,
                                  const StridedSpan* value_output,
                                  const StridedSpan* index_output,
                                  const ReductionGeometry& geometry,
                                  uint32_t packet_bytes,
                                  uint32_t block_threads,
                                  uint64_t shards_per_output,
                                  void* wide_scratch,
                                  bool maximum,
                                  uint64_t row_stride,
                                  Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32AlignedKCarriedArgOccupancy(uint32_t packet_bytes,
                                                                       uint32_t block_threads, bool staged,
                                                                       DataType index_dtype);

/**
 * The RuntimeK rotated first stage: p16 packets, exact (512) or prefix-reserved (497) runtime K tiles, and only the
 * compiled Fp8RuntimeKPolicy::rotated_row_depths. Emits FP32 [outer,shards,K] partials with no K-tile merge; finalizer,
 * scale and output conversion belong to the independently planned continuation.
 * Any other row depth is not compiled and fails with std::logic_error (occupancy: absent candidate).
 */
void launchFp8RuntimeKRotatedFirstStage(
    const StridedSpan& input,
    const StridedSpan& fp32_partials,
    const ReductionGeometry& geometry,
    bool exact_tile,
    uint64_t rows_per_shard,
    uint64_t shards_per_output,
    Stream& stream,
    ReductionStageInputTransform transform = ReductionStageInputTransform::Identity,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

[[nodiscard]] KernelOccupancyInfo queryFp8RuntimeKRotatedOccupancy(
    DataType input_dtype, bool exact_tile, uint64_t rows_per_shard,
    ReductionStageInputTransform transform = ReductionStageInputTransform::Identity,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The RuntimeK rotated first stage on FP32 input, PrefixReserved only: p16 packets (4 components per lane), 4
 *  cooperative warps (the fewest whose revisit stride, 4 rows of K floats, is a whole number of packets), a logical K
 *  tile of 125 components (128 less a 0..3 prefix), and FP8's row depths (Fp8RuntimeKPolicy::rotated_row_depths).
 *  shards_per_output = ceil(R / rows) > 1. Emits unfinalized FP32 [outer, shards, K] partials; every
 *  combine x {identity, |x|, x^2}; the input is 4-byte aligned. */
inline constexpr uint32_t FP32_RUNTIME_K_ROTATED_TILE_COMPONENTS = 125;

[[nodiscard]] bool fp32RuntimeKRotatedSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                               uint64_t rows_per_shard);

void launchFp32RuntimeKRotatedFirstStage(const StridedSpan& input,
                                         const StridedSpan& fp32_partials,
                                         const ReductionGeometry& geometry,
                                         uint64_t rows_per_shard,
                                         uint64_t shards_per_output,
                                         Stream& stream,
                                         ReductionStageInputTransform transform =
                                             ReductionStageInputTransform::Identity,
                                         ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The FP32 RK ARG rotated twin: FP32 VALUE rotated's layout (16-byte packets, 4 warps, the 125-component tile; any
 *  4-byte aligned input) with one ARG candidate per component and any shard depth up to 2^31 - 1 rows.
 *  rows_per_shard >= R is one Complete pass writing the requested outputs; otherwise packed partials go to
 *  `packed_scratch` (fp32ArgPackedScratchBytes with ceil(R / rows) shards) and launchFp32ArgPackedContinuation finishes
 *  them in the same call. Outputs as launchFp32AlignedKArg's. */
[[nodiscard]] bool fp32RuntimeKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint64_t rows_per_shard);

void launchFp32RuntimeKArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           uint64_t rows_per_shard,
                           uint64_t* packed_scratch,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32RuntimeKArgOccupancy(bool staged);

/** The rotated ARG twin on FP16 / BF16 input: the 2-byte rotated layout (16-byte packets of 8 components, 8 cooperative
 *  warps, the 249-component tile; any 2-byte aligned input), each element widened to FP32 and given the FP32 key, so
 *  depths, scratch, the continuation and the outputs are launchFp32RuntimeKArg's. */
[[nodiscard]] bool f16RuntimeKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                          uint64_t rows_per_shard);

void launchF16RuntimeKArg(const StridedSpan& input,
                          const StridedSpan* value_output,
                          const StridedSpan* index_output,
                          const ReductionGeometry& geometry,
                          uint64_t rows_per_shard,
                          uint64_t* packed_scratch,
                          bool maximum,
                          uint64_t domain_stride,
                          Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryF16RuntimeKArgOccupancy(DataType input_dtype, bool staged);

/** The carried-index rotated ARG twin (inputs, outputs and scratch as launchFp32AlignedKCarriedArg's):
 *  launchFp32RuntimeKArg's layout and any depth up to 2^32 - 1 rows (depth >= R: one Complete pass). Both the input and
 *  the carried index are 16-byte aligned. */
[[nodiscard]] bool fp32RuntimeKCarriedArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                                  uint64_t rows_per_shard);

void launchFp32RuntimeKCarriedArg(const StridedSpan& input,
                                  const StridedSpan& carried_index,
                                  const StridedSpan* value_output,
                                  const StridedSpan* index_output,
                                  const ReductionGeometry& geometry,
                                  uint64_t rows_per_shard,
                                  void* wide_scratch,
                                  bool maximum,
                                  uint64_t row_stride,
                                  Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32RuntimeKCarriedArgOccupancy(bool staged, DataType index_dtype);

[[nodiscard]] KernelOccupancyInfo queryFp32RuntimeKRotatedOccupancy(
    uint64_t rows_per_shard,
    ReductionStageInputTransform transform = ReductionStageInputTransform::Identity,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The RuntimeK rotated first stage on FP16 / BF16 input, PrefixReserved: the 2-byte layout (p16 packets of 8
 *  components, 8 cooperative warps, a logical K tile of 249 components, 256 less a 0..7 prefix) at FP8's row depths.
 *  shards_per_output = ceil(R / rows) > 1. Emits unfinalized FP32 [outer, shards, K] partials; every
 *  combine x {identity, |x|, x^2}; the input is 2-byte aligned. */
inline constexpr uint32_t F16_RUNTIME_K_ROTATED_TILE_COMPONENTS = 249;

[[nodiscard]] bool f16RuntimeKRotatedSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                              uint64_t rows_per_shard);

void launchF16RuntimeKRotatedFirstStage(const StridedSpan& input,
                                        const StridedSpan& fp32_partials,
                                        const ReductionGeometry& geometry,
                                        uint64_t rows_per_shard,
                                        uint64_t shards_per_output,
                                        Stream& stream,
                                        ReductionStageInputTransform transform =
                                            ReductionStageInputTransform::Identity,
                                        ReductionStageCombine combine = ReductionStageCombine::Sum);

[[nodiscard]] KernelOccupancyInfo queryF16RuntimeKRotatedOccupancy(
    DataType input_dtype, uint64_t rows_per_shard,
    ReductionStageInputTransform transform = ReductionStageInputTransform::Identity,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The DeepRSmallK FP8 P16 first stage, with typed SUM/PRODUCT/MIN/MAX, typed Identity/ABS/Square, typed E4M3/E5M2, and
 * the selected five depths.
 * Emits UNFINALIZED FP32 [outer, ceil(R/rows), K] partials. The caller must
 * independently plan an FP32 continuation with the same combine semantics.
 */
void launchFp8DeepRSmallKFirstStage(
    const ReductionStageSemantics& semantics, const StridedSpan& input, const StridedSpan& fp32_partials,
    const ReductionGeometry& geometry, uint64_t rows_per_shard, uint64_t shards_per_output,
    Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp8DeepRSmallKOccupancy(
    DataType input_dtype, uint64_t rows_per_shard, uint64_t inner_size,
    const ReductionStageSemantics& semantics);

/** The DeepRSmallK first stage on FP32 input.
 *  FP8's kernels at 4-byte elements: K <= 125 (a P16 window of 128 FP32, less a worst-case prefix), ownerships
 *  TinyK (K <= 4, <= 8) and one / two / four components per lane, FP8's five depths. Emits unfinalized FP32
 *  [outer, ceil(R / rows), K] partials. The input is 16-byte aligned. fp32DeepRSmallKSupports is the legality rule. */
[[nodiscard]] bool fp32DeepRSmallKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           uint64_t rows_per_shard);

void launchFp32DeepRSmallKFirstStage(
    const ReductionStageSemantics& semantics, const StridedSpan& input, const StridedSpan& fp32_partials,
    const ReductionGeometry& geometry, uint64_t rows_per_shard, uint64_t shards_per_output,
    Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32DeepRSmallKOccupancy(
    uint64_t rows_per_shard, uint64_t inner_size, const ReductionStageSemantics& semantics);

/** The DeepRSmallK first stage on FP16 / BF16 input. FP8's kernels and its seven ownerships at 2-byte elements:
 *  K <= 249 (a P16 window of 256, less a worst-case prefix), FP8's five depths. Emits unfinalized FP32 [outer,
 *  ceil(R / rows), K] partials. The input is 16-byte aligned. f16DeepRSmallKSupports is the legality rule. */
[[nodiscard]] bool f16DeepRSmallKSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                          uint64_t rows_per_shard);

void launchF16DeepRSmallKFirstStage(
    const ReductionStageSemantics& semantics, const StridedSpan& input, const StridedSpan& fp32_partials,
    const ReductionGeometry& geometry, uint64_t rows_per_shard, uint64_t shards_per_output,
    Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryF16DeepRSmallKOccupancy(
    DataType input_dtype, uint64_t rows_per_shard, uint64_t inner_size, const ReductionStageSemantics& semantics);





/** The OneWindow Complete kernel. O1 supports W1/W2/W4/W8, while O2/O4/O8 partition a warp among outputs. Applies the
 * runtime finalizer/scale and converts to the final output dtype. The selector's entry, by variant (outputs per warp,
 * warps per output, packet bytes).
 * It runs the bucketed unit for (outputs per warp, packet bytes, K's bucket): only packet_bytes 16 has units, and a
 * request without one throws. */
void launchFp8OneWindowForVariant(
    const StridedSpan& input,
    const StridedSpan& output,
    const ReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream,
    const ReductionStageSemantics& semantics = {},
    float output_scale = 1.0f);

[[nodiscard]] KernelOccupancyInfo queryFp8OneWindowOccupancyForVariant(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The OneWindow kernel by unit (the selector reaches it through launchFp8OneWindowForVariant). A unit is one compiled
 * kernel shape: outputs per warp (1/2/4/8), packet bytes (16), components-per-lane bucket (1/2/4/8/16) and group loop
 * form.
 * fp8OneWindowUnits(K) lists the compiled units whose bucket holds K. warps_per_output is 1/2/4/8 for O1
 * and 1 otherwise. Output must hold outer x K elements (any dims).
 * The VALUE launchers (FP8, FP32, FP16 / BF16) read any element-aligned input base. Off a 16-byte boundary every row
 * group carries a prefix, so K is at most the window less the worst prefix (RK::oneWindowUnalignedCapacity) and the
 * input at most 2^32 - 16 bytes; the launch throws otherwise. */
struct OneWindowUnit {
    uint32_t outputs_per_warp;
    uint32_t packet_bytes;
    uint32_t max_components_per_lane;
    bool load_ahead;
    constexpr bool operator==(const OneWindowUnit&) const = default;
};

[[nodiscard]] std::vector<OneWindowUnit> fp8OneWindowUnits(uint64_t inner_size);

void launchFp8OneWindow(const StridedSpan& input,
                        const StridedSpan& output,
                        const ReductionGeometry& geometry,
                        const OneWindowUnit& unit,
                        uint32_t warps_per_output,
                        Stream& stream,
                        const ReductionStageSemantics& semantics = {},
                        float output_scale = 1.0f);

[[nodiscard]] KernelOccupancyInfo queryFp8OneWindowOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    const OneWindowUnit& unit,
    uint32_t warps_per_output,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The OneWindow kernel on FP32 input. The units are FP8's whose components-per-lane bucket FP32 reaches (<= 4: a
 *  16-byte lane packet holds 4 FP32), so K <= 125 / 61 / 29 / 13 at O1 / O2 / O4 / O8. fp32OneWindowUnits(K) lists the
 *  units whose bucket holds K. The input is at most 2^32 - 1 bytes (any 4-byte aligned base, see above); MIN / MAX fold
 *  with the FP32 masks. */
[[nodiscard]] std::vector<OneWindowUnit> fp32OneWindowUnits(uint64_t inner_size);

void launchFp32OneWindow(const StridedSpan& input,
                         const StridedSpan& output,
                         const ReductionGeometry& geometry,
                         const OneWindowUnit& unit,
                         uint32_t warps_per_output,
                         Stream& stream,
                         const ReductionStageSemantics& semantics = {},
                         float output_scale = 1.0f);

[[nodiscard]] KernelOccupancyInfo queryFp32OneWindowOccupancy(
    uint64_t inner_size,
    const OneWindowUnit& unit,
    uint32_t warps_per_output,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The OneWindow kernel on FP16 / BF16 input. The units are FP8's whose bucket the 2-byte formats reach (<= 8: a
 *  16-byte lane packet holds 8 elements; 14 units), so K <= 249 / 121 / 57 / 25 at O1 / O2 / O4 / O8, and the 8-element
 *  multiples 256 / 128 / 64 / 32 (a 16-byte aligned base only). The input is at most 2^32 - 1 bytes (any 2-byte aligned
 *  base, see above); MIN / MAX fold with the 2-byte masks (fp16CompleteExtremumFold). */
[[nodiscard]] std::vector<OneWindowUnit> f16OneWindowUnits(uint64_t inner_size);

void launchF16OneWindow(const StridedSpan& input,
                        const StridedSpan& output,
                        const ReductionGeometry& geometry,
                        const OneWindowUnit& unit,
                        uint32_t warps_per_output,
                        Stream& stream,
                        const ReductionStageSemantics& semantics = {},
                        float output_scale = 1.0f);

[[nodiscard]] KernelOccupancyInfo queryF16OneWindowOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    const OneWindowUnit& unit,
    uint32_t warps_per_output,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The FP32 RK ARG DeepRSmallK twins: FP32 VALUE DeepRSmallK's kernels and ownership (K <= 125: TinyK at K <= 4 / 8,
 *  then one / two / four components per lane; a compiled depth with R > depth; a 16-byte aligned input) with one ARG
 *  candidate per component. Packed partials [outer, ceil(R / depth), K] go to `packed_scratch`
 *  (fp32ArgPackedScratchBytes) and launchFp32ArgPackedContinuation finishes them in the same call.
 *  Outputs as launchFp32AlignedKArg's. */
[[nodiscard]] bool fp32DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                              uint64_t rows_per_shard);

void launchFp32DeepRSmallKArg(const StridedSpan& input,
                              const StridedSpan* value_output,
                              const StridedSpan* index_output,
                              const ReductionGeometry& geometry,
                              uint64_t rows_per_shard,
                              uint64_t* packed_scratch,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32DeepRSmallKArgOccupancy(uint64_t inner_size);

/** The DeepRSmallK ARG twins on FP16 / BF16 input: launchF16DeepRSmallKFirstStage's depths and windows (K <= 249, a
 *  16-byte aligned input; TinyK at K <= 4 / 8, then 1 / 2 / 4 / 8 components per lane), each element widened to FP32
 *  and given the FP32 key; packed partials, the continuation and the outputs as launchFp32DeepRSmallKArg's. */
[[nodiscard]] bool f16DeepRSmallKArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                             uint64_t rows_per_shard);

void launchF16DeepRSmallKArg(const StridedSpan& input,
                             const StridedSpan* value_output,
                             const StridedSpan* index_output,
                             const ReductionGeometry& geometry,
                             uint64_t rows_per_shard,
                             uint64_t* packed_scratch,
                             bool maximum,
                             uint64_t domain_stride,
                             Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryF16DeepRSmallKArgOccupancy(DataType input_dtype, uint64_t inner_size);

/** The carried-index DeepRSmallK ARG twins (inputs and outputs as launchFp32AlignedKCarriedArg's):
 *  launchFp32DeepRSmallKArg's kernels, ownership and depths (fp32DeepRSmallKSupports: a compiled depth with R > depth),
 *  the input and the carried index 16-byte aligned. Wide partials go to `wide_scratch` (fp32ArgWideScratchBytes with
 *  ceil(R / depth) shards, 256-byte aligned) and launchFp32ArgWideContinuation finishes them in the same call. */
void launchFp32DeepRSmallKCarriedArg(const StridedSpan& input,
                                     const StridedSpan& carried_index,
                                     const StridedSpan* value_output,
                                     const StridedSpan* index_output,
                                     const ReductionGeometry& geometry,
                                     uint64_t rows_per_shard,
                                     void* wide_scratch,
                                     bool maximum,
                                     uint64_t row_stride,
                                     Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32DeepRSmallKCarriedArgOccupancy(uint64_t inner_size,
                                                                          DataType index_dtype);

/** The FP32 RK ARG OneWindow twin: launchFp32OneWindow's units and ownership (O1 W1/W2/W4/W8, O2/O4/O8 W1;
 *  K <= 125 / 61 / 29 / 13) with one ARG candidate per owned component; Complete only, R <= 2^31 - 1, input at most
 *  2^32 - 1 bytes; any 4-byte aligned base, as the VALUE launcher (off a 16-byte boundary K is at most
 *  RK::oneWindowUnalignedCapacity and the input 2^32 - 16 bytes; the launch throws otherwise). Outputs as
 *  launchFp32AlignedKArg's. */
[[nodiscard]] bool fp32OneWindowArgSupports(uint64_t outer_size,
                                            uint64_t reduction_size,
                                            uint64_t inner_size,
                                            const OneWindowUnit& unit,
                                            uint32_t warps_per_output);

void launchFp32OneWindowArg(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const ReductionGeometry& geometry,
                            const OneWindowUnit& unit,
                            uint32_t warps_per_output,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32OneWindowArgOccupancy(uint64_t inner_size,
                                                                 const OneWindowUnit& unit,
                                                                 uint32_t warps_per_output);

/** The OneWindow ARG twin on FP16 / BF16 input: the 2-byte OneWindow units (f16OneWindowUnits), each element widened to
 *  FP32 and given the FP32 key; Complete, R <= 2^31 - 1; any 2-byte aligned base, as launchFp32OneWindowArg; outputs as
 *  launchFp32OneWindowArg's. */
[[nodiscard]] bool f16OneWindowArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                           const OneWindowUnit& unit, uint32_t warps_per_output);

void launchF16OneWindowArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           const OneWindowUnit& unit,
                           uint32_t warps_per_output,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryF16OneWindowArgOccupancy(DataType input_dtype, uint64_t inner_size,
                                                                const OneWindowUnit& unit,
                                                                uint32_t warps_per_output);

/** The carried-index OneWindow ARG twin (inputs and outputs as launchFp32AlignedKCarriedArg's):
 *  launchFp32OneWindowArg's units and ownership; Complete only, R <= 2^32 - 1, at most 2^32 - 1 value bytes, the input
 *  and the carried index 16-byte aligned. */
[[nodiscard]] bool fp32OneWindowCarriedArgSupports(uint64_t outer_size,
                                                   uint64_t reduction_size,
                                                   uint64_t inner_size,
                                                   const OneWindowUnit& unit,
                                                   uint32_t warps_per_output);

void launchFp32OneWindowCarriedArg(const StridedSpan& input,
                                   const StridedSpan& carried_index,
                                   const StridedSpan* value_output,
                                   const StridedSpan* index_output,
                                   const ReductionGeometry& geometry,
                                   const OneWindowUnit& unit,
                                   uint32_t warps_per_output,
                                   bool maximum,
                                   uint64_t row_stride,
                                   Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32OneWindowCarriedArgOccupancy(uint64_t inner_size,
                                                                        const OneWindowUnit& unit,
                                                                        uint32_t warps_per_output,
                                                                        DataType index_dtype);

/** The FP8 OneWindow argmin/argmax twin of a OneWindow unit (one kernel per unit; argmin/argmax and E4M3/E5M2 are
 * runtime). Complete only: writes the winning value (FP8/FP16/BF16/FP32 storage, optional) and the index
 * row * domain_stride (UINT32/UINT64, optional) for each of outer x K outputs. NaN propagates; the lowest row wins
 * ties, -0 ties +0. fp8OneWindowArgSupports() is the legality rule shared by selection and launch (R < 2^23, UINT32
 * element indexing, a compiled unit whose bucket holds K). The launch reads any byte address; off a 16-byte boundary K
 * is at most RK::oneWindowUnalignedCapacity and the input 2^32 - 16 bytes (it throws otherwise; selection:
 * retainedFp8ArgPassInputAlignment). */
[[nodiscard]] bool fp8OneWindowArgSupports(uint64_t outer_size,
                                           uint64_t reduction_size,
                                           uint64_t inner_size,
                                           const OneWindowUnit& unit,
                                           uint32_t warps_per_output);

/** The FP8 RuntimeK PacketComplete argmin/argmax twin: VALUE's persistent [outer, K tile] work items with 1/2/4/8 warps
 * per item over R, at most 256 components per tile (8 per lane; fp8PacketCompleteArgTileComponents keeps VALUE's tile
 * up to 256 unless a 256-component tile needs fewer key words per row). Complete: any supported value storage,
 * UINT32/UINT64 index. Any byte address, as the VALUE kernel (the row prefixes follow the address). */
[[nodiscard]] uint32_t fp8PacketCompleteArgTileComponents(uint64_t inner_size, uint32_t value_tile_components);
[[nodiscard]] bool fp8PacketCompleteArgSupports(uint64_t outer_size,
                                                uint64_t reduction_size,
                                                uint64_t inner_size,
                                                uint32_t warps_per_output,
                                                uint32_t tile_components);
void launchFp8PacketCompleteArg(const StridedSpan& input,
                                const StridedSpan* value_output,
                                const StridedSpan* index_output,
                                const ReductionGeometry& geometry,
                                uint32_t warps_per_output,
                                uint32_t tile_components,
                                bool maximum,
                                uint64_t domain_stride,
                                Stream& stream);

/** The FP8 AlignedK argmin/argmax twin: one thread owns one aligned 2- or 8-byte packet of components (K a multiple of
 * the packet; CTA 32/64/128 threads is a launch operand) over its shard's balanced rows. One shard is Complete (any
 * supported value storage, UINT32/UINT64 index); several write FP32 value + carried index partials. */
[[nodiscard]] bool fp8AlignedKArgSupports(uint64_t outer_size,
                                          uint64_t reduction_size,
                                          uint64_t inner_size,
                                          uint32_t packet_bytes,
                                          uint32_t cta_threads,
                                          uint64_t shards);
void launchFp8AlignedKArgPass(const StridedSpan& input,
                              const StridedSpan* value_output,
                              const StridedSpan* index_output,
                              const ReductionGeometry& geometry,
                              uint32_t packet_bytes,
                              uint32_t cta_threads,
                              uint64_t shards,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream,
                              uint32_t* packed_partials = nullptr);

/** Packed staged partials. A staged twin with `packed_partials` writes one shard-relative lane candidate per partial
 * (key << 24 | shard row << 1 | negative zero, 4 bytes) instead of FP32 value + carried index (8 bytes);
 * launchFp8ArgPackedContinuation reduces them over shards into the requested outputs. Shard s begins at
 * s * uniform_rows_per_shard, or at VALUE's balanced split when uniform_rows_per_shard is 0 (AlignedK). */
void launchFp8DeepRSmallKArgFirstStagePacked(const StridedSpan& input,
                                             uint32_t* packed_partials,
                                             const ReductionGeometry& geometry,
                                             uint64_t rows_per_shard,
                                             uint64_t shards,
                                             bool maximum,
                                             Stream& stream);
/** The packed continuation splits each output's shards across a block (fp8ArgPackedContinuationSplitKernel) when there
 * are many shards and too few outputs for one thread per output to hide the shard loop's latency.
 * Measured against the per-output kernel: at 16-31 shards the split wins below 16 Ki outputs (0.5-0.7x) and loses from
 * there (1.2x at 16-64 Ki, 1.8x at 64-128 Ki); at >= 32 shards it wins below 64 Ki outputs (0.5-0.9x) and loses above
 * (1.5-1.7x). A layout must also fill a warp (columns x groups >= 32): K = 2 at 16 shards left 24 of 32 lanes idle. */
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
                                    const StridedSpan* value_output,
                                    const StridedSpan* index_output,
                                    DataType input_dtype,
                                    uint64_t outer_size,
                                    uint64_t reduction_size,
                                    uint64_t inner_size,
                                    uint64_t shards,
                                    uint64_t uniform_rows_per_shard,
                                    bool maximum,
                                    uint64_t domain_stride,
                                    Stream& stream);

/** The FP8 RuntimeK rotated argmin/argmax twin, one rotated kernel (8-byte packets only; the logical K tile is
 * fp8RuntimeKArgTileComponents) at a runtime shard depth. One shard (rows_per_shard >= R) is a Complete pass writing
 * the requested outputs (value in any supported storage, index UINT32/UINT64); several shards write FP32
 * value + carried index partials [outer, ceil(R / rows), K]. index = row * domain_stride. */
[[nodiscard]] uint32_t fp8RuntimeKArgTileComponents(uint32_t packet_bytes);
[[nodiscard]] bool fp8RuntimeKArgSupports(uint64_t outer_size,
                                          uint64_t reduction_size,
                                          uint64_t inner_size,
                                          uint64_t rows_per_shard,
                                          uint32_t packet_bytes);
void launchFp8RuntimeKArgPass(const StridedSpan& input,
                              const StridedSpan* value_output,
                              const StridedSpan* index_output,
                              const ReductionGeometry& geometry,
                              uint64_t rows_per_shard,
                              uint32_t packet_bytes,
                              bool maximum,
                              uint64_t domain_stride,
                              Stream& stream,
                              uint32_t* packed_partials = nullptr);

/** The FP8 DeepRSmallK argmin/argmax twin: staged first stage at a compiled depth (Fp8DeepRSmallKPolicy) writing FP32
 * value + carried index (UINT32/UINT64) partials [outer, ceil(R / rows), K]; index = row * domain_stride. */
[[nodiscard]] bool fp8DeepRSmallKArgSupports(uint64_t outer_size,
                                             uint64_t reduction_size,
                                             uint64_t inner_size,
                                             uint64_t rows_per_shard);

void launchFp8DeepRSmallKArgFirstStage(const StridedSpan& input,
                                       const StridedSpan& value_partials,
                                       const StridedSpan& index_partials,
                                       const ReductionGeometry& geometry,
                                       uint64_t rows_per_shard,
                                       uint64_t shards,
                                       bool maximum,
                                       uint64_t domain_stride,
                                       Stream& stream);

void launchFp8OneWindowArg(const StridedSpan& input,
                           const StridedSpan* value_output,
                           const StridedSpan* index_output,
                           const ReductionGeometry& geometry,
                           const OneWindowUnit& unit,
                           uint32_t warps_per_output,
                           bool maximum,
                           uint64_t domain_stride,
                           Stream& stream);


/** The RuntimeK PacketP16 Complete kernel. K tiles and warp ownership are runtime geometry; finalizer, scale, and
 * output conversion occur on the completed result. */
void launchFp8PacketComplete(
    const StridedSpan& input,
    const StridedSpan& output,
    const ReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t tile_components,
    Stream& stream,
    const ReductionStageSemantics& semantics = {},
    float output_scale = 1.0f);

[[nodiscard]] KernelOccupancyInfo queryFp8PacketCompleteOccupancy(
    DataType input_dtype,
    ReductionStageInputTransform transform = ReductionStageInputTransform::Identity,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** RuntimeK PacketComplete on FP32 input. The FP8 kernel's topology at 4 components per p16 packet: persistent [outer,
 *  K tile] work items, W1/W2/W4/W8 warps per item over R, a K tile of at most 124 components (any per-row prefix of
 *  0..3 still fits the 32-packet window). Writes the finalized, scaled output. The input base is 16-byte aligned. Every
 *  combine x {identity, |x|, x^2}. */
inline constexpr uint32_t FP32_PACKET_COMPLETE_MAX_TILE_COMPONENTS = 124;

[[nodiscard]] bool fp32PacketCompleteSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                              uint32_t warps_per_output, uint32_t tile_components);

void launchFp32PacketComplete(const StridedSpan& input,
                              const StridedSpan& output,
                              const ReductionGeometry& geometry,
                              uint32_t warps_per_output,
                              uint32_t tile_components,
                              Stream& stream,
                              const ReductionStageSemantics& semantics = {},
                              float output_scale = 1.0f);

/** The FP32 RK ARG PacketComplete twin: launchFp32PacketComplete's topology (W1/W2/W4/W8 warps per [outer, K tile]
 *  item, tiles of 1..124 components; any 4-byte aligned base, as the VALUE kernel) with one ARG candidate per
 *  component; Complete only, R <= 2^31 - 1. Outputs as launchFp32AlignedKArg's. */
[[nodiscard]] bool fp32PacketCompleteArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                                 uint32_t warps_per_output, uint32_t tile_components);

void launchFp32PacketCompleteArg(const StridedSpan& input,
                                 const StridedSpan* value_output,
                                 const StridedSpan* index_output,
                                 const ReductionGeometry& geometry,
                                 uint32_t warps_per_output,
                                 uint32_t tile_components,
                                 bool maximum,
                                 uint64_t domain_stride,
                                 Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryFp32PacketCompleteArgOccupancy();

/** The PacketComplete ARG twin on FP16 / BF16 input: launchF16PacketComplete's ownership (tiles of 1..248 components;
 *  any 2-byte aligned base), each element widened to FP32 and given the FP32 key; Complete only, R <= 2^31 - 1; outputs
 *  as launchFp32PacketCompleteArg's. */
[[nodiscard]] bool f16PacketCompleteArgSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                                uint32_t warps_per_output, uint32_t tile_components);

void launchF16PacketCompleteArg(const StridedSpan& input,
                                const StridedSpan* value_output,
                                const StridedSpan* index_output,
                                const ReductionGeometry& geometry,
                                uint32_t warps_per_output,
                                uint32_t tile_components,
                                bool maximum,
                                uint64_t domain_stride,
                                Stream& stream);

[[nodiscard]] KernelOccupancyInfo queryF16PacketCompleteArgOccupancy(DataType input_dtype);

[[nodiscard]] KernelOccupancyInfo queryFp32PacketCompleteOccupancy(
    ReductionStageInputTransform transform = ReductionStageInputTransform::Identity,
    ReductionStageCombine combine = ReductionStageCombine::Sum);

/** The same PacketComplete kernel on FP16 / BF16 input: 8 components per p16 packet, a K tile of at most 248 components
 *  (31 packets, so any per-row prefix of 0..7 elements still fits the 32-packet window), each widened to FP32.
 *  launchFp32PacketComplete's ownership, output, alignment and semantics. */
inline constexpr uint32_t F16_PACKET_COMPLETE_MAX_TILE_COMPONENTS = 248;

[[nodiscard]] bool f16PacketCompleteSupports(uint64_t outer_size, uint64_t reduction_size, uint64_t inner_size,
                                             uint32_t warps_per_output, uint32_t tile_components);

void launchF16PacketComplete(const StridedSpan& input,
                             const StridedSpan& output,
                             const ReductionGeometry& geometry,
                             uint32_t warps_per_output,
                             uint32_t tile_components,
                             Stream& stream,
                             const ReductionStageSemantics& semantics = {},
                             float output_scale = 1.0f);

[[nodiscard]] KernelOccupancyInfo queryF16PacketCompleteOccupancy(
    DataType input_dtype, ReductionStageInputTransform transform = ReductionStageInputTransform::Identity,
    ReductionStageCombine combine = ReductionStageCombine::Sum);


/**
 * The FP8 R / KR VALUE pass over [shape.rows, shape.row_bytes] (Reduction/KR/KRSegments.h) writing `output` ([rows]
 * elements, any FP8 Complete storage dtype; semantics finalize and output_scale apply once). Pack and Group are one
 * kernel; Split writes chunk partials to `scratch` (krScratchBytes(shape, false) bytes, 16-byte aligned, no
 * initialization) and a finishing kernel combines them in chunk order. Any input address works: loads use the tensor's
 * 4-byte word grid and may read up to 3 bytes on either side of a row (inside the allocation and its trailing padding).
 */
/** The ranker site of an FP8 KR VALUE pass (KRSegments.h), shared by production and the sweep. */
[[nodiscard]] KR::KRRankerSite krValueRankerSite(const ReductionStageSemantics& semantics,
                                                 DataType input_dtype);

/** The semantics the FP8 KR VALUE kernels compile: SUM x {identity, |x|, x^2}, PRODUCT, MIN / MAX x any. */
[[nodiscard]] bool fp8KRValueSupports(const ReductionStageSemantics& semantics);

void launchFp8KRValueSegments(const ReductionStageSemantics& semantics,
                              const StridedSpan& input,
                              const StridedSpan& output,
                              const KR::KRSegmentShape& shape,
                              void* scratch,
                              uint64_t scratch_bytes,
                              float output_scale,
                              Stream& stream,
                              KR::KRSplitStages stages = KR::KRSplitStages::Both);

/** The same R / KR VALUE pass on FP32 input (the FP8 Group / Pack / Block / Split kernels at one element per word).
 *  `shape` is in bytes (row_bytes = 4 * elements per row) and legal for element size 4
 *  (KR::retainedKRShapeLegalForElement: no vector Pack); the input must be 4-byte aligned. The FP32 KR selector picks
 *  the shape. */
void launchFp32KRValueSegments(const ReductionStageSemantics& semantics,
                               const StridedSpan& input,
                               const StridedSpan& output,
                               const KR::KRSegmentShape& shape,
                               void* scratch,
                               uint64_t scratch_bytes,
                               float output_scale,
                               Stream& stream,
                               KR::KRSplitStages stages = KR::KRSplitStages::Both);

/** The same R / KR VALUE pass on FP16 / BF16 input (the FP8 Group / Pack / Block / Split kernels at two elements per
 *  word; Split partials and the finishing kernel are FP32's). `shape` is in bytes (row_bytes = 2 * elements per row)
 *  and legal for element size 2 (KR::retainedKRShapeLegalForElement: no vector Pack); the input must be 2-byte
 *  aligned. */
void launchF16KRValueSegments(const ReductionStageSemantics& semantics,
                              const StridedSpan& input,
                              const StridedSpan& output,
                              const KR::KRSegmentShape& shape,
                              void* scratch,
                              uint64_t scratch_bytes,
                              float output_scale,
                              Stream& stream,
                              KR::KRSplitStages stages = KR::KRSplitStages::Both);

/** The R / KR argmin/argmax pass on FP16 / BF16 input (FP8's Group / Pack / Block / Split / Finish kernels at two
 *  elements per word; one 64-bit candidate, key16 << 48 | row-local index << 1 | -0, F16ArgKey.h).
 *  `shape` is in bytes and legal for it (KR::f16KRArgShapeLegal: no vector Pack); the input is 2-byte aligned; Split
 *  partials take krScratchBytes(shape, true) bytes. Index = row-local index * domain_stride. */
void launchF16KRArgSegments(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const KR::KRSegmentShape& shape,
                            void* scratch,
                            uint64_t scratch_bytes,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            KR::KRSplitStages stages = KR::KRSplitStages::Both);

/** The R / KR argmin/argmax pass on FP32 input (its own kernels: two-word candidates, a 32-bit ordered key and a 64-bit
 *  row-local index). `shape` is in bytes and legal for it (KR::fp32KRArgShapeLegal); the input is 4-byte aligned; Split
 *  partials take krScratchBytes(shape, true) bytes. Index = row-local index * domain_stride. */
void launchFp32KRArgSegments(const StridedSpan& input,
                             const StridedSpan* value_output,
                             const StridedSpan* index_output,
                             const KR::KRSegmentShape& shape,
                             void* scratch,
                             uint64_t scratch_bytes,
                             bool maximum,
                             uint64_t domain_stride,
                             Stream& stream,
                             KR::KRSplitStages stages = KR::KRSplitStages::Both);

/** Wide Split scratch of the carried-index FP32 R / KR ARG pass (fp32ArgWideScratchBytes over [rows, chunks]); 0 for
 *  the other families. */
[[nodiscard]] uint64_t fp32KRCarriedArgScratchBytes(const KR::KRSegmentShape& shape);

/** The carried-index (intermediate) FP32 R / KR ARG pass: launchFp32KRArgSegments's families and shapes
 *  (KR::fp32KRArgShapeLegal) with a UINT32 / UINT64 carried index per element (the input's element count, naturally
 *  aligned). The index output holds the composed index carried + row-local position * row_stride. A Split pass writes
 *  wide partials to `scratch` (fp32KRCarriedArgScratchBytes, 256-byte aligned) and finishes them in the same call
 *  (unless PartialsOnly). */
void launchFp32KRCarriedArgSegments(const StridedSpan& input,
                                    const StridedSpan& carried_index,
                                    const StridedSpan* value_output,
                                    const StridedSpan* index_output,
                                    const KR::KRSegmentShape& shape,
                                    void* scratch,
                                    uint64_t scratch_bytes,
                                    bool maximum,
                                    uint64_t row_stride,
                                    Stream& stream,
                                    KR::KRSplitStages stages = KR::KRSplitStages::Both);

/** The FP8 R / KR argmin/argmax pass (same families; Split partials are 64-bit wide candidates in `scratch`,
 *  krScratchBytes(shape, true) bytes). Index = row-local index * domain_stride. */
void launchFp8KRArgSegments(const StridedSpan& input,
                            const StridedSpan* value_output,
                            const StridedSpan* index_output,
                            const KR::KRSegmentShape& shape,
                            void* scratch,
                            uint64_t scratch_bytes,
                            bool maximum,
                            uint64_t domain_stride,
                            Stream& stream,
                            KR::KRSplitStages stages = KR::KRSplitStages::Both);

}  // namespace ThorImplementation::ReductionInternal
