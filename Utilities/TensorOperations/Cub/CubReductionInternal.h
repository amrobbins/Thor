#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"

#include <cstddef>
#include <optional>

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





/** Retained OneWindow Complete kernel. O1 supports W1/W2/W4/W8, while O2/O4/O8
 * partition a warp among outputs. Applies the runtime finalizer/scale and converts
 * to the final output dtype. This function name predates the retained cutover. */
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

}  // namespace ThorImplementation::CubReductionInternal
