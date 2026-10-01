#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstddef>
#include <optional>

namespace ThorImplementation::CubReductionInternal {

// Physical execution and dense composition share one semantic pass-role vocabulary.
using CubReductionStageRole = CubReductionPassRole;

enum class CubReductionStageInputTransform : uint8_t {
    Identity = 0,
    AbsoluteValue = 1,
    Square = 2,
};

enum class CubReductionStageCombine : uint8_t {
    Sum = 0,
    Product = 1,
    Minimum = 2,
    Maximum = 3,
};

enum class CubReductionStageFinalize : uint8_t {
    Identity = 0,
    Divide = 1,
    SquareRoot = 2,
};

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
 * Mathematical semantics of one physical value-reduction pass.
 *
 * Public CubReductionOp intentionally describes the complete operation, while a composed dense reduction must apply
 * the input transform only on its first pass and the output finalizer only on its last pass.  This descriptor separates
 * those concerns without adding more public reduction operations or changing the direct reducer families.
 *
 * finalize_divisor is consulted only for Divide and is the total original reduction-domain size, not the current
 * physical stage's reduction size.
 */
struct CubReductionStageSemantics {
    CubReductionStageInputTransform input_transform = CubReductionStageInputTransform::Identity;
    CubReductionStageCombine combine = CubReductionStageCombine::Sum;
    CubReductionStageFinalize finalize = CubReductionStageFinalize::Identity;
    uint64_t finalize_divisor = 1;
};

[[nodiscard]] CubReductionStageSemantics makeValueReductionStageSemantics(CubReductionOp op,
                                                                           CubReductionStageRole role,
                                                                           uint64_t total_reduction_size);

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
 * packet_bytes is the naturally aligned per-lane input transaction width (4/8/16 B). block_threads must select one of
 * the compile-time-specialized 32/64/128/256-thread kernels.
 */
void launchKParallelTiledStage(const CubReductionStageSemantics& semantics,
                               const Tensor& input,
                               Tensor& stage_output,
                               const CubReductionGeometry& geometry,
                               const KParallelTiledStagePlan& plan,
                               float output_scale,
                               Stream& stream);

/** Query exact CUDA occupancy metadata for the SUM/identity K-parallel specialization selected by plan. */
[[nodiscard]] CubKernelOccupancyInfo queryKParallelTiledStageOccupancy(DataType input_dtype,
                                                                       const KParallelTiledStagePlan& plan);

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

/** Query exact CUDA occupancy metadata for the SUM/identity rotated R-cooperative staged specialization. */
[[nodiscard]] CubKernelOccupancyInfo queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(
    DataType input_dtype, uint64_t rows_per_shard);

/**
 * Experimental modern R-cooperative primitive for FP16/BF16 K<=32. Each warp reads eight complete logical rows as
 * one flat contiguous 16-byte-packet window, reconstructs [row,K] ownership in shared memory, and emits one FP32
 * partial per [outer,shard,K]. The stage is intentionally one physical pass and owns no continuation.
 *
 * It is production-quality code exposed to the benchmark, but it is not yet admitted to DenseRKInventory or selected
 * by ReducersDenseRK; the focused gap census must first establish that it earns a real performance region.
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
