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

/** Benchmark-only FP8 KParallel Straddled physical plan. The kernel always consumes aligned 16-byte packets from the
 * flattened [R,K] byte stream; block_threads selects how many independent warp work-items share a CTA. */
struct KParallelStraddledStagePlan {
    TiledRKStageTopology topology = TiledRKStageTopology::Staged;
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
 * packet_bytes is the naturally aligned per-lane input transaction width. Established FP16/BF16/FP32 KParallel uses
 * 4/8/16 B; FP8 additionally specializes 2 B so narrow even-K rows can remain on the same modern family. Every global
 * lane transaction remains <=16 B. block_threads must select one of the compile-time-specialized
 * 32/64/128/256-thread kernels.
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
 * Benchmark-only FP8 KParallel Straddled stage for 1 <= K <= 31 with K % 16 != 0. It preserves aligned p16 global
 * loads by allowing physical packets to cross logical row boundaries. A warp-private shared-memory epilogue folds the
 * repeating row phases back to one accumulator per logical K component. Production selector policy never calls this
 * entry point.
 */
void launchFp8KParallelStraddledStage(const CubReductionStageSemantics& semantics,
                                      const Tensor& input,
                                      Tensor& stage_output,
                                      const CubReductionGeometry& geometry,
                                      const KParallelStraddledStagePlan& plan,
                                      float output_scale,
                                      Stream& stream);

/** Query exact CUDA occupancy metadata for the benchmark-only FP8 KParallel Straddled specialization. */
[[nodiscard]] CubKernelOccupancyInfo queryFp8KParallelStraddledStageOccupancy(
    DataType input_dtype, const KParallelStraddledStagePlan& plan);

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
 * Benchmark-only FP8 SUM/identity rotated launch. Calibration may compare prefix-reserved p4/p8/p16 layouts and
 * the exact p16 layout across shallow through deep R. Packet width is benchmark geometry only; production policy is
 * unchanged and continues to call the established production entry point.
 */
void launchFp8AwkwardAlignmentRotatedShardedCalibrationFirstStage(
    const Tensor& input,
    Tensor& fp32_partials,
    const CubReductionGeometry& geometry,
    size_t packet_bytes,
    bool exact_tile,
    uint64_t rows_per_shard,
    uint64_t shards_per_output,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8AwkwardAlignmentRotatedShardedCalibrationOccupancy(
    DataType input_dtype, size_t packet_bytes, bool exact_tile, uint64_t rows_per_shard);

/**
 * Experimental modern R-cooperative primitive for FP8/FP16/BF16 K<=32. Each warp reads exactly 16*K bytes as one
 * flat contiguous 16-byte-packet window: sixteen FP8 rows or eight FP16/BF16 rows. FP8 preserves those packets in
 * packed shared storage through the ownership transpose and converts to FP32 only at the final logical component owner;
 * FP16/BF16 retain the established implementation. The stage emits one FP32 partial per [outer,shard,K], is exactly one
 * physical pass, and owns no continuation. FP8 remains policy-gated pending this calibration.
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

/** Benchmark-only FP8 SUM/identity FlatRows calibration entry point. Packet widths 4/8/16 are exposed only for
 * full-space calibration; each keeps the eight-warp FlatRows topology and adaptively fills its packet-width-specific
 * warp window. Production policy remains unchanged. */
void launchFp8NarrowLowPrecisionFlatRCooperativeCalibrationFirstStage(
    const Tensor& input,
    Tensor& fp32_partials,
    const CubReductionGeometry& geometry,
    size_t packet_bytes,
    uint64_t rows_per_shard,
    uint64_t shards_per_output,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8NarrowLowPrecisionFlatRCooperativeCalibrationOccupancy(
    DataType input_dtype, size_t packet_bytes, uint64_t rows_per_shard, uint64_t inner_size);

/** DEPRECATED EXPERIMENT ONLY -- NOT A PRODUCTION CANDIDATE. Benchmark-only one-output-per-warp FP8 p4 complete
 * reducer retained solely as a calibration baseline. The focused census showed that it wins over older inventory but
 * still underutilizes tiny-K warps and reaches no line-rate cells. Do not promote or schedule this path. */
void launchDeprecatedExperimentalFp8OuterGroupedP4RCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo
queryDeprecatedExperimentalFp8OuterGroupedP4RCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t inner_size);

/** Benchmark-only compact multi-output FP8 SUM/identity complete reducer. Multiple adjacent outputs share one
 * 128-byte p4 warp window. Group strides are chosen to keep both input and output naturally p4 aligned, and complete
 * groups use p4 output stores. Calibration-only; production policy remains unchanged. */
void launchFp8CompactMultiOutputWarpP4RCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8CompactMultiOutputWarpP4RCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t inner_size, uint32_t outputs_per_warp);

[[nodiscard]] uint32_t fp8CompactMultiOutputWarpP4RCooperativeCalibrationOutputsPerWarp(
    uint64_t reduction_size, uint64_t inner_size);


/** Benchmark-only CompactMultiOutput packet-width calibration. p4 routes to the established compact kernel;
 * p8/p16 load one wide packet per producer lane, keep FP8 packed in plane-major shared memory, and decode in the
 * component consumers. R remains runtime and the packet window determines how many adjacent outputs fit per warp. */
void launchFp8CompactMultiOutputWarpRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t packet_bytes,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8CompactMultiOutputWarpRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t packet_bytes);

[[nodiscard]] uint32_t fp8CompactMultiOutputWarpRCooperativeCalibrationOutputsPerWarp(
    uint64_t reduction_size,
    uint64_t inner_size,
    uint32_t packet_bytes);

/** Benchmark-only one-output-per-warp FP8 p4 reducer that spends otherwise-idle lanes across R for each K component. */
void launchFp8RParallelWarpP4RCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8RParallelWarpP4RCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t inner_size);

/** Benchmark-only general RParallel calibration family. One warp is partitioned into 1/2/4/8/16/32 independent
 * output groups while preserving component/R-worker ownership. O1 p4 routes to the established one-output kernel;
 * packet_bytes=8/16 use one wide global packet per producer lane, plane-major packed shared staging, and packed FP8
 * decode in the component consumers. R remains runtime; packet width and outputs-per-warp are physical shapes, not
 * K x R specializations. */
void launchFp8MultiOutputRParallelWarpRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t packet_bytes,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8MultiOutputRParallelWarpRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t inner_size, uint32_t outputs_per_warp, uint32_t packet_bytes);

/** Benchmark-only general Complete RParallel calibration family. Ownership is factored along two mutually exclusive
 * axes: O1 may use W1/W2/W4/W8 warps per output, while O2/O4/O8 partition one W1 warp among outputs. Within each
 * output, spare lanes become R workers/component for narrow K and larger K assigns multiple components/lane.
 * p4/p8/p16 packed shared remapping handles aligned and awkward K in the same physical family. */
void launchFp8GeneralRParallelRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream,
    bool linear_packets = false);

[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t outputs_per_warp,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    bool linear_packets = false);


/** Benchmark-only staged topology of generalized RParallel. S2/S4/S8 balance R across independent CTA shards and
 * emit FP32 [outer, shard, K] partials. Staging is O1-only because S already supplies independent work; W1/W2/W4/W8
 * remains the intra-shard cooperation axis and p4/p8/p16 retain the same packed producer/component ownership as the
 * Complete family. Production policy is unchanged. */
void launchFp8GeneralRParallelRCooperativeCalibrationFirstStage(
    const Tensor& input,
    Tensor& fp32_partials,
    const CubReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    uint32_t shards_per_output,
    Stream& stream,
    bool linear_packets = false);

[[nodiscard]] CubKernelOccupancyInfo queryFp8GeneralRParallelRCooperativeCalibrationFirstStageOccupancy(
    DataType input_dtype,
    uint64_t inner_size,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    bool linear_packets = false);

/** Compatibility wrapper for the original small-K W2/W4/W8 calibration entry point. It now routes through the
 * general Complete RParallel physical kernel above. */
void launchFp8MultiWarpRParallelRCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t warps_per_output,
    uint32_t packet_bytes,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8MultiWarpRParallelRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t inner_size, uint32_t warps_per_output, uint32_t packet_bytes);

/** Compatibility wrappers for the original p4 calibration entry points. */
void launchFp8MultiOutputRParallelWarpP4RCooperativeCalibrationComplete(
    const Tensor& input,
    Tensor& output,
    const CubReductionGeometry& geometry,
    uint32_t outputs_per_warp,
    Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8MultiOutputRParallelWarpP4RCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t inner_size, uint32_t outputs_per_warp);

/** Benchmark-only K=2 RParallel specialization. One p4 load spans both K components across two adjacent R rows;
 * 16 lanes cooperate per output at R=31/32 and a full warp cooperates at R=63/64. Production policy is unchanged. */
void launchFp8K2PairedRowsRParallelP4RCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8K2PairedRowsRParallelP4RCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size);

/** Benchmark-only K=3 phase-stream RParallel specialization. Consecutive aligned p4 loads are canonicalized
 * lane-locally into K=3 FP32 partials before subwarp/warp reduction. Production policy is unchanged. */
void launchFp8K3PhaseStreamRParallelP4RCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8K3PhaseStreamRParallelP4RCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size);

/** Benchmark-only even-R K=3 phase-stream p8-input specialization. Each lane reads one aligned p8 and decodes
 * four FP8x2 pairs; p4 writeback is retained to isolate input packet width. Production policy is unchanged. */
void launchFp8K3PhaseStreamRParallelP8InputRCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8K3PhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size);

/** Benchmark-only K=3 runtime-R phase-stream experiment. R is a runtime kernel argument; PacketBytes is the only
 * physical phase-stream template dimension (besides FP8 dtype). The kernel preserves 4/8/16/32-lane subgroup
 * geometries internally and supports aligned or boundary-masked packet streams. Production policy is unchanged. */
void launchFp8K3RuntimeRPhaseStreamRParallelP4RCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);
void launchFp8K3RuntimeRPhaseStreamRParallelP8InputRCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8K3RuntimeRPhaseStreamRParallelP4RCooperativeCalibrationOccupancy(
    DataType input_dtype);
[[nodiscard]] CubKernelOccupancyInfo queryFp8K3RuntimeRPhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(
    DataType input_dtype);

/** Benchmark-only K=7 phase-stream p8-input specialization. Naturally aligned for R={8,16,32,64}; p4 writeback
 * is retained so the calibration isolates input packet width. Production policy is unchanged. */
void launchFp8K7PhaseStreamRParallelP8InputRCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8K7PhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size);

/** Benchmark-only K=7 phase-stream p16-input specialization. Naturally aligned for R={16,32,64}; p4 writeback
 * is retained to compare p16 directly with the p8 candidate. Production policy is unchanged. */
void launchFp8K7PhaseStreamRParallelP16InputRCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8K7PhaseStreamRParallelP16InputRCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t reduction_size);

/** Benchmark-only direct component-parallel FP8 p4 reducer for the wider-K side of the shallow-R handoff. */
void launchFp8DirectComponentWarpP4RCooperativeCalibrationComplete(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8DirectComponentWarpP4RCooperativeCalibrationOccupancy(
    DataType input_dtype, uint64_t inner_size);

/** Benchmark-only raw p4 read+write roofline reference. Reads exactly the reduction's input/output traffic ratio and
 * writes one raw p4 output packet per R raw p4 input packets, with a tiny exact scalar tail. No reduction semantics. */
void launchFp8P4ReadWriteRooflineCalibrationReference(
    const Tensor& input, Tensor& output, const CubReductionGeometry& geometry, Stream& stream);

[[nodiscard]] CubKernelOccupancyInfo queryFp8P4ReadWriteRooflineCalibrationOccupancy();



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
