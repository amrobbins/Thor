#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {

class ReductionCandidateCoverageGap final : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

/**
 * Launch metadata reported by an experimental reduction candidate.
 *
 * Candidates are benchmark-only implementations.  They are intentionally not represented in CubReductionPath and
 * cannot participate in production path selection.  RUN-* experiments can report their own implementation/strategy
 * names and stamped launch policy here without adding production enums or planner state before the candidate earns a
 * production geometry.
 */
struct CandidateLaunchMetadata {
    std::string_view implementation;
    std::string_view strategy;
    std::optional<uint32_t> index_bits = std::nullopt;
    std::optional<uint32_t> vector_elements_per_load = std::nullopt;
    std::optional<uint32_t> block_threads = std::nullopt;
    std::optional<uint64_t> first_stage_blocks = std::nullopt;
    std::optional<uint64_t> shards_per_output = std::nullopt;
};

/** Stamped benchmark-only candidate.  Construction/planning is outside the timed interval. */
class StampedReductionCandidate {
   public:
    virtual ~StampedReductionCandidate() = default;

    virtual void runOn(Stream& stream) const = 0;
    [[nodiscard]] virtual const Tensor& getOutputTensor() const = 0;
    [[nodiscard]] virtual size_t getWorkspaceSizeInBytes() const = 0;
    [[nodiscard]] virtual CandidateLaunchMetadata getLaunchMetadata() const = 0;
};

/**
 * Benchmark-only reduction candidate factory.
 *
 * Implementations are linked only into thor_cub_reduction_benchmark. supports() must be a pure capability query;
 * stamp() may allocate candidate output/workspace and perform one-time planning, but runOn() must not allocate or plan.
 * Candidates must treat the supplied input as read-only and preserve CubReduction's output shape, dtype, operation, and
 * FP32-accumulation contract for every geometry they advertise as supported.
 */
class ReductionCandidate {
   public:
    virtual ~ReductionCandidate() = default;

    [[nodiscard]] virtual std::string_view getName() const = 0;
    [[nodiscard]] virtual bool supports(CubReductionOp op,
                                        const TensorDescriptor& input_descriptor,
                                        const std::vector<uint32_t>& axes) const = 0;
    [[nodiscard]] virtual std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                           const Tensor& input,
                                                                           const std::vector<uint32_t>& axes,
                                                                           const Stream& stream) const = 0;
};

/** Register an implementation with the benchmark-only candidate registry. */
void registerReductionCandidate(std::unique_ptr<ReductionCandidate> candidate);

/** Candidate registry used by --list-reduction-candidates and --candidate=<name>. */
[[nodiscard]] const std::vector<std::unique_ptr<ReductionCandidate>>& getReductionCandidates();

/**
 * Runs the broad modern dense-RK single-pass sweep. Every timed row contains exactly one production modern RK kernel
 * launch: K-parallel Complete/Staged, aligned R-cooperative Complete/Staged, or rotated R-cooperative Staged. No
 * continuation/remainder reducer is compiled into or launched by a timed row. The sweep is intended to establish each
 * modern kernel's comfortable geometry envelope before selection policy is calibrated.
 */
void runModernRKSinglePassSweep(Tensor& cache_flush, Stream& stream);

/**
 * Runs the second-pass focused modern RK single-pass sweep. The shape set is device-aware: it targets CTA-supply/wave
 * boundaries from the current SM count and fills in retained-width crossover regions identified by the broad sweep.
 * Timed rows still contain exactly one modern production RK physical pass.
 */
void runModernRKSinglePassFocusSweep(Tensor& cache_flush, Stream& stream);

/**
 * Runs the focused modern RK gap census used to validate the native MIN/MAX combine and the experimental narrow-flat
 * low-precision R-cooperative first stage. Every timed row is exactly one physical pass.
 */
void runModernRKGapCensus(Tensor& cache_flush, Stream& stream);

/**
 * Sweeps the complete narrow low-precision selector envelope: K=1..32, FP16/BF16, SUM/MIN/MAX, O=64,R=32768.
 * Every timed row is one physical staged pass. Flat-row and rotated R-cooperative candidates are always measured;
 * K-parallel staged candidates are included whenever the retained width admits a legal >=4-byte packet.
 */
void runRKNarrowLowPrecisionCalibration(Tensor& cache_flush, Stream& stream);

/** Focused low-work benchmark for current vs lean KParallel Complete plus production selector reference. */
void runRKLeanCompleteCalibration(Tensor& cache_flush, Stream& stream);

/** Focused low-output benchmark for lean Complete vs staged KParallel R sharding. */
void runRKKParallelStageCrossoverCalibration(Tensor& cache_flush, Stream& stream);

/** Hold staged KParallel topology fixed and calibrate packet/CTA geometry at 16 rows/shard. */
void runRKKParallelStagedGeometryCalibration(Tensor& cache_flush, Stream& stream);

/** Exact-shape FP8 KParallel packet A/B census: fixed tensor/topology, vary only packet bytes. */
void runFp8KParallelPacketABCensus(Tensor& cache_flush, Stream& stream);

/** Small-reduction FP8 KParallel packet sweep: fixed exact shape, compare every legal packet width. */
void runFp8KParallelSmallPacketSweep(Tensor& cache_flush, Stream& stream);

/** Benchmark-only FP8 KParallel Straddled p16 sweep for packet-misaligned K<=31. */
void runFp8KParallelStraddledSweep(Tensor& cache_flush, Stream& stream);

/** Small/awkward FP8 KParallel Complete layout sweep: optimize CTA width independently per packet size. */
void runFp8KParallelCompleteLayoutSweep(Tensor& cache_flush, Stream& stream);

/** Large/saturating FP8 KParallel staged layout sweep: optimize CTA width and shard count per packet size. */
void runFp8KParallelStagedLayoutSweep(Tensor& cache_flush, Stream& stream);

/**
 * Single-pass deep-R FP8 RCooperative calibration. Measures Complete, p16 FlatRows, and the two proven p16 Rotated
 * layouts in isolation. The primary scheduling floor is 1.50 TB/s input bandwidth, with 1.60 TB/s retained as a peak
 * reference; deepest R consumption above the floor wins. Production selection is unchanged.
 */
void runFp8RCooperativeSinglePassSweep(Tensor& cache_flush, Stream& stream);

/**
 * Full-space FP8 RK calibration over the pruned surviving candidate inventory, including generalized RParallel
 * Complete/staged geometry and unresolved historical baselines whose global niches have not yet been disproven.
 */
void runFp8RCooperativeFullSpaceSweep(Tensor& cache_flush, Stream& stream);

/** Focused runtime-K tiled generalized RParallel census. Uses adaptive K-tile/W/S geometry chosen from exact device
 * occupancy and compares only against the surviving large-K controls. */
void runFp8RCooperativeKTiledSweep(Tensor& cache_flush, Stream& stream);

/** Broad deletion-oriented FP8 selector census. Compares generalized runtime-K RParallel against the historical
 * controls that can still falsify consolidation: production KParallel, PacketAdaptive, FlatRows, old one-window
 * generalized Complete, and the unresolved legacy p4 RParallel path. */
void runFp8RCooperativeSelectorSimplificationSweep(Tensor& cache_flush, Stream& stream);
void runFp8RCooperativeUnifiedSelectorValidationSweep(Tensor& cache_flush, Stream& stream);
/** Fixed out-of-887-training-R policy validation; benchmark and oracle only. */
void runFp8RCooperativeGeometryPolicyHoldoutSweep(Tensor& cache_flush, Stream& stream);

/** Focused RuntimeK PacketStaged vs rotated p16 physical end-to-end inventory census.
 * FP32 staged partials run through stamped production CubReduction, with common FP8 output. */
void runFp8RuntimeKEndToEndSweep(Tensor& cache_flush, Stream& stream);
/** Independent RuntimeK 24/128 MiB inventory holdout with every legal compiled rotated depth. */
void runFp8RuntimeKInventoryHoldoutSweep(Tensor& cache_flush, Stream& stream);
/** Fresh full-oracle RuntimeK selector holdout, 12/96 MiB, five timed samples. */
void runFp8RuntimeKSelectorHoldoutSweep(Tensor& cache_flush, Stream& stream);
/** Fresh 16/112 MiB holdout for RuntimeK five-plan legal shortlist recall. */
void runFp8RuntimeKShortlistHoldoutSweep(Tensor& cache_flush, Stream& stream);
/** Original FP8 epilogue vs runtime-selected output storage on OneWindow and RuntimeK Complete. */
void runFp8CompleteRuntimeOutputRegression(Tensor& cache_flush, Stream& stream);
/** Compare runtime identity/divide/sqrt finalizers against identical-output identity and original FP8 identity. */
void runFp8CompleteRuntimeFinalizeRegression(Tensor& cache_flush, Stream& stream);

/** Benchmark-only AlignedK / production FP8 KParallel Complete vs staged packet/CTA/shard inventory.
 * Staged candidates include the real stamped FP32 continuation in the total time. */
void runFp8AlignedKEndToEndSweep(Tensor& cache_flush, Stream& stream);
/** Expanded lean-inventory calibration, NOT an acceptance holdout. */
void runFp8AlignedKLeanCalibration(Tensor& cache_flush, Stream& stream);
/** Fresh, fixed geometry holdout for the offline topology rule and CTA256 inventory deletion. */
void runFp8AlignedKTopologyHoldout(Tensor& cache_flush, Stream& stream);
/** Experimental nine-inventory AlignedK local-selector holdout, no production changes. */
void runFp8AlignedKSelectorHoldout(Tensor& cache_flush, Stream& stream);

/** FlatRows physical p4/p8/p16 / rows-per-shard sweep, followed by production FP32 continuation.
 * Timed end-to-end plus first-stage/continuation timing; five interleaved samples per candidate. */
void runFp8DeepRSmallKEndToEndSweep(Tensor& cache_flush, Stream& stream);
/** Fixed independent DeepRSmallK local-selector holdout with exact compiled P16 occupancy. */
void runFp8DeepRSmallKFourWayHoldout(Tensor& cache_flush, Stream& stream);
void runFp8DeepRSmallKFiveWayHoldout(Tensor& cache_flush, Stream& stream);

/** Frozen ten-variant OneWindow-only policy; independent larger benchmark (no production changes). */
void runFp8OneWindowSelectorValidationSweep(Tensor& cache_flush, Stream& stream);
void runFp8OneWindowGeneralizationHoldoutSweep(Tensor& cache_flush, Stream& stream);

/** Nine promoted OneWindow physical variants, twelve explicitly demoted oracles: original dense in-domain census. */
void runFp8RCooperativeOneWindowDenseBoundarySweep(Tensor& cache_flush, Stream& stream);

/** Paired 887 OneWindow decode/gather micro-census, benchmark only. */
void runFp8RCooperativeOneWindowDecodeSweep(Tensor& cache_flush, Stream& stream);

/** Focused shallow-R / narrow-K FP8 coverage sweep, including R,K down to 2 and 64..512 MiB actual work. */
void runFp8RCooperativeSmallRSmallKSweep(Tensor& cache_flush, Stream& stream);

/** Focused small-R / medium-K FP8 coverage sweep using the established medium-K nice/awkward census. */
void runFp8RCooperativeSmallRMediumKSweep(Tensor& cache_flush, Stream& stream);

/** Focused medium-R / small-K FP8 coverage sweep using the analyzer's complete nice/awkward domain. */
void runFp8RCooperativeMediumRSmallKSweep(Tensor& cache_flush, Stream& stream);

/** End-to-end low-output benchmark for lean Complete vs forced staged-first followed by normal modern replanning. */
void runRKKParallelEndToEndCrossoverCalibration(Tensor& cache_flush, Stream& stream);

/** End-to-end realistic-work calibration ranked by absolute full-reduction time and microseconds saved. */
void runRKRealisticEndToEndProgressCalibration(Tensor& cache_flush, Stream& stream);
/** Cache-cold FP16 RK production-versus-forced-physical sampled autotuning opportunity census.
 * No production routing changes. */
void runFp16ProductionAutotuneCensus(Tensor& cache_flush, Stream& stream);
void runRKFamilyBoundaryCalibration(Tensor& cache_flush, Stream& stream);

/** Convenience registrar for a candidate translation unit linked into only the benchmark executable. */
template <typename CandidateT>
class ReductionCandidateRegistrar {
   public:
    ReductionCandidateRegistrar() { registerReductionCandidate(std::make_unique<CandidateT>()); }
};

}  // namespace ThorImplementation::CubReductionBenchmarking
