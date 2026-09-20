#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {

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

/** Convenience registrar for a candidate translation unit linked into only the benchmark executable. */
template <typename CandidateT>
class ReductionCandidateRegistrar {
   public:
    ReductionCandidateRegistrar() { registerReductionCandidate(std::make_unique<CandidateT>()); }
};

}  // namespace ThorImplementation::CubReductionBenchmarking
