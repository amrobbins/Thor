#pragma once

#include <cstdint>
#include <optional>

namespace ThorImplementation {

/**
 * How much stamp-time autotuning a reduction may spend on its tuned sites (RK and R / KR).
 * At each site a trained ranker orders the legal passes and the top of that order is timed by the shared contest
 * (ReductionAutotuning::autotune: a cold-L2 probe sets each candidate's launches per measurement, candidates over 3x
 * the best probe are dropped, rounds interleaved, the fastest and slowest rounds trimmed); the winner is cached
 * in-process per problem and depth.
 *
 *   Quick:            RK 6 / KR 6 candidates, 5 rounds, 1 trimmed each side.
 *   Regular (default): RK 12 / KR 12 candidates, 7 rounds, 1 trimmed each side.
 *   Ultra:            RK 24 / KR 12 candidates, 11 rounds, 2 trimmed each side.
 * A cached selection of a deeper depth (Ultra > Regular > Quick) satisfies a shallower request.
 *
 * A reduction constructed without an explicit depth takes the ambient depth: the calling thread's innermost
 * ScopedReductionTuningDepth, else the process default (setDefaultReductionTuningDepth, Regular unless set). So every
 * framework reduction (expressions, losses, metrics) follows one setting without a depth argument of its own; an
 * explicit depth always wins. The depth is resolved when the Reducer / ArgReducer is constructed.
 */
enum class ReductionTuningDepth : uint8_t { Regular = 0, Quick = 1, Ultra = 2 };

/** Depths, shallowest first. */
[[nodiscard]] constexpr int reductionTuningDepthRank(ReductionTuningDepth depth) {
    return depth == ReductionTuningDepth::Quick ? 0 : depth == ReductionTuningDepth::Regular ? 1 : 2;
}

/** The process-wide default depth (Regular until set). Thread-safe. */
[[nodiscard]] ReductionTuningDepth defaultReductionTuningDepth();
void setDefaultReductionTuningDepth(ReductionTuningDepth depth);

/** The depth a reduction without an explicit depth gets on this thread: the innermost scope, else the default. */
[[nodiscard]] ReductionTuningDepth currentReductionTuningDepth();

/** `explicit_depth` if given, else currentReductionTuningDepth(). */
[[nodiscard]] inline ReductionTuningDepth resolveReductionTuningDepth(
    std::optional<ReductionTuningDepth> explicit_depth) {
    return explicit_depth ? *explicit_depth : currentReductionTuningDepth();
}

/**
 * Sets this thread's ambient depth for its lifetime (scopes nest; the previous depth is restored on exit). It does not
 * reach other threads: a stamp on a worker thread follows that thread's scopes or the process default.
 */
class ScopedReductionTuningDepth final {
   public:
    explicit ScopedReductionTuningDepth(ReductionTuningDepth depth);
    ~ScopedReductionTuningDepth();
    ScopedReductionTuningDepth(const ScopedReductionTuningDepth&) = delete;
    ScopedReductionTuningDepth& operator=(const ScopedReductionTuningDepth&) = delete;
    ScopedReductionTuningDepth(ScopedReductionTuningDepth&&) = delete;
    ScopedReductionTuningDepth& operator=(ScopedReductionTuningDepth&&) = delete;

   private:
    std::optional<ReductionTuningDepth> previous_;
};

}  // namespace ThorImplementation
