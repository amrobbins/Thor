#pragma once

#include <cstdint>
#include <optional>

namespace ThorImplementation {

/**
 * How much stamp-time autotuning a reduction may spend on its retained FP8 sites (RK since 1141, R / KR since 1146).
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
 * 1169: a reduction constructed without an explicit depth takes the ambient depth: the calling thread's innermost
 * ScopedCubReductionTuningDepth, else the process default (setDefaultCubReductionTuningDepth, Regular unless set). So
 * every framework reduction (expressions, losses, metrics) follows one setting without a depth argument of its own;
 * an explicit depth always wins. The depth is resolved when the CubReduction / CubArgReduction is constructed.
 */
enum class CubReductionTuningDepth : uint8_t { Regular = 0, Quick = 1, Ultra = 2 };

/** Depths, shallowest first. */
[[nodiscard]] constexpr int cubReductionTuningDepthRank(CubReductionTuningDepth depth) {
    return depth == CubReductionTuningDepth::Quick ? 0 : depth == CubReductionTuningDepth::Regular ? 1 : 2;
}

/** 1169: the process-wide default depth (Regular until set). Thread-safe. */
[[nodiscard]] CubReductionTuningDepth defaultCubReductionTuningDepth();
void setDefaultCubReductionTuningDepth(CubReductionTuningDepth depth);

/** 1169: the depth a reduction without an explicit depth gets on this thread: the innermost scope, else the default. */
[[nodiscard]] CubReductionTuningDepth currentCubReductionTuningDepth();

/** `explicit_depth` if given, else currentCubReductionTuningDepth(). */
[[nodiscard]] inline CubReductionTuningDepth resolveCubReductionTuningDepth(
    std::optional<CubReductionTuningDepth> explicit_depth) {
    return explicit_depth ? *explicit_depth : currentCubReductionTuningDepth();
}

/**
 * 1169: sets this thread's ambient depth for its lifetime (scopes nest; the previous depth is restored on exit). It
 * does not reach other threads: a stamp on a worker thread follows that thread's scopes or the process default.
 */
class ScopedCubReductionTuningDepth final {
   public:
    explicit ScopedCubReductionTuningDepth(CubReductionTuningDepth depth);
    ~ScopedCubReductionTuningDepth();
    ScopedCubReductionTuningDepth(const ScopedCubReductionTuningDepth&) = delete;
    ScopedCubReductionTuningDepth& operator=(const ScopedCubReductionTuningDepth&) = delete;
    ScopedCubReductionTuningDepth(ScopedCubReductionTuningDepth&&) = delete;
    ScopedCubReductionTuningDepth& operator=(ScopedCubReductionTuningDepth&&) = delete;

   private:
    std::optional<CubReductionTuningDepth> previous_;
};

}  // namespace ThorImplementation
