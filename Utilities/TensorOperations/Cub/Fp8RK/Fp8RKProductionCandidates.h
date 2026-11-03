#pragma once

// Production-owned, host-only FP8 dense RK candidate enumeration used by
// the active cache-miss autotuner. It returns physical FIRST STAGES only;
// staged candidates are measured with independently selected hypothetical
// FP32 continuation passes, but production never caches those continuations.
//
// The three occupancy callbacks must query the exact *retained compiled*
// specialization for the supplied format/combine/transform, returning zero
// for any missing/pruned kernel. The RuntimeK span must similarly contain
// only executable, validated physical candidates for those semantics.

#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8GeneralRParallelSelector.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

namespace General = Fp8GeneralRParallelPolicy;
// 1141: the autotuned shortlist per depth (CubReductionTuningDepth): candidates timed and rounds per candidate.
// trimmed: how many of the fastest and of the slowest rounds each candidate drops.
struct Fp8RKTuningPlan {
    size_t candidates = 0;
    size_t rounds = 0;
    size_t trimmed = 1;
};
[[nodiscard]] constexpr Fp8RKTuningPlan fp8RKTuningPlan(CubReductionTuningDepth depth) {
    switch (depth) {
        case CubReductionTuningDepth::Quick: return {6, 5, 1};
        case CubReductionTuningDepth::Regular: return {12, 7, 1};
        case CubReductionTuningDepth::Ultra: return {24, 11, 2};
    }
    return {12, 7, 1};
}

struct ProductionSelectionContext {
    uint64_t outer;
    uint64_t r;
    uint64_t k;
    uint32_t sm_count;
    General::RankerSemantics semantics;
    // Maximum FP32 first-stage partial storage. Candidate measurement also
    // budgets the hypothetical continuation and its physical scratch.
    uint64_t first_stage_scratch_budget_bytes;
    size_t max_candidates = fp8RKTuningPlan(CubReductionTuningDepth::Regular).candidates;
};

// The explicit scratch budget keeps production callers from silently accepting
// the unlimited benchmark default.
template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
[[nodiscard]] std::vector<General::CandidatePlan> selectProductionCandidates(
    const ProductionSelectionContext& context,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    std::span<const Fp8RuntimeKPolicy::Candidate> runtime_candidates) {
    General::GeneralShortlistOptions options;
    options.max_candidates = context.max_candidates;
    options.per_family_limit = 32;
    options.max_scratch_bytes = context.first_stage_scratch_budget_bytes;
    return General::generalShortlist(
        context.outer, context.r, context.k, context.sm_count,
        std::forward<OneWindowResidency>(one_window_residency),
        std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
        std::forward<AlignedKResidency>(aligned_k_residency),
        runtime_candidates, context.semantics, options);
}

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
