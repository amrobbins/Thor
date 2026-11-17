#pragma once

// Production-owned, host-only FP8 dense RK candidate enumeration used by
// the active cache-miss autotuner. It returns physical FIRST STAGES only;
// staged candidates are measured with independently selected hypothetical
// FP32 continuation passes, but production never caches those continuations.
//
// The three occupancy callbacks must query the exact *compiled* specialization for the supplied
// format/combine/transform, returning zero for any missing/pruned kernel. The RuntimeK span must similarly contain only
// executable, validated physical candidates for those semantics.

#include "Utilities/TensorOperations/Reduction/Contest/ReductionTuningPlan.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKSelector.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ThorImplementation::ReductionInternal::RK::Fp8 {

namespace General = Fp8RKPolicy;

struct ProductionSelectionContext {
    uint64_t outer;
    uint64_t r;
    uint64_t k;
    uint32_t sm_count;
    General::RankerSemantics semantics;
    // Maximum FP32 first-stage partial storage. Candidate measurement also
    // budgets the hypothetical continuation and its physical scratch.
    uint64_t first_stage_scratch_budget_bytes;
    size_t max_candidates = reductionTuningPlan(ReductionTuningDepth::Regular).candidates;
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

} // namespace ThorImplementation::ReductionInternal::RK::Fp8
