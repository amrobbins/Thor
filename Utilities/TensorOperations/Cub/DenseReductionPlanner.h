#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"

#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * One iteration of the ordinary-dense planner.
 *
 * considered_candidates is retained for diagnostics/tests only. Selection is based solely on next_bytes followed by
 * the deterministic family/site tie break. No family-local CUDA geometry participates in overall planning.
 */
struct DenseReductionPlanStep {
    DenseReductionProblem input_problem;
    std::vector<DenseReductionCandidate> considered_candidates;
    DenseReductionCandidate selected_candidate;
};

/** Ordered plan produced by repeatedly selecting exactly one physical dense pass. */
struct DenseReductionPlan {
    std::vector<DenseReductionPlanStep> steps;
    DenseReductionProblem final_problem;
};

/** Device/planning facts that are not mathematical tensor state. */
struct DenseReductionPlannerContext {
    DenseRKPlanningContext rk;
};

/**
 * Plans an ordinary dense reduction as a flat sequence of physical passes.
 *
 * A current problem consisting only of [R] is fast-pathed directly to ReducersDenseR and must finish in one pass.
 * Otherwise every current reduction site is nominated independently, its applicable KR/RK family is asked for one
 * candidate, and the candidate with the smallest next_bytes wins. Equal-byte candidates prefer KR to RK and then
 * stable current-site order. The selected next_problem is treated as a fresh problem and planning repeats.
 */
[[nodiscard]] DenseReductionPlan planDenseReduction(const DenseReductionProblem& initial_problem,
                                                     const DenseReductionPlannerContext& context);

/** Human-readable, selection-neutral diagnostic for the plan and every candidate considered. */
[[nodiscard]] std::string formatDenseReductionPlanDiagnostics(const DenseReductionPlan& plan);

}  // namespace ThorImplementation::CubReductionInternal
