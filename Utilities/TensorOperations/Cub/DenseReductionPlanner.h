#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * 1167: one site's lower bound in a planner step: the bytes its next problem would hold if one pass reduced the site
 * fully (`advanceDenseReductionProblem(current, site, 1)`). `selected` says whether the planner asked the physical
 * selector (which autotunes) for this site.
 */
struct DenseReductionSiteBound {
    DenseReductionSite site;
    DenseReducerFamily family;
    uint64_t complete_next_bytes = 0;
    bool selected = false;
};

/**
 * One iteration of the ordinary-dense planner.
 *
 * considered_candidates (the sites actually selected, in visiting order) and site_bounds (every site, in visiting
 * order) are retained for diagnostics/tests only. Selection is based solely on next_bytes followed by the
 * deterministic family/site tie break. No family-local CUDA geometry participates in overall planning.
 */
struct DenseReductionPlanStep {
    DenseReductionProblem input_problem;
    std::vector<DenseReductionSiteBound> site_bounds;
    std::vector<DenseReductionCandidate> considered_candidates;
    DenseReductionCandidate selected_candidate;
};

/** Ordered plan produced by repeatedly selecting exactly one physical dense pass. */
struct DenseReductionPlan {
    std::vector<DenseReductionPlanStep> steps;
    DenseReductionProblem final_problem;
};

/** Device/planning facts that are not mathematical tensor state. */
using DenseReductionPlannerContext = DensePhysicalSelectionContext;

/**
 * Plans an ordinary dense reduction as a flat sequence of physical passes.
 *
 * A current problem consisting only of [R] is fast-pathed through the official physical selector and must finish
 * in one pass.
 * Otherwise the candidate with the smallest next_bytes wins; equal-byte candidates prefer KR to RK and then the outer
 * site (lower run index). 1167: sites are selected lazily, best bound first. Every site's bound is the next problem's
 * bytes if one pass reduced it fully, which no candidate for it can beat. Sites are visited in (bound, KR before RK,
 * outer before inner) order and selected only while the bound could still beat the best candidate so far. A site
 * whose selected pass completes it therefore ends the step after one selection; a partial (staged) pass leaves more
 * bytes and lets the next site be tried if its bound is smaller. The choice equals selecting every site; only the
 * number of selections (autotuning contests) changes. The selected next_problem is treated as a fresh problem and
 * planning repeats; the finished plan is then stamped as a whole.
 */
[[nodiscard]] DenseReductionPlan planDenseReduction(const DenseReductionProblem& initial_problem,
                                                     const DenseReductionPlannerContext& context);

/** Human-readable, selection-neutral diagnostic for the plan and every candidate considered. */
[[nodiscard]] std::string formatDenseReductionPlanDiagnostics(const DenseReductionPlan& plan);

}  // namespace ThorImplementation::CubReductionInternal
