#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation::CubReductionInternal {
namespace {

[[nodiscard]] const char* familyName(DenseReducerFamily family) {
    switch (family) {
        case DenseReducerFamily::R:
            return "R";
        case DenseReducerFamily::KR:
            return "KR";
        case DenseReducerFamily::RK:
            return "RK";
    }
    return "unknown";
}

[[nodiscard]] const char* roleName(DensePassRole role) {
    switch (role) {
        case DensePassRole::Complete:
            return "Complete";
        case DensePassRole::First:
            return "First";
        case DensePassRole::Intermediate:
            return "Intermediate";
        case DensePassRole::Final:
            return "Final";
    }
    return "unknown";
}

[[nodiscard]] const char* implementationName(DenseDirectReductionImplementation implementation) {
    switch (implementation) {
        case DenseDirectReductionImplementation::DeviceTransformReduce:
            return "DeviceTransformReduce";
        case DenseDirectReductionImplementation::ContiguousFixedSegment:
            return "ContiguousFixedSegment";
        case DenseDirectReductionImplementation::TiledFixedSegment:
            return "TiledFixedSegment";
    }
    return "unknown";
}


[[nodiscard]] const char* rkStrategyName(DenseRKStrategy strategy) {
    switch (strategy) {
        case DenseRKStrategy::KParallel:
            return "KParallel";
        case DenseRKStrategy::RCooperative:
            return "RCooperative";
    }
    return "unknown";
}

[[nodiscard]] const char* rkAccessName(DenseRKAccess access) {
    switch (access) {
        case DenseRKAccess::Aligned:
            return "Aligned";
        case DenseRKAccess::Rotated:
            return "Rotated";
        case DenseRKAccess::FlatRows:
            return "FlatRows";
    }
    return "unknown";
}

[[nodiscard]] const char* rkProgressName(DenseRKProgress progress) {
    switch (progress) {
        case DenseRKProgress::Complete:
            return "Complete";
        case DenseRKProgress::Staged:
            return "Staged";
    }
    return "unknown";
}

[[nodiscard]] DenseReductionCandidate proposeForFamily(const DenseReductionProblem& problem,
                                                        DenseReductionSite site,
                                                        DenseReducerFamily family,
                                                        const DenseReductionPlannerContext& context) {
    switch (family) {
        case DenseReducerFamily::R:
            return ReducersDenseR::propose(problem, site);
        case DenseReducerFamily::KR:
            return ReducersDenseKR::propose(problem, site);
        case DenseReducerFamily::RK:
            return ReducersDenseRK::propose(problem, site, context.rk);
    }
    throw std::logic_error("Unknown dense reducer family.");
}

[[nodiscard]] size_t reducedRunCount(const DenseReductionProblem& problem) {
    return static_cast<size_t>(std::count_if(problem.runs.begin(), problem.runs.end(), [](const DenseReductionRun& run) {
        return isDenseReductionRunReduced(run.kind);
    }));
}

void validateCandidateTransition(const DenseReductionProblem& current,
                                 DenseReductionSite nominated_site,
                                 DenseReducerFamily expected_family,
                                 const DenseReductionCandidate& candidate) {
    if (candidate.family != expected_family) {
        throw std::logic_error("Dense family proposal returned a candidate owned by another family.");
    }
    if (candidate.site != nominated_site) {
        throw std::logic_error("Dense family proposal changed the nominated reduction site.");
    }
    if (!candidate.physical_plan) {
        throw std::logic_error("Dense family proposal returned no physical plan.");
    }
    if (candidate.next_problem.aggregate_kind != current.aggregate_kind
        || candidate.next_problem.carried_arg_index_dtype != current.carried_arg_index_dtype
        || candidate.next_problem.final_output != current.final_output) {
        throw std::logic_error("Dense family proposal changed aggregate or final-output semantics.");
    }
    if (candidate.next_problem.input_state != DenseReductionInputState::IntermediateAggregate) {
        throw std::logic_error("Dense family proposal did not produce aggregate input state for the next problem.");
    }
    if (candidate.next_problem.finalizer_pending != hasDenseReductionSites(candidate.next_problem)) {
        throw std::logic_error("Dense family proposal returned inconsistent finalizer state.");
    }
    if (candidate.next_bytes != denseReductionProblemBytes(candidate.next_problem)) {
        throw std::logic_error("Dense family proposal returned inconsistent next_bytes.");
    }

    const size_t current_reduced_runs = reducedRunCount(current);
    const size_t next_reduced_runs = reducedRunCount(candidate.next_problem);
    if (next_reduced_runs > current_reduced_runs || current_reduced_runs == 0) {
        throw std::logic_error("Dense family proposal did not reduce outstanding work.");
    }

    // A legal proposal may only either eliminate the nominated R or replace it by a smaller IntermediateReduction R.
    // Reconstruct that exact transition with the foundation helper so family-local code cannot mutate unrelated sites.
    if (next_reduced_runs + 1 == current_reduced_runs) {
        const DenseReductionProblem expected = advanceDenseReductionProblem(current, nominated_site, 1);
        if (candidate.next_problem != expected) {
            throw std::logic_error("Dense family proposal changed state outside its nominated reduction site.");
        }
        return;
    }
    if (next_reduced_runs != current_reduced_runs) {
        throw std::logic_error("Dense family proposal changed more than one reduction site in one pass.");
    }
    if (nominated_site.run_index >= candidate.next_problem.runs.size()) {
        throw std::logic_error("Dense partial proposal lost its nominated reduction site.");
    }
    const DenseReductionRun& next_run = candidate.next_problem.runs[nominated_site.run_index];
    if (next_run.kind != DenseReductionRunKind::IntermediateReduction) {
        throw std::logic_error("Dense partial proposal must leave an intermediate reduction run.");
    }
    const DenseReductionProblem expected =
        advanceDenseReductionProblem(current, nominated_site, next_run.extent);
    if (candidate.next_problem != expected) {
        throw std::logic_error("Dense family proposal changed state outside its nominated reduction site.");
    }
}

[[nodiscard]] int tieFamilyRank(DenseReducerFamily family) {
    switch (family) {
        case DenseReducerFamily::KR:
            return 0;
        case DenseReducerFamily::RK:
            return 1;
        case DenseReducerFamily::R:
            return 2;  // R is handled by the pure-[R] fast path.
    }
    return 3;
}

[[nodiscard]] bool candidateIsBetter(const DenseReductionCandidate& candidate,
                                     const DenseReductionCandidate& incumbent) {
    if (candidate.next_bytes != incumbent.next_bytes) {
        return candidate.next_bytes < incumbent.next_bytes;
    }
    const int candidate_rank = tieFamilyRank(candidate.family);
    const int incumbent_rank = tieFamilyRank(incumbent.family);
    if (candidate_rank != incumbent_rank) {
        return candidate_rank < incumbent_rank;
    }
    return candidate.site.run_index < incumbent.site.run_index;
}

[[nodiscard]] DensePassRole candidateRole(const DenseReductionProblem& input,
                                          const DenseReductionCandidate& candidate) {
    return deriveDensePassRole(input, candidate.next_problem);
}

[[nodiscard]] std::string problemString(const DenseReductionProblem& problem) {
    if (problem.runs.empty()) {
        return "[]";
    }
    std::ostringstream out;
    out << '[';
    for (size_t index = 0; index < problem.runs.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        const DenseReductionRun& run = problem.runs[index];
        if (run.kind == DenseReductionRunKind::Retained) {
            out << 'K';
        } else if (run.kind == DenseReductionRunKind::OriginalReduction) {
            out << 'R';
        } else {
            out << "Rp";
        }
        out << '=' << run.extent;
    }
    out << ']';
    return out.str();
}

[[nodiscard]] std::string physicalImplementationString(const DenseReductionCandidate& candidate) {
    if (const auto rk = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(candidate.physical_plan); rk) {
        std::ostringstream out;
        out << rkStrategyName(rk->strategy) << '/' << rkAccessName(rk->access) << '/' << rkProgressName(rk->progress)
            << "/packet" << rk->packet_bytes << "/cta" << rk->block_threads
            << "/rows" << rk->rows_per_shard << "/shards" << rk->shards_per_output
            << "/warps" << rk->useful_stage_warps;
        return out.str();
    }
    const auto direct = std::dynamic_pointer_cast<const DenseDirectReductionPhysicalPlan>(candidate.physical_plan);
    if (!direct) {
        return "family-local";
    }
    return implementationName(direct->implementation);
}

}  // namespace

DenseReductionPlan planDenseReduction(const DenseReductionProblem& initial_problem,
                                        const DenseReductionPlannerContext& context) {
    DenseReductionPlan plan;
    DenseReductionProblem current = initial_problem;
    std::vector<DenseReductionProblem> seen;

    while (hasDenseReductionSites(current)) {
        if (std::find(seen.begin(), seen.end(), current) != seen.end()) {
            throw std::logic_error("Dense reduction planner detected a planning cycle.");
        }
        seen.push_back(current);

        DenseReductionPlanStep step;
        step.input_problem = current;

        const std::vector<DenseReductionSite> sites = enumerateDenseReductionSites(current);
        if (sites.empty()) {
            throw std::logic_error("Dense reduction planner found reductions but no reducible sites.");
        }

        // Pure [R] is not a planning competition. Hand the complete domain to the vendored whole-domain reducer and
        // require it to finish. The later ragged overhaul can use the same architectural shortcut independently.
        if (current.runs.size() == 1 && sites.size() == 1
            && classifyDenseReducerFamily(current, sites.front()) == DenseReducerFamily::R) {
            DenseReductionCandidate candidate = ReducersDenseR::propose(current, sites.front());
            validateCandidateTransition(current, sites.front(), DenseReducerFamily::R, candidate);
            if (hasDenseReductionSites(candidate.next_problem)) {
                throw std::logic_error("DenseR pure-[R] proposal must complete the reduction in one pass.");
            }
            step.considered_candidates.push_back(candidate);
            step.selected_candidate = std::move(candidate);
        } else {
            for (DenseReductionSite site : sites) {
                const DenseReducerFamily family = classifyDenseReducerFamily(current, site);
                if (family == DenseReducerFamily::R) {
                    throw std::logic_error("DenseR is only valid for the pure-[R] planner fast path.");
                }
                DenseReductionCandidate candidate = proposeForFamily(current, site, family, context);
                validateCandidateTransition(current, site, family, candidate);
                step.considered_candidates.push_back(std::move(candidate));
            }
            if (step.considered_candidates.empty()) {
                throw std::logic_error("Dense reduction planner received no family candidates.");
            }

            size_t selected_index = 0;
            for (size_t index = 1; index < step.considered_candidates.size(); ++index) {
                if (candidateIsBetter(step.considered_candidates[index], step.considered_candidates[selected_index])) {
                    selected_index = index;
                }
            }
            step.selected_candidate = step.considered_candidates[selected_index];
        }

        const DenseReductionProblem next = step.selected_candidate.next_problem;
        plan.steps.push_back(std::move(step));
        current = next;
    }

    plan.final_problem = std::move(current);
    return plan;
}

std::string formatDenseReductionPlanDiagnostics(const DenseReductionPlan& plan) {
    std::ostringstream out;
    for (size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        const DenseReductionPlanStep& step = plan.steps[step_index];
        out << "dense_plan step=" << step_index << " input=" << problemString(step.input_problem) << '\n';
        for (const DenseReductionCandidate& candidate : step.considered_candidates) {
            const bool selected = candidate.family == step.selected_candidate.family
                                  && candidate.site == step.selected_candidate.site
                                  && candidate.next_problem == step.selected_candidate.next_problem;
            out << "  site=" << candidate.site.run_index << " family=" << familyName(candidate.family)
                << " role=" << roleName(candidateRole(step.input_problem, candidate))
                << " next=" << problemString(candidate.next_problem) << " next_bytes=" << candidate.next_bytes
                << " physical=" << physicalImplementationString(candidate)
                << " selected=" << (selected ? 1 : 0) << '\n';
        }
    }
    out << "dense_plan final=" << problemString(plan.final_problem);
    return out.str();
}

}  // namespace ThorImplementation::CubReductionInternal
