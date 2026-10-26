#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuation.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

SelectedIsolatedContinuation selectIsolatedFp32Continuation(
    const DenseReductionProblem& original_problem,
    DenseReductionSite nominated_site,
    const IsolatedRKSite& site,
    const DensePhysicalSelectionContext& physical_selector) {
    if (site.site != nominated_site ||
        classifyDenseReducerFamily(original_problem, nominated_site) != DenseReducerFamily::RK ||
        site.r != original_problem.runs[nominated_site.run_index].extent)
        throw std::invalid_argument("FP8 lookahead site is inconsistent with its original RK problem");
    const uint64_t shards = site.layout.shards;
    if (site.layout.has_fp32_continuation != (shards > 1))
        throw std::logic_error("FP8 lookahead staged topology disagrees with physical shards");

    SelectedIsolatedContinuation result;
    result.first_stage_output_problem = advanceDenseReductionProblem(original_problem, nominated_site, shards);
    auto current = result.first_stage_output_problem;
    if (shards > 1 && denseReductionProblemBytes(current) != site.layout.partial_bytes)
        throw std::logic_error("FP8 physical partial tensor bytes disagree with the canonical FP32 problem");
    if (shards == 1 && current != site.remaining_problem)
        throw std::logic_error("Complete FP8 kernel did not finish nominated RK site");

    while (shards > 1 && nominated_site.run_index < current.runs.size() &&
           current.runs[nominated_site.run_index].kind == DenseReductionRunKind::IntermediateReduction) {
        const DenseReductionProblem before = current;
        DenseReductionCandidate selected = selectDensePhysicalPass(before, nominated_site, physical_selector);
        if (selected.site != nominated_site ||
            selected.family != classifyDenseReducerFamily(before, nominated_site) ||
            !selected.physical_plan ||
            selected.next_problem.aggregate_kind != before.aggregate_kind ||
            selected.next_problem.final_output != before.final_output ||
            selected.next_problem.carried_arg_index_dtype != before.carried_arg_index_dtype ||
            selected.next_problem.input_state != DenseReductionInputState::IntermediateAggregate ||
            selected.next_problem.finalizer_pending != hasDenseReductionSites(selected.next_problem) ||
            selected.next_bytes != denseReductionProblemBytes(selected.next_problem))
            throw std::logic_error("Official FP32 selector returned an invalid continuation physical pass");

        uint64_t next_extent = 1;
        if (nominated_site.run_index < selected.next_problem.runs.size() &&
            selected.next_problem.runs[nominated_site.run_index].kind == DenseReductionRunKind::IntermediateReduction)
            next_extent = selected.next_problem.runs[nominated_site.run_index].extent;
        const uint64_t old_extent = before.runs[nominated_site.run_index].extent;
        if (next_extent >= old_extent ||
            selected.next_problem != advanceDenseReductionProblem(before, nominated_site, next_extent))
            throw std::logic_error("Official FP32 continuation did not strictly advance only nominated RK site");

        result.fp32_passes.push_back({before, std::move(selected)});
        current = result.fp32_passes.back().selected.next_problem;
    }
    if (current != site.remaining_problem)
        throw std::logic_error("Hypothetical FP32 chain did not complete the nominated RK site");
    result.completed_site_problem = std::move(current);
    return result;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
