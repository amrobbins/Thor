// Host-only: benchmark lookahead must use the official one-pass selector,
// strictly reduce the nominated FP32 shards, and never touch unrelated axes.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuation.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalSelection.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace C = ThorImplementation::CubReductionInternal;
namespace F = ThorImplementation::CubReductionInternal::Fp8RK;
using ThorImplementation::DataType;

namespace ThorImplementation::CubReductionInternal {
DenseReductionCandidate ReducersDenseR::propose(const DenseReductionProblem&, DenseReductionSite) {
    throw std::logic_error("lookahead host smoke accidentally used non-injected R policy");
}
DenseReductionCandidate ReducersDenseKR::propose(const DenseReductionProblem&, DenseReductionSite) {
    throw std::logic_error("lookahead host smoke accidentally used non-injected KR policy");
}
DenseReductionCandidate ReducersDenseRK::propose(const DenseReductionProblem&, DenseReductionSite,
                                                 const DenseRKPlanningContext&) {
    throw std::logic_error("lookahead host smoke accidentally used non-injected RK policy");
}
// DenseReductionPhysicalSelector.cpp also contains the original-input FP8
// production branch. This host-only test injects every FP32 proposal, so its
// unreachable production dependencies deliberately fail if ever exercised.
DenseReductionCandidate makeRetainedFp8RKPhysicalCandidate(
    const DenseReductionProblem&, DenseReductionSite, const Fp8RK::PhysicalKernel&,
    const DensePhysicalSelectionContext&) {
    throw std::logic_error("lookahead host smoke accidentally used retained FP8 production");
}
}  // namespace ThorImplementation::CubReductionInternal

namespace ThorImplementation::CubReductionInternal::Fp8RK {
Fp8RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem&, DenseReductionSite, CubReductionOp, uint64_t,
    float, DataType, const Stream&, uint64_t) {
    throw std::logic_error("lookahead host smoke accidentally entered FP8 autotuning");
}
}  // namespace ThorImplementation::CubReductionInternal::Fp8RK

namespace {
struct Fixture {
    unsigned calls = 0;
    bool multi = false;
    bool invalidate = false;
    bool no_progress = false;
};

C::DenseReductionCandidate proposal(const C::DenseReductionProblem& problem,
                                    C::DenseReductionSite site, C::DenseReducerFamily family,
                                    const C::DenseRKPlanningContext& context, void* data) {
    auto& fixture = *static_cast<Fixture*>(data);
    assert(context.original_input_dtype == DataType::FP8_E4M3);
    assert(problem.input_state == C::DenseReductionInputState::IntermediateAggregate);
    assert(family == C::DenseReducerFamily::RK);
    ++fixture.calls;
    const uint64_t previous = problem.runs[site.run_index].extent;
    const uint64_t next = fixture.no_progress ? previous : fixture.multi && previous > 4 ? 4 : 1;
    C::DenseReductionCandidate candidate;
    candidate.family = family;
    candidate.site = site;
    candidate.physical_plan = std::make_shared<C::DenseDirectReductionPhysicalPlan>();
    if (fixture.no_progress) {
        candidate.next_problem = problem;
    } else {
        candidate.next_problem = C::advanceDenseReductionProblem(problem, site, next);
    }
    if (fixture.invalidate) candidate.next_problem.runs.back().extent = 17;
    candidate.next_bytes = C::denseReductionProblemBytes(candidate.next_problem);
    return candidate;
}

C::DenseReductionProblem problem() {
    C::DenseReductionOutputSpec spec;
    spec.value_dtype = DataType::FP16;
    return C::makeInitialDenseReductionProblem({2048, 32, 3}, {0, 2}, C::DenseReductionAggregateKind::Value, spec);
}

void checkNoTransformOrFinalizer(const F::IsolatedRKSite& site) {
    assert(site.first_stage_semantics.input_transform == C::CubReductionStageInputTransform::Square);
    assert(site.first_stage_semantics.finalize == C::CubReductionStageFinalize::Identity);
    assert(site.continuation_semantics.input_transform == C::CubReductionStageInputTransform::Identity);
    assert(site.continuation_semantics.finalize == C::CubReductionStageFinalize::Identity);
}
}  // namespace

int main() {
    const auto current = problem();
    const F::PhysicalKernel deep{F::PhysicalFamily::DeepRSmallK,
                                 128, 0, 0, 0, 0, 0, 6, 16, 6144};
    C::CubReductionStageSemantics semantics;
    semantics.input_transform = C::CubReductionStageInputTransform::Square;
    const auto site = F::planIsolatedRKSite(current, C::DenseReductionSite{0}, deep, semantics);
    checkNoTransformOrFinalizer(site);
    Fixture fixture;
    C::DensePhysicalSelectionContext context;
    context.rk.original_input_dtype = DataType::FP8_E4M3;
    context.proposal_override = proposal;
    context.proposal_override_context = &fixture;

    // The actual first FP8 physical output, not hypothetical completed R.
    const auto one = F::selectIsolatedFp32Continuation(current, {0}, site, context);
    assert(one.first_stage_output_problem.runs[0].kind == C::DenseReductionRunKind::IntermediateReduction);
    assert(one.first_stage_output_problem.runs[0].extent == 16);
    assert(one.fp32_passes.size() == 1 && fixture.calls == 1);
    assert(one.completed_site_problem == site.remaining_problem);
    assert(C::hasDenseReductionSites(one.completed_site_problem));
    assert(one.completed_site_problem.runs.back().kind == C::DenseReductionRunKind::OriginalReduction);
    assert(one.completed_site_problem.input_state == C::DenseReductionInputState::IntermediateAggregate);
    assert(one.fp32_passes[0].selected.next_bytes ==
           C::denseReductionProblemBytes(one.fp32_passes[0].selected.next_problem));

    fixture = {.multi = true};
    const auto multi = F::selectIsolatedFp32Continuation(current, {0}, site, context);
    assert(multi.fp32_passes.size() == 2 && fixture.calls == 2);
    assert(multi.fp32_passes[0].selected.next_problem.runs[0].extent == 4);
    assert(multi.fp32_passes[1].input_problem == multi.fp32_passes[0].selected.next_problem);
    assert(multi.completed_site_problem == site.remaining_problem);

    // A complete physical FP8 kernel never requests a continuation.
    const auto complete_problem = C::makeInitialDenseReductionProblem(
        {16, 3}, {0}, C::DenseReductionAggregateKind::Value, current.final_output);
    const F::PhysicalKernel complete{F::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 1, 0};
    const auto complete_site = F::planIsolatedRKSite(complete_problem, {0}, complete, semantics);
    fixture = {};
    const auto zero = F::selectIsolatedFp32Continuation(complete_problem, {0}, complete_site, context);
    assert(zero.fp32_passes.empty() && fixture.calls == 0);
    assert(zero.completed_site_problem == complete_site.remaining_problem);

    fixture = {.invalidate = true};
    bool threw = false;
    try { (void)F::selectIsolatedFp32Continuation(current, {0}, site, context); }
    catch (const std::logic_error&) { threw = true; }
    assert(threw && fixture.calls == 1);

    fixture = {.no_progress = true};
    threw = false;
    try { (void)F::selectIsolatedFp32Continuation(current, {0}, site, context); }
    catch (const std::logic_error&) { threw = true; }
    assert(threw && fixture.calls == 1);

    std::cout << "FP8 selected FP32 continuation smoke passed (0/1/2 physical passes, progress, isolation)\n";
}
