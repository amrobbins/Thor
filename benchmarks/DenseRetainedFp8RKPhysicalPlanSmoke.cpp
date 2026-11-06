// P2-B host-only contract: retained FP8 RK contributes exactly one normal
// physical candidate to the existing planner; continuation remains a fresh
// independently chosen FP32 pass. No CUDA, Tensor, tuning, or stamp required.
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalSelection.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace C = ThorImplementation::CubReductionInternal;
namespace F = ThorImplementation::CubReductionInternal::Fp8RK;
using ThorImplementation::DataType;

// 1258: an injected candidate's physical plan (the legacy direct plan the injections used is gone).
struct InjectedPhysicalPlan final : ThorImplementation::CubReductionInternal::DenseReductionPhysicalPlan {};

namespace ThorImplementation::CubReductionInternal::Fp8RK {
Fp8RKSelection selectCachedIsolatedRKSite(const DenseReductionProblem&, DenseReductionSite,
    CubReductionOp, uint64_t, float, DataType, const Stream&, uint64_t, CubReductionTuningDepth) {
    throw std::logic_error("host test unexpectedly invoked FP8 autotuner");
}
}

namespace {
C::DenseReductionProblem problem(std::vector<uint64_t> dimensions,
                                 std::vector<uint32_t> axes,
                                 DataType final_dtype = DataType::FP16) {
    C::DenseReductionOutputSpec output;
    output.value_dtype = final_dtype;
    return C::makeInitialDenseReductionProblem(dimensions, axes,
                                                C::DenseReductionAggregateKind::Value, output);
}

F::PhysicalKernel oneWindow() {
    // O=1, R>=1, K<=512; 8 output groups per CTA, 1 CTA for O=1.
    return {F::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 1, 0};
}
F::PhysicalKernel deepR(uint64_t r, uint64_t k) {
    const uint64_t shards = r / 128 + (r % 128 != 0);
    return {F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0,
            6, shards, shards * k * 4};
}

C::DensePhysicalSelectionContext context(DataType original = DataType::FP8_E4M3) {
    C::DensePhysicalSelectionContext ctx;
    ctx.rk.original_input_dtype = original;
    ctx.original_value_operation = static_cast<ThorImplementation::CubReductionOp>(0);
    ctx.original_reduction_size = 131072;
    return ctx;
}

const C::DenseRetainedFp8RKPhysicalPlan& retained(const C::DenseReductionCandidate& c) {
    const auto* typed = dynamic_cast<const C::DenseRetainedFp8RKPhysicalPlan*>(c.physical_plan.get());
    assert(typed != nullptr);
    return *typed;
}

struct Calls {
    C::DensePhysicalSelectionContext context = ::context();
    F::PhysicalKernel physical = deepR(2048, 32);
    unsigned fp8_calls = 0;
    unsigned fp32_calls = 0;
};

C::DenseReductionCandidate injectedProposal(const C::DenseReductionProblem& current,
                                              C::DenseReductionSite site,
                                              C::DenseReducerFamily family,
                                              const C::DenseRKPlanningContext&,
                                              void* opaque) {
    auto& fixture = *static_cast<Calls*>(opaque);
    if (current.input_state == C::DenseReductionInputState::OriginalInput &&
        family == C::DenseReducerFamily::RK) {
        ++fixture.fp8_calls;
        return C::makeRetainedFp8RKPhysicalCandidate(current, site, fixture.physical, fixture.context);
    }
    if (current.input_state == C::DenseReductionInputState::IntermediateAggregate)
        ++fixture.fp32_calls;
    C::DenseReductionCandidate result;
    result.family = family;
    result.site = site;
    result.physical_plan = std::make_shared<InjectedPhysicalPlan>();
    result.next_problem = C::advanceDenseReductionProblem(current, site, 1);
    result.next_bytes = C::denseReductionProblemBytes(result.next_problem);
    return result;
}

void checkCompleteLast() {
    const auto initial = problem({16, 8}, {0});
    const auto ctx = context();
    const auto candidate = C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, oneWindow(), ctx);
    const auto& pass = retained(candidate);
    assert(candidate.family == C::DenseReducerFamily::RK);
    assert(candidate.scratch_bytes == 0);
    assert(pass.kernel == oneWindow());
    assert(pass.role == C::DensePassRole::Complete);
    assert(pass.input_dimensions == (std::vector<uint64_t>{1, 16, 8}));
    assert(pass.output_dimensions == (std::vector<uint64_t>{1, 1, 8}));
    assert(pass.output_dimensions.at(pass.reduction_axis) == 1 && pass.output_dtype == DataType::FP16);
    assert(candidate.next_bytes == 8 * 2);
    assert(!C::hasDenseReductionSites(candidate.next_problem));
    assert(candidate.next_problem.finalizer_pending == false);
}

void checkCompleteOtherAxes() {
    Calls calls;
    calls.physical = oneWindow();
    calls.context.original_reduction_size = 16 * 3;
    auto ctx = calls.context;
    ctx.proposal_override = injectedProposal;
    ctx.proposal_override_context = &calls;
    const auto plan = C::planDenseReduction(problem({16, 8, 3}, {0, 2}), ctx);
    assert(plan.steps.size() == 2);
    assert(calls.fp8_calls == 1 && calls.fp32_calls == 1);
    const auto& first = plan.steps.front().selected_candidate;
    const auto& phys = retained(first);
    assert(phys.role == C::DensePassRole::First);
    assert(phys.output_dtype == DataType::FP32);
    assert(phys.output_dimensions.at(phys.reduction_axis) == 1 && phys.output_dimensions == (std::vector<uint64_t>{1, 1, 24}));
    assert(first.next_bytes == 24 * 4);
    assert(plan.steps[1].input_problem == first.next_problem);
    assert(plan.steps[1].selected_candidate.family == C::DenseReducerFamily::KR);
    assert(!C::hasDenseReductionSites(plan.final_problem));
}

void checkStagedLast() {
    Calls calls;
    calls.context.original_reduction_size = 2048;
    auto ctx = calls.context;
    ctx.proposal_override = injectedProposal;
    ctx.proposal_override_context = &calls;
    const auto plan = C::planDenseReduction(problem({2048, 32}, {0}), ctx);
    assert(plan.steps.size() == 2);
    assert(calls.fp8_calls == 1 && calls.fp32_calls == 1);
    const auto& first = plan.steps.front().selected_candidate;
    const auto& physical = retained(first);
    assert(physical.kernel.staged() && physical.output_dimensions.at(physical.reduction_axis) == 16);
    assert(physical.output_dtype == DataType::FP32);
    assert(physical.role == C::DensePassRole::First);
    assert(physical.output_dimensions == (std::vector<uint64_t>{1, 16, 32}));
    assert(first.next_problem.runs[0].kind == C::DenseReductionRunKind::IntermediateReduction);
    assert(first.next_problem.runs[0].extent == 16);
    assert(first.next_bytes == 16 * 32 * 4);
    assert(first.scratch_bytes == 0 && physical.kernel.first_stage_scratch_bytes == first.next_bytes);
    assert(plan.steps[1].input_problem == first.next_problem);
    assert(plan.steps[1].selected_candidate.family == C::DenseReducerFamily::RK);
    assert(C::deriveDensePassRole(plan.steps[1].input_problem,
                                  plan.steps[1].selected_candidate.next_problem) == C::DensePassRole::Final);
}

void checkStagedThenDifferentSite() {
    Calls calls;
    calls.physical = deepR(2048, 128);
    calls.context.original_reduction_size = 2048 * 64;
    auto ctx = calls.context;
    ctx.proposal_override = injectedProposal;
    ctx.proposal_override_context = &calls;
    const auto plan = C::planDenseReduction(problem({2048, 2, 64}, {0, 2}), ctx);
    assert(plan.steps.size() == 3);
    const auto& first = plan.steps[0].selected_candidate;
    assert(first.family == C::DenseReducerFamily::RK);
    assert(retained(first).output_dimensions.at(retained(first).reduction_axis) == 16);
    assert(first.next_bytes == 16 * 2 * 64 * 4);
    assert(plan.steps[1].input_problem == first.next_problem);
    // The independently nominated KR site wins: N*2*4 is smaller than
    // 2*64*4, despite FP8 tuning hypothetically finishing the R site first.
    assert(plan.steps[1].selected_candidate.site.run_index == 2);
    assert(plan.steps[1].selected_candidate.family == C::DenseReducerFamily::KR);
    assert(plan.steps[1].selected_candidate.next_bytes == 16 * 2 * 4);
    assert(plan.steps[2].input_problem == plan.steps[1].selected_candidate.next_problem);
    assert(plan.steps[2].selected_candidate.family == C::DenseReducerFamily::RK);
    assert(plan.steps[2].input_problem.runs[0].kind == C::DenseReductionRunKind::IntermediateReduction);
    assert(!C::hasDenseReductionSites(plan.final_problem));
    assert(calls.fp8_calls == 1 && calls.fp32_calls >= 2);
    assert(C::formatDenseReductionPlanDiagnostics(plan).find("retained_fp8_rk") != std::string::npos);
}

void checkBothFp8Formats() {
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const auto ctx = context(dtype);
        const auto complete = C::makeRetainedFp8RKPhysicalCandidate(
            problem({16, 8}, {0}, DataType::FP8_E4M3), {0}, oneWindow(), ctx);
        assert(retained(complete).input_dtype == dtype);
        assert(retained(complete).output_dtype == DataType::FP8_E4M3);
        assert(retained(complete).role == C::DensePassRole::Complete);
        const auto staged = C::makeRetainedFp8RKPhysicalCandidate(
            problem({2048, 32}, {0}, DataType::FP8_E4M3), {0}, deepR(2048, 32), ctx);
        assert(retained(staged).input_dtype == dtype);
        assert(retained(staged).output_dtype == DataType::FP32);
        assert(retained(staged).output_dimensions.at(retained(staged).reduction_axis) == 16);
        assert(staged.next_bytes == 2048);
    }
}

void checkInvalidRequests() {
    auto ctx = context();
    const auto initial = problem({2048, 32}, {0});
    auto expectInvalid = [](auto&& f) {
        bool failed = false;
        try { f(); } catch (const std::invalid_argument&) { failed = true; }
        assert(failed);
    };
    // An inconsistent staged topology is not a valid physical candidate.
    auto invalid = deepR(2048, 32);
    invalid.first_stage_scratch_bytes--;
    expectInvalid([&] { (void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, invalid, ctx); });
    ctx.rk.original_input_dtype = DataType::FP32;
    expectInvalid([&] { (void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, deepR(2048, 32), ctx); });
    ctx = context();
    ctx.output_scale = std::numeric_limits<float>::infinity();
    expectInvalid([&] { (void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, deepR(2048, 32), ctx); });
    ctx = context();
    ctx.original_value_operation.reset();
    expectInvalid([&] { (void)C::makeRetainedFp8RKPhysicalCandidate(initial, {0}, deepR(2048, 32), ctx); });
    ctx = context();
    expectInvalid([&] { (void)C::makeRetainedFp8RKPhysicalCandidate(
        C::advanceDenseReductionProblem(initial, {0}, 16), {0}, deepR(2048, 32), ctx); });
}
}  // namespace

int main() {
    checkCompleteLast();
    checkCompleteOtherAxes();
    checkStagedLast();
    checkStagedThenDifferentSite();
    checkBothFp8Formats();
    checkInvalidRequests();
    std::cout << "P2-B retained FP8 RK physical-plan smoke passed (complete/staged, normalized site changes, FP32, ordering, reject)\n";
}
