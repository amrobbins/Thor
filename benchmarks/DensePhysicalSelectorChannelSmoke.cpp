// Standalone host contract for the common physical selector and the ordinary iterative planner.
// The family methods below deliberately throw: deterministic injection must be the only policy used by this test.
// Production policy parity is separately covered by CubDensePhysicalSelector GTests linked against Thor.
#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalSelection.h"

#ifdef NDEBUG
#undef NDEBUG  // This smoke must validate its contracts even in Release builds.
#endif
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal;

namespace ThorImplementation::CubReductionInternal {
DenseReductionCandidate ReducersDenseR::propose(const DenseReductionProblem&, DenseReductionSite) {
    throw std::logic_error("host test entered non-injected DenseR selection");
}
DenseReductionCandidate ReducersDenseKR::propose(const DenseReductionProblem&, DenseReductionSite) {
    throw std::logic_error("host test entered non-injected DenseKR selection");
}
DenseReductionCandidate ReducersDenseRK::propose(const DenseReductionProblem&,
                                                 DenseReductionSite,
                                                 const DenseRKPlanningContext&) {
    throw std::logic_error("host test entered non-injected DenseRK selection");
}
DenseReductionCandidate makeRetainedFp8RKPhysicalCandidate(
    const DenseReductionProblem&, DenseReductionSite, const Fp8RK::PhysicalKernel&,
    const DensePhysicalSelectionContext&) {
    throw std::logic_error("host test entered production FP8 selection");
}
std::optional<DenseReductionCandidate> ReducersDenseRK::proposeProductionKParallelForCalibration(
    const DenseReductionProblem&, DenseReductionSite, const DenseRKPlanningContext&) {
    throw std::logic_error("host test entered calibration selector");
}
 }  // namespace ThorImplementation::CubReductionInternal

namespace ThorImplementation::CubReductionInternal::Fp8RK {
Fp8RKSelection selectCachedIsolatedRKSite(const DenseReductionProblem&, DenseReductionSite,
    CubReductionOp, uint64_t, float, DataType, const Stream&, uint64_t, CubReductionTuningDepth) {
    throw std::logic_error("host test unexpectedly invoked FP8 autotuner");
}
}

namespace {
struct TestCalls {
    unsigned r = 0;
    unsigned rk = 0;
    unsigned kr = 0;
};

DenseReductionCandidate testProposal(const DenseReductionProblem& problem,
                                     DenseReductionSite site,
                                     DenseReducerFamily family,
                                     const DenseRKPlanningContext&,
                                     void* data) {
    auto& calls = *static_cast<TestCalls*>(data);
    if (family == DenseReducerFamily::R) ++calls.r;
    if (family == DenseReducerFamily::KR) ++calls.kr;
    if (family == DenseReducerFamily::RK) ++calls.rk;
    // A staged physical RK first pass is a legitimate partial output. It is NOT a completed nominated dimension.
    const uint64_t next_extent = family == DenseReducerFamily::RK
                                     && problem.input_state == DenseReductionInputState::OriginalInput ? 2 : 1;
    DenseReductionCandidate result;
    result.family = family;
    result.site = site;
    result.physical_plan = std::make_shared<DenseDirectReductionPhysicalPlan>();
    result.next_problem = advanceDenseReductionProblem(problem, site, next_extent);
    result.next_bytes = denseReductionProblemBytes(result.next_problem);
    return result;
}

DenseReductionProblem inputProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    DenseReductionOutputSpec output;
    output.produce_value = true;
    output.value_dtype = DataType::FP32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Value, output);
}
}  // namespace

int main() {
    TestCalls calls;
    DensePhysicalSelectionContext context;
    context.rk.original_input_dtype = DataType::FP32;
    context.proposal_override = testProposal;
    context.proposal_override_context = &calls;

    const auto pure_r = inputProblem({257}, {0});
    const auto direct = selectDensePhysicalPass(pure_r, DenseReductionSite{0}, context);
    assert(direct.family == DenseReducerFamily::R);
    assert(!hasDenseReductionSites(direct.next_problem));
    const auto direct_plan = planDenseReduction(pure_r, context);
    assert(direct_plan.steps.size() == 1);
    assert(direct_plan.steps[0].considered_candidates.size() == 1);
    assert(calls.r == 2);  // Direct selection and the pure-R planner both used the shared API.

    calls = {};
    const auto input = inputProblem({12, 3, 5}, {0, 2});
    const auto first = selectDensePhysicalPass(input, DenseReductionSite{0}, context);
    assert(first.family == DenseReducerFamily::RK);
    assert(first.next_problem.runs[0].kind == DenseReductionRunKind::IntermediateReduction);
    assert(first.next_problem.runs[0].extent == 2);
    assert(first.next_bytes == denseReductionProblemBytes(first.next_problem));
    const auto next = selectDensePhysicalPass(first.next_problem, DenseReductionSite{0}, context);
    assert(hasDenseReductionSites(next.next_problem));  // Another original R remains.

    calls = {};
    const auto plan = planDenseReduction(input, context);
    assert(plan.steps.size() == 3);
    assert(plan.steps[0].selected_candidate.family == DenseReducerFamily::RK);
    assert(plan.steps[0].selected_candidate.next_problem == first.next_problem);
    // The outer planner chooses KR next, even though the FP32 RK partial remains unfinished: no forced continuation.
    assert(plan.steps[1].selected_candidate.family == DenseReducerFamily::KR);
    assert(plan.steps[2].selected_candidate.family == DenseReducerFamily::KR
           || plan.steps[2].selected_candidate.family == DenseReducerFamily::R
           || plan.steps[2].selected_candidate.family == DenseReducerFamily::RK);
    assert(!hasDenseReductionSites(plan.final_problem));
    assert(calls.rk >= 2 && calls.kr >= 1);
    // The new production route must reject unsupported original FP8 RK
    // requirements before falling through to any legacy RK family. This
    // host test deliberately requires no CUDA stream or FP8 tuning.
    DensePhysicalSelectionContext fp8_context;
    fp8_context.rk.original_input_dtype = DataType::FP8_E4M3;
    fp8_context.original_value_operation = static_cast<CubReductionOp>(0); // public SUM token
    fp8_context.original_reduction_size = 512;
    const auto fp8_problem = inputProblem({16, 512, 64}, {1});
    auto mustRejectRetained = [&](const DenseReductionProblem& p) {
        try {
            (void)selectDensePhysicalPass(p, DenseReductionSite{1}, fp8_context);
        } catch (const std::logic_error& error) {
            if (std::string(error.what()).find("Retained FP8 RK") != std::string::npos)
                return;
            throw;
        }
        throw std::logic_error("Unsupported original FP8 RK was not rejected");
    };
    fp8_context.rk.value_operation = DenseRKValueOperation::Mean;
    mustRejectRetained(fp8_problem);
    fp8_context.rk.value_operation = DenseRKValueOperation::Sum;
    fp8_context.output_scale = 2.0f;
    mustRejectRetained(fp8_problem);
    fp8_context.output_scale = 1.0f;
    auto half_problem = fp8_problem;
    half_problem.final_output.value_dtype = DataType::FP16;
    mustRejectRetained(half_problem);
    std::cout << "Dense physical selector channel smoke passed (R/RK/KR, staged output, "
                 "planner ordering, fail-closed FP8 RK semantics)\n";
}
