#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKContinuationLayout.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal {
namespace {

[[nodiscard]] uint64_t multiplyChecked(uint64_t a, uint64_t b) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("Retained FP8 RK first-stage geometry overflows uint64_t");
    return a * b;
}

[[nodiscard]] bool isFp8(DataType dtype) {
    return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
}

}  // namespace

DenseReductionCandidate makeRetainedFp8RKPhysicalCandidate(
    const DenseReductionProblem& problem, DenseReductionSite site,
    const Fp8RK::PhysicalKernel& first_stage,
    const DensePhysicalSelectionContext& context) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Value ||
        !problem.final_output.produce_value || problem.final_output.produce_index ||
        problem.input_state != DenseReductionInputState::OriginalInput ||
        !isFp8(context.rk.original_input_dtype) ||
        classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK ||
        problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction)
        throw std::invalid_argument("Retained FP8 RK physical pass requires original FP8 VALUE RK input");
    if (!context.original_value_operation || !context.original_reduction_size ||
        !std::isfinite(context.output_scale))
        throw std::invalid_argument("Retained FP8 RK physical pass is missing original VALUE semantics");

    uint64_t outer = 1, inner = 1;
    const uint64_t r = problem.runs[site.run_index].extent;
    for (size_t i = 0; i < problem.runs.size(); ++i) {
        if (i < site.run_index) outer = multiplyChecked(outer, problem.runs[i].extent);
        if (i > site.run_index) inner = multiplyChecked(inner, problem.runs[i].extent);
    }
    // The descriptor is for the real physical kernel, NOT an isolated
    // reduction of the entire nominated R. A staged candidate leaves N>1.
    const Fp8RK::ContinuationLayout physical = Fp8RK::describeContinuationLayout(
        first_stage, outer, r, inner, false);
    const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, physical.shards);
    const bool final_math_pass = !hasDenseReductionSites(next);
    const DataType actual_output_dtype = final_math_pass ? problem.final_output.value_dtype : DataType::FP32;
    const uint64_t next_bytes = denseReductionProblemBytes(next);

    if (physical.has_fp32_continuation && next_bytes != physical.partial_bytes)
        throw std::logic_error("Retained FP8 RK staged first-stage output disagrees with canonical FP32 bytes");
    if (!physical.has_fp32_continuation && !final_math_pass &&
        next_bytes != physical.retained_result_bytes)
        throw std::logic_error("Retained FP8 RK complete first-stage output disagrees with canonical FP32 bytes");

    auto plan = std::make_shared<DenseRetainedFp8RKPhysicalPlan>();
    plan->kernel = first_stage;
    plan->role = deriveDensePassRole(problem, next);
    plan->input_dtype = context.rk.original_input_dtype;
    plan->output_dtype = actual_output_dtype;
    plan->input_dimensions = {outer, r, inner};
    plan->output_dimensions = {outer, physical.shards, inner};
    plan->outer_size = outer;
    plan->reduction_size = r;
    plan->inner_size = inner;

    DenseReductionCandidate candidate;
    candidate.family = DenseReducerFamily::RK;
    candidate.site = site;
    candidate.physical_plan = std::move(plan);
    candidate.next_problem = next;
    candidate.next_bytes = next_bytes;
    // PhysicalKernel::first_stage_scratch_bytes is the *output* FP32 partial
    // tensor, not an additional scratch buffer owned by the physical pass.
    candidate.scratch_bytes = 0;
    return candidate;
}

}  // namespace ThorImplementation::CubReductionInternal
