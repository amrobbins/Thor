#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal {
namespace {

[[nodiscard]] uint64_t multiplyChecked(uint64_t a, uint64_t b) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("Retained FP8 KR geometry overflows uint64_t");
    return a * b;
}

[[nodiscard]] std::vector<uint64_t> runExtents(const DenseReductionProblem& problem) {
    std::vector<uint64_t> extents;
    extents.reserve(problem.runs.size());
    for (const DenseReductionRun& run : problem.runs) extents.push_back(run.extent);
    return extents;
}

void requireShapeAtSite(const DenseReductionProblem& problem, DenseReductionSite site,
                        const Fp8KR::Fp8KRSegmentShape& shape) {
    const Fp8KRSiteGeometry geometry = fp8KRSiteGeometry(problem, site);
    if (!Fp8KR::fp8KRSegmentShapeLegal(shape) || shape.rows != geometry.rows ||
        shape.row_bytes != geometry.row_bytes)
        throw std::invalid_argument("Retained FP8 KR shape does not match its site geometry");
}

}  // namespace

Fp8KRSiteGeometry fp8KRSiteGeometry(const DenseReductionProblem& problem, DenseReductionSite site) {
    const DenseReducerFamily family = classifyDenseReducerFamily(problem, site);
    if (family != DenseReducerFamily::R && family != DenseReducerFamily::KR)
        throw std::invalid_argument("Retained FP8 KR geometry needs an R or KR site");
    Fp8KRSiteGeometry geometry;
    for (size_t i = 0; i < site.run_index; ++i) geometry.rows = multiplyChecked(geometry.rows, problem.runs[i].extent);
    geometry.row_bytes = problem.runs[site.run_index].extent;
    return geometry;
}

bool isOriginalFp8KRSite(const DenseReductionProblem& problem, DenseReductionSite site, DataType original_input_dtype) {
    if (original_input_dtype != DataType::FP8_E4M3 && original_input_dtype != DataType::FP8_E5M2) return false;
    if (problem.input_state != DenseReductionInputState::OriginalInput) return false;
    if (site.run_index >= problem.runs.size()) return false;
    const DenseReducerFamily family = classifyDenseReducerFamily(problem, site);
    return family == DenseReducerFamily::R || family == DenseReducerFamily::KR;
}

DenseReductionCandidate makeRetainedFp8KRValueCandidate(const DenseReductionProblem& problem,
                                                       DenseReductionSite site,
                                                       const Fp8KR::Fp8KRSegmentShape& shape,
                                                       DataType input_dtype) {
    if (!isOriginalFp8KRSite(problem, site, input_dtype) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Value)
        throw std::invalid_argument("Retained FP8 KR VALUE pass requires an original FP8 VALUE R/KR site");
    requireShapeAtSite(problem, site, shape);

    DenseReductionCandidate candidate;
    candidate.family = classifyDenseReducerFamily(problem, site);
    candidate.site = site;
    candidate.next_problem = advanceDenseReductionProblem(problem, site, 1);
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    candidate.scratch_bytes = Fp8KR::fp8KRScratchBytes(shape, false);

    auto plan = std::make_shared<DenseRetainedFp8KRPhysicalPlan>();
    plan->shape = shape;
    plan->role = deriveDensePassRole(problem, candidate.next_problem);
    plan->input_dtype = input_dtype;
    plan->input_dimensions = runExtents(problem);
    plan->reduction_axis = site.run_index;
    candidate.physical_plan = std::move(plan);
    return candidate;
}

DenseReductionCandidate makeRetainedFp32KRValueCandidate(const DenseReductionProblem& problem,
                                                         DenseReductionSite site,
                                                         const Fp8KR::Fp8KRSegmentShape& shape) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Value)
        throw std::invalid_argument("Retained FP32 KR VALUE pass requires a VALUE R/KR site");
    const Fp8KRSiteGeometry geometry = fp8KRSiteGeometry(problem, site);  // the row in elements
    if (!Fp8KR::retainedKRShapeLegalForElement(shape, 4) || shape.rows != geometry.rows ||
        shape.row_bytes != multiplyChecked(4, geometry.row_bytes))
        throw std::invalid_argument("Retained FP32 KR shape does not match its site geometry");

    DenseReductionCandidate candidate;
    candidate.family = classifyDenseReducerFamily(problem, site);
    candidate.site = site;
    candidate.next_problem = advanceDenseReductionProblem(problem, site, 1);
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    candidate.scratch_bytes = Fp8KR::fp8KRScratchBytes(shape, false);

    auto plan = std::make_shared<DenseRetainedFp8KRPhysicalPlan>();
    plan->shape = shape;
    plan->role = deriveDensePassRole(problem, candidate.next_problem);
    plan->input_dtype = DataType::FP32;
    plan->input_dimensions = runExtents(problem);
    plan->reduction_axis = site.run_index;
    candidate.physical_plan = std::move(plan);
    return candidate;
}

DenseReductionCandidate makeRetainedFp8KRArgCandidate(const DenseReductionProblem& problem,
                                                     DenseReductionSite site,
                                                     const Fp8KR::Fp8KRSegmentShape& shape,
                                                     DataType input_dtype) {
    if (!isOriginalFp8KRSite(problem, site, input_dtype) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction)
        throw std::invalid_argument("Retained FP8 KR ARG pass requires an original FP8 ARG R/KR site");
    requireShapeAtSite(problem, site, shape);

    DenseRetainedFp8ArgPhysicalPlan plan;
    plan.kr = shape;
    plan.packed_partials = shape.staged();
    plan.input_dtype = input_dtype;
    plan.input_dimensions = runExtents(problem);
    plan.outer_size = shape.rows;
    plan.reduction_size = shape.row_bytes;
    plan.inner_size = 1;

    DenseReductionCandidate candidate;
    candidate.family = classifyDenseReducerFamily(problem, site);
    candidate.site = site;
    candidate.next_problem = advanceDenseReductionProblem(problem, site, 1);
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    candidate.scratch_bytes = retainedFp8ArgScratchBytes(plan);
    candidate.physical_plan = std::make_shared<DenseRetainedFp8ArgPhysicalPlan>(std::move(plan));
    return candidate;
}

}  // namespace ThorImplementation::CubReductionInternal
