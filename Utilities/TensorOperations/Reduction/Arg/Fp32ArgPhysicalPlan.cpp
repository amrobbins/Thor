#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgPhysicalPlan.h"

#include "Utilities/TensorOperations/Reduction/KR/KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/RK/F16/F16RKArgCandidates.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp32/Fp32RKArgCandidates.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp32/Fp32RKCarriedArgCandidates.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::ReductionInternal {
namespace {

[[nodiscard]] uint64_t multiplyChecked(uint64_t a, uint64_t b) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("Retained FP32 ARG geometry overflows uint64_t");
    return a * b;
}

void requireArgSite(const DenseReductionProblem& problem, bool carried) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        (problem.input_state == DenseReductionInputState::IntermediateAggregate) != carried)
        throw std::invalid_argument(carried ? "Retained FP32 carried ARG pass requires an ARG intermediate site"
                                            : "Retained FP32 ARG pass requires an original FP32 ARG site");
}

DenseReductionCandidate candidateFor(const DenseReductionProblem& problem, DenseReductionSite site,
                                     Fp32ArgPhysicalPlan plan) {
    DenseReductionCandidate candidate;
    candidate.family = classifyDenseReducerFamily(problem, site);
    candidate.site = site;
    candidate.next_problem = advanceDenseReductionProblem(problem, site, 1);  // Complete
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    candidate.scratch_bytes = plan.scratch_bytes;
    candidate.physical_plan = std::make_shared<Fp32ArgPhysicalPlan>(std::move(plan));
    return candidate;
}

const char* rkFamilyName(RK::PhysicalFamily family) {
    switch (family) {
        case RK::PhysicalFamily::OneWindow: return "one_window";
        case RK::PhysicalFamily::DeepRSmallK: return "deep_r_small_k";
        case RK::PhysicalFamily::AlignedK: return "aligned_k";
        case RK::PhysicalFamily::RuntimeK: return "runtime_k";
    }
    return "unknown";
}

}  // namespace

DenseReductionCandidate makeRetainedFp32RKArgCandidate(const DenseReductionProblem& problem, DenseReductionSite site,
                                                       const RK::PhysicalKernel& identity, bool carried) {
    requireArgSite(problem, carried);
    if (classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK)
        throw std::invalid_argument("Retained FP32 RK ARG pass requires an RK site");
    uint64_t outer = 1, inner = 1;
    const uint64_t r = problem.runs[site.run_index].extent;
    for (size_t i = 0; i < problem.runs.size(); ++i) {
        if (i < site.run_index) outer = multiplyChecked(outer, problem.runs[i].extent);
        if (i > site.run_index) inner = multiplyChecked(inner, problem.runs[i].extent);
    }
    const bool legal = carried ? RK::Fp32::fp32RKCarriedArgPassLegal(identity, outer, r, inner)
                               : RK::Fp32::fp32RKArgPassLegal(identity, outer, r, inner);
    if (!legal) throw std::invalid_argument("Retained FP32 RK ARG identity is not legal at its site");
    Fp32ArgPhysicalPlan plan;
    plan.carried = carried;
    plan.identity = identity;
    plan.input_dimensions = {outer, r, inner};
    plan.reduction_axis = 1;
    plan.outer_size = outer;
    plan.reduction_size = r;
    plan.inner_size = inner;
    plan.scratch_bytes = carried ? RK::Fp32::fp32RKCarriedArgScratchBytes(identity, outer, r, inner)
                                 : RK::Fp32::fp32RKArgScratchBytes(identity, outer, r, inner);
    return candidateFor(problem, site, std::move(plan));
}

DenseReductionCandidate makeRetainedFp32KRArgCandidate(const DenseReductionProblem& problem, DenseReductionSite site,
                                                       const KR::KRSegmentShape& shape, bool carried) {
    requireArgSite(problem, carried);
    const KRSiteGeometry geometry = krSiteGeometry(problem, site);  // the row in elements
    if (!KR::fp32KRArgShapeLegal(shape) || shape.rows != geometry.rows ||
        shape.row_bytes != multiplyChecked(4, geometry.row_bytes))
        throw std::invalid_argument("Retained FP32 KR ARG shape does not match its site geometry");
    Fp32ArgPhysicalPlan plan;
    plan.kr = true;
    plan.carried = carried;
    plan.shape = shape;
    for (const DenseReductionRun& run : problem.runs) plan.input_dimensions.push_back(run.extent);
    plan.reduction_axis = site.run_index;
    plan.outer_size = geometry.rows;
    plan.reduction_size = geometry.row_bytes;
    plan.inner_size = 1;
    plan.scratch_bytes = carried ? fp32KRCarriedArgScratchBytes(shape) : KR::krScratchBytes(shape, true);
    return candidateFor(problem, site, std::move(plan));
}

// An original FP16 / BF16 ARG site.
void requireF16ArgSite(const DenseReductionProblem& problem, DataType input_dtype) {
    if ((input_dtype != DataType::FP16 && input_dtype != DataType::BF16) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.input_state != DenseReductionInputState::OriginalInput)
        throw std::invalid_argument("Retained FP16 / BF16 ARG pass requires an original FP16 / BF16 ARG site");
}

DenseReductionCandidate makeRetainedF16RKArgCandidate(const DenseReductionProblem& problem, DenseReductionSite site,
                                                      const RK::PhysicalKernel& identity, DataType input_dtype) {
    requireF16ArgSite(problem, input_dtype);
    if (classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK)
        throw std::invalid_argument("Retained FP16 / BF16 RK ARG pass requires an RK site");
    uint64_t outer = 1, inner = 1;
    const uint64_t r = problem.runs[site.run_index].extent;
    for (size_t i = 0; i < problem.runs.size(); ++i) {
        if (i < site.run_index) outer = multiplyChecked(outer, problem.runs[i].extent);
        if (i > site.run_index) inner = multiplyChecked(inner, problem.runs[i].extent);
    }
    if (!RK::F16::f16RKArgPassLegal(identity, outer, r, inner))
        throw std::invalid_argument("Retained FP16 / BF16 RK ARG identity is not legal at its site");
    Fp32ArgPhysicalPlan plan;
    plan.input_dtype = input_dtype;
    plan.identity = identity;
    plan.input_dimensions = {outer, r, inner};
    plan.reduction_axis = 1;
    plan.outer_size = outer;
    plan.reduction_size = r;
    plan.inner_size = inner;
    plan.scratch_bytes = RK::F16::f16RKArgScratchBytes(identity, outer, r, inner);
    return candidateFor(problem, site, std::move(plan));
}

DenseReductionCandidate makeRetainedF16KRArgCandidate(const DenseReductionProblem& problem, DenseReductionSite site,
                                                      const KR::KRSegmentShape& shape, DataType input_dtype) {
    requireF16ArgSite(problem, input_dtype);
    const KRSiteGeometry geometry = krSiteGeometry(problem, site);  // the row in elements
    if (!KR::f16KRArgShapeLegal(shape) || shape.rows != geometry.rows ||
        shape.row_bytes != multiplyChecked(2, geometry.row_bytes))
        throw std::invalid_argument("Retained FP16 / BF16 KR ARG shape does not match its site geometry");
    Fp32ArgPhysicalPlan plan;
    plan.kr = true;
    plan.input_dtype = input_dtype;
    plan.shape = shape;
    for (const DenseReductionRun& run : problem.runs) plan.input_dimensions.push_back(run.extent);
    plan.reduction_axis = site.run_index;
    plan.outer_size = geometry.rows;
    plan.reduction_size = geometry.row_bytes;
    plan.inner_size = 1;
    plan.scratch_bytes = KR::krScratchBytes(shape, true);
    return candidateFor(problem, site, std::move(plan));
}

void launchRetainedFp32ArgPlan(const Fp32ArgPhysicalPlan& plan, const StridedSpan& input,
                               const StridedSpan* carried_index, const StridedSpan* value_output, const StridedSpan* index_output,
                               const ReductionGeometry& geometry, void* scratch, bool maximum,
                               uint64_t domain_stride, Stream& stream) {
    if ((carried_index != nullptr) != plan.carried)
        throw std::logic_error("Retained FP32 ARG pass received a carried index it does not read, or none");
    if (plan.scratch_bytes != 0 && scratch == nullptr)
        throw std::logic_error("Retained FP32 ARG pass requires its partial scratch");
    if (plan.input_dtype != DataType::FP32) {  // a 2-byte original pass (the launchers check the input dtype)
        if (plan.carried) throw std::logic_error("Retained FP16 / BF16 ARG passes are never carried");
        if (plan.kr) {
            launchF16KRArgSegments(input, value_output, index_output, plan.shape, scratch, plan.scratch_bytes, maximum,
                                   domain_stride, stream);
            return;
        }
        StridedSpan site_input = input;
        site_input.reshape(plan.input_dimensions);
        RK::F16::launchF16RKArgPass(plan.identity, site_input, value_output, index_output, geometry,
                                    static_cast<uint64_t*>(scratch), maximum, domain_stride, stream);
        return;
    }
    if (plan.kr) {
        if (plan.carried) {
            launchFp32KRCarriedArgSegments(input, *carried_index, value_output, index_output, plan.shape, scratch,
                                           plan.scratch_bytes, maximum, domain_stride, stream);
        } else {
            launchFp32KRArgSegments(input, value_output, index_output, plan.shape, scratch, plan.scratch_bytes,
                                    maximum, domain_stride, stream);
        }
        return;
    }
    // The RK launchers take the pass's own [outer, r, inner] geometry.
    StridedSpan site_input = input;
    site_input.reshape(plan.input_dimensions);
    if (plan.carried) {
        StridedSpan site_carried = *carried_index;
        site_carried.reshape(plan.input_dimensions);
        RK::Fp32::launchFp32RKCarriedArgPass(plan.identity, site_input, site_carried, value_output, index_output,
                                             geometry, scratch, maximum, domain_stride, stream);
    } else {
        RK::Fp32::launchFp32RKArgPass(plan.identity, site_input, value_output, index_output, geometry,
                                      static_cast<uint64_t*>(scratch), maximum, domain_stride, stream);
    }
}

std::string retainedFp32ArgPassName(const Fp32ArgPhysicalPlan& plan) {
    std::string name = plan.input_dtype != DataType::FP32 ? "f16_arg_"
                       : plan.carried                      ? "fp32_carried_arg_"
                                                           : "fp32_arg_";
    if (plan.kr) return name + "kr:" + KR::krShapeName(plan.shape);
    const RK::PhysicalKernel& p = plan.identity;
    name += std::string("rk:") + rkFamilyName(p.family) + "_a" + std::to_string(p.a) + "_b" + std::to_string(p.b) +
            "_c" + std::to_string(p.c) + "_e" + std::to_string(p.e);
    return name;
}

}  // namespace ThorImplementation::ReductionInternal
