#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKIsolatedSite.h"

#include <limits>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
[[nodiscard]] uint64_t multiplyChecked(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("Isolated FP8 RK site geometry overflows uint64_t");
    return a * b;
}
}

uint64_t IsolatedRKSite::nextBytes() const {
    return denseReductionProblemBytes(remaining_problem);
}

IsolatedRKSite planIsolatedRKSite(const DenseReductionProblem& current,
                                  DenseReductionSite site,
                                  const PhysicalKernel& first_stage,
                                  const CubReductionStageSemantics& site_semantics) {
    if (current.aggregate_kind != DenseReductionAggregateKind::Value ||
        classifyDenseReducerFamily(current, site) != DenseReducerFamily::RK)
        throw std::invalid_argument("Isolated FP8 RK site requires a VALUE RK reduction run");
    if (!current.final_output.produce_value || current.final_output.produce_index)
        throw std::invalid_argument("Isolated FP8 RK site requires a value-only output");

    IsolatedRKSite result;
    result.site = site;
    result.outer = 1;
    result.k = 1;
    for (size_t i = 0; i < current.runs.size(); ++i) {
        if (i < site.run_index) result.outer = multiplyChecked(result.outer, current.runs[i].extent);
        else if (i > site.run_index) result.k = multiplyChecked(result.k, current.runs[i].extent);
        else result.r = current.runs[i].extent;
    }
    result.remaining_problem = advanceDenseReductionProblem(current, site, 1);
    const bool last_site = !hasDenseReductionSites(result.remaining_problem);
    result.logical_role = deriveDensePassRole(current, result.remaining_problem);
    result.site_output_dtype = last_site ? current.final_output.value_dtype : DataType::FP32;

    if (current.input_state != DenseReductionInputState::OriginalInput &&
        site_semantics.input_transform != CubReductionStageInputTransform::Identity)
        throw std::invalid_argument("An FP8 RK continuation must not reapply the input transform");
    if (!last_site && site_semantics.finalize != CubReductionStageFinalize::Identity)
        throw std::invalid_argument("Only the final mathematical site may apply an output finalizer");
    if (site_semantics.finalize == CubReductionStageFinalize::Divide && !site_semantics.finalize_divisor)
        throw std::invalid_argument("Isolated FP8 RK MEAN requires a nonzero original-domain divisor");

    result.layout = describeContinuationLayout(first_stage, result.outer, result.r, result.k, last_site);
    result.first_stage_semantics = site_semantics;
    result.continuation_semantics = site_semantics;
    result.continuation_semantics.input_transform = CubReductionStageInputTransform::Identity;
    if (result.layout.has_fp32_continuation) {
        result.first_stage_semantics.finalize = CubReductionStageFinalize::Identity;
        result.first_stage_semantics.finalize_divisor = 1;
    } else {
        // Never stamp a fictitious FP32 continuation after a complete kernel.
        result.continuation_semantics.finalize = CubReductionStageFinalize::Identity;
        result.continuation_semantics.finalize_divisor = 1;
    }
    return result;
}
} // namespace ThorImplementation::CubReductionInternal::Fp8RK
