#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"

#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalSelection.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal {

DenseReductionCandidate selectDensePhysicalPass(const DenseReductionProblem& problem,
                                                DenseReductionSite site,
                                                const DensePhysicalSelectionContext& context) {
    const DenseReducerFamily family = classifyDenseReducerFamily(problem, site);
    // No other physical family can nominate R: it is the whole-domain [R] shortcut, not a competing site.
    if (family == DenseReducerFamily::R && problem.runs.size() != 1) {
        throw std::logic_error("DenseR physical selection requires a pure [R] problem.");
    }
    if (context.proposal_override != nullptr) {
        return context.proposal_override(problem, site, family, context.rk, context.proposal_override_context);
    }
    // Original FP8 RK VALUE has exactly one supported production selector:
    // the retained cached physical-first-stage autotuner. Never fall through
    // to the calibrated/generic FP8 RK implementation on missing semantics.
    if (family == DenseReducerFamily::RK &&
        problem.aggregate_kind == DenseReductionAggregateKind::Value &&
        problem.input_state == DenseReductionInputState::OriginalInput &&
        (context.rk.original_input_dtype == DataType::FP8_E4M3 ||
         context.rk.original_input_dtype == DataType::FP8_E5M2)) {
        if (!context.original_value_operation)
            throw std::logic_error("Retained FP8 RK production requires original VALUE semantics");
        if (!problem.final_output.produce_value || problem.final_output.produce_index)
            throw std::logic_error("Retained FP8 RK production requires VALUE-only output");
        // The bound selector, typed first-stage kernels, and normal FP32
        // continuation already support every VALUE operation and the floating
        // output storage dtypes allowed by CubReduction. In particular, do not
        // gate this original-input path to SUM/FP32: Complete applies the
        // finalizer and converts once, while staged passes leave FP32 partials
        // for the independently planned final reduction.
        if (context.tuning_stream == nullptr || context.original_reduction_size == 0)
            throw std::logic_error("Retained FP8 RK production requires a tuning stream and original reduction size");
        const auto selected = Fp8RK::selectCachedIsolatedRKSite(
            problem, site, *context.original_value_operation, context.original_reduction_size,
            context.output_scale, context.rk.original_input_dtype, *context.tuning_stream,
            context.max_workspace_bytes);
        return makeRetainedFp8RKPhysicalCandidate(problem, site, selected.physical, context);
    }
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

}  // namespace ThorImplementation::CubReductionInternal
