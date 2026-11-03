#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"

#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRStampAutotuner.h"
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
            context.max_workspace_bytes, context.tuning_depth);
        return makeRetainedFp8RKPhysicalCandidate(problem, site, selected.physical, context);
    }
    // 1142: original FP8 RK ARG ranks and autotunes its own retained ARG passes (Fp8RK::selectCachedArgRKSite).
    // context.original_value_operation names MIN (argmin) or MAX (argmax).
    if (family == DenseReducerFamily::RK &&
        problem.aggregate_kind == DenseReductionAggregateKind::Arg &&
        problem.input_state == DenseReductionInputState::OriginalInput &&
        problem.runs[site.run_index].kind == DenseReductionRunKind::OriginalReduction &&
        (context.rk.original_input_dtype == DataType::FP8_E4M3 ||
         context.rk.original_input_dtype == DataType::FP8_E5M2)) {
        // 1173: original FP8 RK ARG has no generic fallback (the direct tiled ARG kernels are not compiled for FP8
        // input since 1119); a site the retained ARG passes cannot cover is a planning error.
        if (!context.original_value_operation ||
            (*context.original_value_operation != CubReductionOp::Min &&
             *context.original_value_operation != CubReductionOp::Max))
            throw std::logic_error("Retained FP8 ARG selection requires MIN (argmin) or MAX (argmax)");
        if (context.tuning_stream == nullptr || context.original_reduction_size == 0)
            throw std::logic_error("Retained FP8 ARG selection requires a tuning stream and original reduction size");
        const auto selected = Fp8RK::selectCachedArgRKSite(
            problem, site, *context.original_value_operation == CubReductionOp::Max, context.original_reduction_size,
            context.rk.original_input_dtype, *context.tuning_stream, context.max_workspace_bytes, context.tuning_depth);
        if (selected.candidate_count == 0)
            throw std::logic_error("Retained FP8 ARG selection found no legal retained ARG pass for this RK site "
                                   "(beyond the retained kernels' grid limits); FP8 has no generic ARG fallback");
        uint64_t outer = 1, inner = 1;
        for (size_t i = 0; i < problem.runs.size(); ++i) {
            if (i < site.run_index) outer *= problem.runs[i].extent;
            if (i > site.run_index) inner *= problem.runs[i].extent;
        }
        auto plan = retainedFp8ArgPlanForIdentity(selected.physical, context.rk.original_input_dtype, outer,
                                                  problem.runs[site.run_index].extent, inner);
        if (!plan) throw std::logic_error("Cached FP8 ARG selection names an illegal ARG pass");
        return makeRetainedFp8ArgCandidateForPlan(problem, site, std::move(*plan));
    }
    // 1146-1148: original FP8 R / KR sites autotune the retained R / KR passes only (1148 removed the direct CUB pass
    // from the candidates). 1173: and nothing else. An original FP8 R / KR site without stamp-time tuning information,
    // with semantics the retained kernels do not compile, or with an output combination they do not produce is a
    // planning error, never the direct (CUB) proposal below. (proposal_override, above, is how host tests plan
    // without tuning.)
    if ((family == DenseReducerFamily::R || family == DenseReducerFamily::KR) &&
        isOriginalFp8KRSite(problem, site, context.rk.original_input_dtype)) {
        if (context.tuning_stream == nullptr || !context.original_value_operation ||
            context.original_reduction_size == 0)
            throw std::logic_error("Retained FP8 R/KR selection requires a tuning stream, the original operation and "
                                   "the original reduction size");
        const DataType input_dtype = context.rk.original_input_dtype;
        if (problem.aggregate_kind == DenseReductionAggregateKind::Value) {
            if (!problem.final_output.produce_value || problem.final_output.produce_index)
                throw std::logic_error("Retained FP8 R/KR VALUE selection requires VALUE-only output");
            const auto selected = Fp8KR::selectCachedValueSite(
                problem, site, *context.original_value_operation, context.original_reduction_size, context.output_scale,
                input_dtype, context.rk.multiprocessors, *context.tuning_stream, context.tuning_depth);
            if (!selected.shape)
                throw std::logic_error("Retained FP8 R/KR VALUE kernels do not compile this operation's semantics; "
                                       "FP8 has no direct (CUB) fallback");
            return makeRetainedFp8KRValueCandidate(problem, site, *selected.shape, input_dtype);
        }
        if (problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction ||
            (*context.original_value_operation != CubReductionOp::Min &&
             *context.original_value_operation != CubReductionOp::Max))
            throw std::logic_error("Retained FP8 R/KR ARG selection requires an original run and MIN (argmin) or MAX "
                                   "(argmax)");
        const auto selected = Fp8KR::selectCachedArgSite(
            problem, site, *context.original_value_operation == CubReductionOp::Max, input_dtype,
            context.rk.multiprocessors, *context.tuning_stream, context.tuning_depth);
        if (!selected.shape) throw std::logic_error("Retained FP8 R/KR ARG selection returned no shape");
        return makeRetainedFp8KRArgCandidate(problem, site, *selected.shape, input_dtype);
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
