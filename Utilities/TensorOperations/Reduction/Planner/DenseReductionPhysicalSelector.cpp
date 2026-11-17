#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPhysicalSelector.h"

#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/KR/KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp32ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/KR/F16/F16KRStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/RK/F16/F16RKStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/KR/Fp32/Fp32KRStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp32/Fp32RKStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/KR/Fp8/Fp8KRStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKPhysicalSelection.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <mutex>
#include <stdexcept>

namespace ThorImplementation::ReductionInternal {
namespace {

// The first FP32 RK site whose original input is not 16-byte aligned prints one warning per process: it runs the passes
// that tolerate the alignment (the rotated RuntimeK pass, AlignedK at an 8-byte packet), which can be slower.
void warnMisalignedFp32Input(const DenseReductionProblem& problem, const DensePhysicalSelectionContext& context) {
    if (problem.input_state != DenseReductionInputState::OriginalInput || context.original_input_alignment_bytes >= 16)
        return;
    static std::once_flag once;
    std::call_once(once, [&] {
        std::cerr << "Thor: an FP32 reduction input starts on a " << context.original_input_alignment_bytes
                  << "-byte boundary, not a 16-byte one; its RK passes run on alignment-tolerant kernels, which can be "
                     "slower. Allocate reduction inputs (and views of them) on 16-byte boundaries for the fastest "
                     "kernels. (Printed once per process.)"
                  << std::endl;
    });
}

// The site reads FP32: original FP32 input, or any dtype's FP32 intermediate.
bool siteReadsFp32(const DenseReductionProblem& problem, const DensePhysicalSelectionContext& context) {
    return problem.input_state == DenseReductionInputState::IntermediateAggregate ||
           context.rk.original_input_dtype == DataType::FP32;
}

DenseReductionCandidate selectRetainedFp32ValuePass(const DenseReductionProblem& problem, DenseReductionSite site,
                                                    DenseReducerFamily family,
                                                    const DensePhysicalSelectionContext& context) {
    if (context.tuning_stream == nullptr || !context.original_value_operation || context.original_reduction_size == 0)
        throw std::logic_error("Retained FP32 VALUE selection requires a tuning stream, the original operation and "
                               "the original reduction size");
    if (!problem.final_output.produce_value || problem.final_output.produce_index)
        throw std::logic_error("Retained FP32 VALUE selection requires VALUE-only output");
    if (family == DenseReducerFamily::RK) {
        warnMisalignedFp32Input(problem, context);
        const auto selected = RK::Fp32::selectCachedValueSite(
            problem, site, *context.original_value_operation, context.original_reduction_size, context.output_scale,
            *context.tuning_stream, context.max_workspace_bytes, context.tuning_depth,
            context.original_input_alignment_bytes);
        return makeRetainedFp32RKValueCandidate(problem, site, selected.physical);
    }
    const auto selected = KR::Fp32::selectCachedValueSite(
        problem, site, *context.original_value_operation, context.original_reduction_size, context.output_scale,
        context.rk.multiprocessors, *context.tuning_stream, context.tuning_depth);
    if (!selected.shape)
        throw std::logic_error("Retained FP32 R/KR VALUE kernels do not compile this operation's semantics");
    return makeRetainedFp32KRValueCandidate(problem, site, *selected.shape);
}

// An FP32 ARG site: original FP32 input (the reduced run original) or a carried-index intermediate.
DenseReductionCandidate selectRetainedFp32ArgPass(const DenseReductionProblem& problem, DenseReductionSite site,
                                                  DenseReducerFamily family,
                                                  const DensePhysicalSelectionContext& context) {
    if (context.tuning_stream == nullptr || !context.original_value_operation || context.original_reduction_size == 0 ||
        (*context.original_value_operation != ReductionOp::Min &&
         *context.original_value_operation != ReductionOp::Max))
        throw std::logic_error("Retained FP32 ARG selection requires a tuning stream, MIN (argmin) or MAX (argmax) "
                               "and the original reduction size");
    const bool maximum = *context.original_value_operation == ReductionOp::Max;
    const bool carried = problem.input_state == DenseReductionInputState::IntermediateAggregate;
    const Stream& stream = *context.tuning_stream;
    if (family == DenseReducerFamily::RK) {
        warnMisalignedFp32Input(problem, context);
        const auto selected =
            carried ? RK::Fp32::selectCachedCarriedArgSite(problem, site, maximum, context.original_reduction_size,
                                                           stream, context.max_workspace_bytes, context.tuning_depth)
                    : RK::Fp32::selectCachedArgSite(problem, site, maximum, context.original_reduction_size, stream,
                                                    context.max_workspace_bytes, context.tuning_depth,
                                                    context.original_input_alignment_bytes);
        if (selected.candidate_count == 0)
            throw std::logic_error("Retained FP32 RK ARG selection found no legal pass for this site");
        return makeRetainedFp32RKArgCandidate(problem, site, selected.physical, carried);
    }
    const auto selected =
        carried ? KR::Fp32::selectCachedCarriedArgSite(problem, site, maximum, context.rk.multiprocessors, stream,
                                                       context.tuning_depth)
                : KR::Fp32::selectCachedArgSite(problem, site, maximum, context.rk.multiprocessors, stream,
                                                context.tuning_depth);
    if (!selected.shape) throw std::logic_error("Retained FP32 R/KR ARG selection returned no shape");
    return makeRetainedFp32KRArgCandidate(problem, site, *selected.shape, carried);
}

// As warnMisalignedFp32Input, for an original FP16 / BF16 RK site.
void warnMisalignedF16Input(const DensePhysicalSelectionContext& context) {
    if (context.original_input_alignment_bytes >= 16) return;
    static std::once_flag once;
    std::call_once(once, [&] {
        std::cerr << "Thor: an FP16 / BF16 reduction input starts on a " << context.original_input_alignment_bytes
                  << "-byte boundary, not a 16-byte one; its RK passes run on alignment-tolerant kernels, which can be "
                     "slower. Allocate reduction inputs (and views of them) on 16-byte boundaries for the fastest "
                     "kernels. (Printed once per process.)"
                  << std::endl;
    });
}

// As warnMisalignedFp32Input, for an original FP8 RK site.
void warnMisalignedFp8Input(const DensePhysicalSelectionContext& context) {
    if (context.original_input_alignment_bytes >= 16) return;
    static std::once_flag once;
    std::call_once(once, [&] {
        std::cerr << "Thor: an FP8 reduction input starts on a " << context.original_input_alignment_bytes
                  << "-byte boundary, not a 16-byte one; its RK passes run on alignment-tolerant kernels, which can be "
                     "slower. Allocate reduction inputs (and views of them) on 16-byte boundaries for the fastest "
                     "kernels. (Printed once per process.)"
                  << std::endl;
    });
}

// An original FP16 / BF16 site (the FP16 / BF16 routing's domain).
bool isOriginalF16Site(const DenseReductionProblem& problem, const DensePhysicalSelectionContext& context) {
    return problem.input_state == DenseReductionInputState::OriginalInput &&
           (context.rk.original_input_dtype == DataType::FP16 || context.rk.original_input_dtype == DataType::BF16);
}

DenseReductionCandidate selectRetainedF16Pass(const DenseReductionProblem& problem, DenseReductionSite site,
                                              DenseReducerFamily family,
                                              const DensePhysicalSelectionContext& context) {
    if (context.tuning_stream == nullptr || !context.original_value_operation || context.original_reduction_size == 0)
        throw std::logic_error("Retained FP16 / BF16 selection requires a tuning stream, the original operation and "
                               "the original reduction size");
    const DataType dtype = context.rk.original_input_dtype;
    const Stream& stream = *context.tuning_stream;
    if (problem.aggregate_kind == DenseReductionAggregateKind::Value) {
        if (!problem.final_output.produce_value || problem.final_output.produce_index)
            throw std::logic_error("Retained FP16 / BF16 VALUE selection requires VALUE-only output");
        if (family == DenseReducerFamily::RK) {
            warnMisalignedF16Input(context);
            const auto selected = RK::F16::selectCachedValueSite(
                problem, site, dtype, *context.original_value_operation, context.original_reduction_size,
                context.output_scale, stream, context.max_workspace_bytes, context.tuning_depth,
                std::max<uint64_t>(context.original_input_alignment_bytes, 2));
            return makeRetainedF16RKValueCandidate(problem, site, selected.physical, dtype);
        }
        const auto selected = KR::F16::selectCachedValueSite(
            problem, site, dtype, *context.original_value_operation, context.original_reduction_size,
            context.output_scale, context.rk.multiprocessors, stream, context.tuning_depth);
        if (!selected.shape)
            throw std::logic_error("Retained FP16 / BF16 R/KR VALUE kernels do not compile this operation's semantics");
        return makeRetainedF16KRValueCandidate(problem, site, *selected.shape, dtype);
    }
    if (*context.original_value_operation != ReductionOp::Min &&
        *context.original_value_operation != ReductionOp::Max)
        throw std::logic_error("Retained FP16 / BF16 ARG selection requires MIN (argmin) or MAX (argmax)");
    const bool maximum = *context.original_value_operation == ReductionOp::Max;
    if (family == DenseReducerFamily::RK) {
        warnMisalignedF16Input(context);
        const auto selected = RK::F16::selectCachedArgSite(
            problem, site, dtype, maximum, context.original_reduction_size, stream, context.max_workspace_bytes,
            context.tuning_depth, std::max<uint64_t>(context.original_input_alignment_bytes, 2));
        if (selected.candidate_count == 0)
            throw std::logic_error("Retained FP16 / BF16 RK ARG selection found no legal pass for this site");
        return makeRetainedF16RKArgCandidate(problem, site, selected.physical, dtype);
    }
    const auto selected =
        KR::F16::selectCachedArgSite(problem, site, dtype, maximum, context.rk.multiprocessors, stream,
                                     context.tuning_depth);
    if (!selected.shape) throw std::logic_error("Retained FP16 / BF16 R/KR ARG selection returned no shape");
    return makeRetainedF16KRArgCandidate(problem, site, *selected.shape, dtype);
}

}  // namespace

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
    // Every site that reads FP32 (original FP32 input, or any dtype's FP32 intermediate), VALUE and ARG, is selected by
    // the FP32 selectors, with no fallback.
    if (siteReadsFp32(problem, context)) {
        return problem.aggregate_kind == DenseReductionAggregateKind::Value
                   ? selectRetainedFp32ValuePass(problem, site, family, context)
                   : selectRetainedFp32ArgPass(problem, site, family, context);
    }
    // Every original FP16 / BF16 site, VALUE and ARG, is selected by the FP16 / BF16 selectors, with no fallback.
    if (isOriginalF16Site(problem, context)) {
        return selectRetainedF16Pass(problem, site, family, context);
    }
    // Original FP8 RK VALUE has exactly one supported production selector: the cached physical-first-stage autotuner.
    // There is no fallback on missing semantics.
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
        // output storage dtypes allowed by Reducer. In particular, do not
        // gate this original-input path to SUM/FP32: Complete applies the
        // finalizer and converts once, while staged passes leave FP32 partials
        // for the independently planned final reduction.
        if (context.tuning_stream == nullptr || context.original_reduction_size == 0)
            throw std::logic_error("Retained FP8 RK production requires a tuning stream and original reduction size");
        // The passes that read at the input's alignment (RK::rkPassInputAlignment), as FP32 / FP16.
        warnMisalignedFp8Input(context);
        const auto selected = RK::Fp8::selectCachedIsolatedRKSite(
            problem, site, *context.original_value_operation, context.original_reduction_size,
            context.output_scale, context.rk.original_input_dtype, *context.tuning_stream,
            context.max_workspace_bytes, context.tuning_depth,
            std::max<uint64_t>(context.original_input_alignment_bytes, 1));
        return makeRetainedFp8RKPhysicalCandidate(problem, site, selected.physical, context);
    }
    // Original FP8 RK ARG ranks and autotunes its own ARG passes (RK::Fp8::selectCachedArgRKSite).
    // context.original_value_operation names MIN (argmin) or MAX (argmax).
    if (family == DenseReducerFamily::RK &&
        problem.aggregate_kind == DenseReductionAggregateKind::Arg &&
        problem.input_state == DenseReductionInputState::OriginalInput &&
        problem.runs[site.run_index].kind == DenseReductionRunKind::OriginalReduction &&
        (context.rk.original_input_dtype == DataType::FP8_E4M3 ||
         context.rk.original_input_dtype == DataType::FP8_E5M2)) {
        // Original FP8 RK ARG has no generic fallback; a site the ARG passes cannot cover is a planning error.
        if (!context.original_value_operation ||
            (*context.original_value_operation != ReductionOp::Min &&
             *context.original_value_operation != ReductionOp::Max))
            throw std::logic_error("Retained FP8 ARG selection requires MIN (argmin) or MAX (argmax)");
        if (context.tuning_stream == nullptr || context.original_reduction_size == 0)
            throw std::logic_error("Retained FP8 ARG selection requires a tuning stream and original reduction size");
        warnMisalignedFp8Input(context);
        const auto selected = RK::Fp8::selectCachedArgRKSite(
            problem, site, *context.original_value_operation == ReductionOp::Max, context.original_reduction_size,
            context.rk.original_input_dtype, *context.tuning_stream, context.max_workspace_bytes, context.tuning_depth,
            std::max<uint64_t>(context.original_input_alignment_bytes, 1));
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
    // Original FP8 R / KR sites autotune the R / KR passes only. An original FP8 R / KR site without stamp-time tuning
    // information, with semantics the kernels do not compile, or with an output combination they do not produce is a
    // planning error. (proposal_override, above, is how host tests plan without tuning.)
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
            const auto selected = KR::Fp8::selectCachedValueSite(
                problem, site, *context.original_value_operation, context.original_reduction_size, context.output_scale,
                input_dtype, context.rk.multiprocessors, *context.tuning_stream, context.tuning_depth);
            if (!selected.shape)
                throw std::logic_error("Retained FP8 R/KR VALUE kernels do not compile this operation's semantics; "
                                       "FP8 has no fallback");
            return makeRetainedFp8KRValueCandidate(problem, site, *selected.shape, input_dtype);
        }
        if (problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction ||
            (*context.original_value_operation != ReductionOp::Min &&
             *context.original_value_operation != ReductionOp::Max))
            throw std::logic_error("Retained FP8 R/KR ARG selection requires an original run and MIN (argmin) or MAX "
                                   "(argmax)");
        const auto selected = KR::Fp8::selectCachedArgSite(
            problem, site, *context.original_value_operation == ReductionOp::Max, input_dtype,
            context.rk.multiprocessors, *context.tuning_stream, context.tuning_depth);
        if (!selected.shape) throw std::logic_error("Retained FP8 R/KR ARG selection returned no shape");
        return makeRetainedFp8KRArgCandidate(problem, site, *selected.shape, input_dtype);
    }
    // Every supported dtype's sites are covered above.
    throw std::logic_error("Dense physical selection has no retained selector for this site (input dtype " +
                           std::to_string(static_cast<int>(context.rk.original_input_dtype)) + ")");
}

}  // namespace ThorImplementation::ReductionInternal
