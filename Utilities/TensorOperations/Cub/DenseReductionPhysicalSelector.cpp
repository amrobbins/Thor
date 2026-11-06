#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"

#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp32ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/F16KR/F16KRStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp32KR/Fp32KRStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalSelection.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <mutex>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal {
namespace {

// 1222: the first FP32 RK site whose original input is not 16-byte aligned prints one warning per process: it runs
// the passes that tolerate the alignment (the rotated RuntimeK pass, AlignedK at an 8-byte packet), which can be
// slower.
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

// 1220: the site reads FP32: original FP32 input, or any dtype's FP32 intermediate.
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
        const auto selected = Fp32RK::selectCachedValueSite(
            problem, site, *context.original_value_operation, context.original_reduction_size, context.output_scale,
            *context.tuning_stream, context.max_workspace_bytes, context.tuning_depth,
            context.original_input_alignment_bytes);
        return makeRetainedFp32RKValueCandidate(problem, site, selected.physical);
    }
    const auto selected = Fp32KR::selectCachedValueSite(
        problem, site, *context.original_value_operation, context.original_reduction_size, context.output_scale,
        context.rk.multiprocessors, *context.tuning_stream, context.tuning_depth);
    if (!selected.shape)
        throw std::logic_error("Retained FP32 R/KR VALUE kernels do not compile this operation's semantics");
    return makeRetainedFp32KRValueCandidate(problem, site, *selected.shape);
}

// 1221: an FP32 ARG site: original FP32 input (the reduced run original) or a carried-index intermediate.
DenseReductionCandidate selectRetainedFp32ArgPass(const DenseReductionProblem& problem, DenseReductionSite site,
                                                  DenseReducerFamily family,
                                                  const DensePhysicalSelectionContext& context) {
    if (context.tuning_stream == nullptr || !context.original_value_operation || context.original_reduction_size == 0 ||
        (*context.original_value_operation != CubReductionOp::Min &&
         *context.original_value_operation != CubReductionOp::Max))
        throw std::logic_error("Retained FP32 ARG selection requires a tuning stream, MIN (argmin) or MAX (argmax) "
                               "and the original reduction size");
    const bool maximum = *context.original_value_operation == CubReductionOp::Max;
    const bool carried = problem.input_state == DenseReductionInputState::IntermediateAggregate;
    const Stream& stream = *context.tuning_stream;
    if (family == DenseReducerFamily::RK) {
        warnMisalignedFp32Input(problem, context);
        const auto selected =
            carried ? Fp32RK::selectCachedCarriedArgSite(problem, site, maximum, context.original_reduction_size,
                                                         stream, context.max_workspace_bytes, context.tuning_depth)
                    : Fp32RK::selectCachedArgSite(problem, site, maximum, context.original_reduction_size, stream,
                                                  context.max_workspace_bytes, context.tuning_depth,
                                                  context.original_input_alignment_bytes);
        if (selected.candidate_count == 0)
            throw std::logic_error("Retained FP32 RK ARG selection found no legal pass for this site");
        return makeRetainedFp32RKArgCandidate(problem, site, selected.physical, carried);
    }
    const auto selected =
        carried ? Fp32KR::selectCachedCarriedArgSite(problem, site, maximum, context.rk.multiprocessors, stream,
                                                     context.tuning_depth)
                : Fp32KR::selectCachedArgSite(problem, site, maximum, context.rk.multiprocessors, stream,
                                              context.tuning_depth);
    if (!selected.shape) throw std::logic_error("Retained FP32 R/KR ARG selection returned no shape");
    return makeRetainedFp32KRArgCandidate(problem, site, *selected.shape, carried);
}

// 1254: as warnMisalignedFp32Input, for an original FP16 / BF16 RK site.
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

// 1254: an original FP16 / BF16 site (the retained FP16 / BF16 routing's domain).
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
            const auto selected = F16RK::selectCachedValueSite(
                problem, site, dtype, *context.original_value_operation, context.original_reduction_size,
                context.output_scale, stream, context.max_workspace_bytes, context.tuning_depth,
                std::max<uint64_t>(context.original_input_alignment_bytes, 2));
            return makeRetainedF16RKValueCandidate(problem, site, selected.physical, dtype);
        }
        const auto selected = F16KR::selectCachedValueSite(
            problem, site, dtype, *context.original_value_operation, context.original_reduction_size,
            context.output_scale, context.rk.multiprocessors, stream, context.tuning_depth);
        if (!selected.shape)
            throw std::logic_error("Retained FP16 / BF16 R/KR VALUE kernels do not compile this operation's semantics");
        return makeRetainedF16KRValueCandidate(problem, site, *selected.shape, dtype);
    }
    if (*context.original_value_operation != CubReductionOp::Min &&
        *context.original_value_operation != CubReductionOp::Max)
        throw std::logic_error("Retained FP16 / BF16 ARG selection requires MIN (argmin) or MAX (argmax)");
    const bool maximum = *context.original_value_operation == CubReductionOp::Max;
    if (family == DenseReducerFamily::RK) {
        warnMisalignedF16Input(context);
        const auto selected = F16RK::selectCachedArgSite(
            problem, site, dtype, maximum, context.original_reduction_size, stream, context.max_workspace_bytes,
            context.tuning_depth, std::max<uint64_t>(context.original_input_alignment_bytes, 2));
        if (selected.candidate_count == 0)
            throw std::logic_error("Retained FP16 / BF16 RK ARG selection found no legal pass for this site");
        return makeRetainedF16RKArgCandidate(problem, site, selected.physical, dtype);
    }
    const auto selected =
        F16KR::selectCachedArgSite(problem, site, dtype, maximum, context.rk.multiprocessors, stream,
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
    // 1220 / 1221 / 1227: every site that reads FP32 (original FP32 input, or any dtype's FP32 intermediate), VALUE and
    // ARG, is selected by the retained FP32 selectors, with no fallback to the ordinary families.
    if (siteReadsFp32(problem, context)) {
        return problem.aggregate_kind == DenseReductionAggregateKind::Value
                   ? selectRetainedFp32ValuePass(problem, site, family, context)
                   : selectRetainedFp32ArgPass(problem, site, family, context);
    }
    // 1254 / 1257: every original FP16 / BF16 site, VALUE and ARG, is selected by the retained FP16 / BF16 selectors,
    // with no fallback to the ordinary families.
    if (isOriginalF16Site(problem, context)) {
        return selectRetainedF16Pass(problem, site, family, context);
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
    // 1258 (phase 10): the legacy ordinary families are gone; every supported dtype's sites are covered above.
    throw std::logic_error("Dense physical selection has no retained selector for this site (input dtype " +
                           std::to_string(static_cast<int>(context.rk.original_input_dtype)) + ")");
}

}  // namespace ThorImplementation::CubReductionInternal
