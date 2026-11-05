#pragma once

#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"

#include <limits>
#include <optional>

class Stream;
namespace ThorImplementation { enum class CubReductionOp : uint8_t; }

namespace ThorImplementation::CubReductionInternal {

/**
 * Stamp-time facts shared by the ordinary planner and FP8 cache-miss FP32 lookahead.
 * No Tensor, stream, executable, or completed-plan ownership is part of the physical selection request.
 *
 * The optional proposal hook is for deterministic host tests. Production leaves it null; any replacement is still
 * subject to the ordinary planner's transition validation when used by planDenseReduction().
 */
struct DensePhysicalSelectionContext {
    using ProposalOverride = DenseReductionCandidate (*)(const DenseReductionProblem& problem,
                                                          DenseReductionSite site,
                                                          DenseReducerFamily family,
                                                          const DenseRKPlanningContext& rk,
                                                          void* user_context);

    DenseRKPlanningContext rk;
    // Non-owning stamp-time tuning information for future domain-local autotuners. Existing FP32 policy does not
    // consult it. Never retain the pointer in a physical selection or a cache entry.
    const Stream* tuning_stream = nullptr;
    uint32_t gpu_ordinal = 0;
    // Reserved for domain-local tuning; today's production FP32 policy preserves its existing workspace behavior.
    uint64_t max_workspace_bytes = std::numeric_limits<uint64_t>::max();
    // Original public VALUE semantics required by the retained FP8 RK selector.
    // Optional for FP32 selection and host-injected selectors. The total
    // reduction divisor cannot be inferred from a nominated axis alone.
    std::optional<CubReductionOp> original_value_operation;
    uint64_t original_reduction_size = 0;
    float output_scale = 1.0f;
    // 1141: retained FP8 RK autotuning depth (the reduction's CubReductionTuningDepth).
    CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular;
    // 1222: the original input's base alignment in bytes (Tensor::alignmentBytes(16)); descriptor-only queries assume
    // 16. The retained FP32 RK selectors keep only the passes that tolerate it at original-input sites (intermediates
    // are fresh, aligned allocations).
    uint64_t original_input_alignment_bytes = 16;
    ProposalOverride proposal_override = nullptr;
    void* proposal_override_context = nullptr;
};

/**
 * The authoritative selection of ONE physical pass for a nominated site. The returned next_problem and next_bytes
 * describe the pass's actual output, not a hypothetical completed reduction or a sequence of required continuations.
 * The result owns only an immutable physical-plan description; the caller handles all stamping and allocation.
 *
 * Original-input FP8 RK VALUE is dispatched to the retained domain-local cached physical selector;
 * FP32 and other domains retain their existing production physical-selection policy.
 * Unsupported retained semantics fail explicitly, never falling back to legacy FP8 RK.
 */
[[nodiscard]] DenseReductionCandidate selectDensePhysicalPass(const DenseReductionProblem& problem,
                                                               DenseReductionSite site,
                                                               const DensePhysicalSelectionContext& context);

/**
 * 1220 (plans/Thor_FP32_Dense_Reduction_Plan, phase 8): the retained FP32 routing. FP32 VALUE sites (original FP32
 * input, and the FP32 intermediates of every dtype's earlier passes) are selected by the retained FP32 selectors
 * (Fp32RK::selectCachedValueSite at RK sites, Fp32KR::selectCachedValueSite at R / KR sites) and run their retained
 * passes, with no fallback: a site they cannot cover is a planning error. The context must carry the tuning stream,
 * the original operation and the original reduction size. 1221: and FP32 ARG sites: original FP32 input
 * (Fp32RK / Fp32KR::selectCachedArgSite) and carried-index intermediates of any dtype's earlier ARG passes
 * (selectCachedCarriedArgSite); the planned dense ARG executor then runs FP32 ARG input too.
 *
 * 1225 / 1227 (phase 9): unconditional; the switch that ran the replaced paths is gone. The ordinary families
 * (ReducersDenseR / KR / RK) are never asked for an FP32 site by selectDensePhysicalPass; only a test's
 * proposal_override can still reach them with one.
 */

}  // namespace ThorImplementation::CubReductionInternal
