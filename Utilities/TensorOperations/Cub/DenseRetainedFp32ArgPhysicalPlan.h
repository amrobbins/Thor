#pragma once

// 1221: one retained FP32 ARG pass for the planned dense ARG executor (plans/Thor_FP32_Dense_Reduction_Plan, phase 8;
// the retained FP32 routing, DenseReductionPhysicalSelector.h). Every FP32 ARG pass is Complete to the planner: a
// staged RK pass keeps its packed (original) or wide (carried) partials in the pass's scratch and finishes them in the
// same call, and a Split R / KR shape finishes in the same call too. The pass reads FP32 values, and at a carried site
// (an ARG intermediate aggregate) the carried index of the earlier passes; it writes the public outputs on the final
// pass, else the FP32 value + carried index intermediate. Index: original passes row * the run's original-domain
// stride; carried passes carried + row * that stride (0 for an intermediate run).

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

struct DenseRetainedFp32ArgPhysicalPlan final : DenseReductionPhysicalPlan {
    bool kr = false;       // an R / KR pass (shape) rather than an RK pass (identity)
    bool carried = false;  // reads a carried index (an intermediate site)
    Fp8RK::PhysicalKernel identity{};  // RK: the FP32 RK ARG (or carried ARG) pass identity
    Fp8KR::Fp8KRSegmentShape shape{};  // R / KR: the FP32 ARG shape, in bytes (4 per element)
    std::vector<uint64_t> input_dimensions;  // RK: {outer, r, inner}; R / KR: the site's current runs
    uint32_t reduction_axis = 1;             // RK: 1; R / KR: the last run
    uint64_t outer_size = 0;                 // RK outer; R / KR rows
    uint64_t reduction_size = 0;             // RK R; R / KR row elements
    uint64_t inner_size = 1;                 // RK K; R / KR 1
    uint64_t scratch_bytes = 0;              // the pass's partial scratch (packed or wide), 0 for none
};

/** The planner candidate for an RK ARG site (`carried`: a carried-index intermediate site) running `identity`. */
[[nodiscard]] DenseReductionCandidate makeRetainedFp32RKArgCandidate(const DenseReductionProblem& problem,
                                                                    DenseReductionSite site,
                                                                    const Fp8RK::PhysicalKernel& identity,
                                                                    bool carried);

/** The planner candidate for an R / KR ARG site (`carried` as above) running `shape`. */
[[nodiscard]] DenseReductionCandidate makeRetainedFp32KRArgCandidate(const DenseReductionProblem& problem,
                                                                    DenseReductionSite site,
                                                                    const Fp8KR::Fp8KRSegmentShape& shape,
                                                                    bool carried);

/** Runs the pass: `carried_index` is the carried index tensor at a carried site (else null); `scratch` holds
 *  plan.scratch_bytes (null for none, else 256-byte aligned). */
void launchRetainedFp32ArgPlan(const DenseRetainedFp32ArgPhysicalPlan& plan,
                               const Tensor& input,
                               const Tensor* carried_index,
                               Tensor* value_output,
                               Tensor* index_output,
                               const CubReductionGeometry& geometry,
                               void* scratch,
                               bool maximum,
                               uint64_t domain_stride,
                               Stream& stream);

/** "fp32_arg_rk:aligned_k_p16_cta64_shards4", "fp32_carried_arg_kr:g8", ...: the pass for diagnostics. */
[[nodiscard]] std::string retainedFp32ArgPassName(const DenseRetainedFp32ArgPhysicalPlan& plan);

}  // namespace ThorImplementation::CubReductionInternal
