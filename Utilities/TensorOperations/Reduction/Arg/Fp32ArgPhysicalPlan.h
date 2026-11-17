#pragma once

// One FP32 ARG pass for the planned dense ARG executor (the FP32 routing, DenseReductionPhysicalSelector.h). Every FP32
// ARG pass is Complete to the planner: a staged RK pass keeps its packed (original) or wide (carried) partials in the
// pass's scratch and finishes them in the same call, and a Split R / KR shape finishes in the same call too. The pass
// reads FP32 values, and at a carried site (an ARG intermediate aggregate) the carried index of the earlier passes; it
// writes the public outputs on the final pass, else the FP32 value + carried index intermediate. Index: original passes
// row * the run's original-domain stride; carried passes carried + row * that stride (0 for an intermediate run).
// With input_dtype FP16 / BF16 it is a 2-byte ARG pass at an original FP16 / BF16 site (never carried): the 2-byte RK
// ARG twins (RK::F16::launchF16RKArgPass) or the 2-byte R / KR ARG shapes (launchF16KRArgSegments), with the same
// outputs and intermediates (the FP16 / BF16 routing).

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/TensorOperations/Reduction/KR/KRSegments.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation::ReductionInternal {

struct Fp32ArgPhysicalPlan final : DenseReductionPhysicalPlan {
    bool kr = false;       // an R / KR pass (shape) rather than an RK pass (identity)
    bool carried = false;  // reads a carried index (an intermediate site)
    DataType input_dtype = DataType::FP32;  // FP16 / BF16 for a 2-byte original pass (never carried)
    RK::PhysicalKernel identity{};  // RK: the FP32 RK ARG (or carried ARG) pass identity
    KR::KRSegmentShape shape{};  // R / KR: the ARG shape, in bytes (4 per FP32 element, 2 per FP16 / BF16)
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
                                                                    const RK::PhysicalKernel& identity,
                                                                    bool carried);

/** The planner candidate for an R / KR ARG site (`carried` as above) running `shape`. */
[[nodiscard]] DenseReductionCandidate makeRetainedFp32KRArgCandidate(const DenseReductionProblem& problem,
                                                                    DenseReductionSite site,
                                                                    const KR::KRSegmentShape& shape,
                                                                    bool carried);

/** The planner candidate for an original FP16 / BF16 RK ARG site running the 2-byte ARG pass `identity`
 *  (Reduction/RK/F16/F16RKArgCandidates.h). */
[[nodiscard]] DenseReductionCandidate makeRetainedF16RKArgCandidate(const DenseReductionProblem& problem,
                                                                   DenseReductionSite site,
                                                                   const RK::PhysicalKernel& identity,
                                                                   DataType input_dtype);

/** The planner candidate for an original FP16 / BF16 R / KR ARG site running the 2-byte ARG `shape`. */
[[nodiscard]] DenseReductionCandidate makeRetainedF16KRArgCandidate(const DenseReductionProblem& problem,
                                                                   DenseReductionSite site,
                                                                   const KR::KRSegmentShape& shape,
                                                                   DataType input_dtype);

/** Runs the pass: `carried_index` is the carried index tensor at a carried site (else null); `scratch` holds
 *  plan.scratch_bytes (null for none, else 256-byte aligned). */
void launchRetainedFp32ArgPlan(const Fp32ArgPhysicalPlan& plan,
                               const StridedSpan& input,
                               const StridedSpan* carried_index,
                               const StridedSpan* value_output,
                               const StridedSpan* index_output,
                               const ReductionGeometry& geometry,
                               void* scratch,
                               bool maximum,
                               uint64_t domain_stride,
                               Stream& stream);

/** "fp32_arg_rk:aligned_k_p16_cta64_shards4", "fp32_carried_arg_kr:g8", ...: the pass for diagnostics. A 2-byte pass is
 *  "f16_arg_rk:..." / "f16_arg_kr:...". */
[[nodiscard]] std::string retainedFp32ArgPassName(const Fp32ArgPhysicalPlan& plan);

}  // namespace ThorImplementation::ReductionInternal
