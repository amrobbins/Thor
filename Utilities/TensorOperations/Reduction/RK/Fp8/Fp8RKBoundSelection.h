#pragma once

#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKSelector.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalKernels.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ThorImplementation::ReductionInternal::RK::Fp8 {

// This is a set of compiled FIRST physical stages. Cache-miss measurement times their hypothetical selected FP32
// continuation; production stamps only the selected first stage and lets the ordinary planner choose later passes.
struct BoundSelectionRequest {
    uint64_t outer = 0, r = 0, k = 0;
    uint32_t sm_count = 0;
    DataType input_dtype = DataType::FP8_E4M3;
    DataType complete_output_dtype = DataType::FP8_E4M3;
    ReductionStageSemantics semantics{};
    uint64_t first_stage_scratch_budget_bytes = 0;
    // How many ranked first stages to return (reductionTuningPlan(depth).candidates).
    size_t max_candidates = 12;
    // The input base alignment (a power of two, at most 16). Only passes that read at it are listed
    // (rkPassInputAlignment); at 16 every pass is.
    uint64_t input_alignment_bytes = 16;
};

// RuntimeK physical proposals must be supplied by a compiled inventory, never
// inferred from arbitrary geometry. This bridge REQUERIES the exact occupancy
// and recomputes policy legality; the values supplied in proposal metadata
// cannot make an absent physical kernel become selectable.
[[nodiscard]] std::vector<PhysicalKernel> selectBoundFirstStages(
    const BoundSelectionRequest& request,
    std::span<const PhysicalKernel> registered_runtime_k);

// The COMPLETE callback-proven inventory of `request` (no depth limit), each kernel with its 1-based local rank in the
// mode-local order (a ranker feature). The candidate generator for ranked FP8 ARG selection.
struct BoundInventoryEntry {
    PhysicalKernel kernel;
    uint32_t local_rank = 0;
};
[[nodiscard]] std::vector<BoundInventoryEntry> enumerateBoundInventory(
    const BoundSelectionRequest& request,
    std::span<const PhysicalKernel> registered_runtime_k);

// The inverse of the shortlist's binding: the CandidatePlan whose ranker features (rankerFeatures) describe `kernel`,
// with the given 1-based local rank. Residency/scratch/work metadata are copied. Used for ARG candidates (identities
// from retainedFp8ArgPhysicalIdentity) and for training records.
[[nodiscard]] Fp8RKPolicy::CandidatePlan rankerPlanForKernel(const PhysicalKernel& kernel,
                                                                          uint32_t local_rank);

}  // namespace ThorImplementation::ReductionInternal::RK::Fp8
