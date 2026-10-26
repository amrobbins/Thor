#pragma once

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

// This is a set of retained FIRST physical stages. Cache-miss measurement
// times their hypothetical selected FP32 continuation; production stamps only
// the selected first stage and lets the ordinary planner choose later passes.
struct BoundSelectionRequest {
    uint64_t outer = 0, r = 0, k = 0;
    uint32_t sm_count = 0;
    DataType input_dtype = DataType::FP8_E4M3;
    DataType complete_output_dtype = DataType::FP8_E4M3;
    CubReductionStageSemantics semantics{};
    uint64_t first_stage_scratch_budget_bytes = 0;
};

// RuntimeK physical proposals must be supplied by a compiled inventory, never
// inferred from arbitrary geometry. This bridge REQUERIES the exact occupancy
// and recomputes policy legality; the values supplied in proposal metadata
// cannot make an absent physical kernel become selectable.
[[nodiscard]] std::vector<PhysicalKernel> selectBoundFirstStages(
    const BoundSelectionRequest& request,
    std::span<const PhysicalKernel> registered_runtime_k);

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
