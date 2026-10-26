#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <cstdint>

class Stream;
namespace ThorImplementation { enum class CubReductionOp : uint8_t; }

namespace ThorImplementation::CubReductionInternal::Fp8RK {

// The descriptor-only, cached production answer is ONE FP8 RK physical
// first-stage recipe. Selection requires no caller-owned Tensor or workspace;
// only a cache miss constructs temporary benchmark resources.
[[nodiscard]] Fp8RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    CubReductionOp operation,
    uint64_t original_reduction_size,
    float output_scale,
    DataType input_dtype,
    const Stream& stream,
    uint64_t workspace_budget_bytes);

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
