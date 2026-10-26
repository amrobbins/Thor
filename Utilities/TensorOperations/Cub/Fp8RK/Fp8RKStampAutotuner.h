#pragma once

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalSelection.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ThorImplementation { class Tensor; }

namespace ThorImplementation::CubReductionInternal::Fp8RK {

// Explicit cache-bypassing measurement entry point, useful for benchmarks.
// The returned recipe owns no stamp-local state or allocation.
[[nodiscard]] Fp8RKSelection autotuneIsolatedRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    CubReductionOp operation,
    uint64_t original_reduction_size,
    float output_scale,
    const Tensor& representative_input,
    const Stream& stream,
    uint64_t workspace_budget_bytes);

// Production selection: reuse the GEMM-style, bounded process-wide recipe
// cache and coalesce concurrent misses. This legacy Tensor overload validates
// the supplied original Tensor and, only on a cache miss, uses its contents
// as the representative input. It creates the same key as the descriptor API.
[[nodiscard]] Fp8RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    CubReductionOp operation,
    uint64_t original_reduction_size,
    float output_scale,
    const Tensor& representative_input,
    const Stream& stream,
    uint64_t workspace_budget_bytes);

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
