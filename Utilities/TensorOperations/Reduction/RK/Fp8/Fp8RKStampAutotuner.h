#pragma once

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"

#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKPhysicalSelection.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ThorImplementation { class Tensor; }

namespace ThorImplementation::ReductionInternal::RK::Fp8 {

// Explicit cache-bypassing measurement entry point, useful for benchmarks.
// The returned recipe owns no stamp-local state or allocation.
[[nodiscard]] RKSelection autotuneIsolatedRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    ReductionOp operation,
    uint64_t original_reduction_size,
    float output_scale,
    const StridedSpan& representative_input,
    const Stream& stream,
    uint64_t workspace_budget_bytes,
    ReductionTuningDepth tuning_depth = ReductionTuningDepth::Regular,
    uint64_t input_alignment_bytes = 16);

// Production selection: reuse the GEMM-style, bounded process-wide recipe cache and coalesce concurrent misses. This
// Tensor overload validates the supplied original Tensor and, only on a cache miss, uses its contents as the
// representative input. It creates the same key as the descriptor API.
[[nodiscard]] RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    ReductionOp operation,
    uint64_t original_reduction_size,
    float output_scale,
    const StridedSpan& representative_input,
    const Stream& stream,
    uint64_t workspace_budget_bytes,
    ReductionTuningDepth tuning_depth = ReductionTuningDepth::Regular,
    uint64_t input_alignment_bytes = 16);

// FP8 ARG's own ranked, autotuned selection at an original FP8 ARG RK site (benchmarks may call it directly).
// The recipe's `physical` is the winner's ARG identity (retainedFp8ArgPlanForIdentity rebuilds the pass);
// candidate_count 0 means no ARG pass is legal there.
[[nodiscard]] RKSelection autotuneArgRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    bool maximum,
    uint64_t original_reduction_size,
    const StridedSpan& representative_input,
    const Stream& stream,
    uint64_t workspace_budget_bytes,
    ReductionTuningDepth tuning_depth = ReductionTuningDepth::Regular,
    uint64_t input_alignment_bytes = 16);

} // namespace ThorImplementation::ReductionInternal::RK::Fp8
