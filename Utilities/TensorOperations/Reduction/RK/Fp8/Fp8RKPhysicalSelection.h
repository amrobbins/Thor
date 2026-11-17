#pragma once

#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Reduction/RK/RKSelection.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"

#include <cstdint>

class Stream;
namespace ThorImplementation { enum class ReductionOp : uint8_t; }

namespace ThorImplementation::ReductionInternal::RK::Fp8 {

// The descriptor-only, cached production answer is ONE FP8 RK physical
// first-stage recipe. Selection requires no caller-owned Tensor or workspace;
// only a cache miss constructs temporary benchmark resources.
// `input_alignment_bytes` (the original input's StridedSpan::alignmentBytes(16)) keeps the passes that read at that
// alignment (rkPassInputAlignment) and is part of the cache key, as for FP32 and FP16 / BF16.
// PacketComplete reads any address (any R, any K), so every site has a pass at every alignment.
[[nodiscard]] RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    ReductionOp operation,
    uint64_t original_reduction_size,
    float output_scale,
    DataType input_dtype,
    const Stream& stream,
    uint64_t workspace_budget_bytes,
    ReductionTuningDepth tuning_depth = ReductionTuningDepth::Regular,
    uint64_t input_alignment_bytes = 16);

// The cached FP8 ARG selection (Fp8RKStampAutotuner.h autotuneArgRKSite) at an original FP8 ARG RK site: maximum is
// argmax. Keyed like VALUE (the ARG problem's aggregate, outputs and runs differ) and on the depth.
[[nodiscard]] RKSelection selectCachedArgRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    bool maximum,
    uint64_t original_reduction_size,
    DataType input_dtype,
    const Stream& stream,
    uint64_t workspace_budget_bytes,
    ReductionTuningDepth tuning_depth = ReductionTuningDepth::Regular,
    uint64_t input_alignment_bytes = 16);  // as VALUE's (retainedFp8ArgPassInputAlignment)

}  // namespace ThorImplementation::ReductionInternal::RK::Fp8
