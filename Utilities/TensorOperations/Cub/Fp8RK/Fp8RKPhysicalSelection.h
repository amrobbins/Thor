#pragma once

#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <cstdint>

class Stream;
namespace ThorImplementation { enum class CubReductionOp : uint8_t; }

namespace ThorImplementation::CubReductionInternal::Fp8RK {

// The descriptor-only, cached production answer is ONE FP8 RK physical
// first-stage recipe. Selection requires no caller-owned Tensor or workspace;
// only a cache miss constructs temporary benchmark resources.
// 1378: `input_alignment_bytes` (the original input's StridedSpan::alignmentBytes(16)) keeps the passes that read at
// that alignment (fp8RKPassInputAlignment) and is part of the cache key, as for FP32 (1222) and FP16 / BF16 (1254).
// 1379: PacketComplete reads any address (any R, any K), so every site has a pass at every alignment.
[[nodiscard]] Fp8RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    CubReductionOp operation,
    uint64_t original_reduction_size,
    float output_scale,
    DataType input_dtype,
    const Stream& stream,
    uint64_t workspace_budget_bytes,
    CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular,
    uint64_t input_alignment_bytes = 16);

// 1142: the cached FP8 ARG selection (Fp8RKStampAutotuner.h autotuneArgRKSite) at an original FP8 ARG RK site:
// maximum is argmax. Keyed like VALUE (the ARG problem's aggregate, outputs and runs differ) and on the depth.
[[nodiscard]] Fp8RKSelection selectCachedArgRKSite(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    bool maximum,
    uint64_t original_reduction_size,
    DataType input_dtype,
    const Stream& stream,
    uint64_t workspace_budget_bytes,
    CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular,
    uint64_t input_alignment_bytes = 16);  // 1379: as VALUE's (retainedFp8ArgPassInputAlignment)

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
