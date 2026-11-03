#pragma once

// 1146-1148: stamp-time selection of retained FP8 R / KR passes. The candidates are fp8KRCandidateShapes (best guess
// first; 1160: the tuning depth keeps the ranker's first Quick 6 / Regular 12 / Ultra 12); 1148 removed the direct CUB
// pass. They are timed by the shared contest (ReductionAutotuning::autotune) with the depth's rounds and trim, end to
// end (a Split shape's finishing kernel included). Selections are cached per process and device, keyed on the site
// geometry, semantics, outputs and depth, in the one dense reduction selection cache (1171,
// DenseReductionSelectionCache.h, kind Fp8KR); a deeper cached selection answers a shallower request.

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ThorImplementation::CubReductionInternal::Fp8KR {

struct Fp8KRSelection final : AcceleratorBackendSelectionRecipeTag {
    std::optional<Fp8KRSegmentShape> shape;  // nullopt: semantics the retained kernels do not compile (1173: an error)
    size_t candidate_count = 0;
    std::string winner_name;
};

/** Original FP8 VALUE R / KR site. `operation` is the public operation; output_scale applies on the final pass. */
[[nodiscard]] Fp8KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                                   DenseReductionSite site,
                                                   CubReductionOp operation,
                                                   uint64_t original_reduction_size,
                                                   float output_scale,
                                                   DataType input_dtype,
                                                   uint32_t multiprocessors,
                                                   const Stream& stream,
                                                   CubReductionTuningDepth tuning_depth);

/** Original FP8 ARG R / KR site (argmax when `maximum`). Always returns a retained shape. */
[[nodiscard]] Fp8KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                                 DenseReductionSite site,
                                                 bool maximum,
                                                 DataType input_dtype,
                                                 uint32_t multiprocessors,
                                                 const Stream& stream,
                                                 CubReductionTuningDepth tuning_depth);

}  // namespace ThorImplementation::CubReductionInternal::Fp8KR
