#pragma once

// Stamp-time selection of FP8 R / KR passes. The candidates are fp8KRCandidateShapes (best guess first; the tuning
// depth keeps the ranker's first Quick 6 / Regular 12 / Ultra 12). They are timed by the shared contest
// (ReductionAutotuning::autotune) with the depth's rounds and trim, end to end (a Split shape's finishing kernel
// included). Selections are cached per process and device, keyed on the site geometry, semantics, outputs and depth, in
// the one dense reduction selection cache (DenseReductionSelectionCache.h, kind Fp8KR); a deeper cached selection
// answers a shallower request.

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"
#include "Utilities/TensorOperations/Reduction/ReductionTypes.h"

#include "Utilities/TensorOperations/Reduction/KR/KRSegments.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanning.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ThorImplementation::ReductionInternal::KR::Fp8 {

struct Fp8KRSelection final : AcceleratorBackendSelectionRecipeTag {
    std::optional<KRSegmentShape> shape;  // nullopt: semantics the kernels do not compile (an error)
    size_t candidate_count = 0;
    std::string winner_name;
};

/** Original FP8 VALUE R / KR site. `operation` is the public operation; output_scale applies on the final pass. */
[[nodiscard]] Fp8KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                                   DenseReductionSite site,
                                                   ReductionOp operation,
                                                   uint64_t original_reduction_size,
                                                   float output_scale,
                                                   DataType input_dtype,
                                                   uint32_t multiprocessors,
                                                   const Stream& stream,
                                                   ReductionTuningDepth tuning_depth);

/** Original FP8 ARG R / KR site (argmax when `maximum`). Always returns a shape. */
[[nodiscard]] Fp8KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                                 DenseReductionSite site,
                                                 bool maximum,
                                                 DataType input_dtype,
                                                 uint32_t multiprocessors,
                                                 const Stream& stream,
                                                 ReductionTuningDepth tuning_depth);

}  // namespace ThorImplementation::ReductionInternal::KR::Fp8
