#pragma once

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"

#include <cstddef>
#include <string>
#include <type_traits>

namespace ThorImplementation::ReductionInternal::RK {

// Cached GEMM-style physical selection RECIPE, not a completed reduction plan.
// No Tensor, Stream, executable, CUDA handle, or scratch allocation belongs here.
struct RKSelection final : AcceleratorBackendSelectionRecipeTag {
    PhysicalKernel physical{};
    size_t candidate_count = 0;
    size_t winner_index = 0;
    std::string winner_name;
};
static_assert(AcceleratorBackendSelectionRecipe<RKSelection>);
static_assert(!isAcceleratorBackendLocalExecutionStateV<RKSelection>);

} // namespace ThorImplementation::ReductionInternal::RK
