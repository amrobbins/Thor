#pragma once

#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <cstddef>
#include <string>
#include <type_traits>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

// Cached GEMM-style physical selection RECIPE, not a completed reduction plan.
// No Tensor, Stream, executable, CUDA handle, or scratch allocation belongs here.
struct Fp8RKSelection final : AcceleratorBackendSelectionRecipeTag {
    PhysicalKernel physical{};
    size_t candidate_count = 0;
    size_t winner_index = 0;
    std::string winner_name;
};
static_assert(AcceleratorBackendSelectionRecipe<Fp8RKSelection>);
static_assert(!isAcceleratorBackendLocalExecutionStateV<Fp8RKSelection>);

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
