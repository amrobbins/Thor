// Host-only guard against persisting stamp-local FP8 RK executables or Tensor
// allocations inside the eventual GEMM-style selection repository.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <cassert>
#include <type_traits>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal::Fp8RK;

static_assert(AcceleratorBackendSelectionRecipe<Fp8RKSelection>);
static_assert(std::is_copy_constructible_v<Fp8RKSelection>);
static_assert(std::is_copy_assignable_v<Fp8RKSelection>);
static_assert(!isAcceleratorBackendLocalExecutionStateV<Fp8RKSelection>);
template <typename T>
concept HasCompleteSiteBenchmarkWorkspace = requires(T recipe) { recipe.workspace_bytes; };
static_assert(!HasCompleteSiteBenchmarkWorkspace<Fp8RKSelection>);

int main() {
    Fp8RKSelection selection;
    selection.physical = PhysicalKernel{PhysicalFamily::RuntimeK};
    selection.candidate_count = 28;
    selection.winner_index = 7;
    selection.winner_name = "runtime_k#7";
    auto reused = selection;
    assert(reused.physical == selection.physical);
    assert(reused.candidate_count == 28 && reused.winner_index == 7);
    assert(reused.winner_name == "runtime_k#7");
}
