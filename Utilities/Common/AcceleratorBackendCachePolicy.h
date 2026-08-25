#pragma once

#include <type_traits>

namespace ThorImplementation {

/**
 * Marker for immutable accelerator-backend selection/configuration values that
 * may be shared through a process-global cache.
 *
 * A selection recipe may contain algorithm/engine identifiers, tuning knobs,
 * immutable configuration/replay bytes, and expected workspace byte counts. It must not own
 * backend descriptors, executable graph/plan objects, handles, workspace/temp
 * storage, or state that is mutated by execution.
 *
 * Global backend caches use this vocabulary together with source-audit gates
 * so cacheable selection state and operation-local execution state cannot be
 * silently conflated.
 */
struct AcceleratorBackendSelectionRecipeTag {};

/**
 * Base for backend execution state that belongs to one independently
 * executable stamped operation (or another explicitly serialized execution
 * domain).
 *
 * A backend-specific executable wrapper can derive from this base so its
 * defaulted copy operations are deleted and the type is visibly classified as
 * local execution state. Concrete wrappers must still explicitly delete their
 * own copy operations. Owning pointers to such wrappers must likewise remain
 * operation-local; the tag does not make a global shared_ptr safe.
 */
class AcceleratorBackendLocalExecutionStateTag {
   public:
    AcceleratorBackendLocalExecutionStateTag() = default;
    AcceleratorBackendLocalExecutionStateTag(const AcceleratorBackendLocalExecutionStateTag&) = delete;
    AcceleratorBackendLocalExecutionStateTag& operator=(const AcceleratorBackendLocalExecutionStateTag&) = delete;
    AcceleratorBackendLocalExecutionStateTag(AcceleratorBackendLocalExecutionStateTag&&) noexcept = default;
    AcceleratorBackendLocalExecutionStateTag& operator=(AcceleratorBackendLocalExecutionStateTag&&) noexcept = default;

   protected:
    ~AcceleratorBackendLocalExecutionStateTag() = default;
};

template <typename T>
inline constexpr bool isAcceleratorBackendSelectionRecipeV =
    std::is_base_of_v<AcceleratorBackendSelectionRecipeTag, std::remove_cvref_t<T>>;

template <typename T>
inline constexpr bool isAcceleratorBackendLocalExecutionStateV =
    std::is_base_of_v<AcceleratorBackendLocalExecutionStateTag, std::remove_cvref_t<T>>;

template <typename T>
concept AcceleratorBackendSelectionRecipe = isAcceleratorBackendSelectionRecipeV<T>;

template <typename T>
concept AcceleratorBackendLocalExecutionState = isAcceleratorBackendLocalExecutionStateV<T>;

}  // namespace ThorImplementation
