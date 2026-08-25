#pragma once

#include "Utilities/Cache/LruCache.h"
#include "Utilities/Common/AcceleratorBackendCachePolicy.h"

#include <cudnn.h>

#include <atomic>
#include <condition_variable>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudnn_frontend {
namespace graph {
class Graph;
}
}  // namespace cudnn_frontend

namespace ThorImplementation {

/**
 * Immutable, process-global-safe identity of a selected cuDNN Frontend plan.
 *
 * This is deliberately only a selection recipe.  It contains no graph,
 * execution-plan object, handle, descriptor, workspace allocation, stream, or
 * other runtime state.  Knobs are stored in canonical numeric order so recipe
 * equality is deterministic even when cuDNN reports them through an
 * unordered_map.
 */
struct CudnnFrontendPlanSelection final : AcceleratorBackendSelectionRecipeTag {
    int64_t engine_id = -1;
    std::vector<std::pair<int64_t, int64_t>> knobs;
    uint64_t expected_workspace_bytes = 0;

    // Empty for the preferred structured engine+knob replay path.  When cuDNN
    // exposes a backend knob that this installed Frontend cannot represent in
    // KnobType_t, this contains Frontend's immutable plan-only serialized
    // payload instead.  It is a replay token, never a live descriptor/plan.
    std::vector<uint8_t> serialized_plan;

    CudnnFrontendPlanSelection() = default;
    CudnnFrontendPlanSelection(int64_t engineId,
                               std::vector<std::pair<int64_t, int64_t>> knobValues,
                               uint64_t expectedWorkspaceBytes,
                               std::vector<uint8_t> serializedPlan = {});

    [[nodiscard]] bool usesSerializedReplay() const noexcept { return !serialized_plan.empty(); }

    bool operator==(const CudnnFrontendPlanSelection& other) const {
        return engine_id == other.engine_id && knobs == other.knobs &&
               expected_workspace_bytes == other.expected_workspace_bytes && serialized_plan == other.serialized_plan;
    }
};

/**
 * Move-only local cuDNN Frontend execution state.
 *
 * The underlying graph is intentionally private and is never returned as a
 * shared_ptr.  Independent stamped operations therefore cannot obtain aliases
 * to one another's executable graph through this API.
 */
class CudnnFrontendExecutablePlan final : public AcceleratorBackendLocalExecutionStateTag {
   public:
    CudnnFrontendExecutablePlan(const CudnnFrontendExecutablePlan&) = delete;
    CudnnFrontendExecutablePlan& operator=(const CudnnFrontendExecutablePlan&) = delete;
    CudnnFrontendExecutablePlan(CudnnFrontendExecutablePlan&&) noexcept = default;
    CudnnFrontendExecutablePlan& operator=(CudnnFrontendExecutablePlan&&) noexcept = default;
    ~CudnnFrontendExecutablePlan() = default;

    [[nodiscard]] const CudnnFrontendPlanSelection& selection() const noexcept { return selection_; }
    [[nodiscard]] uint64_t workspaceBytes() const noexcept { return workspace_bytes_; }
    [[nodiscard]] int64_t planIndex() const noexcept { return plan_index_; }

    // Stable only for the lifetime of this local wrapper.  Exposed for
    // ownership diagnostics/tests; it is not an executable graph handle.
    [[nodiscard]] uintptr_t executableId() const noexcept;

    void execute(cudnnHandle_t handle, std::unordered_map<int64_t, void*>& tensorPack, void* workspace) const;

   private:
    CudnnFrontendExecutablePlan(std::shared_ptr<cudnn_frontend::graph::Graph> graph,
                                CudnnFrontendPlanSelection selection,
                                uint64_t workspaceBytes,
                                int64_t planIndex);

    std::shared_ptr<cudnn_frontend::graph::Graph> graph_;
    CudnnFrontendPlanSelection selection_;
    uint64_t workspace_bytes_ = 0;
    int64_t plan_index_ = -1;

    friend CudnnFrontendExecutablePlan replayCudnnFrontendExecutablePlan(
        const std::function<std::shared_ptr<cudnn_frontend::graph::Graph>()>& graphFactory,
        const CudnnFrontendPlanSelection& selection,
        cudnnHandle_t handle,
        std::string_view operationName);
};

using CudnnFrontendGraphFactory = std::function<std::shared_ptr<cudnn_frontend::graph::Graph>()>;

/**
 * Extract a canonical selection recipe from an already-built plan.  Structured
 * engine+knob replay is preferred; if Frontend's enum mapping is lossy for the
 * selected backend plan, an immutable plan-only serialization token is retained
 * instead.
 */
[[nodiscard]] CudnnFrontendPlanSelection cudnnFrontendPlanSelectionAtIndex(cudnn_frontend::graph::Graph& graph,
                                                                            int64_t planIndex,
                                                                            std::string_view operationName);

/**
 * Capture Frontend's currently selected executable plan as an immutable serialized
 * replay recipe.  This is used after empirical autotuning because Frontend may
 * reorder its execution-plan vector without reordering the original heuristic
 * engine-config vector.  A serialized selected-plan recipe therefore intentionally
 * has engine_id == -1: replay identity is carried entirely by serialized_plan.
 */
[[nodiscard]] CudnnFrontendPlanSelection cudnnFrontendSelectedSerializedPlanSelection(
    cudnn_frontend::graph::Graph& graph, std::string_view operationName);

/**
 * Recreate exactly one selected recipe as a fresh operation-local executable.
 * Structured recipes replay onto a pristine graph from graphFactory.  Serialized
 * recipes deserialize into a fresh blank Frontend graph during preparation only.
 * The replayed workspace requirement must exactly match the recipe before the
 * executable is returned.
 */
[[nodiscard]] CudnnFrontendExecutablePlan replayCudnnFrontendExecutablePlan(const CudnnFrontendGraphFactory& graphFactory,
                                                                             const CudnnFrontendPlanSelection& selection,
                                                                             cudnnHandle_t handle,
                                                                             std::string_view operationName);

// Placement-time replay diagnostics.  execute() never increments this counter.
[[nodiscard]] uint64_t cudnnFrontendExecutablePreparationCountForTests() noexcept;

/**
 * Bounded process-global cache whose value type is structurally fixed to the
 * immutable cuDNN Frontend selection recipe.
 *
 * getOrSelect() is single-flight per key: the expensive selector runs outside
 * the cache mutex, unrelated keys can select concurrently, and callers racing
 * on the same key share the one completed selection.  clear() removes ready
 * recipes and detaches in-flight selections from the cache; an in-flight caller
 * may still receive its result, but that result cannot repopulate a cache that
 * was cleared while it was running.
 */
template <class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
class CudnnFrontendPlanSelectionCache {
   public:
    explicit CudnnFrontendPlanSelectionCache(size_t capacity) : ready_cache_(capacity) {}

    CudnnFrontendPlanSelectionCache(const CudnnFrontendPlanSelectionCache&) = delete;
    CudnnFrontendPlanSelectionCache& operator=(const CudnnFrontendPlanSelectionCache&) = delete;
    CudnnFrontendPlanSelectionCache(CudnnFrontendPlanSelectionCache&&) = delete;
    CudnnFrontendPlanSelectionCache& operator=(CudnnFrontendPlanSelectionCache&&) = delete;

    template <typename Selector>
        requires std::invocable<Selector&> &&
                 std::same_as<std::remove_cvref_t<std::invoke_result_t<Selector&>>, CudnnFrontendPlanSelection>
    [[nodiscard]] CudnnFrontendPlanSelection getOrSelect(const Key& key, Selector&& selector) {
        if (std::optional<CudnnFrontendPlanSelection> ready = ready_cache_.get(key); ready.has_value()) {
            hit_count_.fetch_add(1, std::memory_order_relaxed);
            return std::move(ready.value());
        }

        std::shared_ptr<InFlightSelection> flight;
        bool selection_owner = false;
        {
            std::lock_guard<std::mutex> lock(in_flight_mutex_);

            // Close the race between the optimistic ready-cache lookup and
            // registering this key as in-flight.
            if (std::optional<CudnnFrontendPlanSelection> ready = ready_cache_.get(key); ready.has_value()) {
                hit_count_.fetch_add(1, std::memory_order_relaxed);
                return std::move(ready.value());
            }

            const auto existing = in_flight_.find(key);
            if (existing != in_flight_.end()) {
                flight = existing->second;
                hit_count_.fetch_add(1, std::memory_order_relaxed);
            } else {
                flight = std::make_shared<InFlightSelection>();
                flight->generation = generation_;
                in_flight_.emplace(key, flight);
                miss_count_.fetch_add(1, std::memory_order_relaxed);
                selection_owner = true;
            }
        }

        if (!selection_owner) {
            std::unique_lock<std::mutex> flight_lock(flight->mutex);
            flight->completed.wait(flight_lock, [&]() { return flight->done; });
            if (flight->error != nullptr) {
                std::rethrow_exception(flight->error);
            }
            if (!flight->selection.has_value()) {
                throw std::runtime_error("cuDNN Frontend selection single-flight completed without a result.");
            }
            return flight->selection.value();
        }

        std::optional<CudnnFrontendPlanSelection> selected;
        std::exception_ptr selection_error;
        try {
            CudnnFrontendPlanSelection candidate = std::invoke(selector);
            // Reconstruct through the validating constructor before publication.
            // The recipe type stays conveniently copyable, but a selector cannot
            // publish a default-invalid or non-canonicalized value into the cache.
            selected.emplace(candidate.engine_id,
                             std::move(candidate.knobs),
                             candidate.expected_workspace_bytes,
                             std::move(candidate.serialized_plan));
        } catch (...) {
            selection_error = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(in_flight_mutex_);
            const auto current = in_flight_.find(key);
            const bool still_registered = current != in_flight_.end() && current->second == flight;
            if (selected.has_value() && still_registered && flight->generation == generation_) {
                ready_cache_.put(key, selected.value());
            }
            if (still_registered) {
                in_flight_.erase(current);
            }
        }

        {
            std::lock_guard<std::mutex> flight_lock(flight->mutex);
            flight->selection = selected;
            flight->error = selection_error;
            flight->done = true;
        }
        flight->completed.notify_all();

        if (selection_error != nullptr) {
            std::rethrow_exception(selection_error);
        }
        return selected.value();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(in_flight_mutex_);
        ++generation_;
        ready_cache_.clear();
        // Existing callers retain shared_ptrs to their flight entries and will
        // still complete.  Removing them here ensures clear() is a real cache
        // boundary: future callers cannot join or repopulate those old flights.
        in_flight_.clear();
    }

    [[nodiscard]] size_t size() const { return ready_cache_.size(); }
    [[nodiscard]] size_t capacity() const { return ready_cache_.capacity(); }
    [[nodiscard]] uint64_t hitCount() const noexcept { return hit_count_.load(std::memory_order_relaxed); }
    [[nodiscard]] uint64_t missCount() const noexcept { return miss_count_.load(std::memory_order_relaxed); }

   private:
    struct InFlightSelection {
        std::mutex mutex;
        std::condition_variable completed;
        bool done = false;
        uint64_t generation = 0;
        std::optional<CudnnFrontendPlanSelection> selection;
        std::exception_ptr error;
    };

    LruCacheThreadSafe<Key, CudnnFrontendPlanSelection, Hash, KeyEqual> ready_cache_;
    mutable std::mutex in_flight_mutex_;
    std::unordered_map<Key, std::shared_ptr<InFlightSelection>, Hash, KeyEqual> in_flight_;
    uint64_t generation_ = 0;
    std::atomic<uint64_t> hit_count_{0};
    std::atomic<uint64_t> miss_count_{0};
};

static_assert(AcceleratorBackendSelectionRecipe<CudnnFrontendPlanSelection>);
static_assert(std::is_copy_constructible_v<CudnnFrontendPlanSelection>);
static_assert(std::is_copy_assignable_v<CudnnFrontendPlanSelection>);
static_assert(AcceleratorBackendLocalExecutionState<CudnnFrontendExecutablePlan>);
static_assert(!std::is_copy_constructible_v<CudnnFrontendExecutablePlan>);
static_assert(!std::is_copy_assignable_v<CudnnFrontendExecutablePlan>);
static_assert(std::is_move_constructible_v<CudnnFrontendExecutablePlan>);
static_assert(std::is_move_assignable_v<CudnnFrontendExecutablePlan>);

}  // namespace ThorImplementation
