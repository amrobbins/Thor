#pragma once

// GEMM-style cache of immutable FP8 RK selection recipes. Never cache a Tensor,
// Stream, executable, CUDA handle, or stamp-local workspace in this repository.
// This file is intentionally CUDA-independent so concurrency and ownership
// contracts can be tested without a GPU or any generated Thor headers.
#include "Utilities/Cache/LruCache.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

struct Fp8RKReductionRunKey {
    uint64_t extent = 0;
    uint64_t original_arg_domain_stride = 0;
    uint8_t kind = 0;
    bool operator==(const Fp8RKReductionRunKey&) const = default;
};

// Include every parameter that can change the physical shortlist, the
// continuation, the measurement, or the legality of a selected kernel.
// Device ordinal alone is insufficient: device identity and resources matter.
struct Fp8RKSelectionKey {
    std::array<uint8_t, 16> device_uuid{};
    int32_t device_ordinal = -1;
    uint32_t sm_count = 0;
    uint32_t compute_major = 0;
    uint32_t compute_minor = 0;
    uint64_t workspace_budget_bytes = 0;
    uint64_t original_reduction_size = 0;
    uint32_t output_scale_bits = 0;
    uint32_t site_run_index = 0;
    uint8_t input_dtype = 0;
    uint8_t output_dtype = 0;
    uint8_t output_index_dtype = 0;
    uint8_t operation = 0;
    uint8_t aggregate_kind = 0;
    uint8_t input_state = 0;
    uint8_t carried_arg_index_dtype = 0;
    bool produce_value = true;
    bool produce_index = false;
    bool finalizer_pending = true;
    std::vector<Fp8RKReductionRunKey> runs;
    bool operator==(const Fp8RKSelectionKey&) const = default;
};

struct Fp8RKSelectionKeyHash {
    [[nodiscard]] size_t operator()(const Fp8RKSelectionKey& key) const noexcept {
        size_t h = 0x9e3779b9U;
        const auto add = [&](uint64_t v) {
            const size_t x = std::hash<uint64_t>{}(v);
            h ^= x + size_t{0x9e3779b97f4a7c15ULL} + (h << 6) + (h >> 2);
        };
        for (uint8_t b : key.device_uuid) add(b);
        add(static_cast<uint32_t>(key.device_ordinal));
        add(key.sm_count); add(key.compute_major); add(key.compute_minor);
        add(key.workspace_budget_bytes); add(key.original_reduction_size);
        add(key.output_scale_bits); add(key.site_run_index);
        add(key.input_dtype); add(key.output_dtype); add(key.output_index_dtype);
        add(key.operation); add(key.aggregate_kind); add(key.input_state);
        add(key.carried_arg_index_dtype); add(key.produce_value);
        add(key.produce_index); add(key.finalizer_pending);
        add(key.runs.size());
        for (const auto& run : key.runs) {
            add(run.extent); add(run.original_arg_domain_stride); add(run.kind);
        }
        return h;
    }
};

class Fp8RKSelectionCache final {
public:
    explicit Fp8RKSelectionCache(size_t capacity = 25000) : completed_(capacity) {}

    // Same-key concurrent misses share one tuning invocation. Other keys may
    // tune independently; the cache mutex is never held during GPU work.
    // Exceptions are propagated to all waiters and are NOT cached.
    template <class Tune>
    [[nodiscard]] Fp8RKSelection getOrTune(const Fp8RKSelectionKey& key, Tune&& tune) {
        // The current thread must never wait on a flight it ultimately owns.
        // Track active callbacks rather than just in_flight_: clear() can
        // remove a flight while its original callback is still executing.
        if (activeOnThisThread(key))
            throw std::logic_error("Recursive same-key FP8 RK tuning dependency");
        if (auto hit = completed_.get(key)) return *hit;

        std::shared_ptr<InFlight> work;
        bool leader = false;
        uint64_t my_generation = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Prevent a race where another caller finishes between the first
            // completed-cache lookup and acquisition of the flight mutex.
            if (auto hit = completed_.get(key)) return *hit;
            auto it = in_flight_.find(key);
            if (it != in_flight_.end()) {
                work = it->second;
            } else {
                work = std::make_shared<InFlight>();
                in_flight_.emplace(key, work);
                leader = true;
                my_generation = generation_;
                tuning_attempts_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (!leader) return work->future.get();

        try {
            ActiveTune owner(this, key);
            Fp8RKSelection selection = std::forward<Tune>(tune)();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = in_flight_.find(key);
                // A clear() racing the tune invalidates insertion. Avoid
                // erasing a NEW flight that began after the clear.
                if (my_generation == generation_ && it != in_flight_.end() &&
                    it->second == work) {
                    completed_.put(key, selection);
                    in_flight_.erase(it);
                }
            }
            work->promise.set_value(selection);
            return selection;
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = in_flight_.find(key);
                if (it != in_flight_.end() && it->second == work)
                    in_flight_.erase(it);
            }
            work->promise.set_exception(std::current_exception());
            throw;
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++generation_;
        completed_.clear();
        in_flight_.clear(); // existing waiter futures remain valid
    }

    [[nodiscard]] size_t size() const { return completed_.size(); }
    [[nodiscard]] uint64_t tuningAttempts() const {
        return tuning_attempts_.load(std::memory_order_relaxed);
    }

private:
    struct ActiveEntry {
        const Fp8RKSelectionCache* cache;
        Fp8RKSelectionKey key;
    };
    static std::vector<ActiveEntry>& activeTunings() {
        static thread_local std::vector<ActiveEntry> active;
        return active;
    }
    [[nodiscard]] bool activeOnThisThread(const Fp8RKSelectionKey& key) const {
        const auto& active = activeTunings();
        return std::any_of(active.begin(), active.end(), [&](const ActiveEntry& entry) {
            return entry.cache == this && entry.key == key;
        });
    }
    struct ActiveTune {
        ActiveTune(const Fp8RKSelectionCache* cache, const Fp8RKSelectionKey& key) {
            activeTunings().push_back({cache, key});
        }
        ~ActiveTune() { activeTunings().pop_back(); }
        ActiveTune(const ActiveTune&) = delete;
        ActiveTune& operator=(const ActiveTune&) = delete;
    };
    struct InFlight {
        std::promise<Fp8RKSelection> promise;
        std::shared_future<Fp8RKSelection> future = promise.get_future().share();
    };
    LruCacheThreadSafe<Fp8RKSelectionKey, Fp8RKSelection, Fp8RKSelectionKeyHash> completed_;
    mutable std::mutex mutex_;
    std::unordered_map<Fp8RKSelectionKey, std::shared_ptr<InFlight>, Fp8RKSelectionKeyHash> in_flight_;
    uint64_t generation_ = 0;
    std::atomic<uint64_t> tuning_attempts_{0};
};

// One process-wide repository, just as GEMM owns an optimal selection cache.
[[nodiscard]] Fp8RKSelectionCache& fp8RKSelectionCache();

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
