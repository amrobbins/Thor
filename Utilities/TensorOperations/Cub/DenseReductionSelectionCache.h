#pragma once

// 1171: the one stamp-time selection cache of the dense reducer. Every selector that autotunes a pass at stamp time
// (FP8 RK VALUE / ARG, FP8 R / KR VALUE / ARG, and the selectors the other dtypes will add) keeps its winners here,
// so every selector gets the same rules:
//   - one process-wide, bounded LRU of immutable selection recipes (never a Tensor, Stream, executable, CUDA handle
//     or stamp-local workspace: AcceleratorBackendSelectionRecipe);
//   - single flight: concurrent misses on one key run one tuning callback; the others wait and share its result;
//     failures propagate to every waiter and are not cached; clear() while a flight runs does not let it repopulate;
//   - depth: a selection tuned at a deeper depth (Ultra > Regular > Quick) answers a shallower request, completed or
//     in flight; a shallower one never answers a deeper request;
//   - per-kind tuning-attempt counts (misses that ran a callback).
// The cache mutex is never held while a callback runs; GPU timing is serialized by the callbacks' own measurement
// lock (ReductionGpuMeasurementLock). CUDA-independent: host tests use it directly.
//
// A key is (selection kind, device identity, tuning depth, site): the kind names the selector, and so the recipe type
// it stores; the site is the selector's canonical description of everything that can change its shortlist, its
// measurement or the legality of the winner. Not persisted on disk (user decision, 1169).

#include "Utilities/Cache/LruCache.h"
#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"

#include <algorithm>
#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/** The selector an entry belongs to. Each kind stores exactly one recipe type. */
enum class DenseReductionSelectionKind : uint8_t {
    Fp8RK = 1,  // Fp8RK::Fp8RKSelection (VALUE and ARG RK sites)
    Fp8KR = 2,  // Fp8KR::Fp8KRSelection (VALUE and ARG R / KR sites)
    Fp32RK = 3,  // 1178: reserved for the FP32 RK selector (its recipe type arrives with it)
    Fp32KR = 4,  // 1178: reserved for the FP32 R / KR selector
    F16RK = 5,   // 1236: reserved for the FP16 / BF16 RK selector (phase 5)
    F16KR = 6,   // 1236: F16KR::F16KRSelection (FP16 / BF16 R / KR sites; the key holds the input dtype)
};
inline constexpr size_t DenseReductionSelectionKindCount = 7;  // kinds are indexed 1..Count-1

[[nodiscard]] constexpr const char* denseReductionSelectionKindName(DenseReductionSelectionKind kind) {
    switch (kind) {
        case DenseReductionSelectionKind::Fp8RK: return "fp8_rk";
        case DenseReductionSelectionKind::Fp8KR: return "fp8_kr";
        case DenseReductionSelectionKind::Fp32RK: return "fp32_rk";
        case DenseReductionSelectionKind::Fp32KR: return "fp32_kr";
        case DenseReductionSelectionKind::F16RK: return "f16_rk";
        case DenseReductionSelectionKind::F16KR: return "f16_kr";
    }
    return "unknown";
}

/** Device identity and resources: the ordinal alone is not enough to reuse a selection. */
struct DenseReductionDeviceIdentity {
    std::array<uint8_t, 16> uuid{};
    int32_t ordinal = -1;
    uint32_t sm_count = 0;
    uint32_t compute_major = 0;
    uint32_t compute_minor = 0;
    bool operator==(const DenseReductionDeviceIdentity&) const = default;
};

/** The identity of `gpu` (queried once per ordinal per process). Defined in DenseReductionDeviceIdentity.cpp. */
[[nodiscard]] DenseReductionDeviceIdentity denseReductionDeviceIdentity(int gpu);

struct DenseReductionSelectionKey {
    DenseReductionSelectionKind kind = DenseReductionSelectionKind::Fp8RK;
    DenseReductionDeviceIdentity device{};
    CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular;
    std::vector<uint64_t> site;
    bool operator==(const DenseReductionSelectionKey&) const = default;
};

struct DenseReductionSelectionKeyHash {
    [[nodiscard]] size_t operator()(const DenseReductionSelectionKey& key) const noexcept {
        size_t h = 0x9e3779b9U;
        const auto add = [&](uint64_t v) {
            const size_t x = std::hash<uint64_t>{}(v);
            h ^= x + size_t{0x9e3779b97f4a7c15ULL} + (h << 6) + (h >> 2);
        };
        add(static_cast<uint64_t>(key.kind));
        for (uint8_t b : key.device.uuid) add(b);
        add(static_cast<uint32_t>(key.device.ordinal));
        add(key.device.sm_count);
        add(key.device.compute_major);
        add(key.device.compute_minor);
        add(static_cast<uint64_t>(key.tuning_depth));
        add(key.site.size());
        for (uint64_t v : key.site) add(v);
        return h;
    }
};

class DenseReductionSelectionCache final {
   public:
    explicit DenseReductionSelectionCache(size_t capacity = 25000) : completed_(capacity) {}
    DenseReductionSelectionCache(const DenseReductionSelectionCache&) = delete;
    DenseReductionSelectionCache& operator=(const DenseReductionSelectionCache&) = delete;

    /**
     * The selection for `key`: a completed or in-flight entry at the key's depth or deeper, else the result of
     * `tune()` (run once for all concurrent requests of this key). `Selection` must be the kind's recipe type.
     */
    template <class Selection, class Tune>
        requires AcceleratorBackendSelectionRecipe<Selection>
    [[nodiscard]] Selection getOrTune(const DenseReductionSelectionKey& key, Tune&& tune) {
        static_assert(!isAcceleratorBackendLocalExecutionStateV<Selection>);
        // The current thread must never wait on a flight it owns (directly, or through a deeper key it is tuning).
        for (const DenseReductionSelectionKey& answering : answeringKeys(key)) {
            if (activeOnThisThread(answering))
                throw std::logic_error(std::string("Recursive same-site dense reduction tuning dependency (") +
                                       denseReductionSelectionKindName(key.kind) + ")");
        }
        if (auto hit = findAtLeast<Selection>(key)) return *hit;

        std::shared_ptr<InFlight> work;
        bool leader = false;
        uint64_t my_generation = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Deepest first: a completed entry, then a flight, at each depth that can answer.
            for (const DenseReductionSelectionKey& answering : answeringKeys(key)) {
                if (auto hit = completed_.get(answering)) return cast<Selection>(*hit, key);
                if (auto it = in_flight_.find(answering); it != in_flight_.end()) {
                    work = it->second;
                    break;
                }
            }
            if (work == nullptr) {
                work = std::make_shared<InFlight>();
                in_flight_.emplace(key, work);
                leader = true;
                my_generation = generation_;
                ++tuning_attempts_[kindIndex(key.kind)];
            }
        }
        if (!leader) return cast<Selection>(work->future.get(), key);

        try {
            std::any selection;
            {
                ActiveTune owner(this, key);
                selection = std::any(Selection(std::forward<Tune>(tune)()));
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = in_flight_.find(key);
                // A clear() racing the tune invalidates insertion. Never erase a NEW flight that began after it.
                if (my_generation == generation_ && it != in_flight_.end() && it->second == work) {
                    completed_.put(key, selection);
                    in_flight_.erase(it);
                }
            }
            work->promise.set_value(selection);
            return std::any_cast<Selection>(selection);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = in_flight_.find(key);
                if (it != in_flight_.end() && it->second == work) in_flight_.erase(it);
            }
            work->promise.set_exception(std::current_exception());
            throw;
        }
    }

    /** A completed entry for exactly `key` (its depth only), without tuning. */
    template <class Selection>
    [[nodiscard]] std::optional<Selection> find(const DenseReductionSelectionKey& key) {
        if (auto hit = completed_.get(key)) return cast<Selection>(*hit, key);
        return std::nullopt;
    }

    /** A completed entry at the key's depth or deeper (deepest first), without tuning. */
    template <class Selection>
    [[nodiscard]] std::optional<Selection> findAtLeast(const DenseReductionSelectionKey& key) {
        for (const DenseReductionSelectionKey& answering : answeringKeys(key)) {
            if (auto hit = completed_.get(answering)) return cast<Selection>(*hit, key);
        }
        return std::nullopt;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++generation_;
        completed_.clear();
        in_flight_.clear();  // existing waiters keep their futures
    }

    [[nodiscard]] size_t size() const { return completed_.size(); }

    /** Tuning callbacks started (cache misses), all kinds or one kind. */
    [[nodiscard]] uint64_t tuningAttempts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t total = 0;
        for (uint64_t n : tuning_attempts_) total += n;
        return total;
    }
    [[nodiscard]] uint64_t tuningAttempts(DenseReductionSelectionKind kind) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tuning_attempts_[kindIndex(kind)];
    }

   private:
    // The keys whose entry answers `key`: the same site at its depth or deeper, deepest first.
    static std::vector<DenseReductionSelectionKey> answeringKeys(const DenseReductionSelectionKey& key) {
        std::vector<DenseReductionSelectionKey> keys;
        for (const CubReductionTuningDepth depth :
             {CubReductionTuningDepth::Ultra, CubReductionTuningDepth::Regular, CubReductionTuningDepth::Quick}) {
            if (cubReductionTuningDepthRank(depth) < cubReductionTuningDepthRank(key.tuning_depth)) continue;
            DenseReductionSelectionKey answering = key;
            answering.tuning_depth = depth;
            keys.push_back(std::move(answering));
        }
        return keys;
    }

    static size_t kindIndex(DenseReductionSelectionKind kind) {
        const size_t index = static_cast<size_t>(kind);
        if (index == 0 || index >= DenseReductionSelectionKindCount)
            throw std::logic_error("Unknown dense reduction selection kind");
        return index;
    }

    template <class Selection>
    static Selection cast(const std::any& value, const DenseReductionSelectionKey& key) {
        if (const Selection* selection = std::any_cast<Selection>(&value)) return *selection;
        throw std::logic_error(std::string("Dense reduction selection cache: a ") +
                               denseReductionSelectionKindName(key.kind) + " entry holds a different recipe type");
    }

    struct ActiveEntry {
        const DenseReductionSelectionCache* cache;
        DenseReductionSelectionKey key;
    };
    static std::vector<ActiveEntry>& activeTunings() {
        static thread_local std::vector<ActiveEntry> active;
        return active;
    }
    [[nodiscard]] bool activeOnThisThread(const DenseReductionSelectionKey& key) const {
        const auto& active = activeTunings();
        return std::any_of(active.begin(), active.end(),
                           [&](const ActiveEntry& entry) { return entry.cache == this && entry.key == key; });
    }
    struct ActiveTune {
        ActiveTune(const DenseReductionSelectionCache* cache, const DenseReductionSelectionKey& key) {
            activeTunings().push_back({cache, key});
        }
        ~ActiveTune() { activeTunings().pop_back(); }
        ActiveTune(const ActiveTune&) = delete;
        ActiveTune& operator=(const ActiveTune&) = delete;
    };
    struct InFlight {
        std::promise<std::any> promise;
        std::shared_future<std::any> future = promise.get_future().share();
    };

    LruCacheThreadSafe<DenseReductionSelectionKey, std::any, DenseReductionSelectionKeyHash> completed_;
    mutable std::mutex mutex_;
    std::unordered_map<DenseReductionSelectionKey, std::shared_ptr<InFlight>, DenseReductionSelectionKeyHash>
        in_flight_;
    uint64_t generation_ = 0;
    std::array<uint64_t, DenseReductionSelectionKindCount> tuning_attempts_{};
};

/** The process-wide cache every stamp-time dense reduction selector uses. */
[[nodiscard]] DenseReductionSelectionCache& denseReductionSelectionCache();

}  // namespace ThorImplementation::CubReductionInternal
