// CPU-only regression for the cross-stamp FP8 RK recipe repository.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionCache.h"

#include <atomic>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal::Fp8RK;

namespace {
void check(bool ok, const char* msg) {
    if (!ok) throw std::logic_error(msg);
}
Fp8RKSelection recipe(size_t winner) {
    Fp8RKSelection value;
    value.physical.family = PhysicalFamily::RuntimeK;
    value.candidate_count = 28;
    value.winner_index = winner;
    value.winner_name = "runtime_k";
    return value;
}
Fp8RKSelectionKey keyFor(uint64_t outer) {
    Fp8RKSelectionKey key;
    key.device_uuid[0] = 51;
    key.device_ordinal = 0;
    key.compute_major = 12;
    key.sm_count = 170;
    key.workspace_budget_bytes = 256 * 1024 * 1024;
    key.original_reduction_size = 512;
    key.output_scale_bits = 0x3f800000; // 1.0f
    key.input_dtype = 22;
    key.output_dtype = 7;
    key.runs = {{outer, 0, 0}, {512, 0, 1}, {64, 0, 0}};
    key.site_run_index = 1;
    return key;
}
}

int main() {
    static_assert(AcceleratorBackendSelectionRecipe<Fp8RKSelection>);
    static_assert(!isAcceleratorBackendLocalExecutionStateV<Fp8RKSelection>);
    Fp8RKSelectionCache cache(2);
    auto a = keyFor(16);
    auto b = keyFor(32);
    std::atomic<int> calls{0};
    auto selected = cache.getOrTune(a, [&] { ++calls; return recipe(4); });
    check(selected.winner_index == 4 && calls == 1 && cache.size() == 1,
          "First miss must tune and retain one immutable recipe");
    auto repeat = cache.getOrTune(a, [&]() -> Fp8RKSelection {
        throw std::logic_error("Cache hit unexpectedly retuned");
    });
    check(repeat.winner_index == 4 && cache.tuningAttempts() == 1,
          "Repeated identical selections must not retune");
    auto changed = a;
    changed.workspace_budget_bytes++;
    check(!(changed == a), "Workspace budget must be in key");
    changed = a;
    changed.output_scale_bits++;
    check(!(changed == a), "Output scale must be in key");
    changed = a;
    changed.device_uuid[0]++;
    check(!(changed == a), "Device identity must be in key");
    changed = a;
    changed.runs.front().extent++;
    check(!(changed == a), "Geometry must be in key");
    changed = a;
    changed.operation++;
    check(!(changed == a), "Operation must be in key");

    (void)cache.getOrTune(b, [&] { ++calls; return recipe(8); });
    (void)cache.getOrTune(keyFor(48), [&] { ++calls; return recipe(12); });
    check(cache.size() == 2, "Bounded LRU must evict the oldest recipe");
    check(cache.getOrTune(a, [&] { ++calls; return recipe(9); }).winner_index == 9 && calls == 4,
          "Evicted recipe must be retuned");

    // All concurrent requests for a key share a single tuning flight.
    cache.clear();
    const auto before = cache.tuningAttempts();
    std::promise<void> release;
    const auto gate = release.get_future().share();
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    std::atomic<int> concurrent_calls{0};
    std::vector<std::future<Fp8RKSelection>> workers;
    for (int i = 0; i < 12; ++i) {
        workers.push_back(std::async(std::launch::async, [&] {
            return cache.getOrTune(a, [&] {
                if (++concurrent_calls == 1) entered.set_value();
                gate.wait();
                return recipe(13);
            });
        }));
    }
    entered_future.wait();
    release.set_value();
    for (auto& worker : workers)
        check(worker.get().winner_index == 13, "Concurrent waiters must receive the same selection");
    check(concurrent_calls == 1 && cache.tuningAttempts() == before + 1,
          "Same-key concurrent misses must execute exactly one tune");

    // A failed flight does not poison future requests.
    cache.clear();
    bool threw = false;
    try {
        (void)cache.getOrTune(a, []() -> Fp8RKSelection {
            throw std::runtime_error("failed measurement");
        });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw && cache.size() == 0, "Failure must propagate without caching");
    check(cache.getOrTune(a, [] { return recipe(17); }).winner_index == 17,
          "A failed key must be retryable");

    // clear() while tuning must not let an old flight repopulate the cache
    // or remove a newer same-key flight.
    cache.clear();
    std::promise<void> old_release;
    const auto old_gate = old_release.get_future().share();
    std::promise<void> old_started;
    auto old_started_future = old_started.get_future();
    auto old = std::async(std::launch::async, [&] {
        return cache.getOrTune(a, [&] {
            old_started.set_value();
            old_gate.wait();
            return recipe(21);
        });
    });
    old_started_future.wait();
    cache.clear();
    check(cache.getOrTune(a, [] { return recipe(22); }).winner_index == 22,
          "A cache clear must permit a fresh flight");
    old_release.set_value();
    check(old.get().winner_index == 21, "Existing callers may finish old flights");
    check(cache.getOrTune(a, []() -> Fp8RKSelection {
        throw std::logic_error("Stale flight replaced new cached recipe");
    }).winner_index == 22, "Stale tuning completion must not overwrite current cache");
}
