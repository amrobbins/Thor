// Host-only P1-D test. No CUDA initialization or physical kernels required.
// Tests the shared device measurement lock and the FP8 cache lifecycle with
// deterministic, injectable nested selection/measurement callbacks.
#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionCache.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ThorImplementation::CubReductionInternal;
using namespace ThorImplementation::CubReductionInternal::Fp8RK;
using namespace std::chrono_literals;

namespace {
void check(bool okay, const char* message) {
    if (!okay) throw std::logic_error(message);
}
Fp8RKSelectionKey keyFor(uint64_t id) {
    Fp8RKSelectionKey key;
    key.device_ordinal = 0;
    key.device_uuid[0] = 5;
    key.runs = {{id, 0, 0}, {512, 0, 1}, {64, 0, 0}};
    return key;
}
Fp8RKSelection recipe(size_t winner) {
    Fp8RKSelection answer;
    answer.physical.family = PhysicalFamily::RuntimeK;
    answer.candidate_count = 28;
    answer.winner_index = winner;
    answer.winner_name = "runtime_k";
    return answer;
}
template <class T>
T ready(std::future<T>& result, const char* message) {
    check(result.wait_for(5s) == std::future_status::ready, message);
    return result.get();
}
void ready(std::future<void>& result, const char* message) {
    check(result.wait_for(5s) == std::future_status::ready, message);
    result.get();
}

void testNestedDependencyAndCacheHits() {
    Fp8RKSelectionCache fp8;
    Fp8RKSelectionCache mock_fp32; // injected callback, not a second production FP32 policy
    int fp32_contests = 0;
    int fp8_contests = 0;
    const auto result = fp8.getOrTune(keyFor(1), [&] {
        // A future FP32 selector can itself miss and measure before the FP8
        // GPU contest. The same shared mutex is used for both domains.
        auto continuation = mock_fp32.getOrTune(keyFor(2), [&] {
            ScopedReductionGpuMeasurement fp32_measurement(0);
            ++fp32_contests;
            return recipe(2);
        });
        check(continuation.winner_index == 2, "Injected FP32 dependency selection failed");
        ScopedReductionGpuMeasurement fp8_measurement(0);
        ++fp8_contests;
        return recipe(8);
    });
    check(result.winner_index == 8 && fp32_contests == 1 && fp8_contests == 1,
          "Nested selection must resolve dependencies before FP8 measurement");
    auto cached = fp8.getOrTune(keyFor(1), [&]() -> Fp8RKSelection {
        throw std::logic_error("Cached FP8 request resolved hypothetical continuation");
    });
    check(cached.winner_index == 8 && fp8.tuningAttempts() == 1,
          "Cache hit must return without dependency selection or measurement");
    // Contests aren't recursive locks: an accidental nested measurement
    // throws promptly, and doesn't poison the original lock.
    {
        ScopedReductionGpuMeasurement outer(0);
        bool rejected = false;
        try {
            ScopedReductionGpuMeasurement inner(0);
        } catch (const std::logic_error&) {
            rejected = true;
        }
        check(rejected, "Nested same-device measurement must throw rather than deadlock");
        ScopedReductionGpuMeasurement different_device(1);
    }
    ScopedReductionGpuMeasurement reusable(0);
}

void testRecursiveCacheDependencies() {
    Fp8RKSelectionCache cache;
    auto a = keyFor(31);
    auto b = keyFor(32);
    bool rejected = false;
    auto selected = cache.getOrTune(a, [&] {
        try {
            (void)cache.getOrTune(a, [] { return recipe(3); });
        } catch (const std::logic_error&) {
            rejected = true;
        }
        return recipe(4);
    });
    check(rejected && selected.winner_index == 4 && cache.tuningAttempts() == 1,
          "Same-thread same-key recursion must fail without creating a second flight");
    cache.clear();
    rejected = false;
    selected = cache.getOrTune(a, [&] {
        (void)cache.getOrTune(b, [&] {
            try {
                (void)cache.getOrTune(a, [] { return recipe(3); });
            } catch (const std::logic_error&) {
                rejected = true;
            }
            return recipe(5);
        });
        return recipe(6);
    });
    check(rejected && selected.winner_index == 6,
          "Indirect same-thread A-to-B-to-A tuning cycle must fail");
    cache.clear();
    rejected = false;
    selected = cache.getOrTune(a, [&] {
        cache.clear(); // active callback still owns the old generation
        try {
            (void)cache.getOrTune(a, [] { return recipe(7); });
        } catch (const std::logic_error&) {
            rejected = true;
        }
        return recipe(8);
    });
    check(rejected && selected.winner_index == 8 && cache.size() == 0,
          "Cache clear during tune must not permit recursive self-dependency");
    check(cache.getOrTune(a, [] { return recipe(9); }).winner_index == 9,
          "Old generation must not prevent a later fresh selection");
    bool failed = false;
    cache.clear();
    try {
        (void)cache.getOrTune(a, []() -> Fp8RKSelection {
            throw std::runtime_error("measurement failed");
        });
    } catch (const std::runtime_error&) {
        failed = true;
    }
    check(failed && cache.size() == 0 &&
          cache.getOrTune(a, [] { return recipe(11); }).winner_index == 11,
          "Failed tuning must be uncached and retryable");
}

void testPerDeviceSerializationAndNonblockingHits() {
    Fp8RKSelectionCache cache;
    auto a = keyFor(51), b = keyFor(52), hit = keyFor(53);
    (void)cache.getOrTune(hit, [] { return recipe(5); });
    std::promise<void> first_entered, first_release, second_started, second_entered;
    auto first_entered_f = first_entered.get_future();
    auto first_gate = first_release.get_future().share();
    auto second_started_f = second_started.get_future();
    auto second_entered_f = second_entered.get_future();
    auto first = std::async(std::launch::async, [&] {
        return cache.getOrTune(a, [&] {
            ScopedReductionGpuMeasurement measurement(0);
            first_entered.set_value();
            first_gate.wait();
            return recipe(1);
        });
    });
    ready(first_entered_f, "First contest did not enter its GPU timing window");
    auto second = std::async(std::launch::async, [&] {
        return cache.getOrTune(b, [&] {
            // Dependency-selection/CPU preparation is outside the mutex.
            second_started.set_value();
            ScopedReductionGpuMeasurement measurement(0);
            second_entered.set_value();
            return recipe(2);
        });
    });
    ready(second_started_f, "Independent cache miss was serialized before preparation");
    check(second_entered_f.wait_for(25ms) == std::future_status::timeout,
          "Same-device measurement contests overlapped");
    // This hit must not wait for the unrelated still-held GPU measurement lock.
    auto fast_hit = cache.getOrTune(hit, []() -> Fp8RKSelection {
        throw std::logic_error("Cache hit unexpectedly retuned while GPU was busy");
    });
    check(fast_hit.winner_index == 5, "Unrelated cache hit stalled or changed");
    first_release.set_value();
    check(ready(first, "First GPU contest never completed").winner_index == 1,
          "First GPU contest returned wrong result");
    ready(second_entered_f, "Second contest never acquired device after release");
    check(ready(second, "Second GPU contest never completed").winner_index == 2,
          "Second GPU contest returned wrong result");
}

void testIndependentGpus() {
    std::promise<void> entered0, entered1, release0;
    auto entered0_f = entered0.get_future();
    auto entered1_f = entered1.get_future();
    auto release_gate = release0.get_future().share();
    auto first = std::async(std::launch::async, [&] {
        ScopedReductionGpuMeasurement measurement(0);
        entered0.set_value();
        release_gate.wait();
    });
    ready(entered0_f, "GPU 0 contest did not enter");
    auto second = std::async(std::launch::async, [&] {
        ScopedReductionGpuMeasurement measurement(1);
        entered1.set_value();
    });
    // Release GPU 0 even if the other GPU's check fails, so no worker sticks.
    const bool independent = entered1_f.wait_for(500ms) == std::future_status::ready;
    release0.set_value();
    ready(first, "GPU 0 contest did not exit");
    check(independent, "Independent GPU contests must not share a global mutex");
    ready(second, "GPU 1 contest did not exit");
}

void testSameKeySingleFlight() {
    Fp8RKSelectionCache cache;
    const auto key = keyFor(71);
    std::promise<void> started, release;
    auto started_f = started.get_future();
    auto gate = release.get_future().share();
    std::atomic<int> calls{0};
    auto leader = std::async(std::launch::async, [&] {
        return cache.getOrTune(key, [&] {
            ++calls;
            started.set_value();
            gate.wait();
            ScopedReductionGpuMeasurement measurement(2);
            return recipe(3);
        });
    });
    ready(started_f, "Leader didn't begin tuning");
    std::vector<std::future<Fp8RKSelection>> waiters;
    for (int i = 0; i < 8; ++i) {
        waiters.push_back(std::async(std::launch::async, [&] {
            return cache.getOrTune(key, [&] {
                ++calls;
                return recipe(99);
            });
        }));
    }
    // Independently launched callers may observe either the in-flight tune
    // or the freshly populated cache; neither may run a second tune.
    // Clear/new-flight races are covered by Fp8RKSelectionCacheSmoke.
    release.set_value();
    check(ready(leader, "Leader did not finish").winner_index == 3, "Wrong leader selection");
    for (auto& waiter : waiters)
        check(ready(waiter, "Waiter did not finish").winner_index == 3,
              "Waiter must receive same-key winner");
    check(calls == 1 && cache.tuningAttempts() == 1,
          "Concurrent same-key misses must use exactly one tune");
}
} // namespace

int main() {
    try {
        testNestedDependencyAndCacheHits();
        testRecursiveCacheDependencies();
        testPerDeviceSerializationAndNonblockingHits();
        testIndependentGpus();
        testSameKeySingleFlight();
        std::cout << "Reduction GPU measurement lifecycle smoke passed "
                  << "(nested selection, locks, cache cycles, GPU isolation, single-flight)\n";
    } catch (const std::exception& error) {
        std::cerr << "Reduction GPU measurement lifecycle smoke failed: " << error.what() << '\n';
        return 1;
    }
}
