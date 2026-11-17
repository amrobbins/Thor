// The per-device GPU measurement lock that serializes reduction contests: a selection resolves the selections it
// depends on before it measures; a nested measurement on the same device throws instead of deadlocking; contests on
// one device never overlap, while CPU preparation and cache hits do not wait for the lock; devices are independent.
// Host-only (no CUDA).

#include "Utilities/TensorOperations/Reduction/Contest/ReductionGpuMeasurementLock.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/RK/RKSelectionKey.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <future>
#include <stdexcept>

using namespace ThorImplementation::ReductionInternal;
using namespace ThorImplementation::ReductionInternal::RK;
using namespace std::chrono_literals;

namespace {
DenseReductionSelectionKey keyFor(uint64_t id) {
    RKSelectionKey key;
    key.device_ordinal = 0;
    key.device_uuid[0] = 5;
    key.runs = {{id, 0, 0}, {512, 0, 1}, {64, 0, 0}};
    return toDenseReductionSelectionKey(key);
}

RKSelection recipe(size_t winner) {
    RKSelection answer;
    answer.physical.family = PhysicalFamily::RuntimeK;
    answer.candidate_count = 28;
    answer.winner_index = winner;
    answer.winner_name = "runtime_k";
    return answer;
}

template <class T>
bool isReady(std::future<T>& result, std::chrono::milliseconds wait = 5s) {
    return result.wait_for(wait) == std::future_status::ready;
}
}  // namespace

TEST(ReductionGpuMeasurementLock, ADependencyIsSelectedBeforeTheDependentMeasures) {
    DenseReductionSelectionCache staged;
    DenseReductionSelectionCache continuation;  // an injected dependency, not a production selector
    int continuation_contests = 0;
    int staged_contests = 0;
    const auto result = staged.getOrTune<RKSelection>(keyFor(1), [&] {
        const auto dependency = continuation.getOrTune<RKSelection>(keyFor(2), [&] {
            ScopedReductionGpuMeasurement measurement(0);
            ++continuation_contests;
            return recipe(2);
        });
        EXPECT_EQ(dependency.winner_index, 2U);
        ScopedReductionGpuMeasurement measurement(0);
        ++staged_contests;
        return recipe(8);
    });
    EXPECT_EQ(result.winner_index, 8U);
    EXPECT_EQ(continuation_contests, 1);
    EXPECT_EQ(staged_contests, 1);
}

TEST(ReductionGpuMeasurementLock, ANestedSameDeviceMeasurementThrows) {
    {
        ScopedReductionGpuMeasurement outer(0);
        EXPECT_THROW(ScopedReductionGpuMeasurement inner(0), std::logic_error);
        ScopedReductionGpuMeasurement other_device(1);
    }
    // The failed attempt did not poison the lock.
    ScopedReductionGpuMeasurement reusable(0);
}

TEST(ReductionGpuMeasurementLock, SameDeviceContestsSerializeButPreparationAndHitsDoNot) {
    DenseReductionSelectionCache cache;
    const auto a = keyFor(51);
    const auto b = keyFor(52);
    const auto hit = keyFor(53);
    (void)cache.getOrTune<RKSelection>(hit, [] { return recipe(5); });
    std::promise<void> first_entered, first_release, second_started, second_entered;
    auto first_entered_future = first_entered.get_future();
    const auto first_gate = first_release.get_future().share();
    auto second_started_future = second_started.get_future();
    auto second_entered_future = second_entered.get_future();
    auto first = std::async(std::launch::async, [&] {
        return cache.getOrTune<RKSelection>(a, [&] {
            ScopedReductionGpuMeasurement measurement(0);
            first_entered.set_value();
            first_gate.wait();
            return recipe(1);
        });
    });
    ASSERT_TRUE(isReady(first_entered_future));
    auto second = std::async(std::launch::async, [&] {
        return cache.getOrTune<RKSelection>(b, [&] {
            second_started.set_value();  // CPU preparation runs outside the lock
            ScopedReductionGpuMeasurement measurement(0);
            second_entered.set_value();
            return recipe(2);
        });
    });
    EXPECT_TRUE(isReady(second_started_future));
    EXPECT_FALSE(isReady(second_entered_future, 25ms));  // the device is still held by the first contest
    // An unrelated hit does not wait for the held lock.
    EXPECT_EQ(cache.getOrTune<RKSelection>(hit, []() -> RKSelection {
                  throw std::logic_error("a cache hit retuned while the GPU was busy");
              }).winner_index, 5U);
    first_release.set_value();
    ASSERT_TRUE(isReady(first));
    EXPECT_EQ(first.get().winner_index, 1U);
    EXPECT_TRUE(isReady(second_entered_future));
    ASSERT_TRUE(isReady(second));
    EXPECT_EQ(second.get().winner_index, 2U);
}

TEST(ReductionGpuMeasurementLock, DevicesHaveIndependentLocks) {
    std::promise<void> entered0, entered1, release0;
    auto entered0_future = entered0.get_future();
    auto entered1_future = entered1.get_future();
    const auto release_gate = release0.get_future().share();
    auto first = std::async(std::launch::async, [&] {
        ScopedReductionGpuMeasurement measurement(0);
        entered0.set_value();
        release_gate.wait();
    });
    ASSERT_TRUE(isReady(entered0_future));
    auto second = std::async(std::launch::async, [&] {
        ScopedReductionGpuMeasurement measurement(1);
        entered1.set_value();
    });
    const bool independent = isReady(entered1_future, 500ms);
    release0.set_value();  // released before checking, so no worker is left waiting
    EXPECT_TRUE(isReady(first));
    EXPECT_TRUE(independent);
    EXPECT_TRUE(isReady(second));
}
