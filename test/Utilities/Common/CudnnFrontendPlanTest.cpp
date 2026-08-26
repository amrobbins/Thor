#include "Utilities/Common/CudnnFrontendPlan.h"

#include "gtest/gtest.h"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <future>
#include <optional>
#include <thread>
#include <type_traits>
#include <vector>

using namespace ThorImplementation;
using namespace std;

static_assert(AcceleratorBackendSelectionRecipe<CudnnFrontendPlanSelection>);
static_assert(is_copy_constructible_v<CudnnFrontendPlanSelection>);
static_assert(is_copy_assignable_v<CudnnFrontendPlanSelection>);
static_assert(AcceleratorBackendLocalExecutionState<CudnnFrontendExecutablePlan>);
static_assert(!is_copy_constructible_v<CudnnFrontendExecutablePlan>);
static_assert(!is_copy_assignable_v<CudnnFrontendExecutablePlan>);
static_assert(is_move_constructible_v<CudnnFrontendExecutablePlan>);
static_assert(is_move_assignable_v<CudnnFrontendExecutablePlan>);

TEST(CudnnFrontendPlanSelection, CanonicalizesKnobsDeterministically) {
    CudnnFrontendPlanSelection selection(17, {{9, 4}, {2, 8}, {6, -3}}, 4096);

    EXPECT_EQ(selection.engine_id, 17);
    EXPECT_EQ(selection.expected_workspace_bytes, 4096U);
    EXPECT_EQ(selection.knobs, (vector<pair<int64_t, int64_t>>{{2, 8}, {6, -3}, {9, 4}}));

    CudnnFrontendPlanSelection equivalent(17, {{6, -3}, {9, 4}, {2, 8}}, 4096);
    EXPECT_EQ(selection, equivalent);
}

TEST(CudnnFrontendPlanSelection, RejectsInvalidIdentity) {
    EXPECT_THROW((CudnnFrontendPlanSelection(-1, {}, 0)), invalid_argument);
    EXPECT_THROW((CudnnFrontendPlanSelection(3, {{7, 1}, {7, 2}}, 0)), invalid_argument);
    EXPECT_THROW((CudnnFrontendPlanSelection(3, {{0, 1}}, 0)), invalid_argument)
        << "KnobType_t::NOT_SET cannot be replayed through create_execution_plan().";
}

TEST(CudnnFrontendPlanSelection, SerializedReplayTokenCanPreserveFrontendUnrepresentableKnobIdentity) {
    const vector<uint8_t> replayToken{1, 7, 3, 9};
    CudnnFrontendPlanSelection selection(3, {{0, 1}}, 64, replayToken);

    EXPECT_EQ(selection.usesSerializedReplay(), true);
    EXPECT_EQ(selection.engine_id, 3);
    EXPECT_EQ(selection.knobs, (vector<pair<int64_t, int64_t>>{{0, 1}}));
    EXPECT_EQ(selection.serialized_plan, replayToken);
    EXPECT_EQ(selection.expected_workspace_bytes, 64U);
}

TEST(CudnnFrontendPlanSelection, SerializedAutotuneWinnerDoesNotRequireStructuredEngineIdentity) {
    const vector<uint8_t> replayToken{4, 2, 8, 6};
    CudnnFrontendPlanSelection selection(-1, {}, 192, replayToken);

    EXPECT_EQ(selection.usesSerializedReplay(), true);
    EXPECT_EQ(selection.engine_id, -1);
    EXPECT_EQ(selection.knobs.empty(), true);
    EXPECT_EQ(selection.serialized_plan, replayToken);
    EXPECT_EQ(selection.expected_workspace_bytes, 192U);
}

TEST(CudnnFrontendPlanSelectionCache, RejectsInvalidSelectorResultBeforePublication) {
    CudnnFrontendPlanSelectionCache<int> cache(4);

    EXPECT_THROW((void)cache.getOrSelect(1, []() { return CudnnFrontendPlanSelection{}; }), invalid_argument);
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.missCount(), 1U);

    const CudnnFrontendPlanSelection valid = cache.getOrSelect(1, []() { return CudnnFrontendPlanSelection(13, {{8, 2}, {3, 9}}, 128); });
    EXPECT_EQ(valid, (CudnnFrontendPlanSelection(13, {{3, 9}, {8, 2}}, 128)));
    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(cache.missCount(), 2U);
}

TEST(CudnnFrontendPlanSelectionCache, OneMissThenRepeatedHits) {
    CudnnFrontendPlanSelectionCache<int> cache(4);
    atomic<int> selector_calls{0};

    const auto selector = [&]() {
        selector_calls.fetch_add(1, memory_order_relaxed);
        return CudnnFrontendPlanSelection(5, {{3, 11}}, 256);
    };

    const CudnnFrontendPlanSelection first = cache.getOrSelect(42, selector);
    const CudnnFrontendPlanSelection second = cache.getOrSelect(42, selector);
    const CudnnFrontendPlanSelection third = cache.getOrSelect(42, selector);

    EXPECT_EQ(first, second);
    EXPECT_EQ(second, third);
    EXPECT_EQ(selector_calls.load(memory_order_relaxed), 1);
    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(cache.missCount(), 1U);
    EXPECT_EQ(cache.hitCount(), 2U);
}

TEST(CudnnFrontendPlanSelectionCache, ConcurrentEquivalentLookupsSingleFlight) {
    constexpr size_t num_callers = 8;
    CudnnFrontendPlanSelectionCache<int> cache(4);
    atomic<int> selector_calls{0};
    barrier start_gate(static_cast<ptrdiff_t>(num_callers + 1));
    array<optional<CudnnFrontendPlanSelection>, num_callers> results;
    vector<thread> callers;
    callers.reserve(num_callers);

    for (size_t i = 0; i < num_callers; ++i) {
        callers.emplace_back([&, i]() {
            start_gate.arrive_and_wait();
            results[i] = cache.getOrSelect(9, [&]() {
                selector_calls.fetch_add(1, memory_order_relaxed);
                this_thread::sleep_for(chrono::milliseconds(20));
                return CudnnFrontendPlanSelection(31, {{4, 2}, {1, 7}}, 1024);
            });
        });
    }

    start_gate.arrive_and_wait();
    for (thread& caller : callers) {
        caller.join();
    }

    ASSERT_EQ(selector_calls.load(memory_order_relaxed), 1);
    ASSERT_EQ(cache.missCount(), 1U);
    EXPECT_EQ(cache.hitCount(), num_callers - 1);
    EXPECT_EQ(cache.size(), 1U);
    for (const auto& result : results) {
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), (CudnnFrontendPlanSelection(31, {{1, 7}, {4, 2}}, 1024)));
    }
}

TEST(CudnnFrontendPlanSelectionCache, ClearCannotBeUndoneByOlderInFlightSelection) {
    CudnnFrontendPlanSelectionCache<int> cache(4);
    atomic<int> selector_calls{0};
    promise<void> selector_entered;
    promise<void> release_selector;
    shared_future<void> release = release_selector.get_future().share();
    optional<CudnnFrontendPlanSelection> first_result;

    thread first([&]() {
        first_result = cache.getOrSelect(7, [&]() {
            selector_calls.fetch_add(1, memory_order_relaxed);
            selector_entered.set_value();
            release.wait();
            return CudnnFrontendPlanSelection(19, {{2, 5}}, 512);
        });
    });

    selector_entered.get_future().wait();
    cache.clear();
    EXPECT_EQ(cache.size(), 0U);
    release_selector.set_value();
    first.join();

    ASSERT_TRUE(first_result.has_value());
    EXPECT_EQ(first_result.value(), (CudnnFrontendPlanSelection(19, {{2, 5}}, 512)));
    EXPECT_EQ(cache.size(), 0U) << "a selection started before clear() must not repopulate the cleared cache";

    const CudnnFrontendPlanSelection second = cache.getOrSelect(7, [&]() {
        selector_calls.fetch_add(1, memory_order_relaxed);
        return CudnnFrontendPlanSelection(23, {{2, 6}}, 768);
    });
    EXPECT_EQ(second, (CudnnFrontendPlanSelection(23, {{2, 6}}, 768)));
    EXPECT_EQ(selector_calls.load(memory_order_relaxed), 2);
    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(cache.missCount(), 2U);
}
