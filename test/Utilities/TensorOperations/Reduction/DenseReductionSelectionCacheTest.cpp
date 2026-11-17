// The one dense reduction selection cache (DenseReductionSelectionCache.h). Host-only: recipe-only storage, bounded
// LRU, single flight, failure and retry, clear() racing a flight, the depth rule (completed and in flight), kinds kept
// apart, per-kind attempt counts, and the recursion guard (direct, indirect and after a clear); the RK recipe's copy
// contract and the RK descriptor key.

#include "Utilities/TensorOperations/Reduction/KR/Fp8/Fp8KRStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/RK/RKDescriptorSelection.h"
#include "Utilities/TensorOperations/Reduction/RK/RKSelectionKey.h"

#include <gtest/gtest.h>

#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionInternal;
using RK::RKSelection;
using KR::Fp8::Fp8KRSelection;
using Depth = ReductionTuningDepth;
using namespace std::chrono_literals;

namespace {

RKSelection rkRecipe(size_t winner) {
    RKSelection value;
    value.physical.family = RK::PhysicalFamily::RuntimeK;
    value.candidate_count = 28;
    value.winner_index = winner;
    value.winner_name = "runtime_k";
    return value;
}

Fp8KRSelection krRecipe(size_t candidates) {
    Fp8KRSelection value;
    value.candidate_count = candidates;
    value.winner_name = "fp8_kr:test";
    return value;
}

RK::RKSelectionKey rkDescriptor(uint64_t outer) {
    RK::RKSelectionKey key;
    key.device_uuid[0] = 51;
    key.device_ordinal = 0;
    key.compute_major = 12;
    key.sm_count = 170;
    key.workspace_budget_bytes = 256 * 1024 * 1024;
    key.original_reduction_size = 512;
    key.output_scale_bits = 0x3f800000;  // 1.0f
    key.input_dtype = 22;
    key.output_dtype = 7;
    key.runs = {{outer, 0, 0}, {512, 0, 1}, {64, 0, 0}};
    key.site_run_index = 1;
    return key;
}

DenseReductionSelectionKey rkKey(uint64_t outer, Depth depth = Depth::Regular) {
    RK::RKSelectionKey descriptor = rkDescriptor(outer);
    descriptor.tuning_depth = static_cast<uint8_t>(depth);
    return RK::toDenseReductionSelectionKey(descriptor);
}

DenseReductionSelectionKey krKey(std::vector<uint64_t> site, Depth depth = Depth::Regular) {
    DenseReductionSelectionKey key;
    key.kind = DenseReductionSelectionKind::Fp8KR;
    key.device.uuid[0] = 51;
    key.device.ordinal = 0;
    key.tuning_depth = depth;
    key.site = std::move(site);
    return key;
}

template <class T>
concept HasWorkspaceBytes = requires(T recipe) { recipe.workspace_bytes; };

template <class T>
T ready(std::future<T>& result) {
    EXPECT_EQ(result.wait_for(5s), std::future_status::ready);
    return result.get();
}

}  // namespace

TEST(DenseReductionSelectionCache, StoresImmutableRecipesOnly) {
    static_assert(AcceleratorBackendSelectionRecipe<RKSelection>);
    static_assert(AcceleratorBackendSelectionRecipe<Fp8KRSelection>);
    static_assert(!isAcceleratorBackendLocalExecutionStateV<RKSelection>);
    static_assert(!isAcceleratorBackendLocalExecutionStateV<Fp8KRSelection>);
    static_assert(std::is_copy_constructible_v<RKSelection>);
    static_assert(std::is_copy_assignable_v<RKSelection>);
    // A recipe carries no stamp-local workspace size.
    static_assert(!HasWorkspaceBytes<RKSelection>);
    const RKSelection selection = rkRecipe(7);
    const RKSelection copy = selection;
    EXPECT_TRUE(copy.physical == selection.physical);
    EXPECT_EQ(copy.candidate_count, 28U);
    EXPECT_EQ(copy.winner_index, 7U);
    EXPECT_EQ(copy.winner_name, "runtime_k");
}

// The descriptor request builds the whole key: device identity, dtypes, output scale, the nominated site, the
// original runs and the finalizer. Equal requests give equal keys.
TEST(DenseReductionSelectionCache, RkDescriptorRequestsBuildTheWholeKey) {
    constexpr auto sum = static_cast<ReductionOp>(0);
    constexpr uint64_t budget = uint64_t{256} * 1024 * 1024;
    RK::RKDeviceSelectionIdentity device;
    device.uuid[0] = 0x51;
    device.uuid[15] = 0xA7;
    device.ordinal = 0;
    device.sm_count = 170;
    device.compute_major = 12;
    device.compute_minor = 0;
    DenseReductionProblem problem;
    problem.runs = {{DenseReductionRunKind::Retained, 16, 0},
                    {DenseReductionRunKind::OriginalReduction, 512, 0},
                    {DenseReductionRunKind::Retained, 64, 0}};
    problem.final_output.value_dtype = DataType::FP32;
    const auto key =
        RK::makeRKDescriptorSelectionKey(problem, {1}, sum, 512, 1.0f, DataType::FP8_E4M3, device, budget);
    EXPECT_TRUE(key == RK::makeRKDescriptorSelectionKey(problem, {1}, sum, 512, 1.0f, DataType::FP8_E4M3, device,
                                                          budget));
    EXPECT_EQ(key.device_uuid, device.uuid);
    EXPECT_EQ(key.device_ordinal, device.ordinal);
    EXPECT_EQ(key.sm_count, 170U);
    EXPECT_EQ(key.compute_major, 12U);
    EXPECT_EQ(key.compute_minor, 0U);
    EXPECT_EQ(key.input_dtype, 22U);
    EXPECT_EQ(key.output_dtype, 11U);
    EXPECT_EQ(key.output_scale_bits, std::bit_cast<uint32_t>(1.0f));
    EXPECT_EQ(key.original_reduction_size, 512U);
    EXPECT_EQ(key.workspace_budget_bytes, budget);
    EXPECT_EQ(key.site_run_index, 1U);
    ASSERT_EQ(key.runs.size(), 3U);
    EXPECT_EQ(key.runs[1].kind, static_cast<uint8_t>(DenseReductionRunKind::OriginalReduction));
    EXPECT_FALSE(key == RK::makeRKDescriptorSelectionKey(problem, {1}, sum, 512, 1.0f, DataType::FP8_E5M2, device,
                                                          budget));
    problem.finalizer_pending = false;
    EXPECT_FALSE(key == RK::makeRKDescriptorSelectionKey(problem, {1}, sum, 512, 1.0f, DataType::FP8_E4M3, device,
                                                          budget));
}

TEST(DenseReductionSelectionCache, RkDescriptorFieldsAreAllInTheKey) {
    const DenseReductionSelectionKey a = rkKey(16);
    EXPECT_EQ(a, rkKey(16));
    const auto differs = [&](auto edit) {
        RK::RKSelectionKey descriptor = rkDescriptor(16);
        descriptor.tuning_depth = static_cast<uint8_t>(Depth::Regular);
        edit(descriptor);
        return !(RK::toDenseReductionSelectionKey(descriptor) == a);
    };
    EXPECT_TRUE(differs([](auto& k) { k.workspace_budget_bytes++; }));
    EXPECT_TRUE(differs([](auto& k) { k.output_scale_bits++; }));
    EXPECT_TRUE(differs([](auto& k) { k.device_uuid[0]++; }));
    EXPECT_TRUE(differs([](auto& k) { k.sm_count++; }));
    EXPECT_TRUE(differs([](auto& k) { k.runs.front().extent++; }));
    EXPECT_TRUE(differs([](auto& k) { k.runs.push_back({1, 0, 0}); }));
    EXPECT_TRUE(differs([](auto& k) { k.operation++; }));
    EXPECT_TRUE(differs([](auto& k) { k.finalizer_pending = !k.finalizer_pending; }));
    EXPECT_TRUE(differs([](auto& k) { k.tuning_depth = static_cast<uint8_t>(Depth::Ultra); }));
    EXPECT_EQ(rkKey(16).kind, DenseReductionSelectionKind::Fp8RK);
}

TEST(DenseReductionSelectionCache, HitsDoNotRetuneAndTheLruIsBounded) {
    DenseReductionSelectionCache cache(2);
    std::atomic<int> calls{0};
    const auto a = rkKey(16);
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, [&] { ++calls; return rkRecipe(4); }).winner_index, 4U);
    EXPECT_EQ(cache.size(), 1U);
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, []() -> RKSelection {
                  throw std::logic_error("cache hit retuned");
              }).winner_index, 4U);
    EXPECT_EQ(cache.tuningAttempts(), 1U);

    (void)cache.getOrTune<RKSelection>(rkKey(32), [&] { ++calls; return rkRecipe(8); });
    (void)cache.getOrTune<RKSelection>(rkKey(48), [&] { ++calls; return rkRecipe(12); });
    EXPECT_EQ(cache.size(), 2U);  // the oldest (a) was evicted
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, [&] { ++calls; return rkRecipe(9); }).winner_index, 9U);
    EXPECT_EQ(calls.load(), 4);
}

TEST(DenseReductionSelectionCache, ConcurrentMissesShareOneFlight) {
    DenseReductionSelectionCache cache;
    const auto a = rkKey(16);
    std::promise<void> release;
    const auto gate = release.get_future().share();
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    std::atomic<int> concurrent_calls{0};
    std::vector<std::future<RKSelection>> workers;
    for (int i = 0; i < 12; ++i) {
        workers.push_back(std::async(std::launch::async, [&] {
            return cache.getOrTune<RKSelection>(a, [&] {
                if (++concurrent_calls == 1) entered.set_value();
                gate.wait();
                return rkRecipe(13);
            });
        }));
    }
    entered_future.wait();
    release.set_value();
    for (auto& worker : workers) EXPECT_EQ(ready(worker).winner_index, 13U);
    EXPECT_EQ(concurrent_calls.load(), 1);
    EXPECT_EQ(cache.tuningAttempts(), 1U);
}

TEST(DenseReductionSelectionCache, AFailedFlightIsNotCachedAndCanBeRetried) {
    DenseReductionSelectionCache cache;
    const auto a = rkKey(16);
    EXPECT_THROW((void)cache.getOrTune<RKSelection>(
                     a, []() -> RKSelection { throw std::runtime_error("failed measurement"); }),
                 std::runtime_error);
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, [] { return rkRecipe(17); }).winner_index, 17U);
}

TEST(DenseReductionSelectionCache, ClearDuringAFlightDoesNotLetItRepopulate) {
    DenseReductionSelectionCache cache;
    const auto a = rkKey(16);
    std::promise<void> old_release;
    const auto old_gate = old_release.get_future().share();
    std::promise<void> old_started;
    auto old_started_future = old_started.get_future();
    auto old = std::async(std::launch::async, [&] {
        return cache.getOrTune<RKSelection>(a, [&] {
            old_started.set_value();
            old_gate.wait();
            return rkRecipe(21);
        });
    });
    old_started_future.wait();
    cache.clear();
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, [] { return rkRecipe(22); }).winner_index, 22U);
    old_release.set_value();
    EXPECT_EQ(ready(old).winner_index, 21U);  // the old caller still gets its own result
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, []() -> RKSelection {
                  throw std::logic_error("stale flight replaced the new recipe");
              }).winner_index, 22U);
}

TEST(DenseReductionSelectionCache, DeeperAnswersShallowerButNotTheReverse) {
    DenseReductionSelectionCache cache;
    int calls = 0;
    (void)cache.getOrTune<RKSelection>(rkKey(16, Depth::Ultra), [&] { ++calls; return rkRecipe(1); });
    for (const Depth depth : {Depth::Quick, Depth::Regular, Depth::Ultra}) {
        EXPECT_EQ(cache.getOrTune<RKSelection>(rkKey(16, depth), [&] { ++calls; return rkRecipe(99); })
                      .winner_index, 1U);
    }
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(cache.findAtLeast<RKSelection>(rkKey(16, Depth::Quick)).has_value());
    EXPECT_FALSE(cache.find<RKSelection>(rkKey(16, Depth::Quick)).has_value());  // find is exact

    (void)cache.getOrTune<RKSelection>(rkKey(32, Depth::Quick), [&] { ++calls; return rkRecipe(2); });
    EXPECT_FALSE(cache.findAtLeast<RKSelection>(rkKey(32, Depth::Regular)).has_value());
    EXPECT_EQ(cache.getOrTune<RKSelection>(rkKey(32, Depth::Regular), [&] { ++calls; return rkRecipe(3); })
                  .winner_index, 3U);
    EXPECT_EQ(cache.getOrTune<RKSelection>(rkKey(32, Depth::Quick), [&] { ++calls; return rkRecipe(98); })
                  .winner_index, 3U);  // the deeper entry is preferred
    EXPECT_EQ(calls, 3);
}

TEST(DenseReductionSelectionCache, AShallowerRequestJoinsADeeperFlight) {
    DenseReductionSelectionCache cache;
    std::promise<void> release;
    const auto gate = release.get_future().share();
    std::promise<void> started;
    auto started_future = started.get_future();
    auto ultra = std::async(std::launch::async, [&] {
        return cache.getOrTune<RKSelection>(rkKey(16, Depth::Ultra), [&] {
            started.set_value();
            gate.wait();
            return rkRecipe(5);
        });
    });
    started_future.wait();
    auto quick = std::async(std::launch::async, [&] {
        return cache.getOrTune<RKSelection>(rkKey(16, Depth::Quick), []() -> RKSelection {
            throw std::logic_error("a Quick request tuned while an Ultra flight was running");
        });
    });
    EXPECT_EQ(quick.wait_for(100ms), std::future_status::timeout);  // waiting on the Ultra flight
    release.set_value();
    EXPECT_EQ(ready(ultra).winner_index, 5U);
    EXPECT_EQ(ready(quick).winner_index, 5U);
    EXPECT_EQ(cache.tuningAttempts(), 1U);
}

TEST(DenseReductionSelectionCache, KindsAreKeptApartAndCountedSeparately) {
    DenseReductionSelectionCache cache;
    // The same site words under two kinds are two entries, each with its own recipe type.
    DenseReductionSelectionKey rk = rkKey(16);
    DenseReductionSelectionKey kr = rk;
    kr.kind = DenseReductionSelectionKind::Fp8KR;
    (void)cache.getOrTune<RKSelection>(rk, [] { return rkRecipe(7); });
    EXPECT_EQ(cache.getOrTune<Fp8KRSelection>(kr, [] { return krRecipe(6); }).candidate_count, 6U);
    EXPECT_EQ(cache.size(), 2U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp8RK), 1U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp8KR), 1U);
    EXPECT_EQ(cache.tuningAttempts(), 2U);
    (void)cache.getOrTune<Fp8KRSelection>(krKey({1, 2, 3}), [] { return krRecipe(12); });
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp8KR), 2U);
    // Asking a kind's entry for another recipe type is a programming error, not a silent miss.
    EXPECT_THROW((void)cache.find<Fp8KRSelection>(rk), std::logic_error);
}

TEST(DenseReductionSelectionCache, RecursiveSameSiteTuningIsRejected) {
    DenseReductionSelectionCache cache;
    const auto ultra = rkKey(16, Depth::Ultra);
    EXPECT_THROW((void)cache.getOrTune<RKSelection>(ultra, [&] {
                     // A shallower request for the same site inside its own deeper flight would wait on itself.
                     return cache.getOrTune<RKSelection>(rkKey(16, Depth::Quick), [] { return rkRecipe(1); });
                 }),
                 std::logic_error);
    // A different site may be tuned inside a flight (a selector depending on another selection).
    EXPECT_EQ(cache.getOrTune<RKSelection>(rkKey(16), [&] {
                  (void)cache.getOrTune<Fp8KRSelection>(krKey({9}), [] { return krRecipe(3); });
                  return rkRecipe(2);
              }).winner_index, 2U);
    EXPECT_EQ(cache.size(), 2U);
}

TEST(DenseReductionSelectionCache, IndirectAndPostClearSelfDependenciesAreRejected) {
    DenseReductionSelectionCache cache;
    const auto a = rkKey(31);
    const auto b = rkKey(32);
    bool rejected = false;
    // A tunes B, which asks for A again on the same thread.
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, [&] {
                  (void)cache.getOrTune<RKSelection>(b, [&] {
                      try {
                          (void)cache.getOrTune<RKSelection>(a, [] { return rkRecipe(3); });
                      } catch (const std::logic_error&) {
                          rejected = true;
                      }
                      return rkRecipe(5);
                  });
                  return rkRecipe(6);
              }).winner_index, 6U);
    EXPECT_TRUE(rejected);
    // A clear inside A's own flight does not let A depend on itself, and does not cache the old flight.
    cache.clear();
    rejected = false;
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, [&] {
                  cache.clear();
                  try {
                      (void)cache.getOrTune<RKSelection>(a, [] { return rkRecipe(7); });
                  } catch (const std::logic_error&) {
                      rejected = true;
                  }
                  return rkRecipe(8);
              }).winner_index, 8U);
    EXPECT_TRUE(rejected);
    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(cache.getOrTune<RKSelection>(a, [] { return rkRecipe(9); }).winner_index, 9U);
}

TEST(DenseReductionSelectionCache, ProcessCacheIsOneInstance) {
    EXPECT_EQ(&denseReductionSelectionCache(), &denseReductionSelectionCache());
}
