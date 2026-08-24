#include "DeepLearning/Implementation/Data/Residency/DeviceWindowL2Cache.h"
#include "Utilities/Common/PersistingL2Cache.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>

using Thor::DeviceWindowL2CacheLease;
using Thor::DeviceWindowL2CacheLeaseStatus;
using Thor::DeviceWindowL2CacheManager;
using Thor::DeviceWindowL2CacheSource;
using ThorImplementation::PersistingL2Capabilities;
using ThorImplementation::queryPersistingL2Capabilities;

namespace {

bool gpuAvailable() { return MachineEvaluator::instance().getNumGpus() > 0; }

const void *fakeDeviceAddress(uintptr_t value) {
    return reinterpret_cast<const void *>(value);
}

uint64_t expectedBudget(const PersistingL2Capabilities &capabilities) {
    return capabilities.current_persisting_bytes > 0
               ? std::min(capabilities.current_persisting_bytes,
                          capabilities.max_persisting_bytes)
               : capabilities.max_persisting_bytes;
}

}  // namespace

TEST(DeviceWindowL2CacheManagerTest, RejectsInvalidSourceWithoutThrowing) {
    auto &manager = DeviceWindowL2CacheManager::instance();
    manager.resetForTesting();

    DeviceWindowL2CacheLease lease = manager.acquire(
        DeviceWindowL2CacheSource{
            .deviceNum = -1,
            .tensorId = 1,
            .base = fakeDeviceAddress(0x1000),
            .bytes = 4096});
    EXPECT_EQ(lease.snapshot().status,
              DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT);

    lease = manager.acquire(DeviceWindowL2CacheSource{
        .deviceNum = 0,
        .tensorId = 0,
        .base = fakeDeviceAddress(0x1000),
        .bytes = 4096});
    EXPECT_EQ(lease.snapshot().status,
              DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT);

    lease = manager.acquire(DeviceWindowL2CacheSource{
        .deviceNum = 0,
        .tensorId = 1,
        .base = nullptr,
        .bytes = 4096});
    EXPECT_EQ(lease.snapshot().status,
              DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT);

    lease = manager.acquire(DeviceWindowL2CacheSource{
        .deviceNum = 0,
        .tensorId = 1,
        .base = fakeDeviceAddress(0x1000),
        .bytes = 0});
    EXPECT_EQ(lease.snapshot().status,
              DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT);
}

TEST(DeviceWindowL2CacheManagerTest,
     DeduplicatesSharedAllocationAndDoesNotAdvanceGenerationForRefcountOnlyChanges) {
    if (!gpuAvailable())
        GTEST_SKIP() << "window L2 cache manager test requires a GPU";

    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;

    const uint64_t budget = expectedBudget(capabilities);
    const uint64_t sourceBytes = std::min<uint64_t>(
        4096, std::min(budget, capabilities.max_access_policy_window_bytes));
    if (sourceBytes == 0)
        GTEST_SKIP() << "device exposes no usable persisting L2 source size";

    auto &manager = DeviceWindowL2CacheManager::instance();
    manager.resetForTesting();

    const DeviceWindowL2CacheSource source{
        .deviceNum = 0,
        .tensorId = 101,
        .base = fakeDeviceAddress(0x100000),
        .bytes = sourceBytes};
    DeviceWindowL2CacheLease first = manager.acquire(source);
    const auto firstSnapshot = first.snapshot();
    if (!firstSnapshot.active())
        GTEST_SKIP() << "persisting L2 manager unavailable: "
                     << firstSnapshot.detail;
    EXPECT_FLOAT_EQ(firstSnapshot.hitRatio, 1.0f);

    const auto oneLease = manager.telemetry(0);
    EXPECT_TRUE(oneLease.initialized);
    EXPECT_TRUE(oneLease.available);
    EXPECT_EQ(oneLease.activeUniqueSources, 1u);
    EXPECT_EQ(oneLease.activeUniqueBytes, sourceBytes);
    EXPECT_EQ(oneLease.activeLeases, 1u);
    EXPECT_EQ(oneLease.budgetBytes, budget);

    DeviceWindowL2CacheLease duplicate = manager.acquire(source);
    const auto duplicateSnapshot = duplicate.snapshot();
    ASSERT_TRUE(duplicateSnapshot.active()) << duplicateSnapshot.detail;
    EXPECT_FLOAT_EQ(duplicateSnapshot.hitRatio, 1.0f);

    const auto twoLeases = manager.telemetry(0);
    EXPECT_EQ(twoLeases.activeUniqueSources, 1u);
    EXPECT_EQ(twoLeases.activeUniqueBytes, sourceBytes);
    EXPECT_EQ(twoLeases.activeLeases, 2u);
    EXPECT_EQ(twoLeases.generation, oneLease.generation);

    duplicate = DeviceWindowL2CacheLease{};
    const auto backToOne = manager.telemetry(0);
    EXPECT_EQ(backToOne.activeUniqueSources, 1u);
    EXPECT_EQ(backToOne.activeUniqueBytes, sourceBytes);
    EXPECT_EQ(backToOne.activeLeases, 1u);
    EXPECT_EQ(backToOne.generation, oneLease.generation);

    first = DeviceWindowL2CacheLease{};
    const auto empty = manager.telemetry(0);
    EXPECT_EQ(empty.activeUniqueSources, 0u);
    EXPECT_EQ(empty.activeUniqueBytes, 0u);
    EXPECT_EQ(empty.activeLeases, 0u);
    EXPECT_GT(empty.generation, oneLease.generation);
}

TEST(DeviceWindowL2CacheManagerTest,
     OversubscribedSourcesShareBudgetAndRebalanceWhenOneLeaves) {
    if (!gpuAvailable())
        GTEST_SKIP() << "window L2 cache manager test requires a GPU";

    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;

    const uint64_t budget = expectedBudget(capabilities);
    if (budget == 0 || budget > capabilities.max_access_policy_window_bytes)
        GTEST_SKIP() << "test requires one legal access window as large as the persisting budget";

    auto &manager = DeviceWindowL2CacheManager::instance();
    manager.resetForTesting();

    DeviceWindowL2CacheLease first = manager.acquire(
        DeviceWindowL2CacheSource{
            .deviceNum = 0,
            .tensorId = 201,
            .base = fakeDeviceAddress(0x200000),
            .bytes = budget});
    const auto oneSource = first.snapshot();
    if (!oneSource.active())
        GTEST_SKIP() << "persisting L2 manager unavailable: " << oneSource.detail;
    EXPECT_FLOAT_EQ(oneSource.hitRatio, 1.0f);

    const uint64_t generationOneSource = oneSource.generation;
    DeviceWindowL2CacheLease second = manager.acquire(
        DeviceWindowL2CacheSource{
            .deviceNum = 0,
            .tensorId = 202,
            .base = fakeDeviceAddress(0x300000),
            .bytes = budget});
    const auto firstOversubscribed = first.snapshot();
    const auto secondOversubscribed = second.snapshot();
    ASSERT_TRUE(secondOversubscribed.active()) << secondOversubscribed.detail;
    EXPECT_FLOAT_EQ(firstOversubscribed.hitRatio, 0.5f);
    EXPECT_FLOAT_EQ(secondOversubscribed.hitRatio, 0.5f);
    EXPECT_LE(static_cast<long double>(firstOversubscribed.hitRatio) *
                  static_cast<long double>(budget * 2),
              static_cast<long double>(budget));
    EXPECT_EQ(firstOversubscribed.activeUniqueBytes, budget * 2);
    EXPECT_EQ(firstOversubscribed.budgetBytes, budget);
    EXPECT_GT(firstOversubscribed.generation, generationOneSource);

    const auto telemetry = manager.telemetry(0);
    EXPECT_EQ(telemetry.activeUniqueSources, 2u);
    EXPECT_EQ(telemetry.activeUniqueBytes, budget * 2);
    EXPECT_EQ(telemetry.activeLeases, 2u);

    const uint64_t oversubscribedGeneration = firstOversubscribed.generation;
    second = DeviceWindowL2CacheLease{};
    const auto rebalanced = first.snapshot();
    EXPECT_FLOAT_EQ(rebalanced.hitRatio, 1.0f);
    EXPECT_EQ(rebalanced.activeUniqueBytes, budget);
    EXPECT_GT(rebalanced.generation, oversubscribedGeneration);
}

TEST(DeviceWindowL2CacheManagerTest,
     OversizedSourceFallsBackWithoutConsumingBudget) {
    if (!gpuAvailable())
        GTEST_SKIP() << "window L2 cache manager test requires a GPU";

    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;
    if (capabilities.max_access_policy_window_bytes ==
        std::numeric_limits<uint64_t>::max())
        GTEST_SKIP() << "cannot construct an oversized uint64_t window";

    auto &manager = DeviceWindowL2CacheManager::instance();
    manager.resetForTesting();
    DeviceWindowL2CacheLease lease = manager.acquire(
        DeviceWindowL2CacheSource{
            .deviceNum = 0,
            .tensorId = 301,
            .base = fakeDeviceAddress(0x400000),
            .bytes = capabilities.max_access_policy_window_bytes + 1});
    const auto snapshot = lease.snapshot();
    if (snapshot.status == DeviceWindowL2CacheLeaseStatus::UNSUPPORTED ||
        snapshot.status == DeviceWindowL2CacheLeaseStatus::CUDA_ERROR)
        GTEST_SKIP() << "persisting L2 manager unavailable: " << snapshot.detail;
    EXPECT_EQ(snapshot.status,
              DeviceWindowL2CacheLeaseStatus::SOURCE_TOO_LARGE);
    EXPECT_FLOAT_EQ(snapshot.hitRatio, 0.0f);

    const auto telemetry = manager.telemetry(0);
    EXPECT_EQ(telemetry.activeUniqueSources, 0u);
    EXPECT_EQ(telemetry.activeUniqueBytes, 0u);
    EXPECT_EQ(telemetry.activeLeases, 0u);
}

TEST(DeviceWindowL2CacheManagerTest,
     TensorIdentityCannotAliasDifferentAllocationMetadata) {
    if (!gpuAvailable())
        GTEST_SKIP() << "window L2 cache manager test requires a GPU";

    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;

    const uint64_t budget = expectedBudget(capabilities);
    const uint64_t sourceBytes = std::min<uint64_t>(
        4096, std::min(budget, capabilities.max_access_policy_window_bytes));
    if (sourceBytes == 0)
        GTEST_SKIP() << "device exposes no usable persisting L2 source size";

    auto &manager = DeviceWindowL2CacheManager::instance();
    manager.resetForTesting();
    DeviceWindowL2CacheLease first = manager.acquire(
        DeviceWindowL2CacheSource{
            .deviceNum = 0,
            .tensorId = 401,
            .base = fakeDeviceAddress(0x500000),
            .bytes = sourceBytes});
    if (!first.snapshot().active())
        GTEST_SKIP() << "persisting L2 manager unavailable: "
                     << first.snapshot().detail;

    DeviceWindowL2CacheLease alias = manager.acquire(
        DeviceWindowL2CacheSource{
            .deviceNum = 0,
            .tensorId = 401,
            .base = fakeDeviceAddress(0x600000),
            .bytes = sourceBytes});
    EXPECT_EQ(alias.snapshot().status,
              DeviceWindowL2CacheLeaseStatus::INVALID_ARGUMENT);

    const auto telemetry = manager.telemetry(0);
    EXPECT_EQ(telemetry.activeUniqueSources, 1u);
    EXPECT_EQ(telemetry.activeUniqueBytes, sourceBytes);
    EXPECT_EQ(telemetry.activeLeases, 1u);
}
