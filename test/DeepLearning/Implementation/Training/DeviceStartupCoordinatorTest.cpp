#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace {

using ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES;
using ThorImplementation::DeviceStartupGuard;
using ThorImplementation::DeviceStartupMemoryFailureDisposition;
using ThorImplementation::DeviceModelResidencyLease;
using ThorImplementation::DeviceStartupInsufficientMemoryError;
using ThorImplementation::DeviceStartupSafetyReserveError;
using ThorImplementation::acquireDeviceStartupGuard;
using ThorImplementation::decideDeviceStartupMemoryFailureDisposition;
using ThorImplementation::enforceDeviceStartupSafetyReserve;
using ThorImplementation::isDeviceStartupMemoryFailure;

TEST(DeviceStartupCoordinatorTest, SerializesStartupOnTheSameDevice) {
    std::optional<DeviceStartupGuard> first(
        acquireDeviceStartupGuard(0));

    std::promise<void> secondAttemptedPromise;
    std::shared_future<void> secondAttempted =
        secondAttemptedPromise.get_future().share();
    std::promise<void> secondAcquiredPromise;
    std::shared_future<void> secondAcquired =
        secondAcquiredPromise.get_future().share();

    std::future<void> second = std::async(std::launch::async, [&]() {
        secondAttemptedPromise.set_value();
        DeviceStartupGuard guard = acquireDeviceStartupGuard(0);
        secondAcquiredPromise.set_value();
    });

    const std::future_status attemptedStatus =
        secondAttempted.wait_for(std::chrono::seconds(2));
    if (attemptedStatus != std::future_status::ready) {
        first.reset();
        second.wait();
        FAIL() << "Second startup thread did not begin its lock attempt.";
        return;
    }
    EXPECT_EQ(secondAcquired.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);

    first.reset();
    EXPECT_EQ(secondAcquired.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    second.get();
}

TEST(DeviceStartupCoordinatorTest, DifferentDevicesDoNotBlockEachOther) {
    std::optional<DeviceStartupGuard> first(
        acquireDeviceStartupGuard(0));

    std::future<bool> second = std::async(std::launch::async, []() {
        DeviceStartupGuard guard = acquireDeviceStartupGuard(1);
        return guard.ownsLock();
    });

    const std::future_status status =
        second.wait_for(std::chrono::seconds(2));
    first.reset();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_TRUE(second.get());
}

TEST(DeviceStartupCoordinatorTest, CompleteChecksReserveBeforeUnlocking) {
    std::optional<DeviceStartupGuard> first(
        acquireDeviceStartupGuard(8));

    EXPECT_THROW(first->complete(DEVICE_STARTUP_SAFETY_RESERVE_BYTES - 1),
                 DeviceStartupSafetyReserveError);
    ASSERT_TRUE(first->ownsLock());

    std::promise<void> secondAcquiredPromise;
    std::shared_future<void> secondAcquired =
        secondAcquiredPromise.get_future().share();
    std::future<void> second = std::async(std::launch::async, [&]() {
        DeviceStartupGuard guard = acquireDeviceStartupGuard(8);
        secondAcquiredPromise.set_value();
    });

    EXPECT_EQ(secondAcquired.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);
    first.reset();
    EXPECT_EQ(secondAcquired.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    second.get();
}

TEST(DeviceStartupCoordinatorTest, CompleteReleasesAfterSuccessfulReserveCheck) {
    DeviceStartupGuard first = acquireDeviceStartupGuard(9);
    first.complete(DEVICE_STARTUP_SAFETY_RESERVE_BYTES);
    EXPECT_FALSE(first.ownsLock());

    DeviceStartupGuard second = acquireDeviceStartupGuard(9);
    EXPECT_TRUE(second.ownsLock());
}

TEST(DeviceStartupCoordinatorTest, EnforcesOneGiBSafetyReserveBoundary) {
    EXPECT_NO_THROW(enforceDeviceStartupSafetyReserve(
        0, DEVICE_STARTUP_SAFETY_RESERVE_BYTES));
    EXPECT_THROW(enforceDeviceStartupSafetyReserve(
                     0, DEVICE_STARTUP_SAFETY_RESERVE_BYTES - 1),
                 DeviceStartupSafetyReserveError);
}

TEST(DeviceStartupCoordinatorTest, GuardReleasesOnScopeExit) {
    {
        DeviceStartupGuard guard = acquireDeviceStartupGuard(7);
        ASSERT_TRUE(guard.ownsLock());
    }

    DeviceStartupGuard next = acquireDeviceStartupGuard(7);
    EXPECT_TRUE(next.ownsLock());
}


TEST(DeviceStartupCoordinatorTest, TracksLoadedModelsUntilResidencyLeaseIsReleased) {
    DeviceStartupGuard first = acquireDeviceStartupGuard(10);
    std::shared_ptr<DeviceModelResidencyLease> model =
        first.completeAndTrackModel(DEVICE_STARTUP_SAFETY_RESERVE_BYTES);

    DeviceStartupGuard waiting = acquireDeviceStartupGuard(10);
    EXPECT_EQ(waiting.getLoadedModelCount(), 1u);

    std::future<void> release = std::async(
        std::launch::async,
        [model = std::move(model)]() mutable {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            model.reset();
        });

    waiting.waitForModelRelease();
    EXPECT_EQ(waiting.getLoadedModelCount(), 0u);
    waiting.complete(DEVICE_STARTUP_SAFETY_RESERVE_BYTES);
    release.get();
}

TEST(DeviceStartupCoordinatorTest, WaitingStartupRetainsFifoTurnAcrossModelRelease) {
    DeviceStartupGuard first = acquireDeviceStartupGuard(11);
    std::shared_ptr<DeviceModelResidencyLease> loaded =
        first.completeAndTrackModel(DEVICE_STARTUP_SAFETY_RESERVE_BYTES);

    DeviceStartupGuard fourth = acquireDeviceStartupGuard(11);

    std::promise<void> fifthAcquiredPromise;
    std::shared_future<void> fifthAcquired =
        fifthAcquiredPromise.get_future().share();
    std::future<void> fifth = std::async(std::launch::async, [&]() {
        DeviceStartupGuard guard = acquireDeviceStartupGuard(11);
        fifthAcquiredPromise.set_value();
    });

    EXPECT_EQ(fifthAcquired.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);

    std::future<void> release = std::async(
        std::launch::async,
        [loaded = std::move(loaded)]() mutable {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            loaded.reset();
        });
    fourth.waitForModelRelease();
    release.get();

    // The fourth startup woke because memory was freed, but still owns the FIFO
    // startup turn. The fifth cannot attempt placement until the fourth either
    // succeeds or terminates.
    EXPECT_EQ(fifthAcquired.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);

    fourth.complete(DEVICE_STARTUP_SAFETY_RESERVE_BYTES);
    EXPECT_EQ(fifthAcquired.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    fifth.get();
}

TEST(DeviceStartupCoordinatorTest, CanRetryAfterEachLoadedModelRelease) {
    std::vector<std::shared_ptr<DeviceModelResidencyLease>> loaded;
    for (int i = 0; i < 3; ++i) {
        DeviceStartupGuard guard = acquireDeviceStartupGuard(12);
        loaded.push_back(
            guard.completeAndTrackModel(
                DEVICE_STARTUP_SAFETY_RESERVE_BYTES));
    }

    DeviceStartupGuard fourth = acquireDeviceStartupGuard(12);
    EXPECT_EQ(fourth.getLoadedModelCount(), 3u);

    for (uint64_t expectedRemaining : {2u, 1u, 0u}) {
        std::shared_ptr<DeviceModelResidencyLease> released =
            std::move(loaded.back());
        loaded.pop_back();
        std::future<void> release = std::async(
            std::launch::async,
            [released = std::move(released)]() mutable {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(10));
                released.reset();
            });
        fourth.waitForModelRelease();
        EXPECT_EQ(fourth.getLoadedModelCount(), expectedRemaining);
        release.get();
    }

    fourth.complete(DEVICE_STARTUP_SAFETY_RESERVE_BYTES);
}

TEST(DeviceStartupCoordinatorTest, ChoosesWaitThenOneCleanEmptyRetryThenFailure) {
    EXPECT_EQ(decideDeviceStartupMemoryFailureDisposition(
                  3, 3, /*emptyDeviceRetryAlreadyUsed=*/false),
              DeviceStartupMemoryFailureDisposition::WAIT_FOR_MODEL_RELEASE);
    EXPECT_EQ(decideDeviceStartupMemoryFailureDisposition(
                  1, 1, /*emptyDeviceRetryAlreadyUsed=*/true),
              DeviceStartupMemoryFailureDisposition::WAIT_FOR_MODEL_RELEASE);
    EXPECT_EQ(decideDeviceStartupMemoryFailureDisposition(
                  0, 0, /*emptyDeviceRetryAlreadyUsed=*/false),
              DeviceStartupMemoryFailureDisposition::RETRY_EMPTY_DEVICE_ONCE);
    EXPECT_EQ(decideDeviceStartupMemoryFailureDisposition(
                  0, 0, /*emptyDeviceRetryAlreadyUsed=*/true),
              DeviceStartupMemoryFailureDisposition::FAIL);
    EXPECT_EQ(decideDeviceStartupMemoryFailureDisposition(
                  1, 0, /*emptyDeviceRetryAlreadyUsed=*/false),
              DeviceStartupMemoryFailureDisposition::FAIL);
}

TEST(DeviceStartupCoordinatorTest, ClassifiesOnlyMemoryPressureAsRetryable) {
    EXPECT_TRUE(isDeviceStartupMemoryFailure(std::make_exception_ptr(
        DeviceStartupInsufficientMemoryError("device startup out of memory"))));
    EXPECT_TRUE(isDeviceStartupMemoryFailure(std::make_exception_ptr(
        std::runtime_error("cudaErrorMemoryAllocation while allocating tensor"))));
    EXPECT_TRUE(isDeviceStartupMemoryFailure(std::make_exception_ptr(
        std::runtime_error(
            "launchConcatenate failed with cudaErrorMemoryAllocation (2): out of memory"))));
    EXPECT_TRUE(isDeviceStartupMemoryFailure(std::make_exception_ptr(
        std::runtime_error("launch failed: out of memory"))));
    EXPECT_FALSE(isDeviceStartupMemoryFailure(std::make_exception_ptr(
        std::runtime_error("invalid dataset schema"))));
    EXPECT_FALSE(isDeviceStartupMemoryFailure(std::make_exception_ptr(
        std::runtime_error(
            "device dataset materialization failed required_unused_bytes=1073741824"))));
}

}  // namespace
