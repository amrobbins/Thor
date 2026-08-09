#include "Utilities/Common/SharedOwnership.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct SharedState {
    explicit SharedState(std::atomic<uint32_t> &destructions) : destructions(destructions) {}

    ~SharedState() noexcept { destructions.fetch_add(1, std::memory_order_relaxed); }

    std::atomic<uint32_t> &destructions;
};

struct SharedHandle {
    std::shared_ptr<SharedState> state;
};

}  // namespace

TEST(SharedOwnership, DistinctHandlesMayCopyAndDestroySharedStateConcurrently) {
    constexpr uint32_t NUM_THREADS = 8;
    constexpr uint32_t COPIES_PER_THREAD = 10000;

    std::atomic<uint32_t> destructions = 0;
    std::weak_ptr<SharedState> weakState;

    {
        SharedHandle root{std::make_shared<SharedState>(destructions)};
        weakState = root.state;

        std::vector<SharedHandle> stableSources(NUM_THREADS, root);
        std::vector<std::thread> workers;
        workers.reserve(NUM_THREADS);

        for (uint32_t threadIndex = 0; threadIndex < NUM_THREADS; ++threadIndex) {
            workers.emplace_back([source = stableSources[threadIndex]]() mutable {
                for (uint32_t copyIndex = 0; copyIndex < COPIES_PER_THREAD; ++copyIndex) {
                    SharedHandle copy = source;
                    SharedHandle moved = std::move(copy);
                    copy = source;
                    moved = SharedHandle{};
                }
            });
        }

        root = SharedHandle{};

        for (std::thread &worker : workers)
            worker.join();

        // The independently-owned stable handles keep the state alive after all
        // worker-local copies have disappeared.
        EXPECT_FALSE(weakState.expired());
        EXPECT_EQ(destructions.load(std::memory_order_relaxed), 0u);

        stableSources.clear();
        EXPECT_TRUE(weakState.expired());
        EXPECT_EQ(destructions.load(std::memory_order_relaxed), 1u);
    }

    EXPECT_EQ(destructions.load(std::memory_order_relaxed), 1u);
}

TEST(SharedOwnership, CleanupNoThrowAllowsSuccessfulCleanup) {
    bool cleanedUp = false;

    testing::internal::CaptureStderr();
    ThorImplementation::SharedOwnership::cleanupNoThrow("TestResource", "release", [&]() { cleanedUp = true; });
    const std::string stderrOutput = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(cleanedUp);
    EXPECT_TRUE(stderrOutput.empty());
}

TEST(SharedOwnership, CleanupNoThrowReportsStandardExceptionsWithoutEscaping) {
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(ThorImplementation::SharedOwnership::cleanupNoThrow("TestResource", "release", []() {
        throw std::runtime_error("intentional cleanup failure");
    }));
    const std::string stderrOutput = testing::internal::GetCapturedStderr();

    EXPECT_NE(stderrOutput.find("Thor resource cleanup failure [TestResource] release"), std::string::npos);
    EXPECT_NE(stderrOutput.find("intentional cleanup failure"), std::string::npos);
}

TEST(SharedOwnership, CleanupNoThrowReportsUnknownExceptionsWithoutEscaping) {
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(ThorImplementation::SharedOwnership::cleanupNoThrow("TestResource", "release", []() { throw 7; }));
    const std::string stderrOutput = testing::internal::GetCapturedStderr();

    EXPECT_NE(stderrOutput.find("Thor resource cleanup failure [TestResource] release"), std::string::npos);
    EXPECT_NE(stderrOutput.find("unknown exception"), std::string::npos);
}

TEST(SharedOwnershipReleaseQualification, DISABLED_ReleaseGateRequiresCudaDevice) {
    if (std::getenv("THOR_RELEASE_SHARED_OWNERSHIP_GATE") == nullptr)
        GTEST_SKIP() << "Shared-ownership release preflight only runs through check-release-shared-ownership";

    EXPECT_GT(MachineEvaluator::instance().getNumGpus(), 0U)
        << "check-release-shared-ownership requires a CUDA device so GPU-backed ownership tests cannot pass by skipping";
}
