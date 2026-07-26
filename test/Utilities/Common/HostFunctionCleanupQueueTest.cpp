#include "Utilities/Common/HostFunctionCleanupQueue.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <memory>

namespace {

struct CountedHostFunctionArgs : HostFunctionArgsBase {
    CountedHostFunctionArgs(std::atomic<uint32_t> &callbacks,
                            std::atomic<uint32_t> &destructions,
                            std::atomic<uint32_t> &earlyDestructions)
        : callbacks(callbacks), destructions(destructions), earlyDestructions(earlyDestructions) {}

    ~CountedHostFunctionArgs() override {
        if (!callbackRan.load(std::memory_order_acquire))
            earlyDestructions.fetch_add(1, std::memory_order_relaxed);
        destructions.fetch_add(1, std::memory_order_relaxed);
    }

    std::atomic<uint32_t> &callbacks;
    std::atomic<uint32_t> &destructions;
    std::atomic<uint32_t> &earlyDestructions;
    std::atomic<bool> callbackRan = false;
};

void countedHostFunction(void *rawArgs) {
    auto *args = static_cast<CountedHostFunctionArgs *>(rawArgs);
    args->callbacks.fetch_add(1, std::memory_order_relaxed);
    args->callbackRan.store(true, std::memory_order_release);
}

}  // namespace

TEST(HostFunctionCleanupQueue, UsesFixedWorkersAndBoundedPendingStorage) {
    HostFunctionCleanupQueue &queue = HostFunctionCleanupQueue::instance();

    EXPECT_EQ(queue.getWorkerCount(), HostFunctionCleanupQueue::WORKER_COUNT);
    EXPECT_EQ(queue.getQueueCapacity(), HostFunctionCleanupQueue::QUEUE_CAPACITY);
    EXPECT_EQ(queue.getWorkerCount(), 4u);
    EXPECT_EQ(queue.getQueueCapacity(), 4096u);
}

TEST(HostFunctionCleanupQueue, RetainsArgumentsUntilCallbacksComplete) {
    constexpr uint32_t NUM_CALLBACKS = 256;

    Stream stream(0);
    std::atomic<uint32_t> callbacks = 0;
    std::atomic<uint32_t> destructions = 0;
    std::atomic<uint32_t> earlyDestructions = 0;

    for (uint32_t i = 0; i < NUM_CALLBACKS; ++i) {
        stream.enqueueHostFunction(countedHostFunction,
                                   std::make_unique<CountedHostFunctionArgs>(callbacks, destructions, earlyDestructions));
    }

    stream.synchronize();
    HostFunctionCleanupQueue::instance().waitForEmpty();

    EXPECT_EQ(callbacks.load(), NUM_CALLBACKS);
    EXPECT_EQ(destructions.load(), NUM_CALLBACKS);
    EXPECT_EQ(earlyDestructions.load(), 0u);
    EXPECT_EQ(HostFunctionCleanupQueue::instance().getPendingCount(), 0u);
    EXPECT_EQ(HostFunctionCleanupQueue::instance().getActiveCount(), 0u);
}
