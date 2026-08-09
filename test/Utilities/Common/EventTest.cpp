#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

TEST(Event, PreservesBlockingSynchronizationIntentAcrossCopies) {
    Stream stream(0);

    Event pollingEvent = stream.putEvent(
        /*enableTiming=*/false,
        /*expectingHostToWaitOnThisOne=*/false);
    Event blockingEvent = stream.putEvent(
        /*enableTiming=*/false,
        /*expectingHostToWaitOnThisOne=*/true);
    Event copiedBlockingEvent = blockingEvent;

    EXPECT_FALSE(pollingEvent.usesBlockingSync());
    EXPECT_TRUE(blockingEvent.usesBlockingSync());
    EXPECT_TRUE(copiedBlockingEvent.usesBlockingSync());

    blockingEvent.synchronize();
}

TEST(Event, RejectsChangingBlockingSynchronizationIntentWhenReused) {
    Stream stream(0);
    Event event;

    stream.putEvent(event,
                    /*enableTiming=*/false,
                    /*expectingHostToWaitOnThisOne=*/true);
    EXPECT_TRUE(event.usesBlockingSync());

    EXPECT_THROW(stream.putEvent(event,
                                 /*enableTiming=*/false,
                                 /*expectingHostToWaitOnThisOne=*/false),
                 std::logic_error);

    event.synchronize();
}

TEST(Event, DefaultEventIsUninitializedWithZeroIdentity) {
    Event event;

    EXPECT_FALSE(event.isInitialized());
    EXPECT_EQ(event.getId(), 0u);
}

TEST(Event, CopiesAndMovesShareOneEventIdentity) {
    Stream stream(0);
    Event original = stream.putEvent(/*enableTiming=*/false);

    const uint64_t originalId = original.getId();
    const cudaEvent_t originalCudaEvent = original.getEvent();

    Event copied = original;
    EXPECT_EQ(copied.getId(), originalId);
    EXPECT_EQ(copied.getEvent(), originalCudaEvent);

    Event moved = std::move(copied);
    EXPECT_FALSE(copied.isInitialized());
    EXPECT_TRUE(moved.isInitialized());
    EXPECT_EQ(moved.getId(), originalId);
    EXPECT_EQ(moved.getEvent(), originalCudaEvent);

    Event assigned;
    assigned = moved;
    EXPECT_EQ(assigned.getId(), originalId);
    EXPECT_EQ(assigned.getEvent(), originalCudaEvent);

    Event moveAssigned;
    moveAssigned = std::move(assigned);
    EXPECT_FALSE(assigned.isInitialized());
    EXPECT_EQ(moveAssigned.getId(), originalId);
    EXPECT_EQ(moveAssigned.getEvent(), originalCudaEvent);

    original = Event();
    moved = Event();

    // The remaining independently-owned handle must keep the CUDA event alive.
    moveAssigned.synchronize();
}

TEST(Event, IndependentlyCreatedEventsHaveDistinctIdentities) {
    Stream stream(0);

    Event first = stream.putEvent(/*enableTiming=*/false);
    Event second = stream.putEvent(/*enableTiming=*/false);

    EXPECT_NE(first.getId(), second.getId());
    EXPECT_NE(first.getEvent(), second.getEvent());

    first.synchronize();
    second.synchronize();
}

TEST(Event, DistinctHandlesMayCopyAndDestroyOneEventConcurrently) {
    constexpr uint32_t NUM_THREADS = 8;
    constexpr uint32_t COPIES_PER_THREAD = 10000;

    Stream stream(0);
    Event root = stream.putEvent(/*enableTiming=*/false);
    const uint64_t rootId = root.getId();
    const cudaEvent_t rootCudaEvent = root.getEvent();

    std::vector<Event> stableSources(NUM_THREADS, root);
    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    std::atomic<bool> workerFailed = false;
    for (uint32_t threadIndex = 0; threadIndex < NUM_THREADS; ++threadIndex) {
        workers.emplace_back([source = stableSources[threadIndex], rootId, rootCudaEvent, &workerFailed]() mutable {
            for (uint32_t copyIndex = 0; copyIndex < COPIES_PER_THREAD; ++copyIndex) {
                Event copy = source;
                if (copy.getId() != rootId || copy.getEvent() != rootCudaEvent)
                    workerFailed.store(true, std::memory_order_relaxed);

                Event moved = std::move(copy);
                if (copy.isInitialized() || moved.getId() != rootId)
                    workerFailed.store(true, std::memory_order_relaxed);

                copy = source;
                moved = Event();
            }
        });
    }

    // The independently-owned stable source handles keep the resource alive
    // while worker-local handles are copied, moved, assigned, and destroyed.
    root = Event();

    for (std::thread &worker : workers)
        worker.join();

    EXPECT_FALSE(workerFailed.load(std::memory_order_relaxed));
    ASSERT_FALSE(stableSources.empty());
    EXPECT_EQ(stableSources.front().getId(), rootId);
    EXPECT_EQ(stableSources.front().getEvent(), rootCudaEvent);
    stableSources.front().synchronize();
}
