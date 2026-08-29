#include "Utilities/Common/ReusableEventPool.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <utility>

using ThorImplementation::ReusableEventLeases;
using ThorImplementation::ReusableEventPool;
using ThorImplementation::threadLocalReusableEventPool;

TEST(ReusableEventPool, ReleasedEventIsReusedWithinTheSameCreationClass) {
    ReusableEventPool pool;

    Event first = pool.acquire(0,
                               /*enableTiming=*/false,
                               /*expectingHostToWaitOnThisOne=*/false);
    const uint64_t firstId = first.getId();
    const cudaEvent_t firstCudaEvent = first.getEvent();

    pool.release(std::move(first));
    EXPECT_FALSE(first.isInitialized());
    EXPECT_EQ(pool.freeEventCountForTests(0), 1u);

    Event second = pool.acquire(0,
                                /*enableTiming=*/false,
                                /*expectingHostToWaitOnThisOne=*/false);
    EXPECT_EQ(second.getId(), firstId);
    EXPECT_EQ(second.getEvent(), firstCudaEvent);
    EXPECT_EQ(pool.freeEventCountForTests(0), 0u);

    pool.release(std::move(second));
}

TEST(ReusableEventPool, TimingAndBlockingIntentsUseSeparateFreeLists) {
    ReusableEventPool pool;

    Event normal = pool.acquire(0, false, false);
    Event timed = pool.acquire(0, true, false);
    Event blocking = pool.acquire(0, false, true);
    const uint64_t normalId = normal.getId();
    const uint64_t timedId = timed.getId();
    const uint64_t blockingId = blocking.getId();

    pool.release(std::move(normal));
    pool.release(std::move(timed));
    pool.release(std::move(blocking));

    EXPECT_EQ(pool.freeEventCountForTests(0, false, false), 1u);
    EXPECT_EQ(pool.freeEventCountForTests(0, true, false), 1u);
    EXPECT_EQ(pool.freeEventCountForTests(0, false, true), 1u);

    Event normalAgain = pool.acquire(0, false, false);
    Event timedAgain = pool.acquire(0, true, false);
    Event blockingAgain = pool.acquire(0, false, true);

    EXPECT_EQ(normalAgain.getId(), normalId);
    EXPECT_EQ(timedAgain.getId(), timedId);
    EXPECT_EQ(blockingAgain.getId(), blockingId);
    EXPECT_FALSE(normalAgain.usesTiming());
    EXPECT_TRUE(timedAgain.usesTiming());
    EXPECT_TRUE(blockingAgain.usesBlockingSync());
}

TEST(ReusableEventPool, LeaseSetReturnsEventsToTheThreadLocalPoolAfterSubmissionScope) {
    ReusableEventPool &pool = threadLocalReusableEventPool();
    uint64_t leasedId = 0;

    {
        ReusableEventLeases leases(1);
        Event leased = leases.acquire(0);
        leasedId = leased.getId();
        ASSERT_NE(leasedId, 0u);
    }

    // This submission leased one event, so returning the lease makes that same
    // compatible event the next one acquired from the thread-local pool.
    Event reacquired = pool.acquire(0);
    EXPECT_EQ(reacquired.getId(), leasedId);
    pool.release(std::move(reacquired));
}
