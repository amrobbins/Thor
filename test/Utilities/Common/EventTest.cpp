#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

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
