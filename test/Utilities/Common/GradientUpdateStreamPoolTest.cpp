#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

TEST(GradientUpdateStreamPool, AllocatesLazilyAndReusesAtMostThreeStreamsRoundRobin) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Gradient-update stream-pool test requires a GPU";

    GradientUpdateStreamPool pool(0);
    EXPECT_EQ(pool.getNumAllocatedStreams(), 0u);

    Stream stream0 = pool.getNext();
    EXPECT_EQ(pool.getNumAllocatedStreams(), 1u);
    Stream stream1 = pool.getNext();
    EXPECT_EQ(pool.getNumAllocatedStreams(), 2u);
    Stream stream2 = pool.getNext();
    EXPECT_EQ(pool.getNumAllocatedStreams(), 3u);

    EXPECT_NE(stream0.getId(), stream1.getId());
    EXPECT_NE(stream0.getId(), stream2.getId());
    EXPECT_NE(stream1.getId(), stream2.getId());

    Stream stream3 = pool.getNext();
    Stream stream4 = pool.getNext();
    Stream stream5 = pool.getNext();
    EXPECT_EQ(pool.getNumAllocatedStreams(), GradientUpdateStreamPool::MAX_STREAMS);
    EXPECT_EQ(stream3.getId(), stream0.getId());
    EXPECT_EQ(stream4.getId(), stream1.getId());
    EXPECT_EQ(stream5.getId(), stream2.getId());
}

TEST(GradientUpdateStreamPool, DifferentOwnersNeverShareGradientUpdateStreams) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Gradient-update stream-pool test requires a GPU";

    GradientUpdateStreamPool firstModelPool(0);
    GradientUpdateStreamPool secondModelPool(0);

    Stream firstModelStream = firstModelPool.getNext();
    Stream secondModelStream = secondModelPool.getNext();

    EXPECT_NE(firstModelStream.getId(), secondModelStream.getId());
}
