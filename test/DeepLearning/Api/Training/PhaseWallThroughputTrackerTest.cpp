#include "DeepLearning/Implementation/Training/PhaseWallThroughputTracker.h"

#include <gtest/gtest.h>

#include <chrono>

namespace ThorImplementation {
namespace {

TEST(PhaseWallThroughputTracker, ShortPhasesExcludeInactiveGapFromTrainRate) {
    PhaseWallThroughputTracker tracker;
    using Clock = PhaseWallThroughputTracker::Clock;
    using namespace std::chrono_literals;

    const Clock::time_point origin{1s};

    Thor::TrainingStatsSnapshot first;
    first.elapsedSeconds = 0.005;
    tracker.observeCompletedBatch(first,
                                  origin,
                                  origin + 5ms,
                                  /*validExamples=*/3500,
                                  /*floatingPointOperations=*/7000,
                                  /*forceSample=*/true);

    EXPECT_NEAR(first.samplesPerSecond, 700000.0, 1.0);
    EXPECT_NEAR(first.batchesPerSecond, 200.0, 1e-6);
    EXPECT_NEAR(first.floatingPointOperationsPerSecond, 1400000.0, 1.0);

    // The same amount of TRAIN work happens after a long interval occupied by
    // validation/model selection.  elapsedSeconds intentionally includes that
    // gap, while phase throughput must not.
    const Clock::time_point secondPhaseStart = origin + 155ms;
    Thor::TrainingStatsSnapshot second;
    second.elapsedSeconds = 0.160;
    tracker.observeCompletedBatch(second,
                                  secondPhaseStart,
                                  secondPhaseStart + 5ms,
                                  /*validExamples=*/3500,
                                  /*floatingPointOperations=*/7000,
                                  /*forceSample=*/true);

    EXPECT_NEAR(second.samplesPerSecond, 700000.0, 1.0);
    EXPECT_NEAR(second.batchesPerSecond, 200.0, 1e-6);
    EXPECT_NEAR(second.floatingPointOperationsPerSecond, 1400000.0, 1.0);
}

TEST(PhaseWallThroughputTracker, FinalBatchForcesSampleBelowNormalInterval) {
    PhaseWallThroughputTracker tracker;
    using Clock = PhaseWallThroughputTracker::Clock;
    using namespace std::chrono_literals;

    const Clock::time_point phaseStart{1s};

    Thor::TrainingStatsSnapshot first;
    tracker.observeCompletedBatch(first,
                                  phaseStart,
                                  phaseStart + 1ms,
                                  /*validExamples=*/100,
                                  /*floatingPointOperations=*/200,
                                  /*forceSample=*/false);
    EXPECT_DOUBLE_EQ(first.samplesPerSecond, 0.0);

    Thor::TrainingStatsSnapshot final;
    tracker.observeCompletedBatch(final,
                                  phaseStart,
                                  phaseStart + 2ms,
                                  /*validExamples=*/100,
                                  /*floatingPointOperations=*/200,
                                  /*forceSample=*/true);

    EXPECT_NEAR(final.samplesPerSecond, 100000.0, 1.0);
    EXPECT_NEAR(final.batchesPerSecond, 1000.0, 1e-6);
    EXPECT_NEAR(final.floatingPointOperationsPerSecond, 200000.0, 1.0);
}

}  // namespace
}  // namespace ThorImplementation
