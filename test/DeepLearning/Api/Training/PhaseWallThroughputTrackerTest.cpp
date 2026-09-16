#include "DeepLearning/Implementation/Training/PhaseWallThroughputTracker.h"

#include <gtest/gtest.h>

#include <chrono>
#include <optional>

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
                                  LogicalWorkCount{.floatingPointOperations = 7000, .bytes = 3500},
                                  /*forceSample=*/true);

    EXPECT_EQ(first.floatingPointOperationsPerBatch, 7000u);
    EXPECT_EQ(first.logicalBytesPerBatch, 3500u);
    EXPECT_NEAR(first.samplesPerSecond, 700000.0, 1.0);
    EXPECT_NEAR(first.batchesPerSecond, 200.0, 1e-6);
    EXPECT_NEAR(first.floatingPointOperationsPerSecond, 1400000.0, 1.0);
    EXPECT_NEAR(first.logicalBytesPerSecond, 700000.0, 1.0);
    EXPECT_NEAR(first.logicalArithmeticIntensity, 2.0, 1e-12);

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
                                  LogicalWorkCount{.floatingPointOperations = 7000, .bytes = 3500},
                                  /*forceSample=*/true);

    EXPECT_NEAR(second.samplesPerSecond, 700000.0, 1.0);
    EXPECT_NEAR(second.batchesPerSecond, 200.0, 1e-6);
    EXPECT_NEAR(second.floatingPointOperationsPerSecond, 1400000.0, 1.0);
    EXPECT_NEAR(second.logicalBytesPerSecond, 700000.0, 1.0);
    EXPECT_NEAR(second.logicalArithmeticIntensity, 2.0, 1e-12);
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
                                  LogicalWorkCount{.floatingPointOperations = 200, .bytes = 100},
                                  /*forceSample=*/false);
    EXPECT_DOUBLE_EQ(first.samplesPerSecond, 0.0);
    EXPECT_DOUBLE_EQ(first.logicalBytesPerSecond, 0.0);
    EXPECT_DOUBLE_EQ(first.logicalArithmeticIntensity, 0.0);

    Thor::TrainingStatsSnapshot final;
    tracker.observeCompletedBatch(final,
                                  phaseStart,
                                  phaseStart + 2ms,
                                  /*validExamples=*/100,
                                  LogicalWorkCount{.floatingPointOperations = 200, .bytes = 100},
                                  /*forceSample=*/true);

    EXPECT_NEAR(final.samplesPerSecond, 100000.0, 1.0);
    EXPECT_NEAR(final.batchesPerSecond, 1000.0, 1e-6);
    EXPECT_NEAR(final.floatingPointOperationsPerSecond, 200000.0, 1.0);
    EXPECT_NEAR(final.logicalBytesPerSecond, 100000.0, 1.0);
    EXPECT_NEAR(final.logicalArithmeticIntensity, 2.0, 1e-12);
}

TEST(PhaseWallThroughputTracker, LogicalRatesUseSameIntervalAndEmaUpdate) {
    PhaseWallThroughputTracker tracker;
    using Clock = PhaseWallThroughputTracker::Clock;
    using namespace std::chrono_literals;

    const Clock::time_point firstPhaseStart{1s};
    Thor::TrainingStatsSnapshot first;
    tracker.observeCompletedBatch(first,
                                  firstPhaseStart,
                                  firstPhaseStart + 1ms,
                                  /*validExamples=*/100,
                                  LogicalWorkCount{.floatingPointOperations = 200, .bytes = 100},
                                  /*forceSample=*/true);

    EXPECT_NEAR(first.floatingPointOperationsPerSecond, 200000.0, 1.0);
    EXPECT_NEAR(first.logicalBytesPerSecond, 100000.0, 1.0);
    EXPECT_NEAR(first.logicalArithmeticIntensity, 2.0, 1e-12);

    // The second interval has 200 kFLOP/s and 200 kB/s. With the common 0.25
    // EMA update, FLOP/s remains 200k while bytes/s becomes 125k. Arithmetic
    // intensity must be derived from those paired smoothed rates: 1.6 F/B.
    const Clock::time_point secondPhaseStart = firstPhaseStart + 100ms;
    Thor::TrainingStatsSnapshot second;
    tracker.observeCompletedBatch(second,
                                  secondPhaseStart,
                                  secondPhaseStart + 2ms,
                                  /*validExamples=*/100,
                                  LogicalWorkCount{.floatingPointOperations = 400, .bytes = 400},
                                  /*forceSample=*/true);

    EXPECT_NEAR(second.floatingPointOperationsPerSecond, 200000.0, 1.0);
    EXPECT_NEAR(second.logicalBytesPerSecond, 125000.0, 1.0);
    EXPECT_NEAR(second.logicalArithmeticIntensity, 1.6, 1e-12);
}

TEST(PhaseWallThroughputTracker, UnavailableLogicalWorkDoesNotPublishPartialRate) {
    PhaseWallThroughputTracker tracker;
    using Clock = PhaseWallThroughputTracker::Clock;
    using namespace std::chrono_literals;

    const Clock::time_point baselineStart{1s};
    Thor::TrainingStatsSnapshot baseline;
    tracker.observeCompletedBatch(baseline,
                                  baselineStart,
                                  baselineStart + 1ms,
                                  /*validExamples=*/100,
                                  LogicalWorkCount{.floatingPointOperations = 200, .bytes = 100},
                                  /*forceSample=*/true);
    ASSERT_NEAR(baseline.floatingPointOperationsPerSecond, 200000.0, 1.0);
    ASSERT_NEAR(baseline.logicalBytesPerSecond, 100000.0, 1.0);

    const Clock::time_point nextPhaseStart = baselineStart + 100ms;
    Thor::TrainingStatsSnapshot known;
    tracker.observeCompletedBatch(known,
                                  nextPhaseStart,
                                  nextPhaseStart + 1ms,
                                  /*validExamples=*/100,
                                  LogicalWorkCount{.floatingPointOperations = 1000, .bytes = 500},
                                  /*forceSample=*/false);

    Thor::TrainingStatsSnapshot unavailable;
    tracker.observeCompletedBatch(unavailable,
                                  nextPhaseStart,
                                  nextPhaseStart + 2ms,
                                  /*validExamples=*/100,
                                  std::nullopt,
                                  /*forceSample=*/true);

    EXPECT_EQ(unavailable.floatingPointOperationsPerBatch, 0u);
    EXPECT_EQ(unavailable.logicalBytesPerBatch, 0u);
    EXPECT_NEAR(unavailable.floatingPointOperationsPerSecond, 200000.0, 1.0);
    EXPECT_NEAR(unavailable.logicalBytesPerSecond, 100000.0, 1.0);
    EXPECT_NEAR(unavailable.logicalArithmeticIntensity, 2.0, 1e-12);
    EXPECT_GT(unavailable.samplesPerSecond, 0.0);
    EXPECT_GT(unavailable.batchesPerSecond, 0.0);
}

}  // namespace
}  // namespace ThorImplementation
