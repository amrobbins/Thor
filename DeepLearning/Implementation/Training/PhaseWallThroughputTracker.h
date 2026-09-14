#pragma once

#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"

#include <chrono>
#include <cstdint>

namespace ThorImplementation {

/**
 * EMA-smoothed throughput for one logical training phase (TRAIN, VALIDATE, ...).
 *
 * The caller supplies the scheduler start and GPU-completion-callback timestamp
 * for each completed batch.  A new scheduler-start timestamp starts a new active
 * interval, so inactive gaps between occurrences of this phase (for example,
 * validation/model-selection work between TRAIN epochs) are never charged to
 * its rate.  Progress is accumulated until a
 * useful wall interval has elapsed, except that the final batch of every phase
 * forces a sample so short phases still publish a meaningful rate.
 */
class PhaseWallThroughputTracker {
   public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    void observeCompletedBatch(Thor::TrainingStatsSnapshot& snapshot,
                               TimePoint phaseStartedAt,
                               TimePoint completedAt,
                               uint64_t validExamples,
                               uint64_t floatingPointOperations,
                               bool forceSample) noexcept {
        snapshot.floatingPointOperationsPerBatch = floatingPointOperations;

        auto assignCurrentRates = [&]() {
            snapshot.batchesPerSecond = batchesPerSecond;
            snapshot.samplesPerSecond = samplesPerSecond;
            snapshot.floatingPointOperationsPerSecond =
                floatingPointOperationsPerSecond;
        };

        if (validExamples == 0 || phaseStartedAt == TimePoint{} ||
            completedAt == TimePoint{}) {
            assignCurrentRates();
            return;
        }

        if (!activeIntervalInitialized || phaseStartedAt != activeIntervalStartedAt) {
            activeIntervalInitialized = true;
            activeIntervalStartedAt = phaseStartedAt;
            previousCompletionAt = phaseStartedAt;
        }

        // Throughput is diagnostic-only.  Invalid/non-monotonic timing must not
        // make training fail.  Count the work, but add no negative wall time.
        if (completedAt > previousCompletionAt) {
            pendingActiveSeconds +=
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    completedAt - previousCompletionAt)
                    .count();
            previousCompletionAt = completedAt;
        }

        pendingBatches += 1.0;
        pendingSamples += static_cast<double>(validExamples);
        pendingFloatingPointOperations +=
            static_cast<double>(floatingPointOperations);

        if (!forceSample &&
            pendingActiveSeconds < MIN_SAMPLE_INTERVAL_SECONDS) {
            assignCurrentRates();
            return;
        }

        if (pendingActiveSeconds <= 0.0) {
            assignCurrentRates();
            return;
        }

        const double intervalBatchesPerSecond =
            pendingBatches / pendingActiveSeconds;
        const double intervalSamplesPerSecond =
            pendingSamples / pendingActiveSeconds;
        const double intervalFloatingPointOperationsPerSecond =
            pendingFloatingPointOperations / pendingActiveSeconds;

        if (!ratesInitialized) {
            batchesPerSecond = intervalBatchesPerSecond;
            samplesPerSecond = intervalSamplesPerSecond;
            floatingPointOperationsPerSecond =
                intervalFloatingPointOperationsPerSecond;
            ratesInitialized = true;
        } else {
            batchesPerSecond =
                (EMA_ALPHA * intervalBatchesPerSecond) +
                ((1.0 - EMA_ALPHA) * batchesPerSecond);
            samplesPerSecond =
                (EMA_ALPHA * intervalSamplesPerSecond) +
                ((1.0 - EMA_ALPHA) * samplesPerSecond);
            floatingPointOperationsPerSecond =
                (EMA_ALPHA * intervalFloatingPointOperationsPerSecond) +
                ((1.0 - EMA_ALPHA) * floatingPointOperationsPerSecond);
        }

        pendingActiveSeconds = 0.0;
        pendingBatches = 0.0;
        pendingSamples = 0.0;
        pendingFloatingPointOperations = 0.0;
        assignCurrentRates();
    }

   private:
    static constexpr double EMA_ALPHA = 0.25;
    static constexpr double MIN_SAMPLE_INTERVAL_SECONDS = 0.25;

    bool activeIntervalInitialized = false;
    bool ratesInitialized = false;
    TimePoint activeIntervalStartedAt{};
    TimePoint previousCompletionAt{};

    double pendingActiveSeconds = 0.0;
    double pendingBatches = 0.0;
    double pendingSamples = 0.0;
    double pendingFloatingPointOperations = 0.0;

    double samplesPerSecond = 0.0;
    double batchesPerSecond = 0.0;
    double floatingPointOperationsPerSecond = 0.0;
};

}  // namespace ThorImplementation
