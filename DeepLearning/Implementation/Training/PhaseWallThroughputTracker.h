#pragma once

#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"
#include "Utilities/LogicalWork.h"

#include <chrono>
#include <cstdint>
#include <optional>

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
 *
 * Logical FLOP/s and logical bytes/s are accumulated from the same completed
 * batches over the same active wall interval and receive the same EMA update.
 * Arithmetic intensity is derived from those paired rates. If any completed
 * batch in an interval lacks logical-work telemetry, that interval is ignored
 * for logical-work rates rather than treating unavailable work as zero; ordinary
 * samples/s and batches/s throughput remains available.
 */
class PhaseWallThroughputTracker {
   public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    void observeCompletedBatch(Thor::TrainingStatsSnapshot& snapshot,
                               TimePoint phaseStartedAt,
                               TimePoint completedAt,
                               uint64_t validExamples,
                               const std::optional<LogicalWorkCount>& logicalWork,
                               bool forceSample) noexcept {
        snapshot.floatingPointOperationsPerBatch =
            logicalWork.has_value() ? logicalWork->floatingPointOperations : 0;
        snapshot.logicalBytesPerBatch = logicalWork.has_value() ? logicalWork->bytes : 0;

        auto assignCurrentRates = [&]() {
            snapshot.batchesPerSecond = batchesPerSecond;
            snapshot.samplesPerSecond = samplesPerSecond;
            snapshot.floatingPointOperationsPerSecond =
                floatingPointOperationsPerSecond;
            snapshot.logicalBytesPerSecond = logicalBytesPerSecond;
            snapshot.logicalArithmeticIntensity = logicalArithmeticIntensity;
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
        if (logicalWork.has_value() && pendingLogicalWorkComplete) {
            pendingFloatingPointOperations +=
                static_cast<double>(logicalWork->floatingPointOperations);
            pendingLogicalBytes += static_cast<double>(logicalWork->bytes);
        } else if (!logicalWork.has_value()) {
            // One unavailable batch makes the logical-work numerator incomplete
            // for this whole wall-time sample. Do not publish a partial rate.
            pendingLogicalWorkComplete = false;
        }

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

        if (!ratesInitialized) {
            batchesPerSecond = intervalBatchesPerSecond;
            samplesPerSecond = intervalSamplesPerSecond;
            ratesInitialized = true;
        } else {
            batchesPerSecond =
                (EMA_ALPHA * intervalBatchesPerSecond) +
                ((1.0 - EMA_ALPHA) * batchesPerSecond);
            samplesPerSecond =
                (EMA_ALPHA * intervalSamplesPerSecond) +
                ((1.0 - EMA_ALPHA) * samplesPerSecond);
        }

        if (pendingLogicalWorkComplete) {
            const double intervalFloatingPointOperationsPerSecond =
                pendingFloatingPointOperations / pendingActiveSeconds;
            const double intervalLogicalBytesPerSecond =
                pendingLogicalBytes / pendingActiveSeconds;

            if (!logicalWorkRatesInitialized) {
                floatingPointOperationsPerSecond =
                    intervalFloatingPointOperationsPerSecond;
                logicalBytesPerSecond = intervalLogicalBytesPerSecond;
                logicalWorkRatesInitialized = true;
            } else {
                floatingPointOperationsPerSecond =
                    (EMA_ALPHA * intervalFloatingPointOperationsPerSecond) +
                    ((1.0 - EMA_ALPHA) * floatingPointOperationsPerSecond);
                logicalBytesPerSecond =
                    (EMA_ALPHA * intervalLogicalBytesPerSecond) +
                    ((1.0 - EMA_ALPHA) * logicalBytesPerSecond);
            }

            logicalArithmeticIntensity =
                logicalBytesPerSecond > 0.0
                    ? floatingPointOperationsPerSecond / logicalBytesPerSecond
                    : 0.0;
        }

        pendingActiveSeconds = 0.0;
        pendingBatches = 0.0;
        pendingSamples = 0.0;
        pendingFloatingPointOperations = 0.0;
        pendingLogicalBytes = 0.0;
        pendingLogicalWorkComplete = true;
        assignCurrentRates();
    }

   private:
    static constexpr double EMA_ALPHA = 0.25;
    static constexpr double MIN_SAMPLE_INTERVAL_SECONDS = 0.25;

    bool activeIntervalInitialized = false;
    bool ratesInitialized = false;
    bool logicalWorkRatesInitialized = false;
    TimePoint activeIntervalStartedAt{};
    TimePoint previousCompletionAt{};

    double pendingActiveSeconds = 0.0;
    double pendingBatches = 0.0;
    double pendingSamples = 0.0;
    double pendingFloatingPointOperations = 0.0;
    double pendingLogicalBytes = 0.0;
    bool pendingLogicalWorkComplete = true;

    double samplesPerSecond = 0.0;
    double batchesPerSecond = 0.0;
    double floatingPointOperationsPerSecond = 0.0;
    double logicalBytesPerSecond = 0.0;
    double logicalArithmeticIntensity = 0.0;
};

}  // namespace ThorImplementation
