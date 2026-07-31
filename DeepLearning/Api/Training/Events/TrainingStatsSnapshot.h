#pragma once

#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"
#include "DeepLearning/Api/Layers/Metrics/MetricAggregation.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace Thor {

enum class TrainingEventPhase { UNKNOWN, TRAIN, VALIDATE, TEST };

struct MetricBatchStat {
    MetricAggregation aggregation = MetricAggregation::MEAN_BY_EXAMPLE;
    double value = 0.0;
    uint64_t validExamples = 0;
    // Present together exactly when aggregation == RATIO. They are sufficient
    // statistics, not public graph outputs or serialized runtime state.
    std::optional<double> numerator{};
    std::optional<double> denominator{};
};

[[nodiscard]] const char* trainingPhaseName(TrainingEventPhase phase);

struct TrainingStatsSnapshot {
    std::string networkName{};
    std::string datasetName{};
    TrainingEventPhase phase = TrainingEventPhase::UNKNOWN;
    // Set for VALIDATE events. Empty preserves legacy unnamed validation.
    std::string validationPopulation{};
    bool isDefaultValidationPopulation = false;

    uint64_t epoch = 0;
    uint64_t epochs = 0;
    uint64_t step = 0;
    uint64_t stepInEpoch = 0;
    uint64_t stepsPerEpoch = 0;
    // Physical capacity of the placed network batch. Tail batches may have a
    // smaller validExamplesInBatch without changing this value.
    uint64_t batchSize = 0;
    // Number of valid prefix rows in this physical batch.
    uint64_t validExamplesInBatch = 0;
    // Valid examples consumed in the current phase/population epoch.
    uint64_t samplesProcessedInEpoch = 0;
    // Cumulative valid examples consumed for this phase/population.
    uint64_t samplesProcessed = 0;
    uint64_t inFlightBatches = 0;

    // Elapsed wall-clock seconds for the active operation. FIT may include a
    // Trainer-owned offset from earlier sequential training phases so user-facing
    // stats show cumulative training time. EVALUATE reports request-local time.
    double elapsedSeconds = 0.0;

    // Public throughput rates share the same wall-clock basis as elapsedSeconds.
    // Native queued training reports EMA-smoothed exact wall-clock interval rates
    // between same-phase stats snapshots, not CUDA callback/active-kernel rates.
    double samplesPerSecond = 0.0;
    double batchesPerSecond = 0.0;
    uint64_t floatingPointOperationsPerBatch = 0;
    double floatingPointOperationsPerSecond = 0.0;

    std::optional<double> loss{};
    std::optional<double> accuracy{};
    std::optional<double> learningRate{};
    std::optional<double> momentum{};

    // Named graph losses are kept separate from general scalar metrics so
    // model-selection callbacks can choose a validation score that differs
    // from the aggregate training objective. Loss names may also appear in
    // metrics for backwards-compatible reporting.
    std::unordered_map<std::string, double> losses{};
    std::unordered_map<std::string, double> metrics{};

    // Internal sufficient statistics for declared graph metrics. Ordinary
    // observers may ignore this map; it exists so epoch/population aggregation
    // does not have to reconstruct semantics from a display scalar.
    std::unordered_map<std::string, MetricBatchStat> metricBatchStats{};

    DeviceDatasetStorageReport deviceDatasetStorage{};
};

}  // namespace Thor
