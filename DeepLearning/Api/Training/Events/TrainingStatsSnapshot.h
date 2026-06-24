#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace Thor {

enum class TrainingEventPhase { UNKNOWN, TRAIN, VALIDATE, TEST };

[[nodiscard]] const char* trainingPhaseName(TrainingEventPhase phase);

struct TrainingStatsSnapshot {
    std::string networkName{};
    std::string datasetName{};
    TrainingEventPhase phase = TrainingEventPhase::UNKNOWN;

    uint64_t epoch = 0;
    uint64_t epochs = 0;
    uint64_t step = 0;
    uint64_t stepInEpoch = 0;
    uint64_t stepsPerEpoch = 0;
    uint64_t batchSize = 0;
    uint64_t samplesProcessed = 0;
    uint64_t inFlightBatches = 0;

    // Elapsed wall-clock seconds since the current native training/evaluation request started.
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
    std::unordered_map<std::string, double> metrics{};
};

}  // namespace Thor
