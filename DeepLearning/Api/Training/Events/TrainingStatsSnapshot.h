#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace Thor {

enum class TrainingPhase { UNKNOWN, TRAIN, VALIDATE, TEST };

[[nodiscard]] const char* trainingPhaseName(TrainingPhase phase);

struct TrainingStatsSnapshot {
    std::string networkName{};
    std::string datasetName{};
    TrainingPhase phase = TrainingPhase::UNKNOWN;

    uint64_t epoch = 0;
    uint64_t epochs = 0;
    uint64_t step = 0;
    uint64_t stepInEpoch = 0;
    uint64_t stepsPerEpoch = 0;
    uint64_t batchSize = 0;
    uint64_t samplesProcessed = 0;
    uint64_t inFlightBatches = 0;

    double elapsedSeconds = 0.0;
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
