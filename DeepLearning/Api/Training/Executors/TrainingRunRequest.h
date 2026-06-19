#pragma once

#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

class Loader;

namespace Thor {

class Network;
class Optimizer;

enum class TrainingRunExecutionMode { FIT, EVALUATE };

struct TrainingRuntimeConfig {
    uint64_t maxInFlightBatches = 32;
    bool statsEnabled = true;
    double statsIntervalSeconds = 10.0;
    bool statsStderrAlso = false;
    LineStatsColorMode statsColorMode = LineStatsColorMode::AUTO;
    std::set<std::string> scalarTensorsToReport = {"loss", "accuracy"};
};

struct TrainingRunRequest {
    std::shared_ptr<Network> network = nullptr;
    std::shared_ptr<Loader> loader = nullptr;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::shared_ptr<TrainingProgram> trainingProgram = nullptr;
    TrainingRuntimeConfig runtime{};
    uint32_t epochs = 1;
    std::optional<std::string> saveModelDirectory{};
    bool saveModelOverwrite = false;
    bool saveOptimizerState = true;
    TrainingCancellationToken cancellationToken{};

    // FIT preserves the normal train+validate epoch sequence.  EVALUATE reuses the
    // same placed-network/native-queued machinery for one or more forward-only
    // epochs over evaluationExampleType and emits stats as evaluationPhase.
    TrainingRunExecutionMode executionMode = TrainingRunExecutionMode::FIT;
    ExampleType evaluationExampleType = ExampleType::VALIDATE;
    TrainingEventPhase evaluationPhase = TrainingEventPhase::VALIDATE;
};

}  // namespace Thor
