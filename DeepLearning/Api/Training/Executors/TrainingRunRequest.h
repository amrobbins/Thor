#pragma once

#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/EarlyCompletionPolicy.h"
#include "DeepLearning/Api/Training/ModelSelectionScore.h"
#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

class Loader;

namespace Thor {

class Network;
class Optimizer;
class PlacedNetwork;

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

    // Trainer-owned best-candidate checkpoint cadence. During FIT, the native
    // runner evaluates modelSelectionScore every N epochs. The default score is
    // current epoch-average validation loss when present, otherwise current
    // epoch-average training loss. Lower is better. If saveModelDirectory is
    // configured, the best observed candidate is saved from the runtime
    // PlacedNetwork state and becomes the final saved artifact.
    uint32_t checkBestModelEveryEpochs = 1;
    TrainingModelSelectionScore modelSelectionScore{};

    // Checked at the same epoch cadence as best-candidate selection. The
    // candidate manager updates best_score/best_epoch first, then each policy
    // may request early completion from the current_score, best_score,
    // current_epoch, and best_epoch. Lower scores are better.
    std::vector<TrainingEarlyCompletionPolicy> earlyCompletionPolicies{};

    TrainingCancellationToken cancellationToken{};

    // FIT preserves the normal train+validate epoch sequence.  EVALUATE reuses the
    // same placed-network/native-queued machinery for one or more forward-only
    // epochs over evaluationExampleType and emits stats as evaluationPhase.
    TrainingRunExecutionMode executionMode = TrainingRunExecutionMode::FIT;
    ExampleType evaluationExampleType = ExampleType::VALIDATE;
    TrainingEventPhase evaluationPhase = TrainingEventPhase::VALIDATE;

    // Trainer.fit(...) creates a fresh PlacedNetwork for each run so phase-root
    // mutations can recompile against a clean physical graph.  When the same
    // Trainer is fit again, preserve the trained parameter/optimizer state by
    // copying it from the previous placed network into the fresh placement.
    std::shared_ptr<PlacedNetwork> previousPlacedNetwork = nullptr;
    std::shared_ptr<PlacedNetwork>* completedPlacedNetwork = nullptr;
};

}  // namespace Thor
