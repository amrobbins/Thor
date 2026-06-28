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

    // Trainer-owned best-candidate checkpoint cadence. A value of 0 disables
    // best-candidate tracking/snapshotting; the final latest/end-of-fit artifact
    // is still saved when saveModelDirectory is configured. When this is > 0,
    // the native runner evaluates modelSelectionScore every N epochs and also
    // always evaluates the final completed epoch, even if it does not land on
    // the cadence. Lower scores are better.
    uint32_t checkBestModelEveryEpochs = 0;

    // First global/cumulative epoch at which model-selection scoring begins.
    // Epochs count total successful training epochs for the model across
    // Trainer.fit(...) calls. A value of 0 preserves the cadence-only behavior:
    // the first selection happens at checkBestModelEveryEpochs. Early-completion
    // policies are evaluated only on model-selection epochs after the candidate
    // manager updates best_score/best_epoch.
    uint64_t firstModelSelectionEpoch = 0;

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

    // Artifact handoff source used when a previous fit has already finalized a
    // model artifact and the old GPU-resident placement has been released. The
    // runner places this saved source just long enough to copy matching
    // parameter/optimizer state into the fresh execution placement, then drops it
    // before training begins.
    std::optional<std::string> previousModelArtifactDirectory{};
    std::optional<std::string> previousModelNetworkName{};

    std::shared_ptr<PlacedNetwork>* completedPlacedNetwork = nullptr;

    // Cumulative completed training epochs before/after this request. FIT trains
    // `epochs` additional epochs starting after initialCompletedEpochs. The native
    // runner emits and evaluates epoch thresholds using global epoch numbers.
    // Trainer restart handling resets this value to 0 before launching a retry
    // because a restarted model attempt discards the previous trained state.
    uint64_t initialCompletedEpochs = 0;
    uint64_t* completedTrainingEpochs = nullptr;
};

}  // namespace Thor
