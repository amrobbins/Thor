#pragma once

#include "DeepLearning/Api/Data/ExampleType.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"
#include "DeepLearning/Api/Training/EarlyCompletionPolicy.h"
#include "DeepLearning/Api/Training/ModelSelectionScore.h"
#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

class Network;
class Optimizer;
class PlacedNetwork;
class TrainingData;
class BatchSession;

enum class TrainingRunExecutionMode { FIT, EVALUATE };

struct TrainingRuntimeConfig {
    uint64_t maxInFlightBatches = 32;
    double statsIntervalSeconds = 10.0;
    bool statsStderrAlso = false;
    LineStatsColorMode statsColorMode = LineStatsColorMode::AUTO;
    std::set<std::string> scalarTensorsToReport = {"loss", "accuracy"};
};

struct TrainingRunRequest {
    // Present only for standalone-network execution. Explicit TrainingProgram
    // execution owns its model through TrainingPhase networks and leaves this null.
    std::shared_ptr<Network> network = nullptr;
    std::shared_ptr<BatchSession> batchSession = nullptr;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::shared_ptr<TrainingProgram> trainingProgram = nullptr;
    // Strict, dataset-validated bindings compiled by Trainer before any
    // BatchSession is opened or the Network is placed.
    std::vector<TrainingInputBinding> datasetInputBindings{};
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

    // First epoch within this fit request at which model-selection scoring begins.
    // This gate is phase-local: after a previous fit completed N selected epochs,
    // firstModelSelectionEpoch=3 means the next fit first scores at cumulative
    // epoch N + 3. A value of 0 preserves the cadence-only behavior: the first
    // selection happens at checkBestModelEveryEpochs epochs into this request.
    // Model-selection callbacks, early-completion policies, snapshots, and
    // metadata still receive/report cumulative epoch numbers.
    uint64_t firstModelSelectionEpoch = 0;

    // Optional cap for the TRAIN phase only. When unset, a training epoch drains
    // the session's full training split as before. When set, each public training
    // epoch consumes at most this many batches and the session continues from its
    // current position across later public epochs, so large datasets can span
    // several public epochs. Validation/evaluation epochs are intentionally not
    // capped by this option.
    std::optional<uint64_t> maxTrainingBatchesPerEpoch{};

    // Present for training execution so the data recipe can own device-access
    // policy and optionally replace the source session with a resident session.
    std::shared_ptr<const TrainingData> trainingData = nullptr;
    DeviceDatasetStorageReport deviceDatasetStorageReport{};

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

    // Trainer.fit(...) creates a fresh PlacedNetwork for each run so phase
    // activation changes can recompile against a clean physical graph. When the same
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

    // Cumulative continuation epoch before/after this request. FIT trains `epochs`
    // additional epochs starting after initialCompletedEpochs. Public stats,
    // model-selection callbacks, early-completion policies, snapshots, and
    // metadata use cumulative epoch numbers. Phase-local options such as
    // firstModelSelectionEpoch are evaluated relative to initialCompletedEpochs.
    // When a run saves and selects a best candidate, completedTrainingEpochs is
    // set to the selected artifact epoch used for the next handoff; run-finished
    // stats still report the actual epoch where the training attempt stopped.
    // Trainer restart handling resets this value to the phase-initial selected
    // epoch before launching a retry because a restarted attempt discards the
    // failed attempt's trained state.
    uint64_t initialCompletedEpochs = 0;
    uint64_t* completedTrainingEpochs = nullptr;

    // Cumulative wall-clock seconds before/after this request. FIT stats add
    // initialElapsedSeconds to this request's wall time so sequential training
    // phases on the same Trainer report one continuous elapsed timer. EVALUATE
    // requests keep their own request-local timer.
    double initialElapsedSeconds = 0.0;
    double* completedTrainingElapsedSeconds = nullptr;
};

}  // namespace Thor
