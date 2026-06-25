#include "DeepLearning/Api/Training/Trainer.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <unordered_map>
#include <vector>

namespace Thor {

namespace {

std::optional<std::string> trainedArtifactNetworkName(const std::shared_ptr<PlacedNetwork>& placedNetwork,
                                                       const std::shared_ptr<Network>& fallbackNetwork) {
    if (placedNetwork != nullptr) {
        const std::string name = placedNetwork->getNetworkName();
        if (!name.empty()) {
            return name;
        }
    }
    if (fallbackNetwork != nullptr) {
        const std::string name = fallbackNetwork->getNetworkName();
        if (!name.empty()) {
            return name;
        }
    }
    return std::nullopt;
}

}  // namespace

Trainer Trainer::Builder::build() const {
    if (network_ == nullptr) {
        throw std::runtime_error("Trainer requires a Network.");
    }
    if (loader_ == nullptr) {
        throw std::runtime_error("Trainer requires a Loader.");
    }
    if (runtimeConfig_.maxInFlightBatches == 0) {
        throw std::runtime_error("Trainer maxInFlightBatches must be >= 1.");
    }
    if (!std::isfinite(runtimeConfig_.statsIntervalSeconds) || runtimeConfig_.statsIntervalSeconds < 0.0) {
        throw std::runtime_error("Trainer statsIntervalSeconds must be finite and >= 0.");
    }
    if (checkBestModelEveryEpochs_ == 0) {
        throw std::runtime_error("Trainer check_best_model_every_epochs must be >= 1.");
    }
    for (size_t i = 0; i < restartConditions_.size(); ++i) {
        const TrainingRestartCondition& condition = restartConditions_[i];
        if (condition.progressCheckEpochs == 0) {
            throw std::runtime_error("Trainer restart_condition at index " + std::to_string(i) + " must have progress_check_epochs >= 1.");
        }
        if (!std::isfinite(condition.progressImprovementMinPercentage) || condition.progressImprovementMinPercentage < 0.0 || condition.progressImprovementMinPercentage > 100.0) {
            throw std::runtime_error("Trainer restart_condition at index " + std::to_string(i) + " must have progress_improvement_min_percentage in [0, 100].");
        }
    }
    for (size_t i = 0; i < earlyCompletionPolicies_.size(); ++i) {
        if (!earlyCompletionPolicies_[i].completionCondition) {
            throw std::runtime_error("Trainer early_completion_policy at index " + std::to_string(i) + " must have a completion_condition.");
        }
    }

    Trainer trainer;
    trainer.network = network_;
    trainer.loader = loader_;
    trainer.optimizer = optimizer_;
    trainer.trainingProgram = trainingProgram_;
    trainer.runtimeConfig = runtimeConfig_;
    trainer.executor = executor_ ? executor_ : std::make_shared<LocalTrainingExecutor>();
    trainer.observer = observer_;
    trainer.saveModelDirectory = saveModelDirectory_;
    trainer.saveModelOverwrite = saveModelOverwrite_;
    trainer.saveOptimizerState = saveOptimizerState_;
    trainer.checkBestModelEveryEpochs = checkBestModelEveryEpochs_;
    trainer.minEarlyCompletionEpochs = minEarlyCompletionEpochs_;
    trainer.modelSelectionScore = modelSelectionScore_;
    trainer.restartConditions = restartConditions_;
    trainer.earlyCompletionPolicies = earlyCompletionPolicies_;
    if (trainer.observer == nullptr) {
        const LineStatsOutputMode outputMode =
            trainer.runtimeConfig.statsStderrAlso ? LineStatsOutputMode::STDOUT_AND_STDERR : LineStatsOutputMode::STDOUT;
        trainer.observer = std::make_shared<LineStatsReporter>(
            trainer.runtimeConfig.statsIntervalSeconds, true, trainer.runtimeConfig.statsColorMode, outputMode);
    }

    return trainer;
}

namespace {

class ResultCapturingTrainingObserver : public TrainingObserver {
   public:
    explicit ResultCapturingTrainingObserver(TrainingObserver& inner) : inner(inner) {}

    void onTrainingEvent(const TrainingEvent& event) override {
        if (event.type == TrainingEventType::STATS) {
            if (event.stats.phase == TrainingEventPhase::TRAIN) {
                finalTrainingStats = trainingLoss.update(event.stats);
            } else if (event.stats.phase == TrainingEventPhase::VALIDATE) {
                finalValidationStats = validationLoss.update(event.stats);
            } else if (event.stats.phase == TrainingEventPhase::TEST) {
                finalTestStats = testLoss.update(event.stats);
            }
        } else if (event.type == TrainingEventType::RUN_FINISHED) {
            completionReason = event.message == "early_completed" ? TrainingRunCompletionReason::EARLY_COMPLETED
                                                                  : TrainingRunCompletionReason::COMPLETED;
            completedEpoch = uintMetric(event.stats, "completed_epoch");
            if (!completedEpoch.has_value() && event.stats.epoch > 0) {
                completedEpoch = event.stats.epoch;
            }
            bestEpoch = uintMetric(event.stats, "best_epoch");
            bestScore = doubleMetric(event.stats, "best_score");
        }
        inner.onTrainingEvent(event);
    }

    void flush() override { inner.flush(); }
    void close() override { inner.close(); }

    std::optional<TrainingStatsSnapshot> finalTrainingStats{};
    std::optional<TrainingStatsSnapshot> finalValidationStats{};
    std::optional<TrainingStatsSnapshot> finalTestStats{};
    TrainingRunCompletionReason completionReason = TrainingRunCompletionReason::COMPLETED;
    std::optional<uint64_t> completedEpoch{};
    std::optional<uint64_t> bestEpoch{};
    std::optional<double> bestScore{};

   private:
    static std::optional<double> doubleMetric(const TrainingStatsSnapshot& stats, const std::string& name) {
        const auto it = stats.metrics.find(name);
        if (it == stats.metrics.end() || !std::isfinite(it->second)) {
            return std::nullopt;
        }
        return it->second;
    }

    static std::optional<uint64_t> uintMetric(const TrainingStatsSnapshot& stats, const std::string& name) {
        const std::optional<double> value = doubleMetric(stats, name);
        if (!value.has_value() || value.value() < 0.0) {
            return std::nullopt;
        }
        return static_cast<uint64_t>(value.value());
    }

    struct FinalEpochStatsAccumulator {
        struct RunningMean {
            double sum = 0.0;
            uint64_t count = 0;

            void add(double value) {
                sum += value;
                count += 1;
            }

            [[nodiscard]] std::optional<double> mean() const {
                if (count == 0) {
                    return std::nullopt;
                }
                return sum / static_cast<double>(count);
            }
        };

        uint64_t currentEpoch = 0;
        RunningMean lossMean{};
        std::unordered_map<std::string, RunningMean> metricMeans{};

        TrainingStatsSnapshot update(const TrainingStatsSnapshot& stats) {
            TrainingStatsSnapshot finalStats = stats;
            if (currentEpoch != stats.epoch) {
                currentEpoch = stats.epoch;
                lossMean = RunningMean{};
                metricMeans.clear();
            }

            if (stats.loss.has_value() && std::isfinite(stats.loss.value())) {
                lossMean.add(stats.loss.value());
                finalStats.loss = lossMean.mean();
            }

            for (const auto& [name, value] : stats.metrics) {
                if (!std::isfinite(value)) {
                    continue;
                }
                RunningMean& metricMean = metricMeans[name];
                metricMean.add(value);
                std::optional<double> mean = metricMean.mean();
                if (mean.has_value()) {
                    finalStats.metrics[name] = mean.value();
                }
            }

            return finalStats;
        }
    };

    TrainingObserver& inner;
    FinalEpochStatsAccumulator trainingLoss{};
    FinalEpochStatsAccumulator validationLoss{};
    FinalEpochStatsAccumulator testLoss{};
};


struct RestartAttemptProgress {
    uint64_t attempt = 1;
    double firstStepLoss = 0.0;
    double checkLoss = 0.0;
    double requiredLoss = 0.0;
    double observedProgressPercentage = 0.0;
    uint64_t checkedEpoch = 0;
};

std::string formatDoubleForRestartMessage(double value) {
    std::ostringstream out;
    out << std::setprecision(8) << value;
    return out.str();
}

std::string restartConditionName(const TrainingRestartCondition& condition) {
    std::ostringstream out;
    out << "progress_check_epochs=" << condition.progressCheckEpochs
        << " progress_improvement_min_percentage=" << formatDoubleForRestartMessage(condition.progressImprovementMinPercentage)
        << " max_restarts=" << condition.maxRestarts;
    return out.str();
}

std::string restartAttemptProgressToString(const RestartAttemptProgress& progress) {
    std::ostringstream out;
    out << "attempt " << progress.attempt << ": first_step_loss=" << formatDoubleForRestartMessage(progress.firstStepLoss)
        << " check_epoch=" << progress.checkedEpoch << " check_loss=" << formatDoubleForRestartMessage(progress.checkLoss)
        << " required_loss<=" << formatDoubleForRestartMessage(progress.requiredLoss)
        << " observed_progress=" << formatDoubleForRestartMessage(progress.observedProgressPercentage) << "%";
    return out.str();
}

std::string restartAttemptsProgressMessage(const std::string& runName,
                                           const TrainingRestartCondition& condition,
                                           const std::vector<RestartAttemptProgress>& attempts) {
    std::ostringstream out;
    out << "run '" << runName << "' failed restart_condition (" << restartConditionName(condition)
        << "): training loss did not improve by at least "
        << formatDoubleForRestartMessage(condition.progressImprovementMinPercentage) << "% by epoch "
        << condition.progressCheckEpochs << " across " << attempts.size()
        << " attempt" << (attempts.size() == 1 ? "" : "s") << " (max_restarts=" << condition.maxRestarts << ").";
    if (!attempts.empty()) {
        out << " Progress by failed attempt: ";
        for (size_t i = 0; i < attempts.size(); ++i) {
            if (i != 0) {
                out << "; ";
            }
            out << restartAttemptProgressToString(attempts[i]);
        }
    }
    return out.str();
}

class TrainingRestartRequested final : public std::exception {
   public:
    TrainingRestartRequested(const TrainingRestartCondition* condition, RestartAttemptProgress progress, std::string message)
        : condition_(condition), progress_(std::move(progress)), message_(std::move(message)) {}

    const char* what() const noexcept override { return message_.c_str(); }
    [[nodiscard]] const TrainingRestartCondition* condition() const { return condition_; }
    [[nodiscard]] const RestartAttemptProgress& progress() const { return progress_; }

   private:
    const TrainingRestartCondition* condition_ = nullptr;
    RestartAttemptProgress progress_{};
    std::string message_{};
};

class TrainingRestartConditionExceeded final : public std::exception {
   public:
    explicit TrainingRestartConditionExceeded(std::string message) : message_(std::move(message)) {}
    const char* what() const noexcept override { return message_.c_str(); }

   private:
    std::string message_{};
};

class RestartAttemptObserver : public TrainingObserver {
   public:
    RestartAttemptObserver(TrainingObserver& inner,
                           std::vector<const TrainingRestartCondition*> restartConditions,
                           uint64_t attemptNumber)
        : inner(inner), attemptNumber(attemptNumber) {
        restartConditionStates.reserve(restartConditions.size());
        for (const TrainingRestartCondition* condition : restartConditions) {
            if (condition != nullptr) {
                restartConditionStates.push_back(RestartConditionAttemptState{condition});
            }
        }
    }

    void onTrainingEvent(const TrainingEvent& event) override {
        if (event.type == TrainingEventType::STATS) {
            if (event.stats.phase == TrainingEventPhase::TRAIN) {
                finalTrainingStats = trainingLoss.update(event.stats);
            } else if (event.stats.phase == TrainingEventPhase::VALIDATE) {
                finalValidationStats = validationLoss.update(event.stats);
            } else if (event.stats.phase == TrainingEventPhase::TEST) {
                finalTestStats = testLoss.update(event.stats);
            }
        }

        inner.onTrainingEvent(event);

        if (event.type == TrainingEventType::STATS && event.stats.phase == TrainingEventPhase::TRAIN) {
            maybeThrowForRestartConditions(event.stats);
        }
    }

    void flush() override { inner.flush(); }
    void close() override { inner.close(); }

    std::optional<TrainingStatsSnapshot> finalTrainingStats{};
    std::optional<TrainingStatsSnapshot> finalValidationStats{};
    std::optional<TrainingStatsSnapshot> finalTestStats{};

   private:
    struct FinalEpochStatsAccumulator {
        struct RunningMean {
            double sum = 0.0;
            uint64_t count = 0;

            void add(double value) {
                sum += value;
                count += 1;
            }

            [[nodiscard]] std::optional<double> mean() const {
                if (count == 0) {
                    return std::nullopt;
                }
                return sum / static_cast<double>(count);
            }
        };

        uint64_t currentEpoch = 0;
        RunningMean lossMean{};
        std::unordered_map<std::string, RunningMean> metricMeans{};

        TrainingStatsSnapshot update(const TrainingStatsSnapshot& stats) {
            TrainingStatsSnapshot finalStats = stats;
            if (currentEpoch != stats.epoch) {
                currentEpoch = stats.epoch;
                lossMean = RunningMean{};
                metricMeans.clear();
            }

            if (stats.loss.has_value() && std::isfinite(stats.loss.value())) {
                lossMean.add(stats.loss.value());
                finalStats.loss = lossMean.mean();
            }

            for (const auto& [name, value] : stats.metrics) {
                if (!std::isfinite(value)) {
                    continue;
                }
                RunningMean& metricMean = metricMeans[name];
                metricMean.add(value);
                std::optional<double> mean = metricMean.mean();
                if (mean.has_value()) {
                    finalStats.metrics[name] = mean.value();
                }
            }

            return finalStats;
        }
    };

    struct RestartConditionAttemptState {
        const TrainingRestartCondition* condition = nullptr;
        bool checked = false;
        std::optional<double> firstStepLoss{};
    };

    void maybeThrowForRestartConditions(const TrainingStatsSnapshot& stats) {
        if (restartConditionStates.empty() || !stats.loss.has_value()) {
            return;
        }

        for (RestartConditionAttemptState& state : restartConditionStates) {
            if (!state.firstStepLoss.has_value()) {
                state.firstStepLoss = stats.loss.value();
            }
        }

        for (RestartConditionAttemptState& state : restartConditionStates) {
            maybeThrowForRestartCondition(state, stats);
        }
    }

    void maybeThrowForRestartCondition(RestartConditionAttemptState& state, const TrainingStatsSnapshot& stats) {
        const TrainingRestartCondition* restartCondition = state.condition;
        if (restartCondition == nullptr || state.checked || stats.epoch < restartCondition->progressCheckEpochs) {
            return;
        }

        const bool epochComplete = stats.stepsPerEpoch == 0 || stats.stepInEpoch >= stats.stepsPerEpoch;
        if (!epochComplete) {
            return;
        }

        state.checked = true;
        if (!finalTrainingStats.has_value() || !finalTrainingStats->loss.has_value() || !state.firstStepLoss.has_value()) {
            return;
        }

        const double firstStepLoss = state.firstStepLoss.value();
        const double checkLoss = finalTrainingStats->loss.value();
        const double requiredLoss = firstStepLoss * (1.0 - restartCondition->progressImprovementMinPercentage / 100.0);
        const double observedProgressPercentage =
            firstStepLoss == 0.0 ? (checkLoss <= 0.0 ? 100.0 : -std::numeric_limits<double>::infinity())
                                 : ((firstStepLoss - checkLoss) / firstStepLoss) * 100.0;

        if (std::isfinite(firstStepLoss) && std::isfinite(checkLoss) && checkLoss <= requiredLoss) {
            return;
        }

        RestartAttemptProgress progress;
        progress.attempt = attemptNumber;
        progress.firstStepLoss = firstStepLoss;
        progress.checkLoss = checkLoss;
        progress.requiredLoss = requiredLoss;
        progress.observedProgressPercentage = observedProgressPercentage;
        progress.checkedEpoch = stats.epoch;

        throw TrainingRestartRequested(
            restartCondition,
            progress,
            "training loss did not improve enough by restart progress checkpoint: " + restartAttemptProgressToString(progress));
    }

    TrainingObserver& inner;
    uint64_t attemptNumber = 1;
    FinalEpochStatsAccumulator trainingLoss{};
    FinalEpochStatsAccumulator validationLoss{};
    FinalEpochStatsAccumulator testLoss{};
    std::vector<RestartConditionAttemptState> restartConditionStates{};
};


}  // namespace

TrainingRunResult Trainer::fit(uint32_t epochs) { return fit(TrainerFitOptions{epochs}); }

TrainingRunResult Trainer::fit(const TrainerFitOptions& options) {
    TrainingObserver& observer = effectiveObserver();
    ResultCapturingTrainingObserver capturingObserver(observer);
    fitWithRestartConditions(options, capturingObserver, TrainingCancellationToken{}, {}, {}, "trainer");
    capturingObserver.flush();
    return TrainingRunResult::completedResult("trainer",
                                              capturingObserver.finalTrainingStats,
                                              capturingObserver.finalValidationStats,
                                              capturingObserver.finalTestStats,
                                              capturingObserver.completionReason,
                                              capturingObserver.completedEpoch,
                                              capturingObserver.bestEpoch,
                                              capturingObserver.bestScore,
                                              saveModelDirectory,
                                              trainedArtifactNetworkName(placedNetworkAfterLastFit, network));
}


void Trainer::saveModel(const std::string& directory, bool overwrite, bool saveOptimizerState) const {
    if (directory.empty()) {
        throw std::runtime_error("Trainer::saveModel directory must not be empty.");
    }
    if (placedNetworkAfterLastFit == nullptr) {
        throw std::runtime_error("Trainer::saveModel requires a completed fit before saving trained model state.");
    }
    placedNetworkAfterLastFit->save(directory, overwrite, saveOptimizerState);
}

void Trainer::fitInternal(const TrainerFitOptions& options,
                          TrainingObserver& observer,
                          const TrainingCancellationToken& cancellationToken,
                          const std::vector<TrainingEarlyCompletionPolicy>& additionalEarlyCompletionPolicies,
                          const std::set<std::string>& additionalScalarTensorsToReport) {
    validateFitOptions(options);
    std::vector<TrainingEarlyCompletionPolicy> combinedEarlyCompletionPolicies = earlyCompletionPolicies;
    combinedEarlyCompletionPolicies.insert(combinedEarlyCompletionPolicies.end(),
                                           additionalEarlyCompletionPolicies.begin(),
                                           additionalEarlyCompletionPolicies.end());
    validateEarlyCompletionPolicies(combinedEarlyCompletionPolicies);
    cancellationToken.throwIfCancellationRequested();

    TrainingRunRequest request;
    request.network = network;
    request.loader = loader;
    request.optimizer = optimizer;
    request.trainingProgram = trainingProgram;
    request.runtime = runtimeConfig;
    request.runtime.scalarTensorsToReport.insert(additionalScalarTensorsToReport.begin(), additionalScalarTensorsToReport.end());
    request.epochs = options.epochs;
    request.saveModelDirectory = saveModelDirectory;
    request.saveModelOverwrite = saveModelOverwrite;
    request.saveOptimizerState = saveOptimizerState;
    request.checkBestModelEveryEpochs = checkBestModelEveryEpochs;
    request.minEarlyCompletionEpochs = minEarlyCompletionEpochs;
    request.initialCompletedEpochs = completedTrainingEpochs;
    request.modelSelectionScore = modelSelectionScore;
    request.earlyCompletionPolicies = std::move(combinedEarlyCompletionPolicies);
    request.cancellationToken = cancellationToken;
    request.executionMode = TrainingRunExecutionMode::FIT;
    request.previousPlacedNetwork = placedNetworkAfterLastFit;
    request.completedPlacedNetwork = &placedNetworkAfterLastFit;
    request.completedTrainingEpochs = &completedTrainingEpochs;

    executeRequest(request, observer);
}


void Trainer::fitWithRestartConditions(const TrainerFitOptions& options,
                                       TrainingObserver& observer,
                                       const TrainingCancellationToken& cancellationToken,
                                       const std::vector<TrainingRestartCondition>& additionalRestartConditions,
                                       const std::vector<TrainingEarlyCompletionPolicy>& additionalEarlyCompletionPolicies,
                                       const std::string& runNameForMessages,
                                       const std::set<std::string>& additionalScalarTensorsToReport) {
    std::vector<TrainingRestartCondition> combinedConditions = restartConditions;
    combinedConditions.insert(combinedConditions.end(), additionalRestartConditions.begin(), additionalRestartConditions.end());
    validateRestartConditions(combinedConditions);

    struct RestartConditionRunState {
        const TrainingRestartCondition* condition = nullptr;
        std::vector<RestartAttemptProgress> failedAttempts{};
    };

    std::vector<RestartConditionRunState> conditionStates;
    conditionStates.reserve(combinedConditions.size());
    for (const TrainingRestartCondition& condition : combinedConditions) {
        conditionStates.push_back(RestartConditionRunState{&condition});
    }

    auto conditionIsReachableInAttempt = [&](const TrainingRestartCondition& condition, uint64_t attemptInitialCompletedEpochs) {
        const uint64_t finalEpochAfterFit = attemptInitialCompletedEpochs + options.epochs;
        return condition.progressCheckEpochs > attemptInitialCompletedEpochs && condition.progressCheckEpochs <= finalEpochAfterFit;
    };

    const bool anyConditionReachableOnCurrentState = std::any_of(
        combinedConditions.begin(),
        combinedConditions.end(),
        [&](const TrainingRestartCondition& condition) { return conditionIsReachableInAttempt(condition, completedTrainingEpochs); });
    const bool anyConditionReachableAfterFreshRestart = std::any_of(
        combinedConditions.begin(),
        combinedConditions.end(),
        [&](const TrainingRestartCondition& condition) { return conditionIsReachableInAttempt(condition, /*attemptInitialCompletedEpochs=*/0); });

    if (!anyConditionReachableOnCurrentState && !anyConditionReachableAfterFreshRestart) {
        fitInternal(options, observer, cancellationToken, additionalEarlyCompletionPolicies, additionalScalarTensorsToReport);
        return;
    }

    auto findConditionState = [&](const TrainingRestartCondition* condition) -> RestartConditionRunState* {
        for (RestartConditionRunState& state : conditionStates) {
            if (state.condition == condition) {
                return &state;
            }
        }
        return nullptr;
    };

    auto activeConditions = [&](uint64_t attemptInitialCompletedEpochs) {
        std::vector<const TrainingRestartCondition*> active;
        active.reserve(conditionStates.size());
        for (const RestartConditionRunState& state : conditionStates) {
            if (state.condition != nullptr && conditionIsReachableInAttempt(*state.condition, attemptInitialCompletedEpochs)) {
                active.push_back(state.condition);
            }
        }
        return active;
    };

    for (uint64_t attempt = 1;; ++attempt) {
        cancellationToken.throwIfCancellationRequested();
        Trainer attemptTrainer = *this;
        attemptTrainer.restartConditions.clear();
        attemptTrainer.runtimeConfig.scalarTensorsToReport.insert("loss");
        RestartAttemptObserver attemptObserver(observer, activeConditions(attemptTrainer.completedTrainingEpochs), attempt);
        try {
            attemptTrainer.fitInternal(options, attemptObserver, cancellationToken, additionalEarlyCompletionPolicies, additionalScalarTensorsToReport);
            placedNetworkAfterLastFit = attemptTrainer.placedNetworkAfterLastFit;
            completedTrainingEpochs = attemptTrainer.completedTrainingEpochs;
            attemptObserver.flush();
            return;
        } catch (const TrainingRestartRequested& e) {
            attemptObserver.flush();
            RestartConditionRunState* state = findConditionState(e.condition());
            if (state == nullptr || state->condition == nullptr) {
                throw;
            }

            state->failedAttempts.push_back(e.progress());
            if (state->failedAttempts.size() <= state->condition->maxRestarts) {
                // A restart means the current model attempt is discarded. The next
                // attempt must behave like a new model instance: no previously trained
                // PlacedNetwork state is copied in, and epoch/progress accounting starts
                // over from 0 for all restart checks, best-candidate selection, and
                // early-completion policies.
                placedNetworkAfterLastFit.reset();
                completedTrainingEpochs = 0;
                continue;
            }

            throw TrainingRestartConditionExceeded(
                restartAttemptsProgressMessage(runNameForMessages, *state->condition, state->failedAttempts));
        } catch (...) {
            attemptObserver.flush();
            throw;
        }
    }
}

void Trainer::executeRequest(const TrainingRunRequest& request, TrainingObserver& observer) {
    try {
        executor->fit(request, observer);
    } catch (...) {
        observer.flush();
        throw;
    }
    observer.flush();
}

TrainingRunResult Trainer::fitTrainingRun(std::string runName,
                                          const TrainerFitOptions& options,
                                          TrainingObserver& observer,
                                          const TrainingCancellationToken& cancellationToken,
                                          const std::vector<TrainingRestartCondition>& additionalRestartConditions,
                                          const std::vector<TrainingEarlyCompletionPolicy>& additionalEarlyCompletionPolicies,
                                          const std::set<std::string>& additionalScalarTensorsToReport) {
    ResultCapturingTrainingObserver capturingObserver(observer);
    try {
        fitWithRestartConditions(options,
                                 capturingObserver,
                                 cancellationToken,
                                 additionalRestartConditions,
                                 additionalEarlyCompletionPolicies,
                                 runName,
                                 additionalScalarTensorsToReport);
        capturingObserver.flush();
        return TrainingRunResult::completedResult(std::move(runName),
                                                  capturingObserver.finalTrainingStats,
                                                  capturingObserver.finalValidationStats,
                                                  capturingObserver.finalTestStats,
                                                  capturingObserver.completionReason,
                                                  capturingObserver.completedEpoch,
                                                  capturingObserver.bestEpoch,
                                                  capturingObserver.bestScore,
                                                  saveModelDirectory,
                                                  trainedArtifactNetworkName(placedNetworkAfterLastFit, network));
    } catch (const TrainingRestartConditionExceeded& e) {
        capturingObserver.flush();
        TrainingRunResult result;
        result.runName = std::move(runName);
        result.status = TrainingRunStatus::FAILED;
        result.finalTrainingStats = capturingObserver.finalTrainingStats;
        result.finalValidationStats = capturingObserver.finalValidationStats;
        result.finalTestStats = capturingObserver.finalTestStats;
        result.savedModelDirectory = saveModelDirectory;
        result.savedModelNetworkName = trainedArtifactNetworkName(placedNetworkAfterLastFit, network);
        result.exception = TrainingRunExceptionSummary{"TrainingRestartConditionExceeded", e.what()};
        return result;
    } catch (...) {
        capturingObserver.flush();
        return TrainingRunResult::fromException(
            std::move(runName),
            std::current_exception(),
            capturingObserver.finalTrainingStats,
            capturingObserver.finalValidationStats,
            capturingObserver.finalTestStats,
            saveModelDirectory,
            trainedArtifactNetworkName(placedNetworkAfterLastFit, network));
    }
}

TrainingRunResult Trainer::evaluateTrainingRun(std::string runName,
                                               std::shared_ptr<Loader> evaluationLoader,
                                               ExampleType exampleType,
                                               TrainingEventPhase phase,
                                               TrainingObserver& observer,
                                               const TrainingCancellationToken& cancellationToken) {
    if (network == nullptr) {
        throw std::runtime_error("Trainer::evaluate requires a Network.");
    }
    if (evaluationLoader == nullptr) {
        throw std::runtime_error("Trainer::evaluate requires a Loader.");
    }
    if (executor == nullptr) {
        throw std::runtime_error("Trainer::evaluate requires a TrainingExecutor.");
    }
    if (phase == TrainingEventPhase::UNKNOWN) {
        throw std::runtime_error("Trainer::evaluate requires a concrete phase.");
    }

    ResultCapturingTrainingObserver capturingObserver(observer);
    TrainingRunRequest request;
    request.network = network;
    request.loader = std::move(evaluationLoader);
    request.optimizer = optimizer;
    request.trainingProgram = trainingProgram;
    request.runtime = runtimeConfig;
    request.runtime.scalarTensorsToReport.insert("loss");
    request.checkBestModelEveryEpochs = 1;
    request.epochs = 1;
    request.cancellationToken = cancellationToken;
    request.executionMode = TrainingRunExecutionMode::EVALUATE;
    request.evaluationExampleType = exampleType;
    request.evaluationPhase = phase;

    try {
        executeRequest(request, capturingObserver);
        capturingObserver.flush();
        return TrainingRunResult::completedResult(
            std::move(runName), capturingObserver.finalTrainingStats, capturingObserver.finalValidationStats, capturingObserver.finalTestStats);
    } catch (...) {
        capturingObserver.flush();
        return TrainingRunResult::fromException(
            std::move(runName),
            std::current_exception(),
            capturingObserver.finalTrainingStats,
            capturingObserver.finalValidationStats,
            capturingObserver.finalTestStats);
    }
}


TrainingRunResult Trainer::evaluateSavedTrainingRun(std::string runName,
                                                    const std::string& modelArtifactDirectory,
                                                    std::shared_ptr<Loader> evaluationLoader,
                                                    ExampleType exampleType,
                                                    TrainingEventPhase phase,
                                                    TrainingObserver& observer,
                                                    const TrainingCancellationToken& cancellationToken) {
    if (network == nullptr) {
        throw std::runtime_error("Trainer::evaluateSavedTrainingRun requires an original Network for artifact loading context.");
    }
    auto loadedNetwork = std::make_shared<Network>(network->getNetworkName());
    loadedNetwork->load(modelArtifactDirectory);

    Trainer evaluator = *this;
    evaluator.network = std::move(loadedNetwork);
    return evaluator.evaluateTrainingRun(std::move(runName),
                                         std::move(evaluationLoader),
                                         exampleType,
                                         phase,
                                         observer,
                                         cancellationToken);
}

void Trainer::validateFitOptions(const TrainerFitOptions& options) const {
    if (network == nullptr) {
        throw std::runtime_error("Trainer::fit requires a Network.");
    }
    if (loader == nullptr) {
        throw std::runtime_error("Trainer::fit requires a Loader.");
    }
    if (executor == nullptr) {
        throw std::runtime_error("Trainer::fit requires a TrainingExecutor.");
    }
    if (options.epochs == 0) {
        throw std::runtime_error("Trainer::fit epochs must be >= 1.");
    }
    if (checkBestModelEveryEpochs == 0) {
        throw std::runtime_error("Trainer::fit check_best_model_every_epochs must be >= 1.");
    }
    if (saveModelDirectory.has_value()) {
        if (saveModelDirectory->empty()) {
            throw std::runtime_error("Trainer::fit save_model_dir must not be empty.");
        }
        if (!saveModelOverwrite) {
            std::error_code error;
            const std::filesystem::path outputDirectory(*saveModelDirectory);
            const bool exists = std::filesystem::exists(outputDirectory, error);
            if (error) {
                throw std::runtime_error("Trainer::fit could not check save_model_dir '" + outputDirectory.string() +
                                         "': " + error.message());
            }
            if (exists) {
                throw std::runtime_error("Trainer::fit save_model_dir '" + outputDirectory.string() +
                                         "' already exists; set save_model_overwrite=true to replace it.");
            }
        }
    }
}


void Trainer::validateRestartConditions(const std::vector<TrainingRestartCondition>& conditions) const {
    for (size_t i = 0; i < conditions.size(); ++i) {
        const TrainingRestartCondition& condition = conditions[i];
        if (condition.progressCheckEpochs == 0) {
            throw std::runtime_error("Trainer restart_condition at index " + std::to_string(i) + " must have progress_check_epochs >= 1.");
        }
        if (!std::isfinite(condition.progressImprovementMinPercentage) || condition.progressImprovementMinPercentage < 0.0 || condition.progressImprovementMinPercentage > 100.0) {
            throw std::runtime_error("Trainer restart_condition at index " + std::to_string(i) + " must have progress_improvement_min_percentage in [0, 100].");
        }
    }
}

void Trainer::validateEarlyCompletionPolicies(const std::vector<TrainingEarlyCompletionPolicy>& policies) const {
    for (size_t i = 0; i < policies.size(); ++i) {
        if (!policies[i].completionCondition) {
            throw std::runtime_error("Trainer early_completion_policy at index " + std::to_string(i) +
                                     " must have a completion_condition.");
        }
    }
}

TrainingObserver& Trainer::effectiveObserver() {
    if (observer == nullptr) {
        static NullTrainingObserver nullObserver;
        return nullObserver;
    }
    return *observer;
}

}  // namespace Thor
