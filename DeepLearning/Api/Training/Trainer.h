#pragma once

#include "DeepLearning/Api/Training/Executors/DebugSynchronousTrainingExecutor.h"
#include "DeepLearning/Api/Training/Executors/LocalTrainingExecutor.h"
#include "DeepLearning/Api/Training/EarlyCompletionPolicy.h"
#include "DeepLearning/Api/Training/ModelSelectionScore.h"
#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "DeepLearning/Api/Training/Results/TrainingRunResult.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <set>
#include <utility>
#include <vector>

#include "Utilities/Loaders/Shard.h"

class Loader;

namespace Thor {

class Network;
class Optimizer;
class TrainingRuns;

struct TrainingRestartPolicy {
    std::optional<std::string> runName{};
    std::optional<std::string> ensembleGroup{};
    uint32_t progressCheckEpochs = 3;
    double progressImprovementMinPercentage = 5.0;
    uint32_t maxRestarts = 5;

    TrainingRestartPolicy() = default;
    TrainingRestartPolicy(uint32_t progressCheckEpochs, double progressImprovementMinPercentage = 5.0, uint32_t maxRestarts = 5)
        : progressCheckEpochs(progressCheckEpochs), progressImprovementMinPercentage(progressImprovementMinPercentage), maxRestarts(maxRestarts) {}

    static TrainingRestartPolicy forRun(std::string runName,
                                        uint32_t progressCheckEpochs = 3,
                                        double progressImprovementMinPercentage = 5.0,
                                        uint32_t maxRestarts = 5) {
        TrainingRestartPolicy policy;
        policy.runName = std::move(runName);
        policy.progressCheckEpochs = progressCheckEpochs;
        policy.progressImprovementMinPercentage = progressImprovementMinPercentage;
        policy.maxRestarts = maxRestarts;
        return policy;
    }
    static TrainingRestartPolicy forEnsembleGroup(std::string ensembleGroup,
                                                  uint32_t progressCheckEpochs = 3,
                                                  double progressImprovementMinPercentage = 5.0,
                                                  uint32_t maxRestarts = 5) {
        TrainingRestartPolicy policy;
        policy.ensembleGroup = std::move(ensembleGroup);
        policy.progressCheckEpochs = progressCheckEpochs;
        policy.progressImprovementMinPercentage = progressImprovementMinPercentage;
        policy.maxRestarts = maxRestarts;
        return policy;
    }

    [[nodiscard]] TrainingRestartPolicy withoutTarget() const {
        return TrainingRestartPolicy{progressCheckEpochs, progressImprovementMinPercentage, maxRestarts};
    }

    [[nodiscard]] TrainingRestartPolicy toRestartCondition() const { return withoutTarget(); }
};

using TrainingRestartCondition = TrainingRestartPolicy;
using TrainingRunsRestartPolicy = TrainingRestartPolicy;
using TrainingRunsRestartConditionSpec = TrainingRestartPolicy;


struct TrainerFitOptions {
    uint32_t epochs = 1;
    uint32_t checkBestModelEveryEpochs = 0;
    uint64_t firstModelSelectionEpoch = 0;
    std::vector<TrainingRestartCondition> restartConditions{};
    std::vector<TrainingEarlyCompletionPolicy> earlyCompletionPolicies{};
};

class PlacedNetwork;

class Trainer {
   public:
    class Builder;

    Trainer() = default;

    TrainingRunResult fit(uint32_t epochs);
    TrainingRunResult fit(const TrainerFitOptions& options);
    void saveModel(const std::string& directory, bool overwrite = false, bool saveOptimizerState = true) const;
    void releasePlacedNetworkAfterLastFit();

    [[nodiscard]] const TrainingRuntimeConfig& getRuntimeConfig() const { return runtimeConfig; }
    [[nodiscard]] std::shared_ptr<Network> getNetwork() const { return network; }
    [[nodiscard]] const std::optional<std::string>& getSaveModelDirectory() const { return saveModelDirectory; }
    [[nodiscard]] bool getSaveModelOverwrite() const { return saveModelOverwrite; }
    [[nodiscard]] uint64_t getCompletedTrainingEpochs() const { return completedTrainingEpochs; }
    [[nodiscard]] double getCompletedTrainingElapsedSeconds() const { return completedTrainingElapsedSeconds; }
    [[nodiscard]] const TrainingModelSelectionScore& getModelSelectionScore() const { return modelSelectionScore; }

   private:
    void validateFitOptions(const TrainerFitOptions& options) const;
    void validateRestartConditions(const std::vector<TrainingRestartCondition>& conditions) const;
    void validateEarlyCompletionPolicies(const std::vector<TrainingEarlyCompletionPolicy>& policies) const;
    TrainingObserver& effectiveObserver();
    void fitInternal(const TrainerFitOptions& options,
                     TrainingObserver& observer,
                     const TrainingCancellationToken& cancellationToken,
                     const std::vector<TrainingEarlyCompletionPolicy>& additionalEarlyCompletionPolicies = {},
                     const std::set<std::string>& additionalScalarTensorsToReport = {});
    void fitWithRestartConditions(const TrainerFitOptions& options,
                                  TrainingObserver& observer,
                                  const TrainingCancellationToken& cancellationToken,
                                  const std::vector<TrainingRestartCondition>& additionalRestartConditions,
                                  const std::vector<TrainingEarlyCompletionPolicy>& additionalEarlyCompletionPolicies,
                                  const std::string& runNameForMessages,
                                  const std::set<std::string>& additionalScalarTensorsToReport = {});
    void executeRequest(const TrainingRunRequest& request, TrainingObserver& observer);
    TrainingRunResult fitTrainingRun(std::string runName,
                                     const TrainerFitOptions& options,
                                     TrainingObserver& observer,
                                     const TrainingCancellationToken& cancellationToken,
                                     const std::vector<TrainingRestartCondition>& additionalRestartConditions = {},
                                     const std::vector<TrainingEarlyCompletionPolicy>& additionalEarlyCompletionPolicies = {},
                                     const std::set<std::string>& additionalScalarTensorsToReport = {});
    TrainingRunResult evaluateTrainingRun(std::string runName,
                                          std::shared_ptr<Loader> evaluationLoader,
                                          ExampleType exampleType,
                                          TrainingEventPhase phase,
                                          TrainingObserver& observer,
                                          const TrainingCancellationToken& cancellationToken);
    TrainingRunResult evaluateSavedTrainingRun(std::string runName,
                                               const std::string& modelArtifactDirectory,
                                               std::shared_ptr<Loader> evaluationLoader,
                                               ExampleType exampleType,
                                               TrainingEventPhase phase,
                                               TrainingObserver& observer,
                                               const TrainingCancellationToken& cancellationToken);

    friend class TrainingRuns;

    std::shared_ptr<Network> network = nullptr;
    std::shared_ptr<Loader> loader = nullptr;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::shared_ptr<TrainingProgram> trainingProgram = nullptr;
    TrainingRuntimeConfig runtimeConfig{};
    std::shared_ptr<TrainingExecutor> executor = nullptr;
    std::shared_ptr<TrainingObserver> observer = nullptr;
    std::optional<std::string> saveModelDirectory{};
    bool saveModelOverwrite = false;
    TrainingModelSelectionScore modelSelectionScore{};
    std::shared_ptr<PlacedNetwork> placedNetworkAfterLastFit = nullptr;
    std::optional<std::string> lastCompletedArtifactDirectory{};
    std::optional<std::string> lastCompletedArtifactNetworkName{};
    uint64_t completedTrainingEpochs = 0;
    double completedTrainingElapsedSeconds = 0.0;
};

class Trainer::Builder {
   public:
    Builder& network(std::shared_ptr<Network> network) {
        this->network_ = std::move(network);
        return *this;
    }

    Builder& loader(std::shared_ptr<Loader> loader) {
        this->loader_ = std::move(loader);
        return *this;
    }

    Builder& optimizer(std::shared_ptr<Optimizer> optimizer) {
        this->optimizer_ = std::move(optimizer);
        return *this;
    }

    Builder& trainingProgram(std::shared_ptr<TrainingProgram> trainingProgram) {
        this->trainingProgram_ = std::move(trainingProgram);
        return *this;
    }

    Builder& executor(std::shared_ptr<TrainingExecutor> executor) {
        this->executor_ = std::move(executor);
        return *this;
    }

    Builder& debugSynchronousExecutor() {
        this->executor_ = std::make_shared<DebugSynchronousTrainingExecutor>();
        return *this;
    }

    Builder& observer(std::shared_ptr<TrainingObserver> observer) {
        this->observer_ = std::move(observer);
        return *this;
    }

    Builder& statsIntervalSeconds(double statsIntervalSeconds) {
        runtimeConfig_.statsIntervalSeconds = statsIntervalSeconds;
        return *this;
    }

    Builder& statsStderrAlso(bool statsStderrAlso) {
        runtimeConfig_.statsStderrAlso = statsStderrAlso;
        return *this;
    }

    Builder& statsColorMode(LineStatsColorMode colorMode) {
        runtimeConfig_.statsColorMode = colorMode;
        return *this;
    }

    Builder& maxInFlightBatches(uint64_t maxInFlightBatches) {
        runtimeConfig_.maxInFlightBatches = maxInFlightBatches;
        return *this;
    }

    Builder& scalarTensorsToReport(std::set<std::string> scalarTensorsToReport) {
        runtimeConfig_.scalarTensorsToReport = std::move(scalarTensorsToReport);
        return *this;
    }

    Builder& saveModelDirectory(std::optional<std::string> saveModelDirectory) {
        this->saveModelDirectory_ = std::move(saveModelDirectory);
        return *this;
    }

    Builder& saveModelOverwrite(bool saveModelOverwrite) {
        this->saveModelOverwrite_ = saveModelOverwrite;
        return *this;
    }

    Builder& modelSelectionScore(TrainingModelSelectionScore modelSelectionScore) {
        this->modelSelectionScore_ = std::move(modelSelectionScore);
        return *this;
    }



    [[nodiscard]] Trainer build() const;

   private:
    std::shared_ptr<Network> network_ = nullptr;
    std::shared_ptr<Loader> loader_ = nullptr;
    std::shared_ptr<Optimizer> optimizer_ = nullptr;
    std::shared_ptr<TrainingProgram> trainingProgram_ = nullptr;
    TrainingRuntimeConfig runtimeConfig_{};
    std::shared_ptr<TrainingExecutor> executor_ = nullptr;
    std::shared_ptr<TrainingObserver> observer_ = nullptr;
    std::optional<std::string> saveModelDirectory_{};
    bool saveModelOverwrite_ = false;
    TrainingModelSelectionScore modelSelectionScore_{};
};

}  // namespace Thor
