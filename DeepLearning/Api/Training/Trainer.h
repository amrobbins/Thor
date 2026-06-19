#pragma once

#include "DeepLearning/Api/Training/Executors/DebugSynchronousTrainingExecutor.h"
#include "DeepLearning/Api/Training/Executors/LocalTrainingExecutor.h"
#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"
#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "DeepLearning/Api/Training/Results/TrainingRunResult.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "Utilities/Loaders/Shard.h"

class Loader;

namespace Thor {

class Network;
class Optimizer;
class TrainingRuns;

struct TrainerFitOptions {
    uint32_t epochs = 1;
};

class Trainer {
   public:
    class Builder;

    Trainer() = default;

    void fit(uint32_t epochs);
    void fit(const TrainerFitOptions& options);

    [[nodiscard]] const TrainingRuntimeConfig& getRuntimeConfig() const { return runtimeConfig; }
    [[nodiscard]] std::shared_ptr<Network> getNetwork() const { return network; }
    [[nodiscard]] const std::optional<std::string>& getSaveModelDirectory() const { return saveModelDirectory; }
    [[nodiscard]] bool getSaveModelOverwrite() const { return saveModelOverwrite; }
    [[nodiscard]] bool getSaveOptimizerState() const { return saveOptimizerState; }

   private:
    void validateFitOptions(const TrainerFitOptions& options) const;
    TrainingObserver& effectiveObserver();
    void fitInternal(const TrainerFitOptions& options, TrainingObserver& observer, const TrainingCancellationToken& cancellationToken);
    void executeRequest(const TrainingRunRequest& request, TrainingObserver& observer);
    TrainingRunResult fitTrainingRun(std::string runName,
                                     const TrainerFitOptions& options,
                                     TrainingObserver& observer,
                                     const TrainingCancellationToken& cancellationToken);
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
    bool saveOptimizerState = true;
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

    Builder& statsEnabled(bool statsEnabled) {
        runtimeConfig_.statsEnabled = statsEnabled;
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

    Builder& saveOptimizerState(bool saveOptimizerState) {
        this->saveOptimizerState_ = saveOptimizerState;
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
    bool saveOptimizerState_ = true;
};

}  // namespace Thor
