#pragma once

#include "DeepLearning/Api/Training/Executors/DebugSynchronousTrainingExecutor.h"
#include "DeepLearning/Api/Training/Executors/LocalTrainingExecutor.h"
#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"

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

   private:
    void validateFitOptions(const TrainerFitOptions& options) const;
    TrainingObserver& effectiveObserver();

    Network* network = nullptr;
    std::shared_ptr<Loader> loader = nullptr;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::optional<TrainingProgram> trainingProgram{};
    TrainingRuntimeConfig runtimeConfig{};
    std::shared_ptr<TrainingExecutor> executor = nullptr;
    std::shared_ptr<TrainingObserver> observer = nullptr;
};

class Trainer::Builder {
   public:
    Builder& network(Network& network) {
        this->network_ = &network;
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

    Builder& trainingProgram(TrainingProgram trainingProgram) {
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
        this->statsColorMode_ = colorMode;
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

    [[nodiscard]] Trainer build() const;

   private:
    Network* network_ = nullptr;
    std::shared_ptr<Loader> loader_ = nullptr;
    std::shared_ptr<Optimizer> optimizer_ = nullptr;
    std::optional<TrainingProgram> trainingProgram_{};
    TrainingRuntimeConfig runtimeConfig_{};
    LineStatsColorMode statsColorMode_ = LineStatsColorMode::ALWAYS;
    std::shared_ptr<TrainingExecutor> executor_ = nullptr;
    std::shared_ptr<TrainingObserver> observer_ = nullptr;
};

}  // namespace Thor
