#include "DeepLearning/Api/Training/Trainer.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"

#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace Thor {

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
    if (trainer.observer == nullptr) {
        if (trainer.runtimeConfig.statsEnabled) {
            const LineStatsOutputMode outputMode =
                trainer.runtimeConfig.statsStderrAlso ? LineStatsOutputMode::STDOUT_AND_STDERR : LineStatsOutputMode::STDOUT;
            trainer.observer = std::make_shared<LineStatsReporter>(
                trainer.runtimeConfig.statsIntervalSeconds, true, trainer.runtimeConfig.statsColorMode, outputMode);
        } else {
            trainer.observer = std::make_shared<NullTrainingObserver>();
        }
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
        }
        inner.onTrainingEvent(event);
    }

    void flush() override { inner.flush(); }
    void close() override { inner.close(); }

    std::optional<TrainingStatsSnapshot> finalTrainingStats{};
    std::optional<TrainingStatsSnapshot> finalValidationStats{};
    std::optional<TrainingStatsSnapshot> finalTestStats{};

   private:
    struct FinalEpochLossAccumulator {
        uint64_t currentEpoch = 0;
        double currentEpochLossSum = 0.0;
        uint64_t currentEpochLossCount = 0;

        TrainingStatsSnapshot update(const TrainingStatsSnapshot& stats) {
            TrainingStatsSnapshot finalStats = stats;
            if (!stats.loss.has_value()) {
                return finalStats;
            }

            if (currentEpoch != stats.epoch) {
                currentEpoch = stats.epoch;
                currentEpochLossSum = 0.0;
                currentEpochLossCount = 0;
            }

            currentEpochLossSum += stats.loss.value();
            currentEpochLossCount += 1;
            finalStats.loss = currentEpochLossSum / static_cast<double>(currentEpochLossCount);
            return finalStats;
        }
    };

    TrainingObserver& inner;
    FinalEpochLossAccumulator trainingLoss{};
    FinalEpochLossAccumulator validationLoss{};
    FinalEpochLossAccumulator testLoss{};
};


}  // namespace

void Trainer::fit(uint32_t epochs) { fit(TrainerFitOptions{epochs}); }

void Trainer::fit(const TrainerFitOptions& options) {
    TrainingObserver& observer = effectiveObserver();
    fitInternal(options, observer, TrainingCancellationToken{});
}

void Trainer::fitInternal(const TrainerFitOptions& options,
                          TrainingObserver& observer,
                          const TrainingCancellationToken& cancellationToken) {
    validateFitOptions(options);
    cancellationToken.throwIfCancellationRequested();

    TrainingRunRequest request;
    request.network = network;
    request.loader = loader;
    request.optimizer = optimizer;
    request.trainingProgram = trainingProgram;
    request.runtime = runtimeConfig;
    request.epochs = options.epochs;
    request.saveModelDirectory = saveModelDirectory;
    request.saveModelOverwrite = saveModelOverwrite;
    request.saveOptimizerState = saveOptimizerState;
    request.cancellationToken = cancellationToken;
    request.executionMode = TrainingRunExecutionMode::FIT;

    executeRequest(request, observer);
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
                                          const TrainingCancellationToken& cancellationToken) {
    ResultCapturingTrainingObserver capturingObserver(observer);
    try {
        fitInternal(options, capturingObserver, cancellationToken);
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
    request.runtime.statsEnabled = true;
    request.runtime.scalarTensorsToReport.insert("loss");
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
}

TrainingObserver& Trainer::effectiveObserver() {
    if (observer == nullptr) {
        static NullTrainingObserver nullObserver;
        return nullObserver;
    }
    return *observer;
}

}  // namespace Thor
