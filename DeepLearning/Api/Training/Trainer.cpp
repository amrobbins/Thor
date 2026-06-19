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
                finalTrainingStats = event.stats;
            } else if (event.stats.phase == TrainingEventPhase::VALIDATE) {
                finalValidationStats = event.stats;
            } else if (event.stats.phase == TrainingEventPhase::TEST) {
                finalTestStats = event.stats;
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
    TrainingObserver& inner;
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
