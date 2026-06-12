#include "DeepLearning/Api/Training/Trainer.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"

#include <cmath>
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
    if (trainer.observer == nullptr) {
        if (trainer.runtimeConfig.statsEnabled) {
            trainer.observer = std::make_shared<LineStatsReporter>(trainer.runtimeConfig.statsIntervalSeconds, true);
        } else {
            trainer.observer = std::make_shared<NullTrainingObserver>();
        }
    }

    return trainer;
}

void Trainer::fit(uint32_t epochs) { fit(TrainerFitOptions{epochs}); }

void Trainer::fit(const TrainerFitOptions& options) {
    validateFitOptions(options);

    TrainingRunRequest request;
    request.network = network;
    request.loader = loader;
    request.optimizer = optimizer;
    request.trainingProgram = trainingProgram;
    request.runtime = runtimeConfig;
    request.epochs = options.epochs;

    executor->fit(request, effectiveObserver());
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
