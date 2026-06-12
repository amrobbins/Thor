#pragma once

#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"

#include <optional>
#include <string>
#include <utility>

namespace Thor {

enum class TrainingEventType { RUN_STARTED, EPOCH_STARTED, STATS, EPOCH_FINISHED, RUN_FINISHED };

struct TrainingEvent {
    TrainingEventType type = TrainingEventType::STATS;
    TrainingStatsSnapshot stats{};
    std::string message{};

    static TrainingEvent runStarted(TrainingStatsSnapshot stats = {}, std::string message = {}) {
        return TrainingEvent{TrainingEventType::RUN_STARTED, std::move(stats), std::move(message)};
    }

    static TrainingEvent epochStarted(TrainingStatsSnapshot stats = {}, std::string message = {}) {
        return TrainingEvent{TrainingEventType::EPOCH_STARTED, std::move(stats), std::move(message)};
    }

    static TrainingEvent statsUpdated(TrainingStatsSnapshot stats, std::string message = {}) {
        return TrainingEvent{TrainingEventType::STATS, std::move(stats), std::move(message)};
    }

    static TrainingEvent epochFinished(TrainingStatsSnapshot stats = {}, std::string message = {}) {
        return TrainingEvent{TrainingEventType::EPOCH_FINISHED, std::move(stats), std::move(message)};
    }

    static TrainingEvent runFinished(TrainingStatsSnapshot stats = {}, std::string message = {}) {
        return TrainingEvent{TrainingEventType::RUN_FINISHED, std::move(stats), std::move(message)};
    }
};

}  // namespace Thor
