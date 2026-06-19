#pragma once

#include "DeepLearning/Api/Training/Events/TrainingEvent.h"

#include <string>
#include <utility>

namespace Thor {

struct TrainingStatsEvent {
    std::string runName{};
    TrainingEventType type = TrainingEventType::STATS;
    TrainingStatsSnapshot stats{};
    std::string message{};

    static TrainingStatsEvent fromTrainingEvent(TrainingEvent event, std::string runName = {}) {
        TrainingStatsEvent statsEvent;
        statsEvent.runName = std::move(runName);
        statsEvent.type = event.type;
        statsEvent.stats = std::move(event.stats);
        statsEvent.message = std::move(event.message);
        return statsEvent;
    }
};

}  // namespace Thor
