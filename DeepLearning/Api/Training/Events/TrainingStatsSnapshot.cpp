#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"

namespace Thor {

const char* trainingPhaseName(TrainingEventPhase phase) {
    switch (phase) {
        case TrainingEventPhase::TRAIN:
            return "train";
        case TrainingEventPhase::VALIDATE:
            return "validate";
        case TrainingEventPhase::TEST:
            return "test";
        case TrainingEventPhase::UNKNOWN:
        default:
            return "unknown";
    }
}

}  // namespace Thor
