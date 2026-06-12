#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"

namespace Thor {

const char* trainingPhaseName(TrainingPhase phase) {
    switch (phase) {
        case TrainingPhase::TRAIN:
            return "train";
        case TrainingPhase::VALIDATE:
            return "validate";
        case TrainingPhase::TEST:
            return "test";
        case TrainingPhase::UNKNOWN:
        default:
            return "unknown";
    }
}

}  // namespace Thor
