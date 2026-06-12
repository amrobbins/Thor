#pragma once

#include "DeepLearning/Api/Training/Executors/TrainingRunRequest.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"

#include <cstdint>

namespace Thor {

struct NativeQueuedTrainingOptions {
    uint64_t maxInFlightBatches = 32;
    bool synchronizeAfterEveryBatch = false;
};

void runNativeQueuedTraining(const TrainingRunRequest& request,
                             TrainingObserver& observer,
                             const NativeQueuedTrainingOptions& options);

}  // namespace Thor
