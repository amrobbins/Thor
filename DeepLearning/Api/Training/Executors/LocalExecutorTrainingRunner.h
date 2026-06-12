#pragma once

#include "DeepLearning/Api/Training/Executors/TrainingRunRequest.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"

#include <cstdint>

namespace Thor {

struct LocalExecutorTrainingOptions {
    uint64_t maxInFlightBatches = 32;
    bool synchronizeAfterEveryBatch = false;
};

void runLocalExecutorBackedTraining(const TrainingRunRequest& request,
                                    TrainingObserver& observer,
                                    const LocalExecutorTrainingOptions& options);

}  // namespace Thor
