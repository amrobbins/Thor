#include "DeepLearning/Api/Training/Executors/DebugSynchronousTrainingExecutor.h"

#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"

namespace Thor {

void DebugSynchronousTrainingExecutor::fit(const TrainingRunRequest& request, TrainingObserver& observer) {
    NativeQueuedTrainingOptions options;
    options.maxInFlightBatches = 1;
    options.synchronizeAfterEveryBatch = true;
    runNativeQueuedTraining(request, observer, options);
}

}  // namespace Thor
