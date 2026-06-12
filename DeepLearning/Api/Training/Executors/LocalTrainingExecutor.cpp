#include "DeepLearning/Api/Training/Executors/LocalTrainingExecutor.h"

#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"

namespace Thor {

void LocalTrainingExecutor::fit(const TrainingRunRequest& request, TrainingObserver& observer) {
    NativeQueuedTrainingOptions options;
    options.maxInFlightBatches = request.runtime.maxInFlightBatches;
    options.synchronizeAfterEveryBatch = false;
    runNativeQueuedTraining(request, observer, options);
}

}  // namespace Thor
