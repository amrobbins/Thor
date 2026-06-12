#pragma once

#include "DeepLearning/Api/Training/Executors/TrainingExecutor.h"

namespace Thor {

// Debug backend for comparing the normal queue-ahead executor against a conservative
// one-batch-at-a-time path. This intentionally sacrifices throughput for easier
// diagnosis when validating new Trainer/runtime behavior.
class DebugSynchronousTrainingExecutor : public TrainingExecutor {
   public:
    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override;
};

}  // namespace Thor
