#pragma once

#include "DeepLearning/Api/Training/Executors/TrainingExecutor.h"

namespace Thor {

class LocalTrainingExecutor : public TrainingExecutor {
   public:
    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override;
};

}  // namespace Thor
