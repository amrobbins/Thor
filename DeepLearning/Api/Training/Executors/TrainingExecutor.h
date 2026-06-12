#pragma once

#include "DeepLearning/Api/Training/Executors/TrainingRunRequest.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"

namespace Thor {

class TrainingExecutor {
   public:
    virtual ~TrainingExecutor() = default;
    virtual void fit(const TrainingRunRequest& request, TrainingObserver& observer) = 0;
};

}  // namespace Thor
