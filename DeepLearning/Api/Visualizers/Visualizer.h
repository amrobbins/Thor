#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/HyperparameterControllers/HyperparameterController.h"

#include <assert.h>
#include <memory>

namespace Thor {

class Visualizer {
   public:
    Visualizer() {}

    virtual void startUI() {}
    virtual void stopUI() {}

    virtual void connectStateUpdateQueue(std::shared_ptr<AsyncQueue<ExecutionState>> executionStateQueue) {
        this->executionStateQueue = executionStateQueue;
    }
    // virtual void updateState(ExecutionState executionState) = 0;

    virtual ~Visualizer() {}

   protected:
    std::shared_ptr<AsyncQueue<ExecutionState>> executionStateQueue;
};

}  // namespace Thor
