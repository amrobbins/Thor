#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/HyperparameterControllers/HyperparameterController.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Visualizer {
   public:
    Visualizer() {}

    virtual void startUI() {}
    virtual void stopUI() {}

    virtual void connectStateUpdateQueue(shared_ptr<AsyncQueue<ExecutionState>> executionStateQueue) {
        this->executionStateQueue = executionStateQueue;
    }
    // virtual void updateState(ExecutionState executionState) = 0;

    virtual ~Visualizer() {}

   protected:
    shared_ptr<AsyncQueue<ExecutionState>> executionStateQueue;
};

}  // namespace Thor
