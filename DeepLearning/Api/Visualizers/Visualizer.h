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

    virtual void updateState(ExecutionState executionState, HyperparameterController hyperparameterController) = 0;

    virtual ~Visualizer() {}
};

}  // namespace Thor
