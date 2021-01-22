#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"

#include <utility>
#include <vector>

namespace Thor {

class Visualizer;

class ConsoleVisualizer : public Visualizer {
   public:
    class Builder;
    ConsoleVisualizer() : initialized(false) {}

    virtual ~ConsoleVisualizer() {}

    void updateState(ExecutionState executionState, HyperparameterController hyperparameterController);

   private:
    bool initialized;

    Optional<ExecutionState> previousExecutionState;

    void printHeader(ExecutionState executionState, HyperparameterController hyperparameterController);
    void printLine(ExecutionState executionState, HyperparameterController hyperparameterController);
};

class ConsoleVisualizer::Builder {
   public:
    virtual shared_ptr<Visualizer> build();
};

}  // namespace Thor
