#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"

using namespace Thor;

using std::make_shared;
using std::pair;
using std::vector;

shared_ptr<Visualizer> ConsoleVisualizer::Builder::build() {
    shared_ptr<ConsoleVisualizer> consoleVisualizer = make_shared<ConsoleVisualizer>();
    consoleVisualizer->initialized = true;
    return consoleVisualizer;
}

void ConsoleVisualizer::updateState(ExecutionState executionState, HyperparameterController hyperparameterController) {
    if (previousExecutionState.isEmpty()) {
        printHeader(executionState, hyperparameterController);
        printLine(executionState, hyperparameterController);
    } else {
        if (previousExecutionState.get().executionMode != executionState.executionMode) {
            // start a new line
        } else {
            // continue with line
        }
    }

    previousExecutionState = executionState;
}

void ConsoleVisualizer::printHeader(ExecutionState executionState, HyperparameterController hyperparameterController) {
    printf("Thor console visualizer\n");
    printf("Training network FIXME_NEED_NETWORK_NAME\n");
    printf("\nData set stats\n");
    printf("Example Classes:\n");
    printf("Training Examples:\n");
    printf("Validation Examples:\n");
    printf("Test Examples:\n");
    printf("\nTrain session stats\n");
    printf("Epochs FIXME_N\n");
    printf("Batch size FIXME_N\n");
    printf("\n");
    vector<pair<string, string>> hyperparameterDisplayInfo = hyperparameterController.getHeaderDisplayInfo();
    for (uint32_t i = 0; i < hyperparameterDisplayInfo.size(); ++i) {
        printf("%s %s\n", hyperparameterDisplayInfo[i].first.c_str(), hyperparameterDisplayInfo[i].second.c_str());
    }
}

void ConsoleVisualizer::printLine(ExecutionState executionState, HyperparameterController hyperparameterController) {
    vector<pair<string, string>> hyperparameterDisplayInfo = hyperparameterController.getCurrentEpochInfo(executionState);
    for (uint32_t i = 0; i < hyperparameterDisplayInfo.size(); ++i) {
        printf("%s %s\n", hyperparameterDisplayInfo[i].first.c_str(), hyperparameterDisplayInfo[i].second.c_str());
    }
    string epochType;
    if (executionState.executionMode == ExampleType::TRAIN)
        epochType = "Trainining";
    if (executionState.executionMode == ExampleType::TRAIN)
        epochType = "Validating";
    else
        epochType = "Testing";
    printf("%s Epoch %ld, batch %ld of %ld\n",
           epochType.c_str(),
           executionState.epochNum,
           executionState.batchNum + 1,
           executionState.batchesPerEpoch);
    double percentComplete = (executionState.batchNum + 1) / executionState.batchesPerEpoch;
    uint32_t numStars = percentComplete * 100;
    for (uint32_t i = 0; i < numStars; ++i) {
        printf("%s%s\n", std::string(numStars, '*').c_str(), std::string(100 - numStars, 'o').c_str());
    }
}
