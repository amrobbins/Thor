#include "DeepLearning/Api/HyperparameterControllers/HyperparameterController.h"

using std::pair;
using std::string;
using std::vector;

// FIXME: For now just mocking it out.

using namespace Thor;

vector<pair<string, string>> HyperparameterController::getHeaderDisplayInfo() {
    vector<pair<string, string>> displayInfo;
    displayInfo.emplace_back("Learning rate", "Schedule: Epoch[0]:0.001, Epoch[5]:0.003, Epoch[10]:0.01");
    displayInfo.emplace_back("Momentum", "0.35");
    return displayInfo;
}

vector<pair<string, string>> HyperparameterController::getCurrentEpochInfo(ExecutionState executionState) {
    vector<pair<string, string>> displayInfo;
    string rate;
    if (executionState.epochNum < 5)
        rate = "0.01";
    else if (executionState.epochNum < 10)
        rate = "0.003";
    else
        rate = "0.001";

    displayInfo.emplace_back("Learning Rate", rate);
    return displayInfo;
}
