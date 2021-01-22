#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"

#include <assert.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

using std::shared_ptr;

class HyperparameterController {
   public:
    HyperparameterController() {}

    virtual ~HyperparameterController() {}

    std::vector<std::pair<std::string, std::string>> getHeaderDisplayInfo();
    std::vector<std::pair<std::string, std::string>> getCurrentEpochInfo(ExecutionState executionState);
};

}  // namespace Thor
