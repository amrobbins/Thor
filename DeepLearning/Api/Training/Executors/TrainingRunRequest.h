#pragma once

#include "DeepLearning/Api/Training/TrainingProgram.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

class Loader;

namespace Thor {

class Network;
class Optimizer;

struct TrainingRuntimeConfig {
    uint64_t maxInFlightBatches = 32;
    bool statsEnabled = true;
    double statsIntervalSeconds = 10.0;
    bool statsStderrAlso = false;
    std::set<std::string> scalarTensorsToReport = {"loss", "accuracy"};
};

struct TrainingRunRequest {
    Network* network = nullptr;
    std::shared_ptr<Loader> loader = nullptr;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::optional<TrainingProgram> trainingProgram{};
    TrainingRuntimeConfig runtime{};
    uint32_t epochs = 1;
};

}  // namespace Thor
