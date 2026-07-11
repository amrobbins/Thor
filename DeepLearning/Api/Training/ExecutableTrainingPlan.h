#pragma once

#include "DeepLearning/Api/Training/StepExecutable.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace Thor {

class PlacedNetwork;
class TrainingProgram;

class ExecutableTrainingPlan {
   public:
    ExecutableTrainingPlan() = default;
    explicit ExecutableTrainingPlan(std::vector<StepExecutable> steps);

    [[nodiscard]] static ExecutableTrainingPlan compile(const TrainingProgram& program, PlacedNetwork& placedNetwork, bool resolveEmptyUpdateParametersAsAllTrainable = true);

    [[nodiscard]] bool isInitialized() const { return initialized; }
    [[nodiscard]] uint64_t getNumSteps() const { return steps.size(); }
    [[nodiscard]] const StepExecutable& getStep(uint64_t index) const;
    [[nodiscard]] const std::vector<StepExecutable>& getSteps() const { return steps; }
    [[nodiscard]] uint64_t getTotalStepRepeatsPerIteration() const { return totalStepRepeatsPerIteration; }
    [[nodiscard]] const std::vector<std::string>& getRequiredBatchInputNames() const { return requiredBatchInputNames; }

    // Temporary compatibility gate for direct StampedNetwork execution.  The native queued executor owns queue-ahead
    // scheduling now, but the underlying stamped graph still executes the complete network/loss/update surface.
    void validateNativeQueuedExecutorCompatible(const std::vector<ParameterReference>& allTrainableParameters) const;

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] std::string architectureJsonString() const;
    [[nodiscard]] std::string getVersion() const { return "1.0.0"; }

   private:
    void validate() const;
    void rebuildDerivedState();

    std::vector<StepExecutable> steps{};
    std::vector<std::string> requiredBatchInputNames{};
    uint64_t totalStepRepeatsPerIteration = 0;
    bool initialized = false;
};

}  // namespace Thor
