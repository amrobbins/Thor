#include "DeepLearning/Api/Training/ExecutableTrainingPlan.h"

#include "DeepLearning/Api/Training/TrainingProgram.h"

#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

ExecutableTrainingPlan::ExecutableTrainingPlan(std::vector<StepExecutable> steps) : steps(std::move(steps)), initialized(true) {
    rebuildDerivedState();
    validate();
}

ExecutableTrainingPlan ExecutableTrainingPlan::compile(const TrainingProgram& program,
                                                         PlacedNetwork& placedNetwork,
                                                         bool resolveEmptyUpdateParametersAsAllTrainable) {
    return ExecutableTrainingPlan(program.compile(placedNetwork, resolveEmptyUpdateParametersAsAllTrainable));
}

const StepExecutable& ExecutableTrainingPlan::getStep(uint64_t index) const {
    if (index >= steps.size()) {
        throw std::runtime_error("ExecutableTrainingPlan step index out of range.");
    }
    return steps[index];
}

void ExecutableTrainingPlan::rebuildDerivedState() {
    std::set<std::string> requiredBatchInputNameSet;
    totalStepRepeatsPerIteration = 0;

    for (const StepExecutable& step : steps) {
        totalStepRepeatsPerIteration += step.getRepeatCount();
        for (const std::string& batchInputName : step.getRequiredBatchInputNames()) {
            requiredBatchInputNameSet.insert(batchInputName);
        }
    }

    requiredBatchInputNames.assign(requiredBatchInputNameSet.begin(), requiredBatchInputNameSet.end());
}

void ExecutableTrainingPlan::validate() const {
    if (!initialized) {
        return;
    }
    if (steps.empty()) {
        throw std::runtime_error("ExecutableTrainingPlan requires at least one StepExecutable.");
    }
    if (totalStepRepeatsPerIteration == 0) {
        throw std::runtime_error("ExecutableTrainingPlan requires at least one step repeat per iteration.");
    }

    std::set<std::string> stepNames;
    std::set<std::string> requiredBatchInputNameSet;
    uint64_t expectedRepeats = 0;
    for (const StepExecutable& step : steps) {
        if (!step.isInitialized()) {
            throw std::runtime_error("ExecutableTrainingPlan cannot contain an uninitialized StepExecutable.");
        }
        if (!stepNames.insert(step.getName()).second) {
            throw std::runtime_error("ExecutableTrainingPlan contains duplicate step name '" + step.getName() + "'.");
        }
        expectedRepeats += step.getRepeatCount();
        for (const std::string& batchInputName : step.getRequiredBatchInputNames()) {
            requiredBatchInputNameSet.insert(batchInputName);
        }
    }

    if (expectedRepeats != totalStepRepeatsPerIteration) {
        throw std::runtime_error("ExecutableTrainingPlan repeat-count summary is inconsistent.");
    }
    if (requiredBatchInputNames.size() != requiredBatchInputNameSet.size()) {
        throw std::runtime_error("ExecutableTrainingPlan required batch input names contain duplicates.");
    }
    for (const std::string& batchInputName : requiredBatchInputNames) {
        if (!requiredBatchInputNameSet.contains(batchInputName)) {
            throw std::runtime_error("ExecutableTrainingPlan required batch input name '" + batchInputName +
                                     "' is not present in any step.");
        }
    }
}

void ExecutableTrainingPlan::validateNativeQueuedExecutorCompatible(
    const std::vector<ParameterReference>& allTrainableParameters) const {
    validate();

    if (steps.size() != 1) {
        throw std::runtime_error(
            "NativeQueuedTrainingExecutor currently supports only a single-step ExecutableTrainingPlan. "
            "The queue-ahead runtime is native now, but multi-step selective execution still needs explicit per-step graph/update commands.");
    }

    const StepExecutable& step = steps.front();
    if (step.getRepeatCount() != 1) {
        throw std::runtime_error(
            "NativeQueuedTrainingExecutor currently supports only repeat_count=1. "
            "Repeated step execution needs a clear contract for whether each repeat consumes a fresh loader batch.");
    }
    if (step.getGradientClearPolicy() != TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP) {
        throw std::runtime_error(
            "NativeQueuedTrainingExecutor currently supports only clear_before_step gradient policy. "
            "Gradient accumulation needs explicit accumulated-gradient command scheduling.");
    }

    std::set<ParameterReference> expected(allTrainableParameters.begin(), allTrainableParameters.end());
    std::set<ParameterReference> actual(step.getUpdateParameterReferences().begin(), step.getUpdateParameterReferences().end());
    if (expected != actual) {
        throw std::runtime_error(
            "NativeQueuedTrainingExecutor currently requires the TrainingStep update parameter set to match all training-enabled "
            "network parameters. Selective parameter updates need per-step compiled graph/update commands.");
    }
}

json ExecutableTrainingPlan::architectureJson() const {
    validate();

    json j;
    j["version"] = getVersion();
    j["step_count"] = steps.size();
    j["total_step_repeats_per_iteration"] = totalStepRepeatsPerIteration;
    j["required_batch_input_names"] = requiredBatchInputNames;
    j["steps"] = json::array();
    for (const StepExecutable& step : steps) {
        j["steps"].push_back(step.architectureJson());
    }
    return j;
}

std::string ExecutableTrainingPlan::architectureJsonString() const { return architectureJson().dump(); }

}  // namespace Thor
