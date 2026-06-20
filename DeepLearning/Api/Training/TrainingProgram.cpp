#include "DeepLearning/Api/Training/TrainingProgram.h"

#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

TrainingProgram::TrainingProgram(std::vector<std::shared_ptr<TrainingStep>> steps) {
    if (steps.empty()) {
        throw std::runtime_error("TrainingProgram requires at least one TrainingStep.");
    }
    for (std::shared_ptr<TrainingStep>& step : steps) {
        addStep(std::move(step));
    }
}

void TrainingProgram::validate() const {
    if (!initialized || steps.empty()) {
        throw std::runtime_error("TrainingProgram requires at least one TrainingStep.");
    }
    for (const std::shared_ptr<TrainingStep>& step : steps) {
        if (step == nullptr || !step->isInitialized()) {
            throw std::runtime_error("TrainingProgram cannot contain an uninitialized TrainingStep.");
        }
    }
}

void TrainingProgram::validateStepNameIsUnique(const TrainingStep& step) const {
    if (!step.isInitialized()) {
        throw std::runtime_error("TrainingProgram cannot contain an uninitialized TrainingStep.");
    }
    for (const std::shared_ptr<TrainingStep>& existingStep : steps) {
        if (existingStep != nullptr && existingStep->getName() == step.getName()) {
            throw std::runtime_error("TrainingProgram already contains a step named '" + step.getName() + "'.");
        }
    }
}

void TrainingProgram::addStep(std::shared_ptr<TrainingStep> step) {
    if (step == nullptr) {
        throw std::runtime_error("TrainingProgram cannot contain a null TrainingStep reference.");
    }
    validateStepNameIsUnique(*step);
    steps.push_back(std::move(step));
    initialized = true;
}

TrainingStep& TrainingProgram::getStep(uint64_t index) {
    if (index >= steps.size()) {
        throw std::runtime_error("TrainingProgram step index out of range.");
    }
    if (steps[index] == nullptr) {
        throw std::runtime_error("TrainingProgram contains a null TrainingStep reference.");
    }
    return *steps[index];
}

const TrainingStep& TrainingProgram::getStep(uint64_t index) const {
    if (index >= steps.size()) {
        throw std::runtime_error("TrainingProgram step index out of range.");
    }
    if (steps[index] == nullptr) {
        throw std::runtime_error("TrainingProgram contains a null TrainingStep reference.");
    }
    return *steps[index];
}

std::shared_ptr<TrainingStep> TrainingProgram::getStepReference(uint64_t index) const {
    if (index >= steps.size()) {
        throw std::runtime_error("TrainingProgram step index out of range.");
    }
    if (steps[index] == nullptr) {
        throw std::runtime_error("TrainingProgram contains a null TrainingStep reference.");
    }
    return steps[index];
}

json TrainingProgram::architectureJson() const {
    validate();

    json j;
    j["version"] = getVersion();
    j["steps"] = json::array();
    for (const std::shared_ptr<TrainingStep>& step : steps) {
        j["steps"].push_back(step->architectureJson());
    }
    return j;
}

std::string TrainingProgram::architectureJsonString() const { return architectureJson().dump(); }

std::vector<StepExecutable> TrainingProgram::compile(PlacedNetwork& placedNetwork, bool resolveEmptyUpdateParametersAsAllTrainable) const {
    validate();

    std::vector<StepExecutable> executables;
    executables.reserve(steps.size());
    for (const std::shared_ptr<TrainingStep>& step : steps) {
        if (!step->isEnabled()) {
            continue;
        }
        step->validateEnabledPhaseDependencies();
        if (step->getActiveLossRoots().empty()) {
            throw std::runtime_error("TrainingProgram enabled TrainingStep '" + step->getName() +
                                     "' has no active loss roots from enabled TrainingPhases.");
        }
        executables.emplace_back(*step, placedNetwork, resolveEmptyUpdateParametersAsAllTrainable);
    }
    if (executables.empty()) {
        throw std::runtime_error("TrainingProgram has no enabled TrainingStep with active loss roots.");
    }
    return executables;
}


TrainingProgram TrainingProgram::deserialize(const json& j, std::shared_ptr<thor_file::TarReader> archiveReader, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0") {
        throw std::runtime_error("Unsupported TrainingProgram version: " + version);
    }

    std::vector<std::shared_ptr<TrainingStep>> steps;
    for (const json& stepJson : j.at("steps")) {
        steps.push_back(std::make_shared<TrainingStep>(TrainingStep::deserialize(stepJson, archiveReader, network)));
    }

    return TrainingProgram(std::move(steps));
}

}  // namespace Thor
