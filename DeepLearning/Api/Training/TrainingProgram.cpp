#include "DeepLearning/Api/Training/TrainingProgram.h"

#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

TrainingProgram::TrainingProgram(std::vector<TrainingStep> steps) {
    if (steps.empty()) {
        throw std::runtime_error("TrainingProgram requires at least one TrainingStep.");
    }
    for (const TrainingStep& step : steps) {
        addStep(step);
    }
}

void TrainingProgram::validate() const {
    if (!initialized || steps.empty()) {
        throw std::runtime_error("TrainingProgram requires at least one TrainingStep.");
    }
}

void TrainingProgram::validateStepNameIsUnique(const TrainingStep& step) const {
    if (!step.isInitialized()) {
        throw std::runtime_error("TrainingProgram cannot contain an uninitialized TrainingStep.");
    }
    for (const TrainingStep& existingStep : steps) {
        if (existingStep.getName() == step.getName()) {
            throw std::runtime_error("TrainingProgram already contains a step named '" + step.getName() + "'.");
        }
    }
}

void TrainingProgram::addStep(const TrainingStep& step) {
    validateStepNameIsUnique(step);
    steps.push_back(step);
    initialized = true;
}

const TrainingStep& TrainingProgram::getStep(uint64_t index) const {
    if (index >= steps.size()) {
        throw std::runtime_error("TrainingProgram step index out of range.");
    }
    return steps[index];
}

json TrainingProgram::architectureJson() const {
    validate();

    json j;
    j["version"] = getVersion();
    j["steps"] = json::array();
    for (const TrainingStep& step : steps) {
        j["steps"].push_back(step.architectureJson());
    }
    return j;
}

std::string TrainingProgram::architectureJsonString() const { return architectureJson().dump(); }

std::vector<StepExecutable> TrainingProgram::compile(PlacedNetwork& placedNetwork) const {
    validate();

    std::vector<StepExecutable> executables;
    executables.reserve(steps.size());
    for (const TrainingStep& step : steps) {
        executables.emplace_back(step, placedNetwork);
    }
    return executables;
}


TrainingProgram TrainingProgram::deserialize(const json& j, std::shared_ptr<thor_file::TarReader> archiveReader, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0") {
        throw std::runtime_error("Unsupported TrainingProgram version: " + version);
    }

    std::vector<TrainingStep> steps;
    for (const json& stepJson : j.at("steps")) {
        steps.push_back(TrainingStep::deserialize(stepJson, archiveReader, network));
    }

    return TrainingProgram(std::move(steps));
}

}  // namespace Thor
