#pragma once

#include "DeepLearning/Api/Training/TrainingStep.h"
#include "DeepLearning/Api/Training/StepExecutable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace thor_file {
class TarReader;
}

namespace Thor {

class Network;
class PlacedNetwork;

class TrainingProgram {
   public:
    TrainingProgram() = default;
    explicit TrainingProgram(std::vector<std::shared_ptr<TrainingStep>> steps);

    void addStep(std::shared_ptr<TrainingStep> step);

    [[nodiscard]] uint64_t getNumSteps() const { return steps.size(); }
    [[nodiscard]] TrainingStep& getStep(uint64_t index);
    [[nodiscard]] const TrainingStep& getStep(uint64_t index) const;
    [[nodiscard]] std::shared_ptr<TrainingStep> getStepReference(uint64_t index) const;
    [[nodiscard]] const std::vector<std::shared_ptr<TrainingStep>>& getSteps() const { return steps; }
    [[nodiscard]] bool isInitialized() const { return initialized; }

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] std::string architectureJsonString() const;
    [[nodiscard]] std::vector<StepExecutable> compile(PlacedNetwork& placedNetwork) const;
    static TrainingProgram deserialize(const nlohmann::json& j,
                                       std::shared_ptr<thor_file::TarReader> archiveReader = nullptr,
                                       Network* network = nullptr);
    [[nodiscard]] std::string getVersion() const { return "1.0.0"; }

   private:
    void validate() const;
    void validateStepNameIsUnique(const TrainingStep& step) const;

    std::vector<std::shared_ptr<TrainingStep>> steps{};
    bool initialized = false;
};

}  // namespace Thor
