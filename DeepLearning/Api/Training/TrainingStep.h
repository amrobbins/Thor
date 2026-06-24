#pragma once

#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"

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

class TrainingStep {
   public:
    enum class GradientClearPolicy { CLEAR_BEFORE_STEP, ACCUMULATE };

    TrainingStep() = default;
    TrainingStep(std::string name,
                 std::vector<Tensor> lossRoots,
                 std::shared_ptr<Optimizer> optimizer,
                 std::vector<ParameterReference> updateParameters,
                 uint32_t repeatCount = 1,
                 GradientClearPolicy gradientClearPolicy = GradientClearPolicy::CLEAR_BEFORE_STEP,
                 std::vector<TrainingInputBinding> inputBindings = {},
                 bool enabled = true);

    TrainingStep(std::string name,
                 std::vector<std::shared_ptr<TrainingPhase>> phases,
                 std::shared_ptr<Optimizer> optimizer,
                 std::vector<ParameterReference> updateParameters,
                 uint32_t repeatCount = 1,
                 GradientClearPolicy gradientClearPolicy = GradientClearPolicy::CLEAR_BEFORE_STEP,
                 std::vector<TrainingInputBinding> inputBindings = {},
                 bool enabled = true);

    [[nodiscard]] const std::string& getName() const { return name; }
    [[nodiscard]] const std::vector<Tensor>& getLossRoots() const { return lossRoots; }
    [[nodiscard]] std::vector<Tensor> getActiveLossRoots() const;
    [[nodiscard]] std::vector<std::string> getActivePhaseNames() const;
    [[nodiscard]] std::vector<PhaseGraphNetworkSpec> getActivePhaseNetworkSpecs() const;
    [[nodiscard]] const std::vector<std::shared_ptr<TrainingPhase>>& getPhases() const { return phases; }
    [[nodiscard]] std::shared_ptr<Optimizer> getOptimizer() const { return optimizer; }
    [[nodiscard]] const std::vector<ParameterReference>& getUpdateParameters() const { return updateParameters; }
    [[nodiscard]] const std::vector<TrainingInputBinding>& getInputBindings() const { return inputBindings; }
    [[nodiscard]] uint32_t getRepeatCount() const { return repeatCount; }
    [[nodiscard]] GradientClearPolicy getGradientClearPolicy() const { return gradientClearPolicy; }
    [[nodiscard]] bool isEnabled() const { return enabled; }
    [[nodiscard]] bool isInitialized() const { return initialized; }

    void enable();
    void disable();
    void setEnabled(bool enabled);

    void validateEnabledPhaseDependencies() const;

    [[nodiscard]] bool updatesParameter(const ParameterReference& parameter) const;
    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] std::string architectureJsonString() const;
    static TrainingStep deserialize(const nlohmann::json& j,
                                    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr,
                                    Network* network = nullptr);

    [[nodiscard]] std::string getVersion() const { return "1.1.0"; }

   private:
    void validate() const;
    static std::vector<std::shared_ptr<TrainingPhase>> legacyLossRootsToPhases(const std::string& stepName,
                                                                               const std::vector<Tensor>& lossRoots);
    static std::vector<Tensor> collectAllLossRoots(const std::vector<std::shared_ptr<TrainingPhase>>& phases);

    std::string name{};
    std::vector<Tensor> lossRoots{};
    std::vector<std::shared_ptr<TrainingPhase>> phases{};
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::vector<ParameterReference> updateParameters{};
    std::vector<TrainingInputBinding> inputBindings{};
    uint32_t repeatCount = 1;
    GradientClearPolicy gradientClearPolicy = GradientClearPolicy::CLEAR_BEFORE_STEP;
    bool enabled = true;
    bool initialized = false;
};

NLOHMANN_JSON_SERIALIZE_ENUM(TrainingStep::GradientClearPolicy,
                             {{TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP, "clear_before_step"},
                              {TrainingStep::GradientClearPolicy::ACCUMULATE, "accumulate"}})

}  // namespace Thor
