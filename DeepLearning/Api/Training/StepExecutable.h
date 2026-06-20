#pragma once

#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingStep.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace Thor {

class PlacedNetwork;

class StepExecutable {
   public:
    StepExecutable() = default;
    StepExecutable(const TrainingStep& step, PlacedNetwork& placedNetwork, bool resolveEmptyUpdateParametersAsAllTrainable = true);

    [[nodiscard]] const std::string& getName() const { return name; }
    [[nodiscard]] const std::vector<Tensor>& getLossRoots() const { return lossRoots; }
    [[nodiscard]] const std::vector<Tensor>& getResolvedLossRoots() const { return resolvedLossRoots; }
    [[nodiscard]] const std::vector<std::string>& getActivePhaseNames() const { return activePhaseNames; }
    [[nodiscard]] std::shared_ptr<Optimizer> getOptimizer() const { return optimizer; }
    [[nodiscard]] const std::vector<ParameterReference>& getUpdateParameterReferences() const { return updateParameterReferences; }
    [[nodiscard]] const std::vector<BoundParameter>& getResolvedUpdateParameters() const { return resolvedUpdateParameters; }
    [[nodiscard]] const std::vector<TrainingInputBinding>& getInputBindings() const { return inputBindings; }
    [[nodiscard]] const std::vector<TrainingInputBinding>& getResolvedInputBindings() const { return resolvedInputBindings; }
    [[nodiscard]] const std::vector<std::string>& getRequiredBatchInputNames() const { return requiredBatchInputNames; }
    [[nodiscard]] uint32_t getRepeatCount() const { return repeatCount; }
    [[nodiscard]] TrainingStep::GradientClearPolicy getGradientClearPolicy() const { return gradientClearPolicy; }
    [[nodiscard]] bool isInitialized() const { return initialized; }

    [[nodiscard]] nlohmann::json architectureJson() const;
    [[nodiscard]] std::string architectureJsonString() const;
    [[nodiscard]] std::string getVersion() const { return "1.0.0"; }

   private:
    void validate() const;

    std::string name{};
    std::vector<Tensor> lossRoots{};
    std::vector<Tensor> resolvedLossRoots{};
    std::vector<std::string> activePhaseNames{};
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::vector<ParameterReference> updateParameterReferences{};
    std::vector<BoundParameter> resolvedUpdateParameters{};
    std::vector<TrainingInputBinding> inputBindings{};
    std::vector<TrainingInputBinding> resolvedInputBindings{};
    std::vector<std::string> requiredBatchInputNames{};
    uint32_t repeatCount = 1;
    TrainingStep::GradientClearPolicy gradientClearPolicy = TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP;
    bool initialized = false;
};

}  // namespace Thor
