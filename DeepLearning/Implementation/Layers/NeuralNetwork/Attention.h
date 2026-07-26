#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"

#include <optional>
#include <utility>

namespace ThorImplementation {

// Attention-specific semantic wrapper around CustomLayer's generic execution
// variants. CustomLayer knows only which execution variant is active; Attention
// owns the meaning of the stochastic and deterministic variants.
class Attention final : public CustomLayer, public TrainingDropoutControllable {
   public:
    Attention(DynamicExpression expr,
              std::vector<std::string> inputNames,
              std::vector<std::string> outputNames,
              const TensorPlacement& placement,
              const std::vector<std::shared_ptr<PhysicalParameter>>& parameters,
              bool inferenceOnly,
              int64_t stampedId,
              std::vector<DeclaredOutputDescriptor> declaredOutputDescriptors,
              std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId,
              bool trainingDropoutEnabled)
        : CustomLayer(std::move(expr),
                      std::move(inputNames),
                      std::move(outputNames),
                      placement,
                      parameters,
                      inferenceOnly,
                      stampedId,
                      std::move(declaredOutputDescriptors)),
          deterministicTrainingVariantId(deterministicTrainingVariantId) {
        setTrainingDropoutEnabled(trainingDropoutEnabled);
    }

    void setTrainingDropoutEnabled(bool enabled) override {
        if (deterministicTrainingVariantId.has_value()) {
            setActiveTrainingExecutionVariant(
                enabled ? kPrimaryDynamicExpressionVariant : deterministicTrainingVariantId.value());
        }
        trainingDropoutEnabled = enabled;
    }

    [[nodiscard]] bool isTrainingDropoutEnabled() const override {
        return trainingDropoutEnabled;
    }

   private:
    std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId;
    bool trainingDropoutEnabled = true;
};

}  // namespace ThorImplementation
