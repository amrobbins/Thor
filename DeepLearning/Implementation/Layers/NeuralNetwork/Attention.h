#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ThorImplementation {

// Attention-specific semantic wrapper around CustomLayer's generic execution
// variants. CustomLayer knows only which execution variant is active; Attention
// owns the meaning of the stochastic and deterministic variants. Ragged
// execution extent is handled by the compiled physical consumers (active-aware
// expression stages, packed matmul, and cuDNN Attention), so this wrapper does not
// canonicalize packed storage outside the logical row partition.
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
              bool trainingDropoutEnabled,
              std::vector<bool> inputDimensionsIncludeBatch = {},
              std::optional<uint32_t> fixedBatchCapacity = std::nullopt)
        : CustomLayer(std::move(expr),
                      inputNames,
                      outputNames,
                      placement,
                      parameters,
                      inferenceOnly,
                      stampedId,
                      std::move(declaredOutputDescriptors),
                      false,
                      false,
                      std::move(inputDimensionsIncludeBatch),
                      fixedBatchCapacity),
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
