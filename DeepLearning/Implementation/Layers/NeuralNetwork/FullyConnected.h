#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/TrainingDropoutControllable.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ThorImplementation {

// FullyConnected-specific semantic wrapper around CustomLayer execution
// variants. The primary variant may contain stochastic output dropout; the
// deterministic variant omits it and is used for validation/inference or when
// training dropout is explicitly disabled.
class FullyConnected final : public CustomLayer, public TrainingDropoutControllable {
   public:
    FullyConnected(DynamicExpression expression,
                   std::vector<std::string> inputNames,
                   TensorPlacement placement,
                   std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                   bool inferenceOnly,
                   int64_t stampedId,
                   bool preserveInputPrefixDimensions,
                   std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId,
                   bool trainingDropoutEnabled)
        : CustomLayer(std::move(expression),
                      std::move(inputNames),
                      std::vector<std::string>{"feature_output"},
                      placement,
                      physicalParameters,
                      inferenceOnly,
                      stampedId,
                      {},
                      /*usesBatchValidity=*/true,
                      /*requiresFullBatch=*/false),
          preserveInputPrefixDimensions(preserveInputPrefixDimensions),
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

    [[nodiscard]] bool isTrainingDropoutEnabled() const override { return trainingDropoutEnabled; }

   protected:
    std::vector<uint64_t> batchValidityMaskDimensionsForPrimaryInput(const Tensor& primaryInput) const override {
        std::vector<uint64_t> dimensions = primaryInput.getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty());
        if (preserveInputPrefixDimensions) {
            dimensions.back() = 1;
            return dimensions;
        }
        std::vector<uint64_t> maskDimensions(dimensions.size(), 1);
        maskDimensions.front() = dimensions.front();
        return maskDimensions;
    }

   private:
    bool preserveInputPrefixDimensions = false;
    std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId;
    bool trainingDropoutEnabled = true;
};

}  // namespace ThorImplementation
