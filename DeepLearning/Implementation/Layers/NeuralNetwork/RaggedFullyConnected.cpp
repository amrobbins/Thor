#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"

#include <optional>
#include <utility>

namespace ThorImplementation {

RaggedFullyConnected::RaggedFullyConnected(DynamicExpression expression,
                                           TensorPlacement placement,
                                           std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                                           bool inferenceOnly,
                                           int64_t stampedId,
                                           bool useResidual,
                                           std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId,
                                           bool trainingDropoutEnabled)
    : CustomLayer(std::move(expression),
                  useResidual
                      ? std::vector<std::string>{"feature_input", ROW_PARTITION_INPUT_NAME, "residual_input"}
                      : std::vector<std::string>{"feature_input", ROW_PARTITION_INPUT_NAME},
                  std::vector<std::string>{"feature_output"},
                  placement,
                  physicalParameters,
                  inferenceOnly,
                  stampedId,
                  {},
                  false,
                  false,
                  useResidual ? std::vector<bool>{false, true, false} : std::vector<bool>{false, true},
                  std::nullopt),
      deterministicTrainingVariantId(deterministicTrainingVariantId) {
    setTrainingDropoutEnabled(trainingDropoutEnabled);
}

void RaggedFullyConnected::setTrainingDropoutEnabled(bool enabled) {
    if (deterministicTrainingVariantId.has_value()) {
        setActiveTrainingExecutionVariant(
            enabled ? kPrimaryDynamicExpressionVariant : deterministicTrainingVariantId.value());
    }
    trainingDropoutEnabled = enabled;
}

}  // namespace ThorImplementation
