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
                                           std::vector<std::string> epilogueAuxInputNames,
                                           std::optional<DynamicExpressionVariantId> deterministicTrainingVariantId,
                                           bool trainingDropoutEnabled)
    : CustomLayer(std::move(expression),
                  [&]() {
                      std::vector<std::string> names{"feature_input", ROW_PARTITION_INPUT_NAME};
                      if (useResidual) names.push_back("residual_input");
                      names.insert(names.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());
                      return names;
                  }(),
                  std::vector<std::string>{"feature_output"},
                  placement,
                  physicalParameters,
                  inferenceOnly,
                  stampedId,
                  {},
                  false,
                  false,
                  [&]() {
                      std::vector<bool> dimensionsIncludeBatch{false, true};
                      if (useResidual) dimensionsIncludeBatch.push_back(false);
                      dimensionsIncludeBatch.insert(dimensionsIncludeBatch.end(), epilogueAuxInputNames.size(), false);
                      return dimensionsIncludeBatch;
                  }(),
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
