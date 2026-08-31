#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"

#include <optional>
#include <utility>

namespace ThorImplementation {

namespace {

std::vector<bool> raggedRmsNormInputDimensionsIncludeBatch(uint32_t epilogueAuxInputCount) {
    std::vector<bool> dimensionsIncludeBatch{false, true};
    dimensionsIncludeBatch.insert(dimensionsIncludeBatch.end(), epilogueAuxInputCount, false);
    return dimensionsIncludeBatch;
}

}  // namespace

RaggedRMSNorm::RaggedRMSNorm(DynamicExpression expression,
                             std::vector<std::string> inputNames,
                             std::vector<std::string> outputNames,
                             TensorPlacement placement,
                             std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                             bool inferenceOnly,
                             int64_t stampedId,
                             uint32_t epilogueAuxInputCount)
    : CustomLayer(std::move(expression),
                  std::move(inputNames),
                  std::move(outputNames),
                  placement,
                  std::move(physicalParameters),
                  inferenceOnly,
                  stampedId,
                  {},
                  false,
                  false,
                  raggedRmsNormInputDimensionsIncludeBatch(epilogueAuxInputCount),
                  std::nullopt) {}

}  // namespace ThorImplementation
