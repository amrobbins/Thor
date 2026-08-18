#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"

#include <optional>
#include <utility>

namespace ThorImplementation {

RaggedRMSNorm::RaggedRMSNorm(DynamicExpression expression,
                             std::vector<std::string> inputNames,
                             std::vector<std::string> outputNames,
                             TensorPlacement placement,
                             std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                             bool inferenceOnly,
                             int64_t stampedId)
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
                  std::vector<bool>{false, true},
                  std::nullopt) {}

}  // namespace ThorImplementation
