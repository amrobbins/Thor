#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"

#include <optional>
#include <utility>

namespace ThorImplementation {

RaggedFullyConnected::RaggedFullyConnected(DynamicExpression expression,
                                           TensorPlacement placement,
                                           std::vector<std::shared_ptr<PhysicalParameter>> physicalParameters,
                                           bool inferenceOnly,
                                           int64_t stampedId)
    : CustomLayer(std::move(expression),
                  std::vector<std::string>{"feature_input", ROW_PARTITION_INPUT_NAME},
                  std::vector<std::string>{"feature_output"},
                  placement,
                  physicalParameters,
                  inferenceOnly,
                  stampedId,
                  {},
                  false,
                  false,
                  std::vector<bool>{false, true},
                  std::nullopt) {}

}  // namespace ThorImplementation
