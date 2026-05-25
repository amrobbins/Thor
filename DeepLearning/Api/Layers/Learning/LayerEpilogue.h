#pragma once

#include <optional>
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Expression/Expression.h"

#include <stdexcept>
#include <string>

namespace Thor {

class LayerEpilogue {
   public:
    [[nodiscard]] static ThorImplementation::Expression input(
        const std::string& inputName,
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
        return ThorImplementation::Expression::input(inputName, computeDType, outputDType);
    }

   public:
    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeDefinition(const ThorImplementation::Expression& expression,
                                                                                 const std::string& inputName,
                                                                                 const std::string& outputName,
                                                                                 const std::string& layerType) {
        ThorImplementation::ExpressionDefinition definition =
            ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{outputName, expression}}));
        validateDefinition(definition, inputName, outputName, layerType);
        return definition;
    }

    static void validateExpression(const ThorImplementation::Expression& expression,
                                   const std::string& inputName,
                                   const std::string& outputName,
                                   const std::string& layerType) {
        (void)makeDefinition(expression, inputName, outputName, layerType);
    }

    static void validateDefinition(const ThorImplementation::ExpressionDefinition& definition,
                                   const std::string& inputName,
                                   const std::string& outputName,
                                   const std::string& layerType) {
        definition.validate();
        if (definition.outputs.outputs.size() != 1 || definition.outputs.outputs.front().name != outputName) {
            throw std::invalid_argument(layerType + " epilogue expression must have exactly one output named " + outputName + ".");
        }
        if (definition.outputs.expr == nullptr || definition.outputs.expr->inputs.size() != 1 ||
            definition.outputs.expr->inputs.front().name != inputName ||
            definition.outputs.expr->inputs.front().kind != ThorImplementation::NamedInput::Kind::Tensor) {
            throw std::invalid_argument(layerType + " epilogue expression must have exactly one tensor input named " + inputName + ".");
        }
    }

    [[nodiscard]] static ThorImplementation::Expression expressionFromDefinition(const ThorImplementation::ExpressionDefinition& definition,
                                                                                 const std::string& inputName,
                                                                                 const std::string& outputName,
                                                                                 const std::string& layerType) {
        validateDefinition(definition, inputName, outputName, layerType);
        const ThorImplementation::NamedOutput& epilogueOutput = definition.outputs.outputs.front();
        return ThorImplementation::Expression::fromPhysicalNode(definition.outputs.expr, epilogueOutput.node_idx);
    }

    [[nodiscard]] static ThorImplementation::Expression apply(const ThorImplementation::Expression& input,
                                                              const ThorImplementation::Expression& epilogue,
                                                              const std::string& inputName) {
        return epilogue.substituteInput(inputName, input);
    }

    [[nodiscard]] static bool hasSameCanonicalForm(const ThorImplementation::Expression& lhs,
                                                   const ThorImplementation::Expression& rhs,
                                                   const std::string& inputName,
                                                   const std::string& outputName,
                                                   const std::string& layerType) {
        ThorImplementation::ExpressionDefinition lhsDefinition = makeDefinition(lhs, inputName, outputName, layerType);
        ThorImplementation::ExpressionDefinition rhsDefinition = makeDefinition(rhs, inputName, outputName, layerType);
        return ThorImplementation::canonicalize(lhsDefinition.outputs) == ThorImplementation::canonicalize(rhsDefinition.outputs);
    }
};

}  // namespace Thor
