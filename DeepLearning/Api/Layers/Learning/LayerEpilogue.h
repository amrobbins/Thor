#pragma once

#include <optional>
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Expression/Expression.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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
        return makeDefinition(expression, inputName, {}, outputName, layerType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeDefinition(const ThorImplementation::Expression& expression,
                                                                                 const std::string& primaryInputName,
                                                                                 const std::vector<std::string>& auxiliaryInputNames,
                                                                                 const std::string& outputName,
                                                                                 const std::string& layerType) {
        ThorImplementation::ExpressionDefinition definition =
            ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{outputName, expression}}));
        validateDefinition(definition, primaryInputName, auxiliaryInputNames, outputName, layerType);
        return definition;
    }

    static void validateExpression(const ThorImplementation::Expression& expression,
                                   const std::string& inputName,
                                   const std::string& outputName,
                                   const std::string& layerType) {
        (void)makeDefinition(expression, inputName, outputName, layerType);
    }

    static void validateExpression(const ThorImplementation::Expression& expression,
                                   const std::string& primaryInputName,
                                   const std::vector<std::string>& auxiliaryInputNames,
                                   const std::string& outputName,
                                   const std::string& layerType) {
        (void)makeDefinition(expression, primaryInputName, auxiliaryInputNames, outputName, layerType);
    }

    static void validateDefinition(const ThorImplementation::ExpressionDefinition& definition,
                                   const std::string& inputName,
                                   const std::string& outputName,
                                   const std::string& layerType) {
        validateDefinition(definition, inputName, {}, outputName, layerType);
    }

    static void validateDefinition(const ThorImplementation::ExpressionDefinition& definition,
                                   const std::string& primaryInputName,
                                   const std::vector<std::string>& auxiliaryInputNames,
                                   const std::string& outputName,
                                   const std::string& layerType) {
        definition.validate();
        if (definition.outputs.outputs.size() != 1 || definition.outputs.outputs.front().name != outputName) {
            throw std::invalid_argument(layerType + " epilogue expression must have exactly one output named " + outputName + ".");
        }
        if (definition.outputs.expr == nullptr) {
            throw std::invalid_argument(layerType + " epilogue expression must have a backing expression graph.");
        }

        std::set<std::string> expectedInputNames;
        expectedInputNames.insert(primaryInputName);
        for (const std::string& auxiliaryInputName : auxiliaryInputNames) {
            if (auxiliaryInputName.empty()) {
                throw std::invalid_argument(layerType + " epilogue auxiliary input name cannot be empty.");
            }
            if (!expectedInputNames.insert(auxiliaryInputName).second) {
                throw std::invalid_argument(layerType + " epilogue auxiliary input name is duplicated or collides with the primary input: " +
                                            auxiliaryInputName + ".");
            }
        }

        std::set<std::string> actualInputNames;
        bool foundPrimaryInput = false;
        for (const ThorImplementation::NamedInput& input : definition.outputs.expr->inputs) {
            if (input.kind != ThorImplementation::NamedInput::Kind::Tensor) {
                throw std::invalid_argument(layerType + " epilogue expression input " + input.name + " must be a tensor input.");
            }
            actualInputNames.insert(input.name);
            if (input.name == primaryInputName) {
                foundPrimaryInput = true;
            }
        }

        if (!foundPrimaryInput) {
            throw std::invalid_argument(layerType + " epilogue expression must include tensor input named " + primaryInputName + ".");
        }
        if (actualInputNames != expectedInputNames) {
            std::string expected;
            for (const std::string& name : expectedInputNames) {
                if (!expected.empty()) expected += ", ";
                expected += name;
            }
            std::string actual;
            for (const std::string& name : actualInputNames) {
                if (!actual.empty()) actual += ", ";
                actual += name;
            }
            throw std::invalid_argument(layerType + " epilogue expression input mismatch. Expected tensor inputs {" + expected +
                                        "}; got {" + actual + "}.");
        }
    }

    [[nodiscard]] static ThorImplementation::Expression expressionFromDefinition(const ThorImplementation::ExpressionDefinition& definition,
                                                                                 const std::string& inputName,
                                                                                 const std::string& outputName,
                                                                                 const std::string& layerType) {
        return expressionFromDefinition(definition, inputName, {}, outputName, layerType);
    }

    [[nodiscard]] static ThorImplementation::Expression expressionFromDefinition(const ThorImplementation::ExpressionDefinition& definition,
                                                                                 const std::string& primaryInputName,
                                                                                 const std::vector<std::string>& auxiliaryInputNames,
                                                                                 const std::string& outputName,
                                                                                 const std::string& layerType) {
        validateDefinition(definition, primaryInputName, auxiliaryInputNames, outputName, layerType);
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

    [[nodiscard]] static bool hasSameCanonicalForm(const ThorImplementation::Expression& lhs,
                                                   const ThorImplementation::Expression& rhs,
                                                   const std::string& primaryInputName,
                                                   const std::vector<std::string>& auxiliaryInputNames,
                                                   const std::string& outputName,
                                                   const std::string& layerType) {
        ThorImplementation::ExpressionDefinition lhsDefinition =
            makeDefinition(lhs, primaryInputName, auxiliaryInputNames, outputName, layerType);
        ThorImplementation::ExpressionDefinition rhsDefinition =
            makeDefinition(rhs, primaryInputName, auxiliaryInputNames, outputName, layerType);
        return ThorImplementation::canonicalize(lhsDefinition.outputs) == ThorImplementation::canonicalize(rhsDefinition.outputs);
    }
};

}  // namespace Thor
