#pragma once

#include "DeepLearning/Implementation/Layers/Loss/LossWeight.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace detail {

inline void appendConditionalTransformRootInput(PhysicalExpression& root,
                                                const NamedInput& childInput,
                                                const std::string& what) {
    for (const NamedInput& existing : root.inputs) {
        if (existing.name != childInput.name) {
            continue;
        }
        if (existing.kind != childInput.kind) {
            throw std::runtime_error(what + " conditional input kind mismatch for input '" + childInput.name + "'.");
        }
        return;
    }
    root.inputs.push_back(NamedInput{childInput.name, static_cast<uint32_t>(root.inputs.size()), childInput.kind});
}

inline PhysicalOutputs assembleConditionalTransformedOutputs(const PhysicalOutputs& predicate,
                                                              PhysicalOutputs thenBranch,
                                                              PhysicalOutputs elseBranch,
                                                              const std::string& what) {
    if (!predicate.expr || !thenBranch.expr || !elseBranch.expr) {
        throw std::runtime_error(what + " conditional transform requires expression metadata for every child.");
    }
    if (predicate.outputs.size() != 1) {
        throw std::runtime_error(what + " conditional transform requires exactly one predicate output.");
    }
    if (thenBranch.outputs.size() != elseBranch.outputs.size()) {
        throw std::runtime_error(what + " conditional transform produced different branch output counts.");
    }

    PhysicalOutputs result;
    result.expr = std::make_shared<PhysicalExpression>();
    const PhysicalOutputs* children[] = {&predicate, &thenBranch, &elseBranch};
    for (const PhysicalOutputs* child : children) {
        for (const NamedInput& input : child->expr->inputs) {
            appendConditionalTransformRootInput(*result.expr, input, what);
        }
    }

    result.outputs.reserve(thenBranch.outputs.size());
    for (size_t i = 0; i < thenBranch.outputs.size(); ++i) {
        if (thenBranch.outputs[i].name != elseBranch.outputs[i].name) {
            throw std::runtime_error(what + " conditional transform produced different branch output names.");
        }
        result.outputs.push_back(NamedOutput{thenBranch.outputs[i].name, static_cast<uint32_t>(i)});
    }

    result.conditional = std::make_shared<PhysicalConditionalOutputs>();
    result.conditional->predicate = predicate;
    result.conditional->then_branch = std::move(thenBranch);
    result.conditional->else_branch = std::move(elseBranch);
    return result;
}

inline PhysicalOutputs transformDynamicExpressionOutputsRecursively(
    const PhysicalOutputs& rawOutputs,
    const std::function<Expression(const std::string&, const Expression&)>& transform,
    const std::string& what) {
    if (rawOutputs.isConditional()) {
        if (!rawOutputs.conditional) {
            throw std::runtime_error(what + " conditional outputs are missing their conditional payload.");
        }
        const PhysicalConditionalOutputs& conditional = *rawOutputs.conditional;
        PhysicalOutputs thenOutputs = transformDynamicExpressionOutputsRecursively(conditional.then_branch, transform, what);
        PhysicalOutputs elseOutputs = transformDynamicExpressionOutputsRecursively(conditional.else_branch, transform, what);
        return assembleConditionalTransformedOutputs(conditional.predicate, std::move(thenOutputs), std::move(elseOutputs), what);
    }

    if (!rawOutputs.expr) {
        throw std::runtime_error(what + " cannot transform an empty dynamic expression.");
    }

    std::vector<std::pair<std::string, Expression>> transformedOutputs;
    transformedOutputs.reserve(rawOutputs.outputs.size());
    for (const NamedOutput& output : rawOutputs.outputs) {
        if (output.node_idx >= rawOutputs.expr->nodes.size()) {
            throw std::runtime_error(what + " output node index is out of range for '" + output.name + "'.");
        }
        Expression raw = Expression::fromPhysicalNode(rawOutputs.expr, output.node_idx);
        transformedOutputs.emplace_back(output.name, transform(output.name, raw));
    }
    return Expression::outputs(transformedOutputs).physicalOutputs();
}

}  // namespace detail

inline DynamicExpression applyLossWeightToDynamicExpression(
    const DynamicExpression& expression,
    std::unordered_map<std::string, DataType> outputDTypes,
    std::optional<float> lossWeight,
    std::string what) {
    lossWeight = normalizeLossWeight(lossWeight);
    if (!lossWeight.has_value()) {
        return expression;
    }

    return DynamicExpression(
        expression.getExpectedInputNames(),
        expression.getExpectedOutputNames(),
        [expression, outputDTypes = std::move(outputDTypes), lossWeight = lossWeight.value(), what = std::move(what)](
            const DynamicExpression::TensorMap& inputs,
            const DynamicExpression::TensorMap& outputs,
            Stream& stream) {
            DynamicExpressionBuild build = expression.build(inputs, {}, stream);
            const PhysicalOutputs& rawOutputs = build.equation->physicalOutputs();
            PhysicalOutputs weightedPhysicalOutputs = detail::transformDynamicExpressionOutputsRecursively(
                rawOutputs,
                [&outputDTypes, lossWeight, &what](const std::string& outputName, const Expression& raw) {
                    auto dtypeIt = outputDTypes.find(outputName);
                    if (dtypeIt == outputDTypes.end()) {
                        throw std::runtime_error(what + " is missing output dtype for '" + outputName + "'.");
                    }
                    return (raw * Expression::constantScalar(lossWeight)).withOutputDType(dtypeIt->second);
                },
                what);

            return DynamicExpressionBuild{
                .equation = std::make_shared<FusedEquation>(FusedEquation::compile(weightedPhysicalOutputs, stream.getGpuNum())),
                .stamp_inputs = std::move(build.stamp_inputs),
                .tensor_scalar_inputs = std::move(build.tensor_scalar_inputs),
                .preallocated_outputs = outputs,
                .requested_output_shapes = std::move(build.requested_output_shapes),
                .pre_forward_hook = std::move(build.pre_forward_hook),
                .pre_forward_only_inputs = std::move(build.pre_forward_only_inputs),
            };
        });
}

}  // namespace ThorImplementation
