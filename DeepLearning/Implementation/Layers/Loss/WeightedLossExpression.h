#pragma once

#include "DeepLearning/Implementation/Layers/Loss/LossWeight.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {

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
            if (rawOutputs.isConditional()) {
                throw std::runtime_error(what + " cannot apply loss_weight to conditional dynamic expression outputs.");
            }
            if (!rawOutputs.expr) {
                throw std::runtime_error(what + " cannot apply loss_weight to an empty dynamic expression.");
            }

            std::vector<std::pair<std::string, Expression>> weightedOutputs;
            weightedOutputs.reserve(rawOutputs.outputs.size());
            for (const NamedOutput& output : rawOutputs.outputs) {
                auto dtypeIt = outputDTypes.find(output.name);
                if (dtypeIt == outputDTypes.end()) {
                    throw std::runtime_error(what + " is missing output dtype for '" + output.name + "'.");
                }
                if (output.node_idx >= rawOutputs.expr->nodes.size()) {
                    throw std::runtime_error(what + " output node index is out of range for '" + output.name + "'.");
                }

                Expression raw = Expression::fromPhysicalNode(rawOutputs.expr, output.node_idx);
                Expression weighted = (raw * Expression::constantScalar(lossWeight)).withOutputDType(dtypeIt->second);
                weightedOutputs.emplace_back(output.name, std::move(weighted));
            }

            PhysicalOutputs weightedPhysicalOutputs = Expression::outputs(weightedOutputs).physicalOutputs();
            return DynamicExpressionBuild{
                .equation = std::make_shared<FusedEquation>(FusedEquation::compile(weightedPhysicalOutputs, stream.getGpuNum())),
                .stamp_inputs = std::move(build.stamp_inputs),
                .tensor_scalar_inputs = std::move(build.tensor_scalar_inputs),
                .preallocated_outputs = outputs,
                .requested_output_shapes = std::move(build.requested_output_shapes),
                .pre_forward_hook = std::move(build.pre_forward_hook),
            };
        });
}

}  // namespace ThorImplementation
