#include "Utilities/TensorOperations/Einsum/EinsumBackwardPlanner.h"

#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ThorImplementation {
namespace {

struct UniqueOperandLabels {
    std::vector<int32_t> labels;
    std::vector<uint64_t> dimensions;
    std::unordered_map<int32_t, uint32_t> logical_axis_by_label;
    std::unordered_map<int32_t, std::vector<uint32_t>> physical_axes_by_label;
};

[[nodiscard]] UniqueOperandLabels uniqueOperandLabels(const ResolvedEinsumOperand& operand,
                                                      const std::vector<uint64_t>& dimensions) {
    if (operand.axis_labels.size() != dimensions.size()) {
        throw std::logic_error("Internal einsum backward planner error: operand labels do not match feature rank.");
    }

    UniqueOperandLabels result;
    result.labels.reserve(operand.axis_labels.size());
    result.dimensions.reserve(dimensions.size());

    for (size_t physical_axis = 0; physical_axis < operand.axis_labels.size(); ++physical_axis) {
        const int32_t label = operand.axis_labels[physical_axis];
        result.physical_axes_by_label[label].push_back(static_cast<uint32_t>(physical_axis));

        if (result.logical_axis_by_label.contains(label)) {
            continue;
        }
        if (result.labels.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("Einsum operand rank exceeds the supported uint32_t axis range.");
        }
        const uint32_t logical_axis = static_cast<uint32_t>(result.labels.size());
        result.logical_axis_by_label.emplace(label, logical_axis);
        result.labels.push_back(label);
        result.dimensions.push_back(dimensions[physical_axis]);
    }

    return result;
}

[[nodiscard]] uint64_t mergeBackwardInputDimension(uint64_t current, uint64_t incoming) {
    if (current == 0 || current == incoming) {
        return incoming;
    }
    if (current == 1) {
        return incoming;
    }
    if (incoming == 1) {
        return current;
    }
    throw std::logic_error(
        "Internal einsum backward planner error: backward inputs are not broadcast-compatible for a resolved label.");
}

[[nodiscard]] std::vector<std::vector<int32_t>> physicalBackwardInputs(
    const std::vector<int32_t>& upstream_gradient_feature_labels,
    const std::vector<std::vector<int32_t>>& other_operand_feature_axis_labels) {
    std::vector<std::vector<int32_t>> inputs;
    inputs.reserve(1 + other_operand_feature_axis_labels.size());
    inputs.push_back(EinsumLayerBatchContract::prependImplicitBatchLabel(upstream_gradient_feature_labels));
    for (const std::vector<int32_t>& labels : other_operand_feature_axis_labels) {
        inputs.push_back(EinsumLayerBatchContract::prependImplicitBatchLabel(labels));
    }
    return inputs;
}

}  // namespace

std::vector<int32_t> EinsumLayerBatchContract::prependImplicitBatchLabel(const std::vector<int32_t>& feature_labels) {
    std::vector<int32_t> physical_labels;
    physical_labels.reserve(feature_labels.size() + 1);
    physical_labels.push_back(kImplicitBatchLabel);
    physical_labels.insert(physical_labels.end(), feature_labels.begin(), feature_labels.end());
    return physical_labels;
}

std::vector<uint64_t> EinsumLayerBatchContract::prependBatchDimension(
    uint64_t batch_size,
    const std::vector<uint64_t>& feature_dimensions) {
    if (batch_size == 0) {
        throw std::invalid_argument("Einsum layer runtime batch dimension must be non-zero.");
    }

    std::vector<uint64_t> physical_dimensions;
    physical_dimensions.reserve(feature_dimensions.size() + 1);
    physical_dimensions.push_back(batch_size);
    physical_dimensions.insert(physical_dimensions.end(), feature_dimensions.begin(), feature_dimensions.end());
    return physical_dimensions;
}

EinsumLayerBackwardPlan EinsumBackwardPlanner::parseAndPlan(
    const std::string& equation,
    const std::vector<std::vector<uint64_t>>& input_feature_dimensions) {
    EinsumLayerBackwardPlan result;
    result.equation = equation;
    result.feature_equation = EinsumParser::parseAndResolve(equation, input_feature_dimensions);

    const size_t operand_count = result.feature_equation.inputs.size();
    if (operand_count != input_feature_dimensions.size()) {
        throw std::logic_error("Internal einsum backward planner error: resolved operand count mismatch.");
    }
    if (operand_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Einsum operand count exceeds the supported uint32_t range.");
    }

    result.physical_forward_input_axis_labels.reserve(operand_count);
    for (const ResolvedEinsumOperand& operand : result.feature_equation.inputs) {
        result.physical_forward_input_axis_labels.push_back(
            EinsumLayerBatchContract::prependImplicitBatchLabel(operand.axis_labels));
    }
    result.physical_forward_output_axis_labels =
        EinsumLayerBatchContract::prependImplicitBatchLabel(result.feature_equation.output_labels);

    result.operand_gradients.reserve(operand_count);
    for (size_t target_index = 0; target_index < operand_count; ++target_index) {
        EinsumOperandBackwardPlan gradient;
        gradient.operand_index = static_cast<uint32_t>(target_index);
        gradient.final_feature_dimensions = input_feature_dimensions[target_index];

        const ResolvedEinsumOperand& target_operand = result.feature_equation.inputs[target_index];
        const UniqueOperandLabels target_unique =
            uniqueOperandLabels(target_operand, input_feature_dimensions[target_index]);
        gradient.target_unique_feature_labels = target_unique.labels;
        gradient.target_unique_feature_dimensions = target_unique.dimensions;

        EinsumBackwardContractionPlan& contraction = gradient.contraction;
        contraction.upstream_gradient_feature_labels = result.feature_equation.output_labels;
        contraction.other_operand_indices.reserve(operand_count > 0 ? operand_count - 1 : 0);
        contraction.other_operand_feature_axis_labels.reserve(operand_count > 0 ? operand_count - 1 : 0);

        std::unordered_map<int32_t, uint64_t> backward_input_dimensions;
        for (size_t output_axis = 0; output_axis < result.feature_equation.output_labels.size(); ++output_axis) {
            const int32_t label = result.feature_equation.output_labels[output_axis];
            backward_input_dimensions[label] = mergeBackwardInputDimension(
                backward_input_dimensions[label], result.feature_equation.output_dimensions[output_axis]);
        }

        for (size_t other_index = 0; other_index < operand_count; ++other_index) {
            if (other_index == target_index) {
                continue;
            }
            contraction.other_operand_indices.push_back(static_cast<uint32_t>(other_index));
            contraction.other_operand_feature_axis_labels.push_back(result.feature_equation.inputs[other_index].axis_labels);
            for (size_t axis = 0; axis < result.feature_equation.inputs[other_index].axis_labels.size(); ++axis) {
                const int32_t label = result.feature_equation.inputs[other_index].axis_labels[axis];
                backward_input_dimensions[label] = mergeBackwardInputDimension(
                    backward_input_dimensions[label], input_feature_dimensions[other_index][axis]);
            }
        }

        // Raw contraction output follows the target's unique logical order, but
        // can only name labels that occur in a backward input. Missing target-
        // only reduction labels are inserted/broadcast afterward. A label that
        // is present only at singleton extent in the backward inputs remains a
        // singleton raw axis and is expanded separately when the target owns N.
        contraction.output_feature_labels.reserve(target_unique.labels.size());
        contraction.output_feature_dimensions.reserve(target_unique.labels.size());
        for (size_t logical_axis = 0; logical_axis < target_unique.labels.size(); ++logical_axis) {
            const int32_t label = target_unique.labels[logical_axis];
            const auto backward_dimension_it = backward_input_dimensions.find(label);
            if (backward_dimension_it == backward_input_dimensions.end()) {
                gradient.missing_axis_expansions.push_back(
                    EinsumBackwardMissingAxisExpansionPlan{label,
                                                           static_cast<uint32_t>(logical_axis),
                                                           target_unique.dimensions[logical_axis]});
                continue;
            }

            const uint64_t backward_dimension = backward_dimension_it->second;
            const uint32_t contraction_axis = static_cast<uint32_t>(contraction.output_feature_labels.size());
            contraction.output_feature_labels.push_back(label);
            contraction.output_feature_dimensions.push_back(backward_dimension);

            const uint64_t target_dimension = target_unique.dimensions[logical_axis];
            if (target_dimension == 1 && backward_dimension > 1) {
                gradient.broadcast_reductions.push_back(
                    EinsumBackwardBroadcastReductionPlan{label, contraction_axis, backward_dimension, target_dimension});
            } else if (target_dimension > 1 && backward_dimension == 1) {
                gradient.existing_axis_expansions.push_back(EinsumBackwardExistingAxisExpansionPlan{
                    label,
                    contraction_axis,
                    static_cast<uint32_t>(logical_axis),
                    backward_dimension,
                    target_dimension});
            } else if (target_dimension != backward_dimension) {
                throw std::logic_error(
                    "Internal einsum backward planner error: target and backward-input dimensions are not broadcast-compatible.");
            }
        }

        // Repeated labels collapse to one logical gradient axis, then scatter
        // back to the original dense feature tensor.
        for (size_t logical_axis = 0; logical_axis < target_unique.labels.size(); ++logical_axis) {
            const int32_t label = target_unique.labels[logical_axis];
            const auto physical_axes_it = target_unique.physical_axes_by_label.find(label);
            if (physical_axes_it == target_unique.physical_axes_by_label.end()) {
                throw std::logic_error("Internal einsum backward planner error: unique label has no source axes.");
            }
            if (physical_axes_it->second.size() <= 1) {
                continue;
            }
            gradient.diagonal_scatters.push_back(EinsumBackwardDiagonalScatterPlan{
                label,
                static_cast<uint32_t>(logical_axis),
                physical_axes_it->second,
                target_unique.dimensions[logical_axis]});
        }

        contraction.physical_input_axis_labels =
            physicalBackwardInputs(contraction.upstream_gradient_feature_labels,
                                   contraction.other_operand_feature_axis_labels);
        contraction.physical_output_axis_labels =
            EinsumLayerBatchContract::prependImplicitBatchLabel(contraction.output_feature_labels);

        result.operand_gradients.push_back(std::move(gradient));
    }

    return result;
}

}  // namespace ThorImplementation
