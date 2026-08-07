#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Expression/BatchedMatmulPlan.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ThorImplementation {
namespace {

bool isIdentityPermutation(const std::vector<uint32_t>& permutation) {
    for (size_t axis = 0; axis < permutation.size(); ++axis) {
        if (permutation[axis] != axis) {
            return false;
        }
    }
    return true;
}

uint64_t checkedProduct(const std::vector<int32_t>& labels,
                        const std::vector<uint64_t>& label_dimensions,
                        const char* description) {
    uint64_t product = 1;
    for (int32_t label : labels) {
        const size_t label_index = static_cast<size_t>(label);
        if (label_index >= label_dimensions.size() || label_dimensions[label_index] == 0) {
            throw std::logic_error(std::string("Internal einsum planner error: missing dimension for ") + description + ".");
        }
        const uint64_t dimension = label_dimensions[label_index];
        if (product > std::numeric_limits<uint64_t>::max() / dimension) {
            throw std::overflow_error(std::string("Einsum ") + description + " dimension product exceeds uint64_t range.");
        }
        product *= dimension;
    }
    return product;
}

std::vector<uint32_t> permutationToLabels(const std::vector<int32_t>& source_labels,
                                          const std::vector<int32_t>& target_labels,
                                          bool allow_missing_target_labels,
                                          std::vector<uint32_t>* inserted_axes) {
    std::unordered_map<int32_t, uint32_t> source_axis;
    source_axis.reserve(source_labels.size());
    for (size_t axis = 0; axis < source_labels.size(); ++axis) {
        source_axis.emplace(source_labels[axis], static_cast<uint32_t>(axis));
    }

    std::vector<uint32_t> permutation;
    permutation.reserve(source_labels.size());
    if (inserted_axes != nullptr) {
        inserted_axes->clear();
    }

    for (size_t target_axis = 0; target_axis < target_labels.size(); ++target_axis) {
        const auto source = source_axis.find(target_labels[target_axis]);
        if (source == source_axis.end()) {
            if (!allow_missing_target_labels) {
                throw std::logic_error("Internal einsum planner error: canonical label is absent from an operand.");
            }
            if (inserted_axes != nullptr) {
                inserted_axes->push_back(static_cast<uint32_t>(target_axis));
            }
            continue;
        }
        permutation.push_back(source->second);
    }

    if (permutation.size() != source_labels.size()) {
        throw std::logic_error("Internal einsum planner error: operand contains a label outside the canonical iteration order.");
    }
    return permutation;
}

std::unordered_map<int32_t, uint64_t> labelDimensionsForOperand(const std::vector<int32_t>& axis_labels,
                                                                const std::vector<uint64_t>& dimensions) {
    if (axis_labels.size() != dimensions.size()) {
        throw std::invalid_argument("Einsum planner input rank does not match the resolved operand rank.");
    }

    std::unordered_map<int32_t, uint64_t> result;
    result.reserve(axis_labels.size());
    for (size_t axis = 0; axis < axis_labels.size(); ++axis) {
        const int32_t label = axis_labels[axis];
        const uint64_t dimension = dimensions[axis];
        const auto [it, inserted] = result.emplace(label, dimension);
        if (!inserted && it->second != dimension) {
            throw std::invalid_argument("Einsum planner received dimensions inconsistent with a repeated-label diagonal.");
        }
    }
    return result;
}

void validateResolvedEquation(const ResolvedEinsumEquation& equation,
                              const std::vector<std::vector<uint64_t>>& input_dimensions) {
    if (equation.inputs.size() != input_dimensions.size()) {
        throw std::invalid_argument("Einsum planner operand count does not match the supplied input shapes.");
    }

    std::vector<uint64_t> reconstructed_dimensions(equation.label_dimensions.size(), 0);
    for (size_t operand = 0; operand < equation.inputs.size(); ++operand) {
        const auto& labels = equation.inputs[operand].axis_labels;
        const auto& dimensions = input_dimensions[operand];
        if (labels.size() != dimensions.size()) {
            throw std::invalid_argument("Einsum planner input rank does not match the resolved operand rank.");
        }

        const auto local_dimensions = labelDimensionsForOperand(labels, dimensions);
        for (const auto& [label, dimension] : local_dimensions) {
            const size_t index = static_cast<size_t>(label);
            if (index >= reconstructed_dimensions.size() || dimension == 0) {
                throw std::invalid_argument("Einsum planner received dimensions inconsistent with the resolved equation.");
            }
            uint64_t& reconstructed = reconstructed_dimensions[index];
            if (reconstructed == 0 || reconstructed == dimension) {
                reconstructed = dimension;
            } else if (reconstructed == 1) {
                reconstructed = dimension;
            } else if (dimension != 1) {
                throw std::invalid_argument("Einsum planner received mutually incompatible operand dimensions.");
            }
        }
    }

    for (size_t label = 0; label < equation.label_dimensions.size(); ++label) {
        if (equation.label_dimensions[label] != reconstructed_dimensions[label]) {
            throw std::invalid_argument("Einsum planner input shapes do not match the dimensions used to resolve the equation.");
        }
    }
}

EinsumOperandPlan buildOperandPlan(const ResolvedEinsumOperand& operand,
                                   const std::vector<uint64_t>& dimensions,
                                   const std::vector<int32_t>& iteration_labels,
                                   const std::vector<uint64_t>& iteration_dimensions) {
    EinsumOperandPlan plan;

    std::unordered_map<int32_t, uint32_t> first_axis_for_label;
    std::unordered_map<int32_t, size_t> diagonal_index_for_label;
    first_axis_for_label.reserve(operand.axis_labels.size());
    diagonal_index_for_label.reserve(operand.axis_labels.size());

    for (size_t axis = 0; axis < operand.axis_labels.size(); ++axis) {
        const int32_t label = operand.axis_labels[axis];
        const auto [first_it, inserted] = first_axis_for_label.emplace(label, static_cast<uint32_t>(axis));
        if (inserted) {
            plan.diagonalized_labels.push_back(label);
            plan.diagonalized_dimensions.push_back(dimensions[axis]);
            continue;
        }

        size_t diagonal_index;
        const auto diagonal_it = diagonal_index_for_label.find(label);
        if (diagonal_it == diagonal_index_for_label.end()) {
            EinsumDiagonalPlan diagonal;
            diagonal.label = label;
            diagonal.source_axes.push_back(first_it->second);
            plan.diagonals.push_back(std::move(diagonal));
            diagonal_index = plan.diagonals.size() - 1;
            diagonal_index_for_label.emplace(label, diagonal_index);
        } else {
            diagonal_index = diagonal_it->second;
        }
        plan.diagonals[diagonal_index].source_axes.push_back(static_cast<uint32_t>(axis));
    }

    plan.permutation = permutationToLabels(plan.diagonalized_labels, iteration_labels, true, &plan.inserted_axes);
    plan.aligned_dimensions.assign(iteration_labels.size(), 1);

    const auto local_dimensions = labelDimensionsForOperand(operand.axis_labels, dimensions);
    for (size_t axis = 0; axis < iteration_labels.size(); ++axis) {
        const int32_t label = iteration_labels[axis];
        const auto local = local_dimensions.find(label);
        if (local != local_dimensions.end()) {
            plan.aligned_dimensions[axis] = local->second;
        }
        if (plan.aligned_dimensions[axis] != iteration_dimensions[axis]) {
            if (plan.aligned_dimensions[axis] != 1) {
                throw std::logic_error("Internal einsum planner error: a non-singleton operand dimension requires broadcasting.");
            }
            plan.broadcast_axes.push_back(static_cast<uint32_t>(axis));
        }
    }

    return plan;
}

std::vector<int32_t> concatenateLabels(const std::vector<int32_t>& a,
                                       const std::vector<int32_t>& b,
                                       const std::vector<int32_t>& c = {}) {
    std::vector<int32_t> result;
    result.reserve(a.size() + b.size() + c.size());
    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    result.insert(result.end(), c.begin(), c.end());
    return result;
}

bool equalsAfterRemovingMissingLabels(const std::vector<int32_t>& source,
                                      const std::vector<int32_t>& expected,
                                      const std::unordered_set<int32_t>& physically_present) {
    std::vector<int32_t> filtered;
    filtered.reserve(expected.size());
    for (int32_t label : expected) {
        if (physically_present.contains(label)) {
            filtered.push_back(label);
        }
    }
    return source == filtered;
}

std::vector<uint64_t> denseStridesForLogicalOperand(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> strides(dimensions.size(), 1);
    for (size_t axis = dimensions.size(); axis > 1; --axis) {
        const uint64_t dimension = dimensions[axis - 1];
        if (dimension == 0) {
            throw std::invalid_argument("Einsum logical operand dimensions must be non-zero.");
        }
        if (strides[axis - 1] > std::numeric_limits<uint64_t>::max() / dimension) {
            throw std::overflow_error("Einsum logical operand dense stride exceeds uint64_t range.");
        }
        strides[axis - 2] = strides[axis - 1] * dimension;
    }
    if (!dimensions.empty() && dimensions.front() == 0) {
        throw std::invalid_argument("Einsum logical operand dimensions must be non-zero.");
    }
    return strides;
}

EinsumLogicalOperandPlan buildLogicalInputOperandPlan(const ResolvedEinsumOperand& operand,
                                                      const std::vector<uint64_t>& input_dimensions,
                                                      const EinsumOperandPlan& operand_plan,
                                                      uint32_t source_operand_index) {
    if (operand.axis_labels.size() != input_dimensions.size()) {
        throw std::invalid_argument("Einsum logical input rank does not match the resolved operand rank.");
    }

    const std::vector<uint64_t> source_strides = denseStridesForLogicalOperand(input_dimensions);
    std::vector<uint64_t> logical_strides;
    logical_strides.reserve(operand_plan.diagonalized_labels.size());
    for (int32_t label : operand_plan.diagonalized_labels) {
        uint64_t combined_stride = 0;
        bool found = false;
        for (size_t source_axis = 0; source_axis < operand.axis_labels.size(); ++source_axis) {
            if (operand.axis_labels[source_axis] != label) {
                continue;
            }
            if (combined_stride > std::numeric_limits<uint64_t>::max() - source_strides[source_axis]) {
                throw std::overflow_error("Einsum logical diagonal stride exceeds uint64_t range.");
            }
            combined_stride += source_strides[source_axis];
            found = true;
        }
        if (!found || combined_stride == 0) {
            throw std::logic_error("Internal einsum planner error: failed to map a diagonalized logical label.");
        }
        logical_strides.push_back(combined_stride);
    }

    EinsumLogicalOperandPlan result;
    result.labels = operand_plan.diagonalized_labels;
    result.dimensions = operand_plan.diagonalized_dimensions;
    result.strides_elements = std::move(logical_strides);
    result.source_operand_indices = {source_operand_index};
    result.dense_storage = !operand_plan.requiresDiagonalExtraction();
    result.diagonal_view = operand_plan.requiresDiagonalExtraction();
    return result;
}

void validateLogicalOperand(const EinsumLogicalOperandPlan& operand, const char* side) {
    if (operand.labels.size() != operand.dimensions.size() || operand.labels.size() != operand.strides_elements.size()) {
        throw std::invalid_argument(std::string("Einsum pair planner ") + side +
                                    " logical operand metadata has inconsistent rank.");
    }

    std::unordered_set<int32_t> seen;
    seen.reserve(operand.labels.size());
    for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
        if (operand.labels[axis] < 0 || operand.dimensions[axis] == 0 || operand.strides_elements[axis] == 0) {
            throw std::invalid_argument(std::string("Einsum pair planner ") + side +
                                        " logical operand contains invalid label/dimension/stride metadata.");
        }
        if (!seen.insert(operand.labels[axis]).second) {
            throw std::invalid_argument(std::string("Einsum pair planner ") + side +
                                        " logical operand repeats a label; diagonals must already be collapsed.");
        }
    }

    if (!std::is_sorted(operand.source_operand_indices.begin(), operand.source_operand_indices.end()) ||
        std::adjacent_find(operand.source_operand_indices.begin(), operand.source_operand_indices.end()) !=
            operand.source_operand_indices.end()) {
        throw std::invalid_argument(std::string("Einsum pair planner ") + side +
                                    " provenance must be sorted and contain unique source operand indices.");
    }
}

std::unordered_map<int32_t, uint64_t> logicalOperandDimensions(const EinsumLogicalOperandPlan& operand) {
    std::unordered_map<int32_t, uint64_t> result;
    result.reserve(operand.labels.size());
    for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
        result.emplace(operand.labels[axis], operand.dimensions[axis]);
    }
    return result;
}

uint64_t mergePairDimension(uint64_t current, uint64_t incoming) {
    if (current == 0 || current == incoming) {
        return incoming;
    }
    if (current == 1) {
        return incoming;
    }
    if (incoming == 1) {
        return current;
    }
    throw std::invalid_argument("Einsum pair planner received mutually incompatible logical operand dimensions.");
}

std::vector<uint64_t> buildPairLabelDimensions(const EinsumLogicalOperandPlan& lhs,
                                               const EinsumLogicalOperandPlan& rhs) {
    int32_t max_label = -1;
    for (int32_t label : lhs.labels) max_label = std::max(max_label, label);
    for (int32_t label : rhs.labels) max_label = std::max(max_label, label);

    std::vector<uint64_t> dimensions(max_label < 0 ? 0 : static_cast<size_t>(max_label) + 1, 0);
    const auto merge_operand = [&dimensions](const EinsumLogicalOperandPlan& operand) {
        for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
            const size_t label_index = static_cast<size_t>(operand.labels[axis]);
            dimensions[label_index] = mergePairDimension(dimensions[label_index], operand.dimensions[axis]);
        }
    };
    merge_operand(lhs);
    merge_operand(rhs);
    return dimensions;
}

std::vector<int32_t> pairReductionLabels(const EinsumLogicalOperandPlan& lhs,
                                         const EinsumLogicalOperandPlan& rhs,
                                         const std::vector<int32_t>& surviving_labels,
                                         const std::vector<uint64_t>& pair_label_dimensions) {
    std::unordered_set<int32_t> surviving;
    surviving.reserve(surviving_labels.size());
    for (int32_t label : surviving_labels) {
        const size_t index = static_cast<size_t>(label);
        if (label < 0 || index >= pair_label_dimensions.size() || pair_label_dimensions[index] == 0) {
            throw std::invalid_argument("Einsum pair planner surviving label is absent from both operands.");
        }
        if (!surviving.insert(label).second) {
            throw std::invalid_argument("Einsum pair planner surviving labels must be unique.");
        }
    }

    std::unordered_set<int32_t> present;
    present.reserve(lhs.labels.size() + rhs.labels.size());
    present.insert(lhs.labels.begin(), lhs.labels.end());
    present.insert(rhs.labels.begin(), rhs.labels.end());

    std::vector<int32_t> reductions;
    reductions.reserve(present.size());
    for (int32_t label : present) {
        if (!surviving.contains(label)) {
            reductions.push_back(label);
        }
    }
    std::sort(reductions.begin(), reductions.end());
    return reductions;
}

std::vector<uint64_t> logicalDimensionsForLabels(const std::vector<int32_t>& labels,
                                                 const std::vector<uint64_t>& label_dimensions) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(labels.size());
    for (int32_t label : labels) {
        const size_t index = static_cast<size_t>(label);
        if (label < 0 || index >= label_dimensions.size() || label_dimensions[index] == 0) {
            throw std::logic_error("Internal einsum pair planner error: missing logical result dimension.");
        }
        dimensions.push_back(label_dimensions[index]);
    }
    return dimensions;
}

std::optional<DataType> pairResultStorageDtype(const EinsumLogicalOperandPlan& lhs,
                                               const EinsumLogicalOperandPlan& rhs) {
    if (lhs.storage_dtype.has_value() && rhs.storage_dtype.has_value() && lhs.storage_dtype != rhs.storage_dtype) {
        throw std::invalid_argument("Einsum pair planner requires identical operand storage dtypes.");
    }
    return lhs.storage_dtype.has_value() ? lhs.storage_dtype : rhs.storage_dtype;
}

void requireDisjointPairProvenance(const EinsumLogicalOperandPlan& lhs,
                                   const EinsumLogicalOperandPlan& rhs) {
    std::vector<uint32_t> overlap;
    std::set_intersection(lhs.source_operand_indices.begin(),
                          lhs.source_operand_indices.end(),
                          rhs.source_operand_indices.begin(),
                          rhs.source_operand_indices.end(),
                          std::back_inserter(overlap));
    if (!overlap.empty()) {
        throw std::invalid_argument("Einsum pair planner operands must have disjoint source provenance.");
    }
}

std::vector<uint32_t> mergePairProvenance(const EinsumLogicalOperandPlan& lhs,
                                          const EinsumLogicalOperandPlan& rhs) {
    std::vector<uint32_t> result;
    result.reserve(lhs.source_operand_indices.size() + rhs.source_operand_indices.size());
    std::set_union(lhs.source_operand_indices.begin(),
                   lhs.source_operand_indices.end(),
                   rhs.source_operand_indices.begin(),
                   rhs.source_operand_indices.end(),
                   std::back_inserter(result));
    return result;
}

EinsumMatrixOperandPlan buildMatrixOperandPlan(const std::vector<int32_t>& post_reduction_labels,
                                                const EinsumLogicalOperandPlan& logical_operand,
                                                const std::vector<int32_t>& batch_labels,
                                                const std::vector<int32_t>& first_matrix_group,
                                                const std::vector<int32_t>& second_matrix_group,
                                                const std::vector<uint64_t>& label_dimensions) {
    EinsumMatrixOperandPlan plan;
    plan.canonical_labels = concatenateLabels(batch_labels, first_matrix_group, second_matrix_group);
    plan.permutation = permutationToLabels(post_reduction_labels, plan.canonical_labels, true, &plan.inserted_axes);
    for (uint32_t inserted_axis : plan.inserted_axes) {
        if (inserted_axis >= batch_labels.size()) {
            throw std::logic_error("Internal einsum planner error: only matrix batch axes may be absent from an operand.");
        }
    }

    const auto local_dimensions = logicalOperandDimensions(logical_operand);
    for (size_t axis = 0; axis < plan.canonical_labels.size(); ++axis) {
        const int32_t label = plan.canonical_labels[axis];
        const auto local = local_dimensions.find(label);
        const uint64_t local_dimension = local == local_dimensions.end() ? 1 : local->second;
        const uint64_t target_dimension = label_dimensions.at(static_cast<size_t>(label));
        if (local_dimension != target_dimension) {
            if (local_dimension != 1) {
                throw std::logic_error("Internal einsum planner error: invalid matrix-operand broadcast dimension.");
            }
            plan.broadcast_axes.push_back(static_cast<uint32_t>(axis));
        }
    }

    std::unordered_set<int32_t> physically_present(post_reduction_labels.begin(), post_reduction_labels.end());
    const std::vector<int32_t> no_transpose_order = concatenateLabels(batch_labels, first_matrix_group, second_matrix_group);
    const std::vector<int32_t> transpose_order = concatenateLabels(batch_labels, second_matrix_group, first_matrix_group);

    if (equalsAfterRemovingMissingLabels(post_reduction_labels, no_transpose_order, physically_present)) {
        plan.transpose = false;
        plan.requires_materialized_permutation = false;
    } else if (!first_matrix_group.empty() && !second_matrix_group.empty() &&
               equalsAfterRemovingMissingLabels(post_reduction_labels, transpose_order, physically_present)) {
        plan.transpose = true;
        plan.requires_materialized_permutation = false;
    } else {
        plan.transpose = false;
        plan.requires_materialized_permutation = true;
    }

    return plan;
}

enum class SharedReductionDisposition {
    CONTRACT,
    REDUCE_LHS_ELIDE_RHS,
    REDUCE_RHS_ELIDE_LHS,
};

SharedReductionDisposition classifySharedReductionLabel(
    int32_t label,
    const std::unordered_map<int32_t, uint64_t>& lhs_dimensions,
    const std::unordered_map<int32_t, uint64_t>& rhs_dimensions,
    const std::vector<uint64_t>& resolved_dimensions) {
    const auto lhs_it = lhs_dimensions.find(label);
    const auto rhs_it = rhs_dimensions.find(label);
    const size_t label_index = static_cast<size_t>(label);
    if (lhs_it == lhs_dimensions.end() || rhs_it == rhs_dimensions.end() || label_index >= resolved_dimensions.size()) {
        throw std::logic_error("Internal einsum planner error: shared reduction label is missing operand dimensions.");
    }

    const uint64_t lhs_dimension = lhs_it->second;
    const uint64_t rhs_dimension = rhs_it->second;
    const uint64_t resolved_dimension = resolved_dimensions[label_index];
    if (lhs_dimension == resolved_dimension && rhs_dimension == resolved_dimension) {
        return SharedReductionDisposition::CONTRACT;
    }
    if (resolved_dimension > 1 && lhs_dimension == resolved_dimension && rhs_dimension == 1) {
        return SharedReductionDisposition::REDUCE_LHS_ELIDE_RHS;
    }
    if (resolved_dimension > 1 && lhs_dimension == 1 && rhs_dimension == resolved_dimension) {
        return SharedReductionDisposition::REDUCE_RHS_ELIDE_LHS;
    }
    throw std::logic_error("Internal einsum planner error: invalid shared reduction broadcast dimensions.");
}

std::vector<int32_t> labelsAfterOperandNormalization(const EinsumLogicalOperandPlan& operand,
                                                     const std::vector<int32_t>& reduction_labels,
                                                     const std::vector<int32_t>& broadcast_elision_labels) {
    if (reduction_labels.empty() && broadcast_elision_labels.empty()) {
        return operand.labels;
    }
    std::unordered_set<int32_t> removed(reduction_labels.begin(), reduction_labels.end());
    removed.insert(broadcast_elision_labels.begin(), broadcast_elision_labels.end());
    std::vector<int32_t> retained;
    retained.reserve(operand.labels.size() - removed.size());
    for (int32_t label : operand.labels) {
        if (!removed.contains(label)) {
            retained.push_back(label);
        }
    }
    return retained;
}

std::optional<EinsumMatrixMultiplyPlan> tryBuildMatrixMultiplyPlan(
    const EinsumLogicalOperandPlan& lhs_operand,
    const EinsumLogicalOperandPlan& rhs_operand,
    const std::vector<int32_t>& output_labels_order,
    const std::vector<int32_t>& reduction_labels,
    const std::vector<uint64_t>& label_dimensions) {
    const std::unordered_set<int32_t> lhs_labels(lhs_operand.labels.begin(), lhs_operand.labels.end());
    const std::unordered_set<int32_t> rhs_labels(rhs_operand.labels.begin(), rhs_operand.labels.end());
    const std::unordered_set<int32_t> output_labels(output_labels_order.begin(), output_labels_order.end());
    const auto lhs_dimensions = logicalOperandDimensions(lhs_operand);
    const auto rhs_dimensions = logicalOperandDimensions(rhs_operand);

    EinsumMatrixMultiplyPlan plan;

    // Expanded ellipsis labels are broadcast batch dimensions even when a
    // lower-rank operand has no corresponding physical axis. Regular output
    // labels are batch labels only when physically present in both operands.
    for (int32_t label : output_labels_order) {
        if (EinsumParser::isEllipsisLabel(label) || (lhs_labels.contains(label) && rhs_labels.contains(label))) {
            plan.batch_labels.push_back(label);
        } else if (lhs_labels.contains(label)) {
            plan.lhs_free_labels.push_back(label);
        } else if (rhs_labels.contains(label)) {
            plan.rhs_free_labels.push_back(label);
        } else {
            throw std::logic_error("Internal einsum planner error: output label is absent from both operands.");
        }
    }

    // Reduction-label order is not externally observable. Preserve each
    // operand's physical label order for operand-local reductions and K.
    for (int32_t label : lhs_operand.labels) {
        if (output_labels.contains(label)) {
            continue;
        }
        if (!rhs_labels.contains(label)) {
            plan.lhs_reduction_labels.push_back(label);
            continue;
        }
        switch (classifySharedReductionLabel(label, lhs_dimensions, rhs_dimensions, label_dimensions)) {
            case SharedReductionDisposition::CONTRACT:
                plan.contraction_labels.push_back(label);
                break;
            case SharedReductionDisposition::REDUCE_LHS_ELIDE_RHS:
                plan.lhs_reduction_labels.push_back(label);
                break;
            case SharedReductionDisposition::REDUCE_RHS_ELIDE_LHS:
                plan.lhs_broadcast_elision_labels.push_back(label);
                break;
        }
    }
    for (int32_t label : rhs_operand.labels) {
        if (output_labels.contains(label)) {
            continue;
        }
        if (!lhs_labels.contains(label)) {
            plan.rhs_reduction_labels.push_back(label);
            continue;
        }
        switch (classifySharedReductionLabel(label, lhs_dimensions, rhs_dimensions, label_dimensions)) {
            case SharedReductionDisposition::CONTRACT:
                break;
            case SharedReductionDisposition::REDUCE_LHS_ELIDE_RHS:
                plan.rhs_broadcast_elision_labels.push_back(label);
                break;
            case SharedReductionDisposition::REDUCE_RHS_ELIDE_LHS:
                plan.rhs_reduction_labels.push_back(label);
                break;
        }
    }

    if (plan.lhs_reduction_labels.size() + plan.contraction_labels.size() + plan.rhs_reduction_labels.size() !=
        reduction_labels.size()) {
        throw std::logic_error("Internal einsum planner error: matrix lowering does not cover every reduction label.");
    }

    if (plan.contraction_labels.empty()) {
        return std::nullopt;
    }

    std::unordered_set<int32_t> matrix_labels;
    for (int32_t label : plan.batch_labels) matrix_labels.insert(label);
    for (int32_t label : plan.lhs_free_labels) matrix_labels.insert(label);
    for (int32_t label : plan.lhs_reduction_labels) matrix_labels.insert(label);
    for (int32_t label : plan.lhs_broadcast_elision_labels) matrix_labels.insert(label);
    for (int32_t label : plan.contraction_labels) matrix_labels.insert(label);
    for (int32_t label : plan.rhs_broadcast_elision_labels) matrix_labels.insert(label);
    for (int32_t label : plan.rhs_reduction_labels) matrix_labels.insert(label);
    for (int32_t label : plan.rhs_free_labels) matrix_labels.insert(label);
    for (int32_t label : lhs_labels) {
        if (!matrix_labels.contains(label)) return std::nullopt;
    }
    for (int32_t label : rhs_labels) {
        if (!matrix_labels.contains(label)) return std::nullopt;
    }

    plan.canonical_output_labels = concatenateLabels(plan.batch_labels, plan.lhs_free_labels, plan.rhs_free_labels);
    plan.output_permutation = permutationToLabels(plan.canonical_output_labels, output_labels_order, false, nullptr);
    plan.requires_output_permutation = !isIdentityPermutation(plan.output_permutation);

    plan.batch_count = checkedProduct(plan.batch_labels, label_dimensions, "matrix-multiply batch");
    plan.m = checkedProduct(plan.lhs_free_labels, label_dimensions, "matrix-multiply M");
    plan.n = checkedProduct(plan.rhs_free_labels, label_dimensions, "matrix-multiply N");
    plan.k = checkedProduct(plan.contraction_labels, label_dimensions, "matrix-multiply K");

    const std::vector<int32_t> lhs_post_reduction_labels =
        labelsAfterOperandNormalization(lhs_operand, plan.lhs_reduction_labels, plan.lhs_broadcast_elision_labels);
    const std::vector<int32_t> rhs_post_reduction_labels =
        labelsAfterOperandNormalization(rhs_operand, plan.rhs_reduction_labels, plan.rhs_broadcast_elision_labels);
    plan.lhs = buildMatrixOperandPlan(lhs_post_reduction_labels,
                                      lhs_operand,
                                      plan.batch_labels,
                                      plan.lhs_free_labels,
                                      plan.contraction_labels,
                                      label_dimensions);
    plan.rhs = buildMatrixOperandPlan(rhs_post_reduction_labels,
                                      rhs_operand,
                                      plan.batch_labels,
                                      plan.contraction_labels,
                                      plan.rhs_free_labels,
                                      label_dimensions);

    const bool pre_reduction = !plan.lhs_reduction_labels.empty() || !plan.rhs_reduction_labels.empty();
    const bool physical_view = !lhs_operand.dense_storage || !rhs_operand.dense_storage;
    const bool input_permutation = plan.lhs.requires_materialized_permutation || plan.rhs.requires_materialized_permutation;
    const bool shared_label_broadcast = !plan.lhs.broadcast_axes.empty() || !plan.rhs.broadcast_axes.empty();
    plan.direct = !pre_reduction && !physical_view && !input_permutation && !shared_label_broadcast &&
                  !plan.requires_output_permutation;

    return plan;
}

std::optional<EinsumPairProductPlan> tryBuildPairProductPlan(
    const EinsumLogicalOperandPlan& lhs_operand,
    const EinsumLogicalOperandPlan& rhs_operand,
    const std::vector<int32_t>& output_labels_order,
    const std::vector<int32_t>& reduction_labels,
    const std::vector<uint64_t>& label_dimensions) {
    const std::unordered_set<int32_t> lhs_labels(lhs_operand.labels.begin(), lhs_operand.labels.end());
    const std::unordered_set<int32_t> rhs_labels(rhs_operand.labels.begin(), rhs_operand.labels.end());
    const std::unordered_set<int32_t> output_labels(output_labels_order.begin(), output_labels_order.end());
    const auto lhs_dimensions = logicalOperandDimensions(lhs_operand);
    const auto rhs_dimensions = logicalOperandDimensions(rhs_operand);

    EinsumPairProductPlan plan;
    for (int32_t label : lhs_operand.labels) {
        if (output_labels.contains(label)) {
            continue;
        }
        if (!rhs_labels.contains(label)) {
            plan.lhs_reduction_labels.push_back(label);
            continue;
        }
        switch (classifySharedReductionLabel(label, lhs_dimensions, rhs_dimensions, label_dimensions)) {
            case SharedReductionDisposition::CONTRACT:
                return std::nullopt;
            case SharedReductionDisposition::REDUCE_LHS_ELIDE_RHS:
                plan.lhs_reduction_labels.push_back(label);
                break;
            case SharedReductionDisposition::REDUCE_RHS_ELIDE_LHS:
                plan.lhs_broadcast_elision_labels.push_back(label);
                break;
        }
    }
    for (int32_t label : rhs_operand.labels) {
        if (output_labels.contains(label)) {
            continue;
        }
        if (!lhs_labels.contains(label)) {
            plan.rhs_reduction_labels.push_back(label);
            continue;
        }
        switch (classifySharedReductionLabel(label, lhs_dimensions, rhs_dimensions, label_dimensions)) {
            case SharedReductionDisposition::CONTRACT:
                return std::nullopt;
            case SharedReductionDisposition::REDUCE_LHS_ELIDE_RHS:
                plan.rhs_broadcast_elision_labels.push_back(label);
                break;
            case SharedReductionDisposition::REDUCE_RHS_ELIDE_LHS:
                plan.rhs_reduction_labels.push_back(label);
                break;
        }
    }

    if (plan.lhs_reduction_labels.size() + plan.rhs_reduction_labels.size() != reduction_labels.size()) {
        return std::nullopt;
    }
    return plan;
}


uint64_t checkedMultiplyPhysicalCost(uint64_t lhs, uint64_t rhs, const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string("Einsum ") + description + " exceeds uint64_t range.");
    }
    return lhs * rhs;
}

uint64_t checkedAddPhysicalCost(uint64_t lhs, uint64_t rhs, const char* description) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        throw std::overflow_error(std::string("Einsum ") + description + " exceeds uint64_t range.");
    }
    return lhs + rhs;
}

uint64_t logicalElementCount(const EinsumLogicalOperandPlan& operand) {
    uint64_t count = 1;
    for (uint64_t dimension : operand.dimensions) {
        count = checkedMultiplyPhysicalCost(count, dimension, "logical operand element count");
    }
    return count;
}

std::optional<uint64_t> storageElementBytes(std::optional<DataType> dtype) {
    if (!dtype.has_value()) {
        return std::nullopt;
    }
    switch (*dtype) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::FP32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FP64:
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::TF32:
            // TF32 is compute-only and therefore is not a persistent tensor
            // storage format.
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<uint64_t> storageBytesForElements(std::optional<DataType> dtype,
                                                uint64_t elements,
                                                const char* description) {
    const std::optional<uint64_t> bytes_per_element = storageElementBytes(dtype);
    if (!bytes_per_element.has_value()) {
        return std::nullopt;
    }
    return checkedMultiplyPhysicalCost(elements, *bytes_per_element, description);
}

void populatePhysicalByteCosts(EinsumPairPhysicalCost& cost, std::optional<DataType> dtype) {
    cost.lhs_materialization_bytes =
        storageBytesForElements(dtype, cost.lhs_materialization_elements, "lhs materialization byte count");
    cost.rhs_materialization_bytes =
        storageBytesForElements(dtype, cost.rhs_materialization_elements, "rhs materialization byte count");
    cost.output_materialization_bytes =
        storageBytesForElements(dtype, cost.output_materialization_elements, "output materialization byte count");
    cost.result_bytes = storageBytesForElements(dtype, cost.result_elements, "pair result byte count");
    cost.peak_temporary_bytes =
        storageBytesForElements(dtype, cost.peak_temporary_elements, "pair temporary byte count");
}

std::optional<size_t> logicalAxisForPlannerLabel(const EinsumLogicalOperandPlan& operand, int32_t label) {
    for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
        if (operand.labels[axis] == label) {
            return axis;
        }
    }
    return std::nullopt;
}

struct PlannerFlattenedLogicalGroup {
    uint64_t extent = 1;
    uint64_t stride = 1;
};

std::optional<PlannerFlattenedLogicalGroup> flattenPlannerLogicalGroup(
    const EinsumLogicalOperandPlan& operand,
    const std::vector<int32_t>& labels) {
    PlannerFlattenedLogicalGroup result;
    if (labels.empty()) {
        return result;
    }

    bool have_non_singleton_axis = false;
    uint64_t expected_outer_stride = 0;
    for (size_t label_index = labels.size(); label_index-- > 0;) {
        const std::optional<size_t> axis = logicalAxisForPlannerLabel(operand, labels[label_index]);
        if (!axis.has_value()) {
            return std::nullopt;
        }
        const uint64_t dimension = operand.dimensions[*axis];
        const uint64_t stride = operand.strides_elements[*axis];
        result.extent = checkedMultiplyPhysicalCost(result.extent, dimension, "matrix-group flatten extent");

        if (dimension == 1) {
            continue;
        }
        if (!have_non_singleton_axis) {
            result.stride = stride;
            expected_outer_stride =
                checkedMultiplyPhysicalCost(dimension, stride, "matrix-group flatten stride");
            have_non_singleton_axis = true;
            continue;
        }
        if (stride != expected_outer_stride) {
            return std::nullopt;
        }
        expected_outer_stride =
            checkedMultiplyPhysicalCost(dimension, stride, "matrix-group flatten stride");
    }

    if (!have_non_singleton_axis) {
        result.stride = 1;
    }
    return result;
}

bool containsPlannerAxis(const std::vector<uint32_t>& axes, uint32_t axis) {
    return std::find(axes.begin(), axes.end(), axis) != axes.end();
}

bool matrixDimensionsFitPlannerBackend(const EinsumMatrixMultiplyPlan& matrix) {
    constexpr uint64_t kMax = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    return matrix.m <= kMax && matrix.n <= kMax && matrix.k <= kMax && matrix.batch_count <= kMax;
}

bool hasOnlyPlannerBatchBroadcasts(const EinsumMatrixOperandPlan& operand, size_t batch_rank) {
    return std::all_of(operand.broadcast_axes.begin(), operand.broadcast_axes.end(), [batch_rank](uint32_t axis) {
        return axis < batch_rank;
    });
}

EinsumLogicalOperandPlan normalizedPlannerMatrixOperand(
    const EinsumLogicalOperandPlan& source,
    const std::vector<int32_t>& reduction_labels,
    const std::vector<int32_t>& broadcast_elision_labels) {
    const std::unordered_set<int32_t> reductions(reduction_labels.begin(), reduction_labels.end());
    const std::unordered_set<int32_t> elisions(broadcast_elision_labels.begin(), broadcast_elision_labels.end());

    EinsumLogicalOperandPlan result = source;
    if (!reductions.empty()) {
        std::vector<int32_t> labels;
        std::vector<uint64_t> dimensions;
        labels.reserve(source.labels.size());
        dimensions.reserve(source.dimensions.size());
        for (size_t axis = 0; axis < source.labels.size(); ++axis) {
            if (reductions.contains(source.labels[axis])) {
                continue;
            }
            labels.push_back(source.labels[axis]);
            dimensions.push_back(source.dimensions[axis]);
        }
        result.labels = std::move(labels);
        result.dimensions = std::move(dimensions);
        result.strides_elements = denseStridesForLogicalOperand(result.dimensions);
        result.dense_storage = true;
        result.diagonal_view = false;
    }

    if (!elisions.empty()) {
        std::vector<int32_t> labels;
        std::vector<uint64_t> dimensions;
        std::vector<uint64_t> strides;
        labels.reserve(result.labels.size());
        dimensions.reserve(result.dimensions.size());
        strides.reserve(result.strides_elements.size());
        for (size_t axis = 0; axis < result.labels.size(); ++axis) {
            if (elisions.contains(result.labels[axis])) {
                if (result.dimensions[axis] != 1) {
                    throw std::logic_error(
                        "Internal einsum planner error: broadcast elision requires a singleton logical axis.");
                }
                continue;
            }
            labels.push_back(result.labels[axis]);
            dimensions.push_back(result.dimensions[axis]);
            strides.push_back(result.strides_elements[axis]);
        }
        result.labels = std::move(labels);
        result.dimensions = std::move(dimensions);
        result.strides_elements = std::move(strides);
        if (result.strides_elements == denseStridesForLogicalOperand(result.dimensions)) {
            result.dense_storage = true;
        }
    }

    return result;
}

std::vector<uint64_t> plannerMatrixBatchDimensions(const EinsumMatrixMultiplyPlan& matrix,
                                                   const EinsumMatrixOperandPlan& operand,
                                                   const std::vector<uint64_t>& label_dimensions) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(matrix.batch_labels.size());
    for (size_t axis = 0; axis < matrix.batch_labels.size(); ++axis) {
        if (containsPlannerAxis(operand.broadcast_axes, static_cast<uint32_t>(axis))) {
            dimensions.push_back(1);
        } else {
            dimensions.push_back(label_dimensions.at(static_cast<size_t>(matrix.batch_labels[axis])));
        }
    }
    return dimensions;
}

std::vector<uint64_t> plannerMatrixStoredShape(const EinsumMatrixMultiplyPlan& matrix,
                                               const EinsumMatrixOperandPlan& operand,
                                               const std::vector<uint64_t>& label_dimensions,
                                               uint64_t logical_rows,
                                               uint64_t logical_columns) {
    std::vector<uint64_t> dimensions = plannerMatrixBatchDimensions(matrix, operand, label_dimensions);
    dimensions.push_back(logical_rows);
    dimensions.push_back(logical_columns);
    return dimensions;
}

struct PlannerMatrixOperandLayout {
    MatmulTensorLayout layout;
    bool transpose = false;
    bool materialize = false;
};

std::optional<MatmulTensorLayout> directPlannerMatrixOperandLayout(
    const EinsumMatrixMultiplyPlan& matrix,
    const EinsumMatrixOperandPlan& operand_plan,
    const EinsumLogicalOperandPlan& source,
    const std::vector<int32_t>& first_matrix_group,
    const std::vector<int32_t>& second_matrix_group,
    uint64_t logical_rows,
    uint64_t logical_columns) {
    if (operand_plan.requires_materialized_permutation) {
        return std::nullopt;
    }

    std::vector<int32_t> expected_source_labels;
    expected_source_labels.reserve(source.labels.size());
    for (int32_t label : matrix.batch_labels) {
        if (logicalAxisForPlannerLabel(source, label).has_value()) {
            expected_source_labels.push_back(label);
        }
    }
    const auto append_present = [&](const std::vector<int32_t>& labels) {
        for (int32_t label : labels) {
            if (logicalAxisForPlannerLabel(source, label).has_value()) {
                expected_source_labels.push_back(label);
            }
        }
    };
    if (operand_plan.transpose) {
        append_present(second_matrix_group);
        append_present(first_matrix_group);
    } else {
        append_present(first_matrix_group);
        append_present(second_matrix_group);
    }
    if (expected_source_labels != source.labels) {
        return std::nullopt;
    }

    const std::optional<PlannerFlattenedLogicalGroup> first =
        flattenPlannerLogicalGroup(source, first_matrix_group);
    const std::optional<PlannerFlattenedLogicalGroup> second =
        flattenPlannerLogicalGroup(source, second_matrix_group);
    if (!first.has_value() || !second.has_value() || first->extent != logical_rows ||
        second->extent != logical_columns) {
        return std::nullopt;
    }

    MatmulTensorLayout layout;
    layout.dimensions.reserve(matrix.batch_labels.size() + 2);
    layout.strides_elements.reserve(matrix.batch_labels.size() + 2);
    for (int32_t label : matrix.batch_labels) {
        const std::optional<size_t> axis = logicalAxisForPlannerLabel(source, label);
        if (axis.has_value()) {
            layout.dimensions.push_back(source.dimensions[*axis]);
            layout.strides_elements.push_back(source.strides_elements[*axis]);
        } else {
            layout.dimensions.push_back(1);
            layout.strides_elements.push_back(0);
        }
    }

    if (operand_plan.transpose) {
        layout.dimensions.push_back(second->extent);
        layout.dimensions.push_back(first->extent);
        layout.strides_elements.push_back(second->stride);
        layout.strides_elements.push_back(first->stride);
    } else {
        layout.dimensions.push_back(first->extent);
        layout.dimensions.push_back(second->extent);
        layout.strides_elements.push_back(first->stride);
        layout.strides_elements.push_back(second->stride);
    }
    return layout;
}

PlannerMatrixOperandLayout plannerMatrixOperandLayout(
    const EinsumMatrixMultiplyPlan& matrix,
    const EinsumMatrixOperandPlan& operand,
    const EinsumLogicalOperandPlan& source,
    const std::vector<int32_t>& first_matrix_group,
    const std::vector<int32_t>& second_matrix_group,
    const std::vector<uint64_t>& label_dimensions,
    uint64_t logical_rows,
    uint64_t logical_columns) {
    const std::optional<MatmulTensorLayout> direct = directPlannerMatrixOperandLayout(
        matrix, operand, source, first_matrix_group, second_matrix_group, logical_rows, logical_columns);
    if (direct.has_value()) {
        return PlannerMatrixOperandLayout{*direct, operand.transpose, false};
    }
    return PlannerMatrixOperandLayout{
        denseMatmulTensorLayout(
            plannerMatrixStoredShape(matrix, operand, label_dimensions, logical_rows, logical_columns)),
        false,
        true};
}

std::vector<uint64_t> plannerMatrixOutputShape(const EinsumMatrixMultiplyPlan& matrix,
                                               const std::vector<uint64_t>& label_dimensions,
                                               bool swapped_orientation) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(matrix.batch_labels.size() + 2);
    for (int32_t label : matrix.batch_labels) {
        dimensions.push_back(label_dimensions.at(static_cast<size_t>(label)));
    }
    dimensions.push_back(swapped_orientation ? matrix.n : matrix.m);
    dimensions.push_back(swapped_orientation ? matrix.m : matrix.n);
    return dimensions;
}

struct PlannerMatrixPhysicalLayout {
    PlannerMatrixOperandLayout lhs;
    PlannerMatrixOperandLayout rhs;
    BatchedMatmulLayoutPlan matmul;
};

std::optional<PlannerMatrixPhysicalLayout> planPlannerMatrixPhysicalLayout(
    const EinsumMatrixMultiplyPlan& matrix,
    const EinsumLogicalOperandPlan& lhs_operand,
    const EinsumLogicalOperandPlan& rhs_operand,
    const std::vector<uint64_t>& label_dimensions,
    bool swapped_orientation) {
    if (!matrixDimensionsFitPlannerBackend(matrix)) {
        return std::nullopt;
    }
    const size_t batch_rank = matrix.batch_labels.size();
    if (!hasOnlyPlannerBatchBroadcasts(matrix.lhs, batch_rank) ||
        !hasOnlyPlannerBatchBroadcasts(matrix.rhs, batch_rank)) {
        return std::nullopt;
    }

    const EinsumLogicalOperandPlan lhs_source = normalizedPlannerMatrixOperand(
        lhs_operand, matrix.lhs_reduction_labels, matrix.lhs_broadcast_elision_labels);
    const EinsumLogicalOperandPlan rhs_source = normalizedPlannerMatrixOperand(
        rhs_operand, matrix.rhs_reduction_labels, matrix.rhs_broadcast_elision_labels);

    PlannerMatrixPhysicalLayout result;
    result.lhs = plannerMatrixOperandLayout(matrix,
                                            matrix.lhs,
                                            lhs_source,
                                            matrix.lhs_free_labels,
                                            matrix.contraction_labels,
                                            label_dimensions,
                                            matrix.m,
                                            matrix.k);
    result.rhs = plannerMatrixOperandLayout(matrix,
                                            matrix.rhs,
                                            rhs_source,
                                            matrix.contraction_labels,
                                            matrix.rhs_free_labels,
                                            label_dimensions,
                                            matrix.k,
                                            matrix.n);

    const MatmulTensorLayout output_layout =
        denseMatmulTensorLayout(plannerMatrixOutputShape(matrix, label_dimensions, swapped_orientation));
    result.matmul = swapped_orientation
                        ? planBatchedMatmulLayout(result.rhs.layout,
                                                  result.lhs.layout,
                                                  output_layout,
                                                  !result.rhs.transpose,
                                                  !result.lhs.transpose)
                        : planBatchedMatmulLayout(result.lhs.layout,
                                                  result.rhs.layout,
                                                  output_layout,
                                                  result.lhs.transpose,
                                                  result.rhs.transpose);

    const bool lhs_addressable =
        swapped_orientation ? result.matmul.rhs_matrix.isBlasAddressable() : result.matmul.lhs_matrix.isBlasAddressable();
    const bool rhs_addressable =
        swapped_orientation ? result.matmul.lhs_matrix.isBlasAddressable() : result.matmul.rhs_matrix.isBlasAddressable();

    bool replan = false;
    if (!lhs_addressable && !result.lhs.materialize) {
        result.lhs = PlannerMatrixOperandLayout{
            denseMatmulTensorLayout(
                plannerMatrixStoredShape(matrix, matrix.lhs, label_dimensions, matrix.m, matrix.k)),
            false,
            true};
        replan = true;
    }
    if (!rhs_addressable && !result.rhs.materialize) {
        result.rhs = PlannerMatrixOperandLayout{
            denseMatmulTensorLayout(
                plannerMatrixStoredShape(matrix, matrix.rhs, label_dimensions, matrix.k, matrix.n)),
            false,
            true};
        replan = true;
    }
    if (replan) {
        result.matmul = swapped_orientation
                            ? planBatchedMatmulLayout(result.rhs.layout,
                                                      result.lhs.layout,
                                                      output_layout,
                                                      !result.rhs.transpose,
                                                      !result.lhs.transpose)
                            : planBatchedMatmulLayout(result.lhs.layout,
                                                      result.rhs.layout,
                                                      output_layout,
                                                      result.lhs.transpose,
                                                      result.rhs.transpose);
    }

    if (!result.matmul.canLowerWithoutMaterialization()) {
        return std::nullopt;
    }
    return result;
}

std::vector<int32_t> plannerMatrixPhysicalOutputLabels(const EinsumMatrixMultiplyPlan& matrix,
                                                       bool swapped_orientation) {
    if (!swapped_orientation) {
        return matrix.canonical_output_labels;
    }
    std::vector<int32_t> labels;
    labels.reserve(matrix.canonical_output_labels.size());
    labels.insert(labels.end(), matrix.batch_labels.begin(), matrix.batch_labels.end());
    labels.insert(labels.end(), matrix.rhs_free_labels.begin(), matrix.rhs_free_labels.end());
    labels.insert(labels.end(), matrix.lhs_free_labels.begin(), matrix.lhs_free_labels.end());
    return labels;
}

EinsumLogicalOperandPlan plannerResultViewForPhysicalLabels(
    const std::vector<int32_t>& logical_labels,
    const std::vector<int32_t>& physical_labels,
    const std::vector<uint64_t>& label_dimensions,
    std::optional<DataType> dtype,
    const std::vector<uint32_t>& provenance) {
    EinsumLogicalOperandPlan result;
    result.labels = logical_labels;
    result.dimensions = logicalDimensionsForLabels(logical_labels, label_dimensions);
    result.storage_dtype = dtype;
    result.source_operand_indices = provenance;
    result.diagonal_view = false;

    const std::vector<uint64_t> physical_dimensions =
        logicalDimensionsForLabels(physical_labels, label_dimensions);
    const std::vector<uint64_t> physical_strides = denseStridesForLogicalOperand(physical_dimensions);
    result.strides_elements.reserve(logical_labels.size());
    for (int32_t label : logical_labels) {
        const auto it = std::find(physical_labels.begin(), physical_labels.end(), label);
        if (it == physical_labels.end()) {
            throw std::logic_error("Internal einsum planner error: physical result is missing a surviving label.");
        }
        const size_t physical_axis = static_cast<size_t>(std::distance(physical_labels.begin(), it));
        result.strides_elements.push_back(physical_strides[physical_axis]);
    }
    result.dense_storage = result.strides_elements == denseStridesForLogicalOperand(result.dimensions);
    return result;
}

EinsumLogicalOperandPlan densePlannerPairResult(const std::vector<int32_t>& surviving_labels,
                                                const std::vector<uint64_t>& label_dimensions,
                                                std::optional<DataType> dtype,
                                                const std::vector<uint32_t>& provenance) {
    return plannerResultViewForPhysicalLabels(
        surviving_labels, surviving_labels, label_dimensions, dtype, provenance);
}

uint64_t reducedPlannerOperandResultElements(const EinsumLogicalOperandPlan& operand,
                                             const std::vector<int32_t>& reduction_labels,
                                             const std::vector<int32_t>& elision_labels) {
    return logicalElementCount(normalizedPlannerMatrixOperand(operand, reduction_labels, elision_labels));
}

EinsumPairPhysicalCost plannerMatrixCandidateCost(
    const EinsumMatrixMultiplyPlan& matrix,
    const EinsumLogicalOperandPlan& lhs,
    const EinsumLogicalOperandPlan& rhs,
    const PlannerMatrixPhysicalLayout& layout,
    bool output_materialized,
    uint64_t result_elements,
    std::optional<DataType> dtype) {
    EinsumPairPhysicalCost cost;
    cost.matmul_fma_count = checkedMultiplyPhysicalCost(
        checkedMultiplyPhysicalCost(
            checkedMultiplyPhysicalCost(matrix.batch_count, matrix.m, "matmul FMA count"),
            matrix.n,
            "matmul FMA count"),
        matrix.k,
        "matmul FMA count");

    if (!matrix.lhs_reduction_labels.empty()) {
        cost.reduction_input_elements = checkedAddPhysicalCost(
            cost.reduction_input_elements, logicalElementCount(lhs), "reduction work");
        cost.reduction_op_count = checkedAddPhysicalCost(
            cost.reduction_op_count, 1, "reduction operation count");
    }
    if (!matrix.rhs_reduction_labels.empty()) {
        cost.reduction_input_elements = checkedAddPhysicalCost(
            cost.reduction_input_elements, logicalElementCount(rhs), "reduction work");
        cost.reduction_op_count = checkedAddPhysicalCost(
            cost.reduction_op_count, 1, "reduction operation count");
    }

    const uint64_t lhs_normalized_elements =
        reducedPlannerOperandResultElements(lhs, matrix.lhs_reduction_labels, matrix.lhs_broadcast_elision_labels);
    const uint64_t rhs_normalized_elements =
        reducedPlannerOperandResultElements(rhs, matrix.rhs_reduction_labels, matrix.rhs_broadcast_elision_labels);
    cost.lhs_materialization_elements = layout.lhs.materialize ? lhs_normalized_elements : 0;
    cost.rhs_materialization_elements = layout.rhs.materialize ? rhs_normalized_elements : 0;
    cost.output_materialization_elements = output_materialized ? result_elements : 0;
    cost.materialization_op_count = static_cast<uint64_t>(layout.lhs.materialize) +
                                    static_cast<uint64_t>(layout.rhs.materialize) +
                                    static_cast<uint64_t>(output_materialized);
    cost.result_elements = result_elements;
    cost.matmul_group_count = layout.matmul.grouping.group_count;

    cost.peak_temporary_elements = result_elements;
    cost.peak_temporary_elements = std::max(cost.peak_temporary_elements, cost.lhs_materialization_elements);
    cost.peak_temporary_elements = std::max(cost.peak_temporary_elements, cost.rhs_materialization_elements);
    if (!matrix.lhs_reduction_labels.empty()) {
        cost.peak_temporary_elements = std::max(cost.peak_temporary_elements, lhs_normalized_elements);
    }
    if (!matrix.rhs_reduction_labels.empty()) {
        cost.peak_temporary_elements = std::max(cost.peak_temporary_elements, rhs_normalized_elements);
    }

    populatePhysicalByteCosts(cost, dtype);
    return cost;
}

EinsumPairPhysicalCandidate plannerMatrixCandidate(
    const EinsumMatrixMultiplyPlan& matrix,
    const EinsumLogicalOperandPlan& lhs,
    const EinsumLogicalOperandPlan& rhs,
    const std::vector<int32_t>& surviving_labels,
    const std::vector<uint64_t>& label_dimensions,
    const PlannerMatrixPhysicalLayout& physical_layout,
    bool swapped_orientation,
    bool output_materialized,
    std::optional<DataType> dtype,
    const std::vector<uint32_t>& provenance) {
    const std::vector<int32_t> physical_labels =
        plannerMatrixPhysicalOutputLabels(matrix, swapped_orientation);

    EinsumPairPhysicalCandidate candidate;
    candidate.kind = matrix.batch_labels.empty() ? EinsumPlanKind::GEMM : EinsumPlanKind::BATCHED_GEMM;
    candidate.matrix_multiply = matrix;
    candidate.swapped_gemm_orientation = swapped_orientation;
    candidate.lhs_materialized = physical_layout.lhs.materialize;
    candidate.rhs_materialized = physical_layout.rhs.materialize;
    candidate.output_materialized = output_materialized;
    candidate.physical_result_labels = output_materialized ? surviving_labels : physical_labels;
    candidate.result = output_materialized
                           ? densePlannerPairResult(surviving_labels, label_dimensions, dtype, provenance)
                           : plannerResultViewForPhysicalLabels(
                                 surviving_labels, physical_labels, label_dimensions, dtype, provenance);
    candidate.cost = plannerMatrixCandidateCost(matrix,
                                                lhs,
                                                rhs,
                                                physical_layout,
                                                output_materialized,
                                                logicalElementCount(candidate.result),
                                                dtype);
    return candidate;
}

bool samePlannerPhysicalResult(const EinsumLogicalOperandPlan& lhs, const EinsumLogicalOperandPlan& rhs) {
    return lhs.labels == rhs.labels && lhs.dimensions == rhs.dimensions &&
           lhs.strides_elements == rhs.strides_elements && lhs.storage_dtype == rhs.storage_dtype &&
           lhs.source_operand_indices == rhs.source_operand_indices && lhs.dense_storage == rhs.dense_storage &&
           lhs.diagonal_view == rhs.diagonal_view;
}

uint64_t totalPlannerMaterializationElements(const EinsumPairPhysicalCost& cost) {
    return checkedAddPhysicalCost(
        checkedAddPhysicalCost(cost.lhs_materialization_elements,
                               cost.rhs_materialization_elements,
                               "materialization element count"),
        cost.output_materialization_elements,
        "materialization element count");
}

bool plannerCandidateDominates(const EinsumPairPhysicalCandidate& lhs,
                               const EinsumPairPhysicalCandidate& rhs) {
    if (!samePlannerPhysicalResult(lhs.result, rhs.result)) {
        return false;
    }

    const bool no_worse = lhs.cost.matmul_fma_count <= rhs.cost.matmul_fma_count &&
                          lhs.cost.fused_elementwise_count <= rhs.cost.fused_elementwise_count &&
                          lhs.cost.reduction_input_elements <= rhs.cost.reduction_input_elements &&
                          lhs.cost.fused_kernel_count <= rhs.cost.fused_kernel_count &&
                          lhs.cost.reduction_op_count <= rhs.cost.reduction_op_count &&
                          lhs.cost.materialization_op_count <= rhs.cost.materialization_op_count &&
                          totalPlannerMaterializationElements(lhs.cost) <=
                              totalPlannerMaterializationElements(rhs.cost) &&
                          lhs.cost.peak_temporary_elements <= rhs.cost.peak_temporary_elements &&
                          lhs.cost.matmul_group_count <= rhs.cost.matmul_group_count;
    const bool strictly_better = lhs.cost.matmul_fma_count < rhs.cost.matmul_fma_count ||
                                 lhs.cost.fused_elementwise_count < rhs.cost.fused_elementwise_count ||
                                 lhs.cost.reduction_input_elements < rhs.cost.reduction_input_elements ||
                                 lhs.cost.fused_kernel_count < rhs.cost.fused_kernel_count ||
                                 lhs.cost.reduction_op_count < rhs.cost.reduction_op_count ||
                                 lhs.cost.materialization_op_count < rhs.cost.materialization_op_count ||
                                 totalPlannerMaterializationElements(lhs.cost) <
                                     totalPlannerMaterializationElements(rhs.cost) ||
                                 lhs.cost.peak_temporary_elements < rhs.cost.peak_temporary_elements ||
                                 lhs.cost.matmul_group_count < rhs.cost.matmul_group_count;
    return no_worse && strictly_better;
}

bool plannerCandidateCostEqual(const EinsumPairPhysicalCandidate& lhs,
                               const EinsumPairPhysicalCandidate& rhs) {
    return samePlannerPhysicalResult(lhs.result, rhs.result) &&
           lhs.cost.matmul_fma_count == rhs.cost.matmul_fma_count &&
           lhs.cost.fused_elementwise_count == rhs.cost.fused_elementwise_count &&
           lhs.cost.reduction_input_elements == rhs.cost.reduction_input_elements &&
           lhs.cost.fused_kernel_count == rhs.cost.fused_kernel_count &&
           lhs.cost.reduction_op_count == rhs.cost.reduction_op_count &&
           lhs.cost.materialization_op_count == rhs.cost.materialization_op_count &&
           totalPlannerMaterializationElements(lhs.cost) == totalPlannerMaterializationElements(rhs.cost) &&
           lhs.cost.peak_temporary_elements == rhs.cost.peak_temporary_elements &&
           lhs.cost.matmul_group_count == rhs.cost.matmul_group_count;
}

void appendPlannerCandidateWithDominancePruning(
    std::vector<EinsumPairPhysicalCandidate>& candidates,
    EinsumPairPhysicalCandidate candidate) {
    for (const EinsumPairPhysicalCandidate& existing : candidates) {
        if (plannerCandidateDominates(existing, candidate) || plannerCandidateCostEqual(existing, candidate)) {
            return;
        }
    }
    candidates.erase(std::remove_if(candidates.begin(),
                                    candidates.end(),
                                    [&](const EinsumPairPhysicalCandidate& existing) {
                                        return plannerCandidateDominates(candidate, existing);
                                    }),
                     candidates.end());
    candidates.push_back(std::move(candidate));
}

EinsumPairPhysicalCandidate plannerPairProductCandidate(
    EinsumPlanKind kind,
    const EinsumPairProductPlan& pair_product,
    const EinsumLogicalOperandPlan& lhs,
    const EinsumLogicalOperandPlan& rhs,
    const std::vector<int32_t>& surviving_labels,
    const std::vector<uint64_t>& label_dimensions,
    std::optional<DataType> dtype,
    const std::vector<uint32_t>& provenance) {
    EinsumPairPhysicalCandidate candidate;
    candidate.kind = kind;
    candidate.pair_product = pair_product;
    candidate.physical_result_labels = surviving_labels;
    candidate.result = densePlannerPairResult(surviving_labels, label_dimensions, dtype, provenance);

    candidate.cost.result_elements = logicalElementCount(candidate.result);
    candidate.cost.fused_elementwise_count = candidate.cost.result_elements;
    candidate.cost.fused_kernel_count = 1;
    if (!pair_product.lhs_reduction_labels.empty()) {
        candidate.cost.reduction_input_elements = checkedAddPhysicalCost(
            candidate.cost.reduction_input_elements, logicalElementCount(lhs), "reduction work");
        candidate.cost.reduction_op_count = checkedAddPhysicalCost(
            candidate.cost.reduction_op_count, 1, "reduction operation count");
    }
    if (!pair_product.rhs_reduction_labels.empty()) {
        candidate.cost.reduction_input_elements = checkedAddPhysicalCost(
            candidate.cost.reduction_input_elements, logicalElementCount(rhs), "reduction work");
        candidate.cost.reduction_op_count = checkedAddPhysicalCost(
            candidate.cost.reduction_op_count, 1, "reduction operation count");
    }

    candidate.cost.peak_temporary_elements = candidate.cost.result_elements;
    if (!pair_product.lhs_reduction_labels.empty()) {
        candidate.cost.peak_temporary_elements =
            std::max(candidate.cost.peak_temporary_elements,
                     reducedPlannerOperandResultElements(
                         lhs, pair_product.lhs_reduction_labels, pair_product.lhs_broadcast_elision_labels));
    }
    if (!pair_product.rhs_reduction_labels.empty()) {
        candidate.cost.peak_temporary_elements =
            std::max(candidate.cost.peak_temporary_elements,
                     reducedPlannerOperandResultElements(
                         rhs, pair_product.rhs_reduction_labels, pair_product.rhs_broadcast_elision_labels));
    }

    populatePhysicalByteCosts(candidate.cost, dtype);
    return candidate;
}

EinsumPairPhysicalCandidate plannerGenericPairCandidate(
    const EinsumLogicalOperandPlan& lhs,
    const EinsumLogicalOperandPlan& rhs,
    const std::vector<int32_t>& surviving_labels,
    const std::vector<uint64_t>& label_dimensions,
    const std::vector<int32_t>& reduction_labels,
    std::optional<DataType> dtype,
    const std::vector<uint32_t>& provenance) {
    EinsumPairPhysicalCandidate candidate;
    candidate.kind = EinsumPlanKind::GENERAL;
    candidate.physical_result_labels = surviving_labels;
    candidate.result = densePlannerPairResult(surviving_labels, label_dimensions, dtype, provenance);

    std::unordered_set<int32_t> all_labels;
    all_labels.insert(lhs.labels.begin(), lhs.labels.end());
    all_labels.insert(rhs.labels.begin(), rhs.labels.end());
    uint64_t iteration_elements = 1;
    for (int32_t label : all_labels) {
        iteration_elements = checkedMultiplyPhysicalCost(
            iteration_elements,
            label_dimensions.at(static_cast<size_t>(label)),
            "generic pair iteration element count");
    }

    candidate.cost.fused_elementwise_count = iteration_elements;
    candidate.cost.fused_kernel_count = 1;
    candidate.cost.reduction_input_elements = reduction_labels.empty() ? 0 : iteration_elements;
    candidate.cost.reduction_op_count = reduction_labels.empty() ? 0 : 1;
    candidate.cost.result_elements = logicalElementCount(candidate.result);
    candidate.cost.peak_temporary_elements =
        reduction_labels.empty() ? candidate.cost.result_elements : iteration_elements;
    populatePhysicalByteCosts(candidate.cost, dtype);
    return candidate;
}

void populatePlannerPhysicalCandidates(EinsumPairContractionPlan& plan,
                                       const EinsumLogicalOperandPlan& lhs,
                                       const EinsumLogicalOperandPlan& rhs) {
    plan.physical_candidates.clear();
    const std::optional<DataType> dtype = pairResultStorageDtype(lhs, rhs);
    const std::vector<uint32_t> provenance = mergePairProvenance(lhs, rhs);

    if (plan.matrix_multiply.has_value()) {
        const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
        for (bool swapped_orientation : {false, true}) {
            const std::optional<PlannerMatrixPhysicalLayout> physical_layout =
                planPlannerMatrixPhysicalLayout(
                    matrix, lhs, rhs, plan.pair_label_dimensions, swapped_orientation);
            if (!physical_layout.has_value()) {
                continue;
            }

            const std::vector<int32_t> physical_labels =
                plannerMatrixPhysicalOutputLabels(matrix, swapped_orientation);
            appendPlannerCandidateWithDominancePruning(
                plan.physical_candidates,
                plannerMatrixCandidate(matrix,
                                       lhs,
                                       rhs,
                                       plan.surviving_labels,
                                       plan.pair_label_dimensions,
                                       *physical_layout,
                                       swapped_orientation,
                                       false,
                                       dtype,
                                       provenance));
            if (physical_labels != plan.surviving_labels) {
                appendPlannerCandidateWithDominancePruning(
                    plan.physical_candidates,
                    plannerMatrixCandidate(matrix,
                                           lhs,
                                           rhs,
                                           plan.surviving_labels,
                                           plan.pair_label_dimensions,
                                           *physical_layout,
                                           swapped_orientation,
                                           true,
                                           dtype,
                                           provenance));
            }
        }
    } else if (plan.pair_product.has_value()) {
        appendPlannerCandidateWithDominancePruning(
            plan.physical_candidates,
            plannerPairProductCandidate(
                plan.kind, *plan.pair_product, lhs, rhs, plan.surviving_labels, plan.pair_label_dimensions, dtype, provenance));
    }

    if (plan.physical_candidates.empty()) {
        plan.physical_candidates.push_back(plannerGenericPairCandidate(lhs,
                                                                       rhs,
                                                                       plan.surviving_labels,
                                                                       plan.pair_label_dimensions,
                                                                       plan.reduction_labels,
                                                                       dtype,
                                                                       provenance));
    }

    // Preserve the exact current two-operand execution policy as the preferred
    // candidate: use swapped GEMM only for the complete [batch,N,M] output
    // group swap, otherwise use natural GEMM orientation and materialize a
    // general requested permutation.
    size_t preferred_index = 0;
    if (plan.matrix_multiply.has_value()) {
        const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
        const std::vector<int32_t> swapped_labels =
            plannerMatrixPhysicalOutputLabels(matrix, true);
        const bool prefer_swapped =
            matrix.requires_output_permutation && swapped_labels == plan.surviving_labels;
        const std::vector<int32_t> preferred_physical_labels =
            plannerMatrixPhysicalOutputLabels(matrix, prefer_swapped);
        const bool prefer_output_materialization =
            preferred_physical_labels != plan.surviving_labels;

        bool found = false;
        for (size_t index = 0; index < plan.physical_candidates.size(); ++index) {
            const EinsumPairPhysicalCandidate& candidate = plan.physical_candidates[index];
            if (candidate.kind == EinsumPlanKind::GENERAL) {
                continue;
            }
            if (candidate.swapped_gemm_orientation == prefer_swapped &&
                candidate.output_materialized == prefer_output_materialization &&
                candidate.result.dense_storage) {
                preferred_index = index;
                found = true;
                break;
            }
        }
        if (!found) {
            // Dominance pruning may have removed an equivalent orientation. In
            // that case prefer any dense candidate matching the current logical
            // result, otherwise the only viable fallback candidate.
            for (size_t index = 0; index < plan.physical_candidates.size(); ++index) {
                if (plan.physical_candidates[index].result.dense_storage) {
                    preferred_index = index;
                    found = true;
                    break;
                }
            }
            if (!found) {
                preferred_index = 0;
            }
        }
    }
    plan.preferred_physical_candidate = static_cast<uint32_t>(preferred_index);
}

// Broad primitive-class weights calibrated from production Thor backends on a
// modern GPU, then intentionally rounded to architecture-level powers of two.
// The planner should model fundamental work/traffic classes, not cuBLASLt
// shape-specific efficiency on one GPU. result writes are charged separately,
// so the fused weight excludes the output write measured by the calibration.
constexpr uint64_t kExactFusedElementwiseWeight = 128;
constexpr uint64_t kExactReductionWeight = 64;
constexpr uint64_t kExactMaterializationWeight = 128;
constexpr uint64_t kExactResultWriteWeight = 64;

uint64_t weightedExactCost(uint64_t value, uint64_t weight, const char* description) {
    return checkedMultiplyPhysicalCost(value, weight, description);
}

uint64_t estimatedExactExecutionUnits(const EinsumExactContractionCost& cost) {
    uint64_t result = cost.matmul_fma_count;
    result = checkedAddPhysicalCost(
        result,
        weightedExactCost(cost.fused_elementwise_count,
                          kExactFusedElementwiseWeight,
                          "exact fused-elementwise weighted cost"),
        "exact estimated execution cost");
    result = checkedAddPhysicalCost(
        result,
        weightedExactCost(cost.reduction_input_elements,
                          kExactReductionWeight,
                          "exact reduction weighted cost"),
        "exact estimated execution cost");
    result = checkedAddPhysicalCost(
        result,
        weightedExactCost(cost.materialization_elements,
                          kExactMaterializationWeight,
                          "exact materialization weighted cost"),
        "exact estimated execution cost");
    result = checkedAddPhysicalCost(
        result,
        weightedExactCost(cost.result_write_elements,
                          kExactResultWriteWeight,
                          "exact result-write weighted cost"),
        "exact estimated execution cost");
    return result;
}

EinsumExactContractionCost combineExactCosts(const EinsumExactContractionCost& lhs,
                                             const EinsumExactContractionCost& rhs,
                                             const EinsumPairPhysicalCandidate& candidate) {
    EinsumExactContractionCost result;
    result.matmul_fma_count = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.matmul_fma_count, rhs.matmul_fma_count, "exact matmul FMA count"),
        candidate.cost.matmul_fma_count,
        "exact matmul FMA count");
    result.fused_elementwise_count = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.fused_elementwise_count,
                               rhs.fused_elementwise_count,
                               "exact fused-elementwise count"),
        candidate.cost.fused_elementwise_count,
        "exact fused-elementwise count");
    result.reduction_input_elements = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.reduction_input_elements,
                               rhs.reduction_input_elements,
                               "exact reduction work"),
        candidate.cost.reduction_input_elements,
        "exact reduction work");
    result.materialization_elements = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.materialization_elements,
                               rhs.materialization_elements,
                               "exact materialization count"),
        totalPlannerMaterializationElements(candidate.cost),
        "exact materialization count");
    result.result_write_elements = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.result_write_elements,
                               rhs.result_write_elements,
                               "exact result-write count"),
        candidate.cost.result_elements,
        "exact result-write count");
    result.matmul_group_count = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.matmul_group_count,
                               rhs.matmul_group_count,
                               "exact matmul group count"),
        candidate.cost.matmul_group_count,
        "exact matmul group count");
    result.fused_kernel_count = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.fused_kernel_count,
                               rhs.fused_kernel_count,
                               "exact fused-kernel count"),
        candidate.cost.fused_kernel_count,
        "exact fused-kernel count");
    result.reduction_op_count = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.reduction_op_count,
                               rhs.reduction_op_count,
                               "exact reduction-operation count"),
        candidate.cost.reduction_op_count,
        "exact reduction-operation count");
    result.materialization_op_count = checkedAddPhysicalCost(
        checkedAddPhysicalCost(lhs.materialization_op_count,
                               rhs.materialization_op_count,
                               "exact materialization-operation count"),
        candidate.cost.materialization_op_count,
        "exact materialization-operation count");

    result.peak_temporary_elements =
        std::max({lhs.peak_temporary_elements,
                  rhs.peak_temporary_elements,
                  candidate.cost.peak_temporary_elements});
    result.peak_intermediate_elements =
        std::max({lhs.peak_intermediate_elements,
                  rhs.peak_intermediate_elements,
                  candidate.cost.result_elements});
    result.estimated_execution_units = estimatedExactExecutionUnits(result);
    return result;
}

bool exactCostNoWorse(const EinsumExactContractionCost& lhs,
                      const EinsumExactContractionCost& rhs) {
    return lhs.matmul_fma_count <= rhs.matmul_fma_count &&
           lhs.fused_elementwise_count <= rhs.fused_elementwise_count &&
           lhs.reduction_input_elements <= rhs.reduction_input_elements &&
           lhs.materialization_elements <= rhs.materialization_elements &&
           lhs.result_write_elements <= rhs.result_write_elements &&
           lhs.matmul_group_count <= rhs.matmul_group_count &&
           lhs.fused_kernel_count <= rhs.fused_kernel_count &&
           lhs.reduction_op_count <= rhs.reduction_op_count &&
           lhs.materialization_op_count <= rhs.materialization_op_count &&
           lhs.peak_temporary_elements <= rhs.peak_temporary_elements &&
           lhs.peak_intermediate_elements <= rhs.peak_intermediate_elements;
}

bool exactCostStrictlyBetterComponent(const EinsumExactContractionCost& lhs,
                                      const EinsumExactContractionCost& rhs) {
    return lhs.matmul_fma_count < rhs.matmul_fma_count ||
           lhs.fused_elementwise_count < rhs.fused_elementwise_count ||
           lhs.reduction_input_elements < rhs.reduction_input_elements ||
           lhs.materialization_elements < rhs.materialization_elements ||
           lhs.result_write_elements < rhs.result_write_elements ||
           lhs.matmul_group_count < rhs.matmul_group_count ||
           lhs.fused_kernel_count < rhs.fused_kernel_count ||
           lhs.reduction_op_count < rhs.reduction_op_count ||
           lhs.materialization_op_count < rhs.materialization_op_count ||
           lhs.peak_temporary_elements < rhs.peak_temporary_elements ||
           lhs.peak_intermediate_elements < rhs.peak_intermediate_elements;
}

bool exactCostDominates(const EinsumExactContractionCost& lhs,
                        const EinsumExactContractionCost& rhs) {
    return exactCostNoWorse(lhs, rhs) && exactCostStrictlyBetterComponent(lhs, rhs);
}

bool exactCostEqual(const EinsumExactContractionCost& lhs,
                    const EinsumExactContractionCost& rhs) {
    return lhs.matmul_fma_count == rhs.matmul_fma_count &&
           lhs.fused_elementwise_count == rhs.fused_elementwise_count &&
           lhs.reduction_input_elements == rhs.reduction_input_elements &&
           lhs.materialization_elements == rhs.materialization_elements &&
           lhs.result_write_elements == rhs.result_write_elements &&
           lhs.matmul_group_count == rhs.matmul_group_count &&
           lhs.fused_kernel_count == rhs.fused_kernel_count &&
           lhs.reduction_op_count == rhs.reduction_op_count &&
           lhs.materialization_op_count == rhs.materialization_op_count &&
           lhs.peak_temporary_elements == rhs.peak_temporary_elements &&
           lhs.peak_intermediate_elements == rhs.peak_intermediate_elements;
}

bool exactCostPreferred(const EinsumExactContractionCost& lhs,
                        const EinsumExactContractionCost& rhs) {
    if (lhs.estimated_execution_units != rhs.estimated_execution_units) {
        return lhs.estimated_execution_units < rhs.estimated_execution_units;
    }
    if (lhs.peak_intermediate_elements != rhs.peak_intermediate_elements) {
        return lhs.peak_intermediate_elements < rhs.peak_intermediate_elements;
    }
    if (lhs.materialization_elements != rhs.materialization_elements) {
        return lhs.materialization_elements < rhs.materialization_elements;
    }
    if (lhs.peak_temporary_elements != rhs.peak_temporary_elements) {
        return lhs.peak_temporary_elements < rhs.peak_temporary_elements;
    }
    if (lhs.matmul_group_count != rhs.matmul_group_count) {
        return lhs.matmul_group_count < rhs.matmul_group_count;
    }
    const uint64_t lhs_non_matmul_ops = lhs.fused_kernel_count + lhs.reduction_op_count +
                                        lhs.materialization_op_count;
    const uint64_t rhs_non_matmul_ops = rhs.fused_kernel_count + rhs.reduction_op_count +
                                        rhs.materialization_op_count;
    if (lhs_non_matmul_ops != rhs_non_matmul_ops) {
        return lhs_non_matmul_ops < rhs_non_matmul_ops;
    }
    if (lhs.reduction_op_count != rhs.reduction_op_count) {
        return lhs.reduction_op_count < rhs.reduction_op_count;
    }
    if (lhs.materialization_op_count != rhs.materialization_op_count) {
        return lhs.materialization_op_count < rhs.materialization_op_count;
    }
    if (lhs.fused_kernel_count != rhs.fused_kernel_count) {
        return lhs.fused_kernel_count < rhs.fused_kernel_count;
    }
    if (lhs.matmul_fma_count != rhs.matmul_fma_count) {
        return lhs.matmul_fma_count < rhs.matmul_fma_count;
    }
    if (lhs.fused_elementwise_count != rhs.fused_elementwise_count) {
        return lhs.fused_elementwise_count < rhs.fused_elementwise_count;
    }
    if (lhs.reduction_input_elements != rhs.reduction_input_elements) {
        return lhs.reduction_input_elements < rhs.reduction_input_elements;
    }
    return lhs.result_write_elements < rhs.result_write_elements;
}

struct ExactPlannerRealization {
    EinsumLogicalOperandPlan result;
    EinsumExactContractionCost cost;
    std::vector<EinsumExactContractionStep> steps;
};

void appendExactRealizationWithDominancePruning(std::vector<ExactPlannerRealization>& realizations,
                                                ExactPlannerRealization candidate) {
    for (const ExactPlannerRealization& existing : realizations) {
        if (!samePlannerPhysicalResult(existing.result, candidate.result)) {
            continue;
        }
        if (exactCostDominates(existing.cost, candidate.cost) ||
            exactCostEqual(existing.cost, candidate.cost)) {
            return;
        }
    }

    realizations.erase(std::remove_if(realizations.begin(),
                                      realizations.end(),
                                      [&](const ExactPlannerRealization& existing) {
                                          return samePlannerPhysicalResult(existing.result, candidate.result) &&
                                                 exactCostDominates(candidate.cost, existing.cost);
                                      }),
                       realizations.end());
    realizations.push_back(std::move(candidate));
}

std::vector<uint64_t> exactLabelOperandMasks(const ResolvedEinsumEquation& equation) {
    std::vector<uint64_t> masks(equation.label_dimensions.size(), 0);
    for (size_t operand_index = 0; operand_index < equation.inputs.size(); ++operand_index) {
        const uint64_t operand_bit = uint64_t{1} << operand_index;
        for (int32_t label : equation.inputs[operand_index].axis_labels) {
            const size_t label_index = static_cast<size_t>(label);
            if (label < 0 || label_index >= masks.size()) {
                throw std::logic_error("Internal einsum exact planner error: invalid input label.");
            }
            masks[label_index] |= operand_bit;
        }
    }
    return masks;
}

std::vector<int32_t> exactSurvivingLabels(uint64_t subset_mask,
                                          uint64_t full_mask,
                                          const std::vector<int32_t>& iteration_labels,
                                          const std::vector<uint64_t>& label_operand_masks,
                                          const std::unordered_set<int32_t>& output_labels) {
    std::vector<int32_t> result;
    result.reserve(iteration_labels.size());
    const uint64_t outside_mask = full_mask ^ subset_mask;
    for (int32_t label : iteration_labels) {
        const size_t label_index = static_cast<size_t>(label);
        if (label < 0 || label_index >= label_operand_masks.size()) {
            throw std::logic_error("Internal einsum exact planner error: invalid iteration label.");
        }
        const uint64_t operand_mask = label_operand_masks[label_index];
        if ((operand_mask & subset_mask) == 0) {
            continue;
        }
        if (output_labels.contains(label) || (operand_mask & outside_mask) != 0) {
            result.push_back(label);
        }
    }
    return result;
}

std::vector<EinsumExactContractionStep> concatenateExactSteps(
    const std::vector<EinsumExactContractionStep>& lhs,
    const std::vector<EinsumExactContractionStep>& rhs,
    EinsumExactContractionStep current) {
    std::vector<EinsumExactContractionStep> result;
    result.reserve(lhs.size() + rhs.size() + 1);
    result.insert(result.end(), lhs.begin(), lhs.end());
    result.insert(result.end(), rhs.begin(), rhs.end());
    result.push_back(std::move(current));
    return result;
}

uint64_t exactSourceMask(const EinsumLogicalOperandPlan& operand, size_t source_operand_count) {
    uint64_t result = 0;
    for (uint32_t source_operand : operand.source_operand_indices) {
        if (source_operand >= source_operand_count ||
            source_operand >= EinsumPlanner::MAX_SOURCE_OPERANDS) {
            throw std::logic_error("Internal einsum exact planner error: invalid logical-operand provenance.");
        }
        result |= uint64_t{1} << source_operand;
    }
    if (result == 0) {
        throw std::logic_error("Internal einsum exact planner error: logical operand has empty provenance.");
    }
    return result;
}

std::vector<uint64_t> exactActiveSubsetSourceMasks(
    const std::vector<ExactPlannerRealization>& initial_realizations,
    size_t source_operand_count) {
    const size_t active_operand_count = initial_realizations.size();
    const uint64_t active_full_mask = (uint64_t{1} << active_operand_count) - 1;
    std::vector<uint64_t> leaf_source_masks(active_operand_count, 0);
    uint64_t covered_source_mask = 0;
    for (size_t active_index = 0; active_index < active_operand_count; ++active_index) {
        const uint64_t source_mask = exactSourceMask(initial_realizations[active_index].result,
                                                     source_operand_count);
        if ((covered_source_mask & source_mask) != 0) {
            throw std::logic_error("Internal einsum exact planner error: active operands have overlapping provenance.");
        }
        covered_source_mask |= source_mask;
        leaf_source_masks[active_index] = source_mask;
    }

    const uint64_t expected_source_mask = (uint64_t{1} << source_operand_count) - 1;
    if (covered_source_mask != expected_source_mask) {
        throw std::logic_error("Internal einsum exact planner error: active operands do not cover all source operands.");
    }

    std::vector<uint64_t> result(static_cast<size_t>(active_full_mask) + 1, 0);
    for (uint64_t active_mask = 1; active_mask <= active_full_mask; ++active_mask) {
        uint64_t source_mask = 0;
        for (size_t active_index = 0; active_index < active_operand_count; ++active_index) {
            if ((active_mask & (uint64_t{1} << active_index)) != 0) {
                source_mask |= leaf_source_masks[active_index];
            }
        }
        result[static_cast<size_t>(active_mask)] = source_mask;
    }
    return result;
}

std::optional<EinsumExactContractionPlan> planExactActiveContraction(
    const ResolvedEinsumEquation& equation,
    const std::vector<ExactPlannerRealization>& initial_realizations,
    const std::vector<int32_t>& iteration_labels) {
    const size_t active_operand_count = initial_realizations.size();
    if (active_operand_count < 2 || active_operand_count > EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS) {
        return std::nullopt;
    }
    const size_t source_operand_count = equation.inputs.size();
    if (source_operand_count == 0 || source_operand_count > EinsumPlanner::MAX_SOURCE_OPERANDS) {
        throw std::invalid_argument(
            "Einsum exact planner supports at most " +
            std::to_string(EinsumPlanner::MAX_SOURCE_OPERANDS) +
            " source operands.");
    }

    const uint64_t active_full_mask = (uint64_t{1} << active_operand_count) - 1;
    const uint64_t source_full_mask = (uint64_t{1} << source_operand_count) - 1;
    const std::vector<uint64_t> active_subset_source_masks =
        exactActiveSubsetSourceMasks(initial_realizations, source_operand_count);
    const std::vector<uint64_t> label_operand_masks = exactLabelOperandMasks(equation);
    const std::unordered_set<int32_t> output_labels(equation.output_labels.begin(),
                                                    equation.output_labels.end());
    bool saw_overflowed_realization = false;

    std::vector<std::vector<ExactPlannerRealization>> states(static_cast<size_t>(active_full_mask) + 1);
    for (size_t active_index = 0; active_index < active_operand_count; ++active_index) {
        const uint64_t mask = uint64_t{1} << active_index;
        states[static_cast<size_t>(mask)].push_back(initial_realizations[active_index]);
    }

    for (uint64_t mask = 1; mask <= active_full_mask; ++mask) {
        if ((mask & (mask - 1)) == 0) {
            continue;
        }

        const uint64_t source_subset_mask = active_subset_source_masks[static_cast<size_t>(mask)];
        const std::vector<int32_t> surviving_labels =
            exactSurvivingLabels(source_subset_mask,
                                 source_full_mask,
                                 iteration_labels,
                                 label_operand_masks,
                                 output_labels);

        for (uint64_t lhs_mask = (mask - 1) & mask; lhs_mask != 0; lhs_mask = (lhs_mask - 1) & mask) {
            const uint64_t rhs_mask = mask ^ lhs_mask;
            if (rhs_mask == 0 || lhs_mask >= rhs_mask) {
                continue;
            }

            const auto try_orientation = [&](uint64_t oriented_lhs_mask,
                                             uint64_t oriented_rhs_mask) {
                const auto& lhs_realizations = states[static_cast<size_t>(oriented_lhs_mask)];
                const auto& rhs_realizations = states[static_cast<size_t>(oriented_rhs_mask)];
                for (const ExactPlannerRealization& lhs : lhs_realizations) {
                    for (const ExactPlannerRealization& rhs : rhs_realizations) {
                        try {
                            const EinsumPairContractionPlan pair =
                                EinsumPlanner::planPair(lhs.result, rhs.result, surviving_labels);
                            for (const EinsumPairPhysicalCandidate& physical_candidate :
                                 pair.physical_candidates) {
                                try {
                                    ExactPlannerRealization realization;
                                    realization.result = physical_candidate.result;
                                    realization.cost = combineExactCosts(lhs.cost, rhs.cost, physical_candidate);

                                    EinsumExactContractionStep step;
                                    step.lhs_source_mask =
                                        active_subset_source_masks[static_cast<size_t>(oriented_lhs_mask)];
                                    step.rhs_source_mask =
                                        active_subset_source_masks[static_cast<size_t>(oriented_rhs_mask)];
                                    step.result_source_mask = source_subset_mask;
                                    step.lhs = lhs.result;
                                    step.rhs = rhs.result;
                                    step.surviving_labels = surviving_labels;
                                    step.eliminated_labels = pair.reduction_labels;
                                    step.physical_candidate = physical_candidate;
                                    step.incremental_estimated_execution_units =
                                        combineExactCosts({}, {}, physical_candidate).estimated_execution_units;
                                    step.cumulative_cost = realization.cost;
                                    realization.steps = concatenateExactSteps(
                                        lhs.steps, rhs.steps, std::move(step));

                                    appendExactRealizationWithDominancePruning(
                                        states[static_cast<size_t>(mask)], std::move(realization));
                                } catch (const std::overflow_error&) {
                                    // Exact search is speculative. A physically enormous
                                    // contraction order that cannot be represented in the
                                    // uint64_t cost model is simply infeasible; it must not
                                    // poison a different, representable tree.
                                    saw_overflowed_realization = true;
                                }
                            }
                        } catch (const std::overflow_error&) {
                            saw_overflowed_realization = true;
                        }
                    }
                }
            };

            try_orientation(lhs_mask, rhs_mask);
            try_orientation(rhs_mask, lhs_mask);
        }

        // Some speculative subsets can be physically/cost-wise unrepresentable
        // even though a complete tree that never forms that subset is valid.
        // Leave those states empty and let the root determine whether planning
        // as a whole succeeded.
    }

    const std::vector<ExactPlannerRealization>& roots = states[static_cast<size_t>(active_full_mask)];
    const ExactPlannerRealization* best = nullptr;
    for (const ExactPlannerRealization& root : roots) {
        // Thor's public einsum result is a dense tensor in requested output
        // order. Intermediate states may remain arbitrary zero-copy views, but
        // a root physical realization must satisfy that final contract.
        if (!root.result.dense_storage || root.result.labels != equation.output_labels) {
            continue;
        }
        if (best == nullptr || exactCostPreferred(root.cost, best->cost)) {
            best = &root;
        }
    }
    if (best == nullptr) {
        if (saw_overflowed_realization) {
            throw std::overflow_error(
                "Einsum exact planner has no complete contraction tree representable by the uint64_t cost model.");
        }
        throw std::logic_error("Internal einsum exact planner error: no dense final-output realization.");
    }

    EinsumExactContractionPlan result;
    result.steps = best->steps;
    result.result = best->result;
    result.cost = best->cost;
    return result;
}

std::optional<EinsumExactContractionPlan> planExactSmallContraction(
    const ResolvedEinsumEquation& equation,
    const std::vector<EinsumLogicalOperandPlan>& logical_operands,
    const std::vector<int32_t>& iteration_labels) {
    const size_t operand_count = logical_operands.size();
    if (operand_count < 3 || operand_count > EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS) {
        return std::nullopt;
    }

    std::vector<ExactPlannerRealization> initial_realizations;
    initial_realizations.reserve(operand_count);
    for (const EinsumLogicalOperandPlan& operand : logical_operands) {
        initial_realizations.push_back(ExactPlannerRealization{operand, {}, {}});
    }

    std::optional<EinsumExactContractionPlan> result =
        planExactActiveContraction(equation, initial_realizations, iteration_labels);
    if (result.has_value()) {
        result->planning_mode = EinsumContractionPlanningMode::EXACT;
        result->bridge_seed_pair_mask = 0;
    }
    return result;
}

std::optional<EinsumExactContractionPlan> planSixOperandBridge(
    const ResolvedEinsumEquation& equation,
    const std::vector<EinsumLogicalOperandPlan>& logical_operands,
    const std::vector<int32_t>& iteration_labels) {
    if (logical_operands.size() != EinsumPlanner::MAX_BRIDGED_ACTIVE_OPERANDS ||
        equation.inputs.size() != logical_operands.size()) {
        return std::nullopt;
    }

    const size_t operand_count = logical_operands.size();
    const uint64_t source_full_mask = (uint64_t{1} << operand_count) - 1;
    const std::vector<uint64_t> label_operand_masks = exactLabelOperandMasks(equation);
    const std::unordered_set<int32_t> output_labels(equation.output_labels.begin(),
                                                    equation.output_labels.end());

    std::optional<EinsumExactContractionPlan> best;
    bool saw_overflowed_realization = false;

    // There are exactly C(6,2)=15 bridge seed pairs.  For each unordered pair,
    // retain both operand orientations and every viable physical pair result,
    // then solve the resulting five-active-operand problem with the exhaustive
    // planner above. This intentionally avoids a separate six-operand DP while
    // preserving all physical-layout information across the bridge boundary.
    for (size_t first = 0; first < operand_count; ++first) {
        for (size_t second = first + 1; second < operand_count; ++second) {
            const uint64_t first_bit = uint64_t{1} << first;
            const uint64_t second_bit = uint64_t{1} << second;
            const uint64_t seed_pair_mask = first_bit | second_bit;
            const std::vector<int32_t> surviving_labels =
                exactSurvivingLabels(seed_pair_mask,
                                     source_full_mask,
                                     iteration_labels,
                                     label_operand_masks,
                                     output_labels);

            std::vector<ExactPlannerRealization> seed_realizations;
            const auto collect_orientation = [&](size_t lhs_index, size_t rhs_index) {
                try {
                    const EinsumPairContractionPlan pair =
                        EinsumPlanner::planPair(logical_operands[lhs_index],
                                                logical_operands[rhs_index],
                                                surviving_labels);
                    for (const EinsumPairPhysicalCandidate& physical_candidate : pair.physical_candidates) {
                        try {
                            const EinsumExactContractionCost seed_cost =
                                combineExactCosts({}, {}, physical_candidate);

                            EinsumExactContractionStep seed_step;
                            seed_step.lhs_source_mask = uint64_t{1} << lhs_index;
                            seed_step.rhs_source_mask = uint64_t{1} << rhs_index;
                            seed_step.result_source_mask = seed_pair_mask;
                            seed_step.lhs = logical_operands[lhs_index];
                            seed_step.rhs = logical_operands[rhs_index];
                            seed_step.surviving_labels = surviving_labels;
                            seed_step.eliminated_labels = pair.reduction_labels;
                            seed_step.physical_candidate = physical_candidate;
                            seed_step.incremental_estimated_execution_units =
                                seed_cost.estimated_execution_units;
                            seed_step.cumulative_cost = seed_cost;

                            appendExactRealizationWithDominancePruning(
                                seed_realizations,
                                ExactPlannerRealization{
                                    physical_candidate.result, seed_cost, {std::move(seed_step)}});
                        } catch (const std::overflow_error&) {
                            saw_overflowed_realization = true;
                        }
                    }
                } catch (const std::overflow_error&) {
                    saw_overflowed_realization = true;
                }
            };

            // Opposite operand orientations frequently rediscover the same
            // natural/swapped physical result layouts.  Deduplicate those seed
            // states before paying for an exact five-active-operand tail.
            collect_orientation(first, second);
            collect_orientation(second, first);

            for (const ExactPlannerRealization& seed : seed_realizations) {
                try {
                    std::vector<ExactPlannerRealization> active_leaves;
                    active_leaves.reserve(EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS);
                    active_leaves.push_back(seed);
                    for (size_t operand_index = 0; operand_index < operand_count; ++operand_index) {
                        if (operand_index == first || operand_index == second) {
                            continue;
                        }
                        active_leaves.push_back(
                            ExactPlannerRealization{logical_operands[operand_index], {}, {}});
                    }

                    std::optional<EinsumExactContractionPlan> tail =
                        planExactActiveContraction(equation, active_leaves, iteration_labels);
                    if (!tail.has_value()) {
                        continue;
                    }
                    tail->planning_mode = EinsumContractionPlanningMode::SIX_OPERAND_BRIDGE;
                    tail->bridge_seed_pair_mask = seed_pair_mask;
                    if (!best.has_value() || exactCostPreferred(tail->cost, best->cost)) {
                        best = std::move(tail);
                    }
                } catch (const std::overflow_error&) {
                    saw_overflowed_realization = true;
                }
            }
        }
    }

    if (!best.has_value()) {
        if (saw_overflowed_realization) {
            throw std::overflow_error(
                "Einsum six-operand bridge has no complete contraction tree representable by the uint64_t cost model.");
        }
        throw std::logic_error("Internal einsum six-operand bridge error: no complete contraction tree.");
    }
    return best;
}

struct BeamPlannerState {
    std::vector<ExactPlannerRealization> active;
    EinsumExactContractionCost committed_cost;
    std::string physical_signature;
};

EinsumExactContractionCost aggregateBeamForestCost(
    const std::vector<ExactPlannerRealization>& active) {
    EinsumExactContractionCost result;
    for (const ExactPlannerRealization& realization : active) {
        result.matmul_fma_count = checkedAddPhysicalCost(
            result.matmul_fma_count, realization.cost.matmul_fma_count, "beam matmul FMA count");
        result.fused_elementwise_count = checkedAddPhysicalCost(
            result.fused_elementwise_count,
            realization.cost.fused_elementwise_count,
            "beam fused-elementwise count");
        result.reduction_input_elements = checkedAddPhysicalCost(
            result.reduction_input_elements,
            realization.cost.reduction_input_elements,
            "beam reduction work");
        result.materialization_elements = checkedAddPhysicalCost(
            result.materialization_elements,
            realization.cost.materialization_elements,
            "beam materialization count");
        result.result_write_elements = checkedAddPhysicalCost(
            result.result_write_elements,
            realization.cost.result_write_elements,
            "beam result-write count");
        result.matmul_group_count = checkedAddPhysicalCost(
            result.matmul_group_count,
            realization.cost.matmul_group_count,
            "beam matmul group count");
        result.fused_kernel_count = checkedAddPhysicalCost(
            result.fused_kernel_count,
            realization.cost.fused_kernel_count,
            "beam fused-kernel count");
        result.reduction_op_count = checkedAddPhysicalCost(
            result.reduction_op_count,
            realization.cost.reduction_op_count,
            "beam reduction-operation count");
        result.materialization_op_count = checkedAddPhysicalCost(
            result.materialization_op_count,
            realization.cost.materialization_op_count,
            "beam materialization-operation count");
        result.peak_temporary_elements =
            std::max(result.peak_temporary_elements, realization.cost.peak_temporary_elements);
        result.peak_intermediate_elements =
            std::max(result.peak_intermediate_elements, realization.cost.peak_intermediate_elements);
    }
    result.estimated_execution_units = estimatedExactExecutionUnits(result);
    return result;
}

void canonicalizeBeamActive(std::vector<ExactPlannerRealization>& active,
                            size_t source_operand_count) {
    std::sort(active.begin(),
              active.end(),
              [&](const ExactPlannerRealization& lhs, const ExactPlannerRealization& rhs) {
                  return exactSourceMask(lhs.result, source_operand_count) <
                         exactSourceMask(rhs.result, source_operand_count);
              });
}

std::string beamPhysicalSignature(const std::vector<ExactPlannerRealization>& active,
                                  size_t source_operand_count) {
    const auto appendNumbers = [](std::ostringstream& out, const auto& values) {
        out << '[';
        for (size_t index = 0; index < values.size(); ++index) {
            if (index != 0) {
                out << ',';
            }
            out << values[index];
        }
        out << ']';
    };

    std::ostringstream out;
    for (const ExactPlannerRealization& realization : active) {
        const EinsumLogicalOperandPlan& operand = realization.result;
        out << exactSourceMask(operand, source_operand_count) << ':';
        appendNumbers(out, operand.labels);
        appendNumbers(out, operand.dimensions);
        appendNumbers(out, operand.strides_elements);
        appendNumbers(out, operand.source_operand_indices);
        out << ':';
        if (operand.storage_dtype.has_value()) {
            out << static_cast<int>(*operand.storage_dtype);
        } else {
            out << '-';
        }
        out << ':' << (operand.dense_storage ? 1 : 0)
            << ':' << (operand.diagonal_view ? 1 : 0) << ';';
    }
    return out.str();
}

BeamPlannerState makeBeamState(std::vector<ExactPlannerRealization> active,
                               size_t source_operand_count) {
    canonicalizeBeamActive(active, source_operand_count);
    BeamPlannerState result;
    result.active = std::move(active);
    result.committed_cost = aggregateBeamForestCost(result.active);
    result.physical_signature = beamPhysicalSignature(result.active, source_operand_count);
    return result;
}

bool beamStatePreferred(const BeamPlannerState& lhs, const BeamPlannerState& rhs) {
    if (exactCostPreferred(lhs.committed_cost, rhs.committed_cost)) {
        return true;
    }
    if (exactCostPreferred(rhs.committed_cost, lhs.committed_cost)) {
        return false;
    }
    return lhs.physical_signature < rhs.physical_signature;
}

std::optional<EinsumBeamContractionPlan> planBeamContraction(
    const ResolvedEinsumEquation& equation,
    const std::vector<EinsumLogicalOperandPlan>& logical_operands,
    const std::vector<int32_t>& iteration_labels,
    uint32_t beam_width) {
    const size_t source_operand_count = logical_operands.size();
    if (source_operand_count <= EinsumPlanner::MAX_BRIDGED_ACTIVE_OPERANDS ||
        source_operand_count > EinsumPlanner::MAX_BEAM_SOURCE_OPERANDS ||
        equation.inputs.size() != source_operand_count || beam_width == 0) {
        return std::nullopt;
    }

    const uint64_t source_full_mask = (uint64_t{1} << source_operand_count) - 1;
    const std::vector<uint64_t> label_operand_masks = exactLabelOperandMasks(equation);
    const std::unordered_set<int32_t> output_labels(equation.output_labels.begin(),
                                                    equation.output_labels.end());

    std::vector<ExactPlannerRealization> initial_active;
    initial_active.reserve(source_operand_count);
    for (const EinsumLogicalOperandPlan& operand : logical_operands) {
        initial_active.push_back(ExactPlannerRealization{operand, {}, {}});
    }

    EinsumBeamContractionPlan diagnostics;
    diagnostics.beam_width = beam_width;
    diagnostics.exact_tail_active_operands = EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS;

    std::vector<BeamPlannerState> beam;
    try {
        beam.push_back(makeBeamState(std::move(initial_active), source_operand_count));
    } catch (const std::overflow_error&) {
        return std::nullopt;
    }

    while (!beam.empty() &&
           beam.front().active.size() > EinsumPlanner::MAX_EXACT_ACTIVE_OPERANDS) {
        ++diagnostics.beam_levels;
        diagnostics.expanded_state_count = checkedAddPhysicalCost(
            diagnostics.expanded_state_count,
            static_cast<uint64_t>(beam.size()),
            "beam expanded-state count");

        std::vector<BeamPlannerState> next_states;
        std::unordered_map<std::string, size_t> signature_to_index;

        const auto append_state = [&](BeamPlannerState candidate) {
            diagnostics.generated_state_count = checkedAddPhysicalCost(
                diagnostics.generated_state_count, 1, "beam generated-state count");
            auto existing = signature_to_index.find(candidate.physical_signature);
            if (existing == signature_to_index.end()) {
                const size_t index = next_states.size();
                signature_to_index.emplace(candidate.physical_signature, index);
                next_states.push_back(std::move(candidate));
                return;
            }

            diagnostics.deduplicated_state_count = checkedAddPhysicalCost(
                diagnostics.deduplicated_state_count, 1, "beam deduplicated-state count");
            BeamPlannerState& current = next_states[existing->second];
            if (exactCostPreferred(candidate.committed_cost, current.committed_cost)) {
                current = std::move(candidate);
            }
        };

        for (const BeamPlannerState& state : beam) {
            const size_t active_count = state.active.size();
            for (size_t first = 0; first < active_count; ++first) {
                for (size_t second = first + 1; second < active_count; ++second) {
                    const uint64_t first_source_mask =
                        exactSourceMask(state.active[first].result, source_operand_count);
                    const uint64_t second_source_mask =
                        exactSourceMask(state.active[second].result, source_operand_count);
                    const uint64_t pair_source_mask = first_source_mask | second_source_mask;
                    const std::vector<int32_t> surviving_labels =
                        exactSurvivingLabels(pair_source_mask,
                                             source_full_mask,
                                             iteration_labels,
                                             label_operand_masks,
                                             output_labels);

                    const auto expand_orientation = [&](size_t lhs_index, size_t rhs_index) {
                        const ExactPlannerRealization& lhs = state.active[lhs_index];
                        const ExactPlannerRealization& rhs = state.active[rhs_index];
                        try {
                            const EinsumPairContractionPlan pair =
                                EinsumPlanner::planPair(lhs.result, rhs.result, surviving_labels);
                            for (const EinsumPairPhysicalCandidate& physical_candidate :
                                 pair.physical_candidates) {
                                try {
                                    ExactPlannerRealization merged;
                                    merged.result = physical_candidate.result;
                                    merged.cost = combineExactCosts(lhs.cost, rhs.cost, physical_candidate);

                                    EinsumExactContractionStep step;
                                    step.lhs_source_mask =
                                        exactSourceMask(lhs.result, source_operand_count);
                                    step.rhs_source_mask =
                                        exactSourceMask(rhs.result, source_operand_count);
                                    step.result_source_mask = pair_source_mask;
                                    step.lhs = lhs.result;
                                    step.rhs = rhs.result;
                                    step.surviving_labels = surviving_labels;
                                    step.eliminated_labels = pair.reduction_labels;
                                    step.physical_candidate = physical_candidate;
                                    step.incremental_estimated_execution_units =
                                        combineExactCosts({}, {}, physical_candidate)
                                            .estimated_execution_units;
                                    step.cumulative_cost = merged.cost;
                                    merged.steps = concatenateExactSteps(
                                        lhs.steps, rhs.steps, std::move(step));

                                    std::vector<ExactPlannerRealization> next_active;
                                    next_active.reserve(active_count - 1);
                                    for (size_t index = 0; index < active_count; ++index) {
                                        if (index == first || index == second) {
                                            continue;
                                        }
                                        next_active.push_back(state.active[index]);
                                    }
                                    next_active.push_back(std::move(merged));
                                    append_state(makeBeamState(std::move(next_active),
                                                               source_operand_count));
                                } catch (const std::overflow_error&) {
                                    // Beam planning remains speculative. An
                                    // unrepresentable branch is pruned; failure to
                                    // find any representable beam must not break the
                                    // whole-equation generic execution fallback.
                                }
                            }
                        } catch (const std::overflow_error&) {
                            // Same speculative-overflow policy as above.
                        }
                    };

                    expand_orientation(first, second);
                    expand_orientation(second, first);
                }
            }
        }

        if (next_states.empty()) {
            return std::nullopt;
        }

        std::sort(next_states.begin(), next_states.end(), beamStatePreferred);
        if (next_states.size() > beam_width) {
            next_states.resize(beam_width);
        }
        diagnostics.retained_state_count = checkedAddPhysicalCost(
            diagnostics.retained_state_count,
            static_cast<uint64_t>(next_states.size()),
            "beam retained-state count");
        beam = std::move(next_states);
    }

    std::optional<EinsumExactContractionPlan> best_tail;
    for (const BeamPlannerState& state : beam) {
        try {
            std::optional<EinsumExactContractionPlan> tail =
                planExactActiveContraction(equation, state.active, iteration_labels);
            if (!tail.has_value()) {
                continue;
            }
            diagnostics.exact_tail_count = checkedAddPhysicalCost(
                diagnostics.exact_tail_count, 1, "beam exact-tail count");
            if (!best_tail.has_value() || exactCostPreferred(tail->cost, best_tail->cost)) {
                best_tail = std::move(tail);
            }
        } catch (const std::overflow_error&) {
            // Leave execution on the generic path if every exact tail overflows
            // the planner cost model.
        }
    }

    if (!best_tail.has_value()) {
        return std::nullopt;
    }

    diagnostics.steps = std::move(best_tail->steps);
    diagnostics.result = std::move(best_tail->result);
    diagnostics.cost = best_tail->cost;
    return diagnostics;
}


}  // namespace


bool EinsumOperandPlan::requiresPermutation() const {
    return !isIdentityPermutation(permutation);
}

EinsumPairContractionPlan EinsumPlanner::planPair(const EinsumLogicalOperandPlan& lhs,
                                                       const EinsumLogicalOperandPlan& rhs,
                                                       const std::vector<int32_t>& surviving_labels) {
    validateLogicalOperand(lhs, "lhs");
    validateLogicalOperand(rhs, "rhs");
    requireDisjointPairProvenance(lhs, rhs);

    EinsumPairContractionPlan plan;
    plan.reduction_accumulation_dtype = DataType::FP32;
    plan.surviving_labels = surviving_labels;
    plan.pair_label_dimensions = buildPairLabelDimensions(lhs, rhs);
    plan.reduction_labels = pairReductionLabels(lhs, rhs, surviving_labels, plan.pair_label_dimensions);

    plan.matrix_multiply =
        tryBuildMatrixMultiplyPlan(lhs, rhs, surviving_labels, plan.reduction_labels, plan.pair_label_dimensions);
    if (!plan.matrix_multiply.has_value()) {
        plan.pair_product =
            tryBuildPairProductPlan(lhs, rhs, surviving_labels, plan.reduction_labels, plan.pair_label_dimensions);
    }

    if (plan.matrix_multiply.has_value()) {
        plan.kind =
            plan.matrix_multiply->batch_labels.empty() ? EinsumPlanKind::GEMM : EinsumPlanKind::BATCHED_GEMM;
    } else if (plan.pair_product.has_value() && !plan.reduction_labels.empty()) {
        plan.kind = EinsumPlanKind::PAIR_PRODUCT;
    } else if (plan.pair_product.has_value() || plan.reduction_labels.empty()) {
        plan.kind = EinsumPlanKind::ELEMENTWISE;
    } else {
        plan.kind = EinsumPlanKind::GENERAL;
    }

    plan.result.labels = surviving_labels;
    plan.result.dimensions = logicalDimensionsForLabels(surviving_labels, plan.pair_label_dimensions);
    plan.result.strides_elements = denseStridesForLogicalOperand(plan.result.dimensions);
    plan.result.storage_dtype = pairResultStorageDtype(lhs, rhs);
    plan.result.source_operand_indices = mergePairProvenance(lhs, rhs);
    plan.result.dense_storage = true;
    plan.result.diagonal_view = false;

    populatePlannerPhysicalCandidates(plan, lhs, rhs);
    return plan;
}

EinsumPlan EinsumPlanner::plan(const ResolvedEinsumEquation& equation,
                               const std::vector<std::vector<uint64_t>>& input_dimensions) {
    if (equation.inputs.size() > MAX_SOURCE_OPERANDS) {
        throw std::invalid_argument(
            "Einsum supports at most " + std::to_string(MAX_SOURCE_OPERANDS) +
            " operands; received " + std::to_string(equation.inputs.size()) + ".");
    }

    validateResolvedEquation(equation, input_dimensions);

    EinsumPlan plan;
    plan.equation = equation;
    plan.iteration_labels = equation.output_labels;
    plan.iteration_labels.insert(plan.iteration_labels.end(), equation.reduction_labels.begin(), equation.reduction_labels.end());
    plan.iteration_dimensions.reserve(plan.iteration_labels.size());
    for (int32_t label : plan.iteration_labels) {
        plan.iteration_dimensions.push_back(equation.label_dimensions.at(static_cast<size_t>(label)));
    }

    for (size_t axis = equation.output_labels.size(); axis < plan.iteration_labels.size(); ++axis) {
        plan.reduction_axes.push_back(static_cast<uint32_t>(axis));
    }

    plan.operands.reserve(equation.inputs.size());
    plan.logical_operands.reserve(equation.inputs.size());
    for (size_t operand = 0; operand < equation.inputs.size(); ++operand) {
        plan.operands.push_back(
            buildOperandPlan(equation.inputs[operand], input_dimensions[operand], plan.iteration_labels, plan.iteration_dimensions));
        plan.logical_operands.push_back(buildLogicalInputOperandPlan(equation.inputs[operand],
                                                                    input_dimensions[operand],
                                                                    plan.operands.back(),
                                                                    static_cast<uint32_t>(operand)));
    }

    if (equation.inputs.size() == 2) {
        plan.pair_contraction = planPair(plan.logical_operands[0], plan.logical_operands[1], equation.output_labels);
        plan.matrix_multiply = plan.pair_contraction->matrix_multiply;
        plan.pair_product = plan.pair_contraction->pair_product;
        plan.kind = plan.pair_contraction->kind;
    } else if (equation.inputs.size() == 1) {
        plan.kind = equation.reduction_labels.empty() ? EinsumPlanKind::UNARY : EinsumPlanKind::REDUCTION;
    } else {
        // Three through five active operands use exhaustive subset-DP search.
        // Six uses the bounded first-pair bridge into that same exact five-
        // active-operand engine. Seven through 63 are planned by bounded beam
        // search with an exact five-active-operand tail; runtime consumes the
        // selected postorder tree through the same pair lowering machinery.
        if (equation.inputs.size() <= MAX_EXACT_ACTIVE_OPERANDS) {
            plan.exact_contraction =
                planExactSmallContraction(equation, plan.logical_operands, plan.iteration_labels);
        } else if (equation.inputs.size() == MAX_BRIDGED_ACTIVE_OPERANDS) {
            plan.exact_contraction =
                planSixOperandBridge(equation, plan.logical_operands, plan.iteration_labels);
        } else if (equation.inputs.size() <= MAX_BEAM_SOURCE_OPERANDS) {
            plan.beam_contraction =
                planBeamContraction(equation,
                                    plan.logical_operands,
                                    plan.iteration_labels,
                                    DEFAULT_BEAM_WIDTH);
        }
        plan.kind = equation.reduction_labels.empty() ? EinsumPlanKind::ELEMENTWISE : EinsumPlanKind::GENERAL;
    }

    return plan;
}

EinsumPlan EinsumPlanner::parseAndPlan(const std::string& equation,
                                       const std::vector<std::vector<uint64_t>>& input_dimensions) {
    return plan(EinsumParser::parseAndResolve(equation, input_dimensions), input_dimensions);
}

std::string EinsumPlanner::describeExactContraction(const EinsumPlan& plan) {
    if (!plan.exact_contraction.has_value()) {
        return {};
    }

    const auto labelName = [](int32_t label) -> std::string {
        if (EinsumParser::isEllipsisLabel(label)) {
            return "..." + std::to_string(label - EinsumParser::kEllipsisLabelBase);
        }
        return std::string(1, EinsumParser::labelCharacter(label));
    };
    const auto labelsText = [&](const std::vector<int32_t>& labels) {
        std::ostringstream out;
        out << '[';
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i != 0) out << ',';
            out << labelName(labels[i]);
        }
        out << ']';
        return out.str();
    };
    const auto numbersText = [](const auto& values) {
        std::ostringstream out;
        out << '[';
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) out << ',';
            out << values[i];
        }
        out << ']';
        return out.str();
    };
    const auto kindText = [](EinsumPlanKind kind) -> const char* {
        switch (kind) {
            case EinsumPlanKind::UNARY: return "UNARY";
            case EinsumPlanKind::ELEMENTWISE: return "ELEMENTWISE";
            case EinsumPlanKind::REDUCTION: return "REDUCTION";
            case EinsumPlanKind::GEMM: return "GEMM";
            case EinsumPlanKind::BATCHED_GEMM: return "BATCHED_GEMM";
            case EinsumPlanKind::GENERAL: return "GENERAL";
            case EinsumPlanKind::PAIR_PRODUCT: return "PAIR_PRODUCT";
        }
        return "UNKNOWN";
    };
    const auto operandText = [&](const EinsumLogicalOperandPlan& operand) {
        std::ostringstream out;
        out << "sources=" << numbersText(operand.source_operand_indices)
            << " labels=" << labelsText(operand.labels)
            << " dims=" << numbersText(operand.dimensions)
            << " strides=" << numbersText(operand.strides_elements)
            << " dense=" << (operand.dense_storage ? 1 : 0)
            << " diagonal=" << (operand.diagonal_view ? 1 : 0);
        return out.str();
    };
    const auto costText = [](const EinsumExactContractionCost& cost) {
        std::ostringstream out;
        out << "estimated:" << cost.estimated_execution_units
            << " fma:" << cost.matmul_fma_count
            << " fused:" << cost.fused_elementwise_count
            << " reduction:" << cost.reduction_input_elements
            << " materialization:" << cost.materialization_elements
            << " writes:" << cost.result_write_elements
            << " ops={gemm_groups:" << cost.matmul_group_count
            << ",fused:" << cost.fused_kernel_count
            << ",reduction:" << cost.reduction_op_count
            << ",materialization:" << cost.materialization_op_count << '}'
            << " peak_temp:" << cost.peak_temporary_elements
            << " peak_intermediate:" << cost.peak_intermediate_elements;
        return out.str();
    };

    const EinsumExactContractionPlan& exact = *plan.exact_contraction;
    const char* planning_mode =
        exact.planning_mode == EinsumContractionPlanningMode::SIX_OPERAND_BRIDGE
            ? "six_operand_bridge"
            : "exact";
    std::ostringstream out;
    out << "exact_contraction operands=" << plan.logical_operands.size()
        << " steps=" << exact.steps.size()
        << " mode=" << planning_mode;
    if (exact.planning_mode == EinsumContractionPlanningMode::SIX_OPERAND_BRIDGE) {
        out << " bridge_seed_pair_mask=" << exact.bridge_seed_pair_mask;
    }
    out << " weights={fma:1,fused:" << kExactFusedElementwiseWeight
        << ",reduction:" << kExactReductionWeight
        << ",materialization:" << kExactMaterializationWeight
        << ",writes:" << kExactResultWriteWeight << "}"
        << " total{" << costText(exact.cost) << "}";

    for (size_t index = 0; index < exact.steps.size(); ++index) {
        const EinsumExactContractionStep& step = exact.steps[index];
        const EinsumPairPhysicalCandidate& candidate = step.physical_candidate;
        const EinsumPairPhysicalCost& pair_cost = candidate.cost;
        out << '\n'
            << "step " << index
            << " lhs_mask=" << step.lhs_source_mask
            << " rhs_mask=" << step.rhs_source_mask
            << " result_mask=" << step.result_source_mask
            << " lhs{" << operandText(step.lhs) << "}"
            << " rhs{" << operandText(step.rhs) << "}"
            << " surviving=" << labelsText(step.surviving_labels)
            << " eliminated=" << labelsText(step.eliminated_labels)
            << " kind=" << kindText(candidate.kind)
            << " physical_labels=" << labelsText(candidate.physical_result_labels)
            << " result{" << operandText(candidate.result) << "}"
            << " orientation=" << (candidate.swapped_gemm_orientation ? "swapped" : "natural")
            << " materialize={lhs:" << (candidate.lhs_materialized ? 1 : 0)
            << ",rhs:" << (candidate.rhs_materialized ? 1 : 0)
            << ",output:" << (candidate.output_materialized ? 1 : 0) << '}';
        if (candidate.matrix_multiply.has_value()) {
            const EinsumMatrixMultiplyPlan& matrix = *candidate.matrix_multiply;
            out << " gemm={batch:" << matrix.batch_count
                << ",m:" << matrix.m
                << ",n:" << matrix.n
                << ",k:" << matrix.k << '}';
        }
        out << " pair_cost={estimated:" << step.incremental_estimated_execution_units
            << ",fma:" << pair_cost.matmul_fma_count
            << ",fused:" << pair_cost.fused_elementwise_count
            << ",reduction:" << pair_cost.reduction_input_elements
            << ",lhs_materialization:" << pair_cost.lhs_materialization_elements
            << ",rhs_materialization:" << pair_cost.rhs_materialization_elements
            << ",output_materialization:" << pair_cost.output_materialization_elements
            << ",writes:" << pair_cost.result_elements
            << ",ops={gemm_groups:" << pair_cost.matmul_group_count
            << ",fused:" << pair_cost.fused_kernel_count
            << ",reduction:" << pair_cost.reduction_op_count
            << ",materialization:" << pair_cost.materialization_op_count << '}'
            << ",peak_temp:" << pair_cost.peak_temporary_elements << '}'
            << " cumulative={" << costText(step.cumulative_cost) << '}';
    }
    return out.str();
}

std::string EinsumPlanner::describeBeamContraction(const EinsumPlan& plan) {
    if (!plan.beam_contraction.has_value()) {
        return {};
    }

    const EinsumBeamContractionPlan& beam = *plan.beam_contraction;
    std::ostringstream out;
    out << "beam_contraction operands=" << plan.logical_operands.size()
        << " steps=" << beam.steps.size()
        << " beam_width=" << beam.beam_width
        << " exact_tail_active_operands=" << beam.exact_tail_active_operands
        << " beam_levels=" << beam.beam_levels
        << " expanded_states=" << beam.expanded_state_count
        << " generated_states=" << beam.generated_state_count
        << " deduplicated_states=" << beam.deduplicated_state_count
        << " retained_states=" << beam.retained_state_count
        << " exact_tails=" << beam.exact_tail_count
        << " weights={fma:1,fused:" << kExactFusedElementwiseWeight
        << ",reduction:" << kExactReductionWeight
        << ",materialization:" << kExactMaterializationWeight
        << ",writes:" << kExactResultWriteWeight << "}"
        << " total_estimated=" << beam.cost.estimated_execution_units
        << " peak_intermediate=" << beam.cost.peak_intermediate_elements;

    for (size_t index = 0; index < beam.steps.size(); ++index) {
        const EinsumExactContractionStep& step = beam.steps[index];
        out << '\n'
            << "step " << index
            << " lhs_mask=" << step.lhs_source_mask
            << " rhs_mask=" << step.rhs_source_mask
            << " result_mask=" << step.result_source_mask
            << " kind=" << static_cast<int>(step.physical_candidate.kind)
            << " estimated=" << step.incremental_estimated_execution_units
            << " cumulative=" << step.cumulative_cost.estimated_execution_units;
    }
    return out.str();
}

}  // namespace ThorImplementation
