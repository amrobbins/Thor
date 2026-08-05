#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include <algorithm>
#include <limits>
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

EinsumMatrixOperandPlan buildMatrixOperandPlan(const EinsumOperandPlan& generic_operand,
                                                const ResolvedEinsumOperand& resolved_operand,
                                                const std::vector<uint64_t>& dimensions,
                                                const std::vector<int32_t>& batch_labels,
                                                const std::vector<int32_t>& first_matrix_group,
                                                const std::vector<int32_t>& second_matrix_group,
                                                const std::vector<uint64_t>& label_dimensions) {
    EinsumMatrixOperandPlan plan;
    plan.canonical_labels = concatenateLabels(batch_labels, first_matrix_group, second_matrix_group);
    plan.permutation = permutationToLabels(generic_operand.diagonalized_labels, plan.canonical_labels, true, &plan.inserted_axes);
    for (uint32_t inserted_axis : plan.inserted_axes) {
        if (inserted_axis >= batch_labels.size()) {
            throw std::logic_error("Internal einsum planner error: only matrix batch axes may be absent from an operand.");
        }
    }

    const auto local_dimensions = labelDimensionsForOperand(resolved_operand.axis_labels, dimensions);
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

    std::unordered_set<int32_t> physically_present(generic_operand.diagonalized_labels.begin(), generic_operand.diagonalized_labels.end());
    const std::vector<int32_t> no_transpose_order = concatenateLabels(batch_labels, first_matrix_group, second_matrix_group);
    const std::vector<int32_t> transpose_order = concatenateLabels(batch_labels, second_matrix_group, first_matrix_group);

    if (equalsAfterRemovingMissingLabels(generic_operand.diagonalized_labels, no_transpose_order, physically_present)) {
        plan.transpose = false;
        plan.requires_materialized_permutation = false;
    } else if (!first_matrix_group.empty() && !second_matrix_group.empty() &&
               equalsAfterRemovingMissingLabels(generic_operand.diagonalized_labels, transpose_order, physically_present)) {
        plan.transpose = true;
        plan.requires_materialized_permutation = false;
    } else {
        plan.transpose = false;
        plan.requires_materialized_permutation = true;
    }

    return plan;
}

std::optional<EinsumMatrixMultiplyPlan> tryBuildMatrixMultiplyPlan(
    const ResolvedEinsumEquation& equation,
    const std::vector<std::vector<uint64_t>>& input_dimensions,
    const std::vector<EinsumOperandPlan>& operands) {
    if (equation.inputs.size() != 2) {
        return std::nullopt;
    }

    const std::unordered_set<int32_t> lhs_labels(operands[0].diagonalized_labels.begin(), operands[0].diagonalized_labels.end());
    const std::unordered_set<int32_t> rhs_labels(operands[1].diagonalized_labels.begin(), operands[1].diagonalized_labels.end());
    const std::unordered_set<int32_t> output_labels(equation.output_labels.begin(), equation.output_labels.end());

    EinsumMatrixMultiplyPlan plan;

    // Expanded ellipsis labels are broadcast batch dimensions even when a
    // lower-rank operand has no corresponding physical axis.  Regular output
    // labels are batch labels only when physically present in both operands.
    for (int32_t label : equation.output_labels) {
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

    for (int32_t label : equation.reduction_labels) {
        if (!lhs_labels.contains(label) || !rhs_labels.contains(label)) {
            // A one-sided reduction requires a pre-reduction and is not a pure
            // matrix multiplication lowering.
            return std::nullopt;
        }
    }

    // Reduction-label order is not externally observable, so choose the lhs
    // physical order.  This maximizes the chance that a multi-axis K group can
    // be flattened without permuting lhs; rhs is then checked against the same
    // order to preserve contraction indexing.
    for (int32_t label : operands[0].diagonalized_labels) {
        if (!output_labels.contains(label) && rhs_labels.contains(label)) {
            plan.contraction_labels.push_back(label);
        }
    }
    if (plan.contraction_labels.size() != equation.reduction_labels.size()) {
        throw std::logic_error("Internal einsum planner error: matrix contraction labels do not cover every reduction label.");
    }

    if (plan.contraction_labels.empty()) {
        // Outer products and pure elementwise multiplication are better
        // represented by the generic alignment plan than by inventing K=1.
        return std::nullopt;
    }

    // Every physically present label must now belong to exactly one matrix
    // group.  This catches labels that would require an additional reduction.
    std::unordered_set<int32_t> matrix_labels;
    for (int32_t label : plan.batch_labels) matrix_labels.insert(label);
    for (int32_t label : plan.lhs_free_labels) matrix_labels.insert(label);
    for (int32_t label : plan.rhs_free_labels) matrix_labels.insert(label);
    for (int32_t label : plan.contraction_labels) matrix_labels.insert(label);
    for (int32_t label : lhs_labels) {
        if (!matrix_labels.contains(label)) return std::nullopt;
    }
    for (int32_t label : rhs_labels) {
        if (!matrix_labels.contains(label)) return std::nullopt;
    }

    plan.canonical_output_labels = concatenateLabels(plan.batch_labels, plan.lhs_free_labels, plan.rhs_free_labels);
    plan.output_permutation = permutationToLabels(plan.canonical_output_labels, equation.output_labels, false, nullptr);
    plan.requires_output_permutation = !isIdentityPermutation(plan.output_permutation);

    plan.batch_count = checkedProduct(plan.batch_labels, equation.label_dimensions, "matrix-multiply batch");
    plan.m = checkedProduct(plan.lhs_free_labels, equation.label_dimensions, "matrix-multiply M");
    plan.n = checkedProduct(plan.rhs_free_labels, equation.label_dimensions, "matrix-multiply N");
    plan.k = checkedProduct(plan.contraction_labels, equation.label_dimensions, "matrix-multiply K");

    plan.lhs = buildMatrixOperandPlan(operands[0],
                                      equation.inputs[0],
                                      input_dimensions[0],
                                      plan.batch_labels,
                                      plan.lhs_free_labels,
                                      plan.contraction_labels,
                                      equation.label_dimensions);
    plan.rhs = buildMatrixOperandPlan(operands[1],
                                      equation.inputs[1],
                                      input_dimensions[1],
                                      plan.batch_labels,
                                      plan.contraction_labels,
                                      plan.rhs_free_labels,
                                      equation.label_dimensions);

    const bool diagonal_extraction = operands[0].requiresDiagonalExtraction() || operands[1].requiresDiagonalExtraction();
    const bool input_permutation = plan.lhs.requires_materialized_permutation || plan.rhs.requires_materialized_permutation;
    const bool shared_label_broadcast = !plan.lhs.broadcast_axes.empty() || !plan.rhs.broadcast_axes.empty();
    plan.direct = !diagonal_extraction && !input_permutation && !shared_label_broadcast && !plan.requires_output_permutation;

    return plan;
}

}  // namespace

bool EinsumOperandPlan::requiresPermutation() const {
    return !isIdentityPermutation(permutation);
}

EinsumPlan EinsumPlanner::plan(const ResolvedEinsumEquation& equation,
                               const std::vector<std::vector<uint64_t>>& input_dimensions) {
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
    for (size_t operand = 0; operand < equation.inputs.size(); ++operand) {
        plan.operands.push_back(
            buildOperandPlan(equation.inputs[operand], input_dimensions[operand], plan.iteration_labels, plan.iteration_dimensions));
    }

    plan.matrix_multiply = tryBuildMatrixMultiplyPlan(equation, input_dimensions, plan.operands);
    if (plan.matrix_multiply.has_value()) {
        plan.kind = plan.matrix_multiply->batch_labels.empty() ? EinsumPlanKind::GEMM : EinsumPlanKind::BATCHED_GEMM;
    } else if (equation.inputs.size() == 1) {
        plan.kind = equation.reduction_labels.empty() ? EinsumPlanKind::UNARY : EinsumPlanKind::REDUCTION;
    } else if (equation.reduction_labels.empty()) {
        plan.kind = EinsumPlanKind::ELEMENTWISE;
    } else {
        plan.kind = EinsumPlanKind::GENERAL;
    }

    return plan;
}

EinsumPlan EinsumPlanner::parseAndPlan(const std::string& equation,
                                       const std::vector<std::vector<uint64_t>>& input_dimensions) {
    return plan(EinsumParser::parseAndResolve(equation, input_dimensions), input_dimensions);
}

}  // namespace ThorImplementation
