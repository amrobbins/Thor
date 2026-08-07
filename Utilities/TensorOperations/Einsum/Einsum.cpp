#include "Utilities/TensorOperations/Einsum/Einsum.h"

#include "Utilities/Expression/BatchedMatmulPlan.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace {

constexpr const char* kOutputName = "einsum_output";

[[nodiscard]] bool isSupportedStorageDType(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

[[nodiscard]] uint64_t checkedProduct(const std::vector<uint64_t>& dimensions, const char* role) {
    uint64_t product = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument(std::string("Einsum ") + role + " dimensions must be non-zero.");
        }
        if (product > std::numeric_limits<uint64_t>::max() / dimension) {
            throw std::invalid_argument(std::string("Einsum ") + role + " element count overflows uint64_t.");
        }
        product *= dimension;
    }
    return product;
}

[[nodiscard]] uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* role) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::invalid_argument(std::string("Einsum ") + role + " size overflows uint64_t.");
    }
    return lhs * rhs;
}

[[nodiscard]] std::vector<uint64_t> physicalOutputDimensions(const EinsumPlan& plan) {
    if (plan.equation.output_dimensions.empty()) {
        return {1};
    }
    return plan.equation.output_dimensions;
}

void requireSupportedInputs(const std::vector<Tensor>& inputs, const Stream& stream) {
    if (inputs.empty()) {
        throw std::invalid_argument("Einsum requires at least one input tensor.");
    }
    if (!stream.isInitialized()) {
        throw std::invalid_argument("Einsum requires an initialized GPU stream.");
    }

    if (!inputs.front().isInitialized()) {
        throw std::invalid_argument("Einsum input tensor 0 is uninitialized.");
    }
    const TensorPlacement expected_placement = inputs.front().getPlacement();
    const DataType expected_dtype = inputs.front().getDataType();
    if (expected_placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Einsum execution currently requires GPU input tensors.");
    }
    if (expected_placement.getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("Einsum input tensors and stream must use the same GPU.");
    }
    if (!isSupportedStorageDType(expected_dtype)) {
        throw std::invalid_argument("Einsum execution supports FP16, BF16, and FP32 storage dtypes.");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        const Tensor& input = inputs[i];
        if (!input.isInitialized()) {
            throw std::invalid_argument("Einsum input tensor " + std::to_string(i) + " is uninitialized.");
        }
        if (input.getPlacement() != expected_placement) {
            throw std::invalid_argument("Einsum input tensors must all use the same GPU placement.");
        }
        if (input.getDataType() != expected_dtype) {
            throw std::invalid_argument("Einsum execution requires all input tensors to have the same storage dtype.");
        }
        if (!input.isDenseContiguous()) {
            throw std::invalid_argument("Einsum execution currently requires dense contiguous input tensors.");
        }
    }
}

[[nodiscard]] bool tensorStorageOverlaps(const Tensor& lhs, const Tensor& rhs) {
    const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs.getMemPtr<void>());
    const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs.getMemPtr<void>());
    const uintptr_t lhs_end = lhs_begin + lhs.getArraySizeInBytes();
    const uintptr_t rhs_end = rhs_begin + rhs.getArraySizeInBytes();
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

void requireOutput(const Tensor& output,
                   const std::vector<Tensor>& inputs,
                   const std::vector<uint64_t>& expected_dimensions,
                   DataType expected_dtype) {
    if (!output.isInitialized()) {
        throw std::invalid_argument("Einsum preallocated output tensor is uninitialized.");
    }
    if (output.getPlacement() != inputs.front().getPlacement()) {
        throw std::invalid_argument("Einsum preallocated output must use the input GPU placement.");
    }
    if (output.getDataType() != expected_dtype) {
        throw std::invalid_argument("Einsum preallocated output dtype must match the input dtype.");
    }
    if (!output.isDenseContiguous()) {
        throw std::invalid_argument("Einsum preallocated output must be dense contiguous.");
    }
    if (output.getDimensions() != expected_dimensions) {
        throw std::invalid_argument("Einsum preallocated output dimensions do not match the resolved equation.");
    }
    for (const Tensor& input : inputs) {
        if (tensorStorageOverlaps(input, output)) {
            throw std::invalid_argument("Einsum input and output storage must not overlap.");
        }
    }
}

[[nodiscard]] std::vector<uint64_t> denseStrides(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> strides(dimensions.size(), 1);
    if (dimensions.empty()) {
        return strides;
    }
    for (size_t axis = dimensions.size() - 1; axis > 0; --axis) {
        strides[axis - 1] = checkedMultiply(strides[axis], dimensions[axis], "dense stride");
    }
    return strides;
}

[[nodiscard]] std::vector<uint64_t> diagonalizedStrides(const ResolvedEinsumOperand& operand,
                                                         const std::vector<uint64_t>& input_dimensions,
                                                         const EinsumOperandPlan& operand_plan) {
    const std::vector<uint64_t> source_strides = denseStrides(input_dimensions);
    std::vector<uint64_t> result;
    result.reserve(operand_plan.diagonalized_labels.size());

    for (int32_t label : operand_plan.diagonalized_labels) {
        uint64_t combined_stride = 0;
        bool found = false;
        for (size_t source_axis = 0; source_axis < operand.axis_labels.size(); ++source_axis) {
            if (operand.axis_labels[source_axis] != label) {
                continue;
            }
            if (combined_stride > std::numeric_limits<uint64_t>::max() - source_strides[source_axis]) {
                throw std::invalid_argument("Einsum diagonal stride overflows uint64_t.");
            }
            combined_stride += source_strides[source_axis];
            found = true;
        }
        if (!found || combined_stride == 0) {
            throw std::runtime_error("Einsum planner produced an invalid diagonalized label mapping.");
        }
        result.push_back(combined_stride);
    }
    return result;
}

// Logical einsum operand state at a pair-contraction lowering boundary.
// It carries the Expression together with the ordered logical labels and their
// physical dimensions/strides, so the same lowering helpers can consume either
// an original input or, in later multi-operand planning, an intermediate
// Expression. Repeated-label diagonals remain zero-copy strided views here;
// matrix lowering decides from the real physical strides whether cuBLASLt can
// consume the view directly or whether only this operand must be materialized.
struct LogicalEinsumOperand {
    Expression expression;
    std::vector<int32_t> labels;
    std::vector<uint64_t> dimensions;
    std::vector<uint64_t> strides;
    bool dense_storage = false;
};

[[nodiscard]] LogicalEinsumOperand logicalInputOperand(size_t operand_index,
                                                       const std::vector<uint64_t>& input_dimensions,
                                                       const EinsumPlan& plan) {
    const EinsumOperandPlan& operand_plan = plan.operands.at(operand_index);
    const ResolvedEinsumOperand& resolved_operand = plan.equation.inputs.at(operand_index);

    LogicalEinsumOperand result{Expression::input("einsum_input_" + std::to_string(operand_index)),
                                operand_plan.diagonalized_labels,
                                operand_plan.diagonalized_dimensions,
                                diagonalizedStrides(resolved_operand, input_dimensions, operand_plan),
                                !operand_plan.requiresDiagonalExtraction()};

    if (operand_plan.requiresDiagonalExtraction()) {
        if (result.dimensions.empty()) {
            throw std::runtime_error("Einsum execution cannot construct an empty-rank diagonal matrix view.");
        }
        result.expression = result.expression.stridedView(result.dimensions, result.strides);
    }
    return result;
}

[[nodiscard]] std::vector<uint64_t> logicalReductionAxes(const LogicalEinsumOperand& operand,
                                                         const std::vector<int32_t>& reduction_labels) {
    if (reduction_labels.empty()) {
        return {};
    }

    const std::unordered_set<int32_t> reduced(reduction_labels.begin(), reduction_labels.end());
    std::vector<uint64_t> axes;
    axes.reserve(reduction_labels.size());
    for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
        if (reduced.contains(operand.labels[axis])) {
            axes.push_back(static_cast<uint64_t>(axis));
        }
    }
    if (axes.size() != reduction_labels.size()) {
        throw std::runtime_error("Einsum matrix pre-reduction labels do not map one-to-one to logical axes.");
    }
    return axes;
}

void markLogicalOperandPreReduced(LogicalEinsumOperand& operand,
                                    const std::vector<int32_t>& reduction_labels) {
    if (reduction_labels.empty()) {
        return;
    }

    const std::unordered_set<int32_t> reduced(reduction_labels.begin(), reduction_labels.end());
    std::vector<int32_t> retained_labels;
    std::vector<uint64_t> retained_dimensions;
    retained_labels.reserve(operand.labels.size() - reduction_labels.size());
    retained_dimensions.reserve(operand.dimensions.size() - reduction_labels.size());
    for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
        if (!reduced.contains(operand.labels[axis])) {
            retained_labels.push_back(operand.labels[axis]);
            retained_dimensions.push_back(operand.dimensions[axis]);
        }
    }

    operand.labels = std::move(retained_labels);
    operand.dimensions = std::move(retained_dimensions);
    operand.strides = denseStrides(operand.dimensions);
    operand.dense_storage = true;
}

[[nodiscard]] LogicalEinsumOperand preReduceLogicalOperand(LogicalEinsumOperand operand,
                                                              const std::vector<int32_t>& reduction_labels,
                                                              DataType output_dtype) {
    const std::vector<uint64_t> reduction_axes = logicalReductionAxes(operand, reduction_labels);
    if (reduction_axes.empty()) {
        return operand;
    }

    // Accumulate through centralized CubReduction in FP32 and store in the
    // dtype requested by the downstream lowering. Matrix lowering requests the
    // einsum storage dtype to satisfy Thor's no-implicit-conversion matmul
    // contract; pair-product lowering can retain FP32 through its final fused
    // multiply. CubReduction receives the logical view strides directly, so
    // diagonal extraction stays zero-copy and removable axes disappear before
    // any possible operand materialization.
    operand.expression =
        operand.expression.reduce_sum(reduction_axes, reduction_axes, DataType::FP32).withOutputDType(output_dtype);
    markLogicalOperandPreReduced(operand, reduction_labels);
    return operand;
}

void markLogicalSingletonLabelsElided(LogicalEinsumOperand& operand,
                                      const std::vector<int32_t>& broadcast_elision_labels) {
    if (broadcast_elision_labels.empty()) {
        return;
    }

    const std::unordered_set<int32_t> elided(broadcast_elision_labels.begin(), broadcast_elision_labels.end());
    std::vector<int32_t> retained_labels;
    std::vector<uint64_t> retained_dimensions;
    std::vector<uint64_t> retained_strides;
    retained_labels.reserve(operand.labels.size() - broadcast_elision_labels.size());
    retained_dimensions.reserve(operand.dimensions.size() - broadcast_elision_labels.size());
    retained_strides.reserve(operand.strides.size() - broadcast_elision_labels.size());

    size_t removed = 0;
    for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
        if (elided.contains(operand.labels[axis])) {
            if (operand.dimensions[axis] != 1) {
                throw std::runtime_error("Einsum attempted to elide a non-singleton broadcast-contraction axis.");
            }
            ++removed;
            continue;
        }
        retained_labels.push_back(operand.labels[axis]);
        retained_dimensions.push_back(operand.dimensions[axis]);
        retained_strides.push_back(operand.strides[axis]);
    }
    if (removed != broadcast_elision_labels.size()) {
        throw std::runtime_error("Einsum broadcast-contraction elision labels do not map one-to-one to logical axes.");
    }

    operand.labels = std::move(retained_labels);
    operand.dimensions = std::move(retained_dimensions);
    operand.strides = std::move(retained_strides);
    operand.dense_storage = operand.dimensions.empty() || operand.strides == denseStrides(operand.dimensions);
}

[[nodiscard]] LogicalEinsumOperand elideLogicalSingletonLabels(
    LogicalEinsumOperand operand,
    const std::vector<int32_t>& broadcast_elision_labels) {
    if (broadcast_elision_labels.empty()) {
        return operand;
    }

    markLogicalSingletonLabelsElided(operand, broadcast_elision_labels);

    // Removing a singleton contraction axis is a view-only operation. Keep a
    // physical {1} representation when no logical axes survive so downstream
    // scalar handling remains consistent with Thor's reduction conventions.
    operand.expression = operand.dimensions.empty()
                             ? operand.expression.reshape({1})
                             : operand.expression.stridedView(operand.dimensions, operand.strides);
    return operand;
}

[[nodiscard]] Expression alignLogicalPairOperandToOutput(LogicalEinsumOperand operand,
                                                          const std::vector<int32_t>& output_labels) {
    // Thor represents a logical scalar as physical {1}. Preserve that
    // representation when there is no output axis, or reshape it to the
    // all-singleton rank needed for ordinary broadcast alignment.
    if (operand.labels.empty()) {
        if (output_labels.empty()) {
            return operand.expression;
        }
        return operand.expression.reshape(std::vector<uint64_t>(output_labels.size(), 1));
    }

    std::vector<uint32_t> present_axes;
    std::vector<uint64_t> present_dimensions;
    std::vector<uint64_t> present_strides;
    std::vector<uint64_t> inserted_axes;
    present_axes.reserve(operand.labels.size());
    present_dimensions.reserve(operand.labels.size());
    present_strides.reserve(operand.labels.size());
    inserted_axes.reserve(output_labels.size());

    for (size_t output_axis = 0; output_axis < output_labels.size(); ++output_axis) {
        const int32_t label = output_labels[output_axis];
        auto it = std::find(operand.labels.begin(), operand.labels.end(), label);
        if (it == operand.labels.end()) {
            inserted_axes.push_back(static_cast<uint64_t>(output_axis));
            continue;
        }
        const size_t source_axis = static_cast<size_t>(it - operand.labels.begin());
        present_axes.push_back(static_cast<uint32_t>(source_axis));
        present_dimensions.push_back(operand.dimensions[source_axis]);
        present_strides.push_back(operand.strides[source_axis]);
    }

    if (present_axes.size() != operand.labels.size()) {
        throw std::runtime_error("Einsum pair-product operand retained a label absent from the requested output.");
    }

    bool identity_permutation = true;
    for (size_t axis = 0; axis < present_axes.size(); ++axis) {
        if (present_axes[axis] != axis) {
            identity_permutation = false;
            break;
        }
    }
    if (!identity_permutation) {
        operand.expression = operand.expression.stridedView(present_dimensions, present_strides);
    }
    if (!inserted_axes.empty()) {
        operand.expression = operand.expression.unsqueeze(inserted_axes);
    }
    return operand.expression;
}

[[nodiscard]] Expression pairProductExpression(const EinsumPlan& pair_plan,
                                                LogicalEinsumOperand lhs,
                                                LogicalEinsumOperand rhs,
                                                DataType output_dtype) {
    if (!pair_plan.pair_product.has_value()) {
        throw std::runtime_error("Einsum pair-product lowering requires a pair-product plan.");
    }
    const EinsumPairProductPlan& pair = *pair_plan.pair_product;

    // Unlike GEMM, the final pair product has no storage-dtype contract on its
    // inputs. Keep independently reduced values in FP32 so the optimized
    // factorization retains the generic path's FP32 accumulation semantics.
    lhs = preReduceLogicalOperand(std::move(lhs), pair.lhs_reduction_labels, DataType::FP32);
    rhs = preReduceLogicalOperand(std::move(rhs), pair.rhs_reduction_labels, DataType::FP32);
    lhs = elideLogicalSingletonLabels(std::move(lhs), pair.lhs_broadcast_elision_labels);
    rhs = elideLogicalSingletonLabels(std::move(rhs), pair.rhs_broadcast_elision_labels);

    Expression lhs_aligned = alignLogicalPairOperandToOutput(std::move(lhs), pair_plan.equation.output_labels);
    Expression rhs_aligned = alignLogicalPairOperandToOutput(std::move(rhs), pair_plan.equation.output_labels);
    Expression result = lhs_aligned.cast(DataType::FP32) * rhs_aligned.cast(DataType::FP32);
    if (output_dtype != DataType::FP32) {
        result = result.cast(output_dtype);
    }
    return result;
}

[[nodiscard]] Expression alignedOperandExpression(size_t operand_index,
                                                  const Tensor& input,
                                                  const EinsumPlan& plan) {
    Expression expression = Expression::input("einsum_input_" + std::to_string(operand_index));
    const EinsumOperandPlan& operand_plan = plan.operands.at(operand_index);
    const ResolvedEinsumOperand& resolved_operand = plan.equation.inputs.at(operand_index);

    if (operand_plan.requiresDiagonalExtraction() || operand_plan.requiresPermutation()) {
        const std::vector<uint64_t> logical_strides =
            diagonalizedStrides(resolved_operand, input.getDimensions(), operand_plan);
        std::vector<uint64_t> view_dimensions;
        std::vector<uint64_t> view_strides;
        view_dimensions.reserve(operand_plan.permutation.size());
        view_strides.reserve(operand_plan.permutation.size());
        for (uint32_t source_axis : operand_plan.permutation) {
            if (source_axis >= operand_plan.diagonalized_dimensions.size() || source_axis >= logical_strides.size()) {
                throw std::runtime_error("Einsum planner produced an out-of-range operand permutation.");
            }
            view_dimensions.push_back(operand_plan.diagonalized_dimensions[source_axis]);
            view_strides.push_back(logical_strides[source_axis]);
        }
        if (view_dimensions.empty()) {
            throw std::runtime_error("Einsum execution cannot construct an empty-rank strided view.");
        }
        expression = expression.stridedView(view_dimensions, view_strides);
    }

    if (!operand_plan.inserted_axes.empty()) {
        std::vector<uint64_t> inserted_axes;
        inserted_axes.reserve(operand_plan.inserted_axes.size());
        for (uint32_t axis : operand_plan.inserted_axes) {
            inserted_axes.push_back(axis);
        }
        expression = expression.unsqueeze(inserted_axes);
    }

    return expression;
}

[[nodiscard]] std::unordered_map<std::string, Tensor> expressionInputs(const std::vector<Tensor>& inputs) {
    std::unordered_map<std::string, Tensor> named_inputs;
    named_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        named_inputs.emplace("einsum_input_" + std::to_string(i), inputs[i]);
    }
    return named_inputs;
}

[[nodiscard]] bool matrixDimensionsFitBackend(const EinsumMatrixMultiplyPlan& matrix_plan) {
    constexpr uint64_t kMax = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    return matrix_plan.m <= kMax && matrix_plan.n <= kMax && matrix_plan.k <= kMax && matrix_plan.batch_count <= kMax;
}

[[nodiscard]] bool containsAxis(const std::vector<uint32_t>& axes, uint32_t axis) {
    for (uint32_t candidate : axes) {
        if (candidate == axis) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool hasOnlyBatchBroadcasts(const EinsumMatrixOperandPlan& operand, size_t batch_rank) {
    for (uint32_t axis : operand.broadcast_axes) {
        if (axis >= batch_rank) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::vector<uint64_t> matrixBatchDimensions(const EinsumPlan& plan,
                                                           const EinsumMatrixOperandPlan& operand) {
    if (!plan.matrix_multiply.has_value()) {
        throw std::runtime_error("Einsum matrix batch dimensions require a matrix plan.");
    }
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
    std::vector<uint64_t> dimensions;
    dimensions.reserve(matrix.batch_labels.size());
    for (size_t axis = 0; axis < matrix.batch_labels.size(); ++axis) {
        if (containsAxis(operand.broadcast_axes, static_cast<uint32_t>(axis))) {
            dimensions.push_back(1);
        } else {
            const int32_t label = matrix.batch_labels[axis];
            dimensions.push_back(plan.equation.label_dimensions.at(static_cast<size_t>(label)));
        }
    }
    return dimensions;
}

[[nodiscard]] std::vector<uint64_t> matrixOperandStoredShape(const EinsumPlan& plan,
                                                              const EinsumMatrixOperandPlan& operand,
                                                              uint64_t logical_rows,
                                                              uint64_t logical_columns,
                                                              bool transpose) {
    if (logical_rows == 0 || logical_columns == 0) {
        throw std::invalid_argument("Einsum matrix dimensions must be non-zero.");
    }

    std::vector<uint64_t> dimensions = matrixBatchDimensions(plan, operand);
    if (transpose) {
        dimensions.push_back(logical_columns);
        dimensions.push_back(logical_rows);
    } else {
        dimensions.push_back(logical_rows);
        dimensions.push_back(logical_columns);
    }
    return dimensions;
}

struct MatrixOperandLayoutPlan {
    MatmulTensorLayout layout;
    bool transpose = false;
    bool materialize = false;
};

struct MatrixExpressionLayoutPlan {
    MatrixOperandLayoutPlan lhs;
    MatrixOperandLayoutPlan rhs;
    BatchedMatmulLayoutPlan matmul;
    bool swapped_orientation = false;
};

[[nodiscard]] std::optional<size_t> logicalAxisForLabel(const LogicalEinsumOperand& operand, int32_t label) {
    for (size_t axis = 0; axis < operand.labels.size(); ++axis) {
        if (operand.labels[axis] == label) {
            return axis;
        }
    }
    return std::nullopt;
}

struct FlattenedLogicalGroup {
    uint64_t extent = 1;
    uint64_t stride = 1;
};

[[nodiscard]] std::optional<FlattenedLogicalGroup> flattenLogicalGroup(
    const LogicalEinsumOperand& operand,
    const std::vector<int32_t>& labels) {
    FlattenedLogicalGroup result;
    if (labels.empty()) {
        return result;
    }

    bool have_non_singleton_axis = false;
    uint64_t expected_outer_stride = 0;
    for (size_t label_index = labels.size(); label_index-- > 0;) {
        const std::optional<size_t> axis = logicalAxisForLabel(operand, labels[label_index]);
        if (!axis.has_value()) {
            return std::nullopt;
        }
        const uint64_t dimension = operand.dimensions[*axis];
        const uint64_t stride = operand.strides[*axis];
        result.extent = checkedMultiply(result.extent, dimension, "matrix-group flatten");

        // Singleton axes do not contribute to addressing and therefore place no
        // contiguity requirement on the surrounding logical axes.
        if (dimension == 1) {
            continue;
        }
        if (!have_non_singleton_axis) {
            result.stride = stride;
            expected_outer_stride = checkedMultiply(dimension, stride, "matrix-group flatten stride");
            have_non_singleton_axis = true;
            continue;
        }
        if (stride != expected_outer_stride) {
            return std::nullopt;
        }
        expected_outer_stride = checkedMultiply(dimension, stride, "matrix-group flatten stride");
    }

    if (!have_non_singleton_axis) {
        result.stride = 1;
    }
    return result;
}

[[nodiscard]] std::optional<MatmulTensorLayout> directMatrixOperandLayout(
    const EinsumPlan& plan,
    const EinsumMatrixOperandPlan& operand_plan,
    const LogicalEinsumOperand& source,
    const std::vector<int32_t>& first_matrix_group,
    const std::vector<int32_t>& second_matrix_group,
    uint64_t logical_rows,
    uint64_t logical_columns) {
    if (!plan.matrix_multiply.has_value() || operand_plan.requires_materialized_permutation) {
        return std::nullopt;
    }
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;

    // The planner's transpose flag means the physical source label order is
    // [batch..., second-group..., first-group...] rather than canonical
    // [batch..., first-group..., second-group...]. Verify that the current
    // logical operand still has exactly that order after any pre-reduction.
    std::vector<int32_t> expected_source_labels;
    expected_source_labels.reserve(source.labels.size());
    for (int32_t label : matrix.batch_labels) {
        if (logicalAxisForLabel(source, label).has_value()) {
            expected_source_labels.push_back(label);
        }
    }
    const auto append_present = [&](const std::vector<int32_t>& labels) {
        for (int32_t label : labels) {
            if (logicalAxisForLabel(source, label).has_value()) {
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

    const std::optional<FlattenedLogicalGroup> first = flattenLogicalGroup(source, first_matrix_group);
    const std::optional<FlattenedLogicalGroup> second = flattenLogicalGroup(source, second_matrix_group);
    if (!first.has_value() || !second.has_value() || first->extent != logical_rows ||
        second->extent != logical_columns) {
        return std::nullopt;
    }

    MatmulTensorLayout layout;
    layout.dimensions.reserve(matrix.batch_labels.size() + 2);
    layout.strides_elements.reserve(matrix.batch_labels.size() + 2);
    for (int32_t label : matrix.batch_labels) {
        const std::optional<size_t> axis = logicalAxisForLabel(source, label);
        if (axis.has_value()) {
            layout.dimensions.push_back(source.dimensions[*axis]);
            layout.strides_elements.push_back(source.strides[*axis]);
        } else {
            // Only batch labels may be physically absent. The matmul planner
            // interprets extent-one batch axes as zero-stride broadcasts.
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

[[nodiscard]] MatrixOperandLayoutPlan materializedMatrixOperandLayout(const EinsumPlan& plan,
                                                                       const EinsumMatrixOperandPlan& operand,
                                                                       uint64_t logical_rows,
                                                                       uint64_t logical_columns) {
    return MatrixOperandLayoutPlan{
        denseMatmulTensorLayout(matrixOperandStoredShape(plan, operand, logical_rows, logical_columns, false)),
        false,
        true};
}

[[nodiscard]] MatrixOperandLayoutPlan matrixOperandLayoutPlan(const EinsumPlan& plan,
                                                               const EinsumMatrixOperandPlan& operand,
                                                               const LogicalEinsumOperand& source,
                                                               const std::vector<int32_t>& first_matrix_group,
                                                               const std::vector<int32_t>& second_matrix_group,
                                                               uint64_t logical_rows,
                                                               uint64_t logical_columns) {
    const std::optional<MatmulTensorLayout> direct = directMatrixOperandLayout(
        plan, operand, source, first_matrix_group, second_matrix_group, logical_rows, logical_columns);
    if (direct.has_value()) {
        return MatrixOperandLayoutPlan{*direct, operand.transpose, false};
    }
    return materializedMatrixOperandLayout(plan, operand, logical_rows, logical_columns);
}

[[nodiscard]] std::vector<uint64_t> matrixOutputShape(const EinsumPlan& plan, bool swapped_orientation = false) {
    if (!plan.matrix_multiply.has_value()) {
        throw std::runtime_error("Einsum matrix output shape requires a matrix plan.");
    }
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
    std::vector<uint64_t> dimensions;
    dimensions.reserve(matrix.batch_labels.size() + 2);
    for (int32_t label : matrix.batch_labels) {
        dimensions.push_back(plan.equation.label_dimensions.at(static_cast<size_t>(label)));
    }
    dimensions.push_back(swapped_orientation ? matrix.n : matrix.m);
    dimensions.push_back(swapped_orientation ? matrix.m : matrix.n);
    return dimensions;
}

[[nodiscard]] bool matrixOutputPermutationUsesSwappedGemmOrientation(const EinsumPlan& plan) {
    if (!plan.matrix_multiply.has_value()) {
        return false;
    }
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
    if (!matrix.requires_output_permutation) {
        return false;
    }

    // The canonical GEMM result is [batch..., flattened(lhs_free),
    // flattened(rhs_free)]. If the requested output keeps batch labels in place
    // and swaps the complete N and M label groups, compute the transpose
    // algebraically as rhs^T @ lhs^T so cuBLASLt writes [batch..., N, M]
    // directly. Reshaping then restores the individual einsum axes without a
    // transpose/materialization stage.
    std::vector<int32_t> transposed_labels;
    transposed_labels.reserve(matrix.canonical_output_labels.size());
    transposed_labels.insert(transposed_labels.end(), matrix.batch_labels.begin(), matrix.batch_labels.end());
    transposed_labels.insert(transposed_labels.end(), matrix.rhs_free_labels.begin(), matrix.rhs_free_labels.end());
    transposed_labels.insert(transposed_labels.end(), matrix.lhs_free_labels.begin(), matrix.lhs_free_labels.end());
    return transposed_labels == plan.equation.output_labels;
}

[[nodiscard]] BatchedMatmulLayoutPlan buildExpressionMatmulLayoutPlan(const MatrixOperandLayoutPlan& lhs,
                                                                          const MatrixOperandLayoutPlan& rhs,
                                                                          const MatmulTensorLayout& output_layout,
                                                                          bool swapped_orientation) {
    return swapped_orientation
               ? planBatchedMatmulLayout(
                     rhs.layout, lhs.layout, output_layout, !rhs.transpose, !lhs.transpose)
               : planBatchedMatmulLayout(lhs.layout, rhs.layout, output_layout, lhs.transpose, rhs.transpose);
}

[[nodiscard]] std::optional<MatrixExpressionLayoutPlan> expressionMatmulLayoutPlan(
    const EinsumPlan& pair_plan,
    LogicalEinsumOperand lhs_source,
    LogicalEinsumOperand rhs_source,
    std::optional<bool> forced_swapped_orientation = std::nullopt) {
    if (!pair_plan.matrix_multiply.has_value()) {
        return std::nullopt;
    }
    const EinsumPlan& plan = pair_plan;
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
    if (!matrixDimensionsFitBackend(matrix)) {
        return std::nullopt;
    }

    const size_t batch_rank = matrix.batch_labels.size();
    if (!hasOnlyBatchBroadcasts(matrix.lhs, batch_rank) || !hasOnlyBatchBroadcasts(matrix.rhs, batch_rank)) {
        // Expression matmul broadcasts batch axes. Shared K broadcasts should
        // already have been normalized algebraically into an operand-local
        // reduction plus singleton elision. Any remaining non-batch broadcast
        // is not representable by the current matrix lowering.
        return std::nullopt;
    }

    markLogicalOperandPreReduced(lhs_source, matrix.lhs_reduction_labels);
    markLogicalOperandPreReduced(rhs_source, matrix.rhs_reduction_labels);
    markLogicalSingletonLabelsElided(lhs_source, matrix.lhs_broadcast_elision_labels);
    markLogicalSingletonLabelsElided(rhs_source, matrix.rhs_broadcast_elision_labels);

    MatrixExpressionLayoutPlan result;
    result.lhs = matrixOperandLayoutPlan(plan,
                                         matrix.lhs,
                                         lhs_source,
                                         matrix.lhs_free_labels,
                                         matrix.contraction_labels,
                                         matrix.m,
                                         matrix.k);
    result.rhs = matrixOperandLayoutPlan(plan,
                                         matrix.rhs,
                                         rhs_source,
                                         matrix.contraction_labels,
                                         matrix.rhs_free_labels,
                                         matrix.k,
                                         matrix.n);
    result.swapped_orientation =
        forced_swapped_orientation.value_or(matrixOutputPermutationUsesSwappedGemmOrientation(plan));
    const MatmulTensorLayout output_layout =
        denseMatmulTensorLayout(matrixOutputShape(plan, result.swapped_orientation));

    result.matmul = buildExpressionMatmulLayoutPlan(result.lhs, result.rhs, output_layout, result.swapped_orientation);

    // A diagonalized logical view may flatten cleanly into matrix groups yet
    // still have a matrix-plane stride pattern that cuBLASLt cannot address.
    // Materialize only the offending operand into canonical dense matrix-group
    // order, then re-plan. This is deliberately decided from physical strides,
    // not from the presence of a diagonal itself.
    const bool lhs_addressable = result.swapped_orientation ? result.matmul.rhs_matrix.isBlasAddressable()
                                                            : result.matmul.lhs_matrix.isBlasAddressable();
    const bool rhs_addressable = result.swapped_orientation ? result.matmul.lhs_matrix.isBlasAddressable()
                                                            : result.matmul.rhs_matrix.isBlasAddressable();
    bool replan = false;
    if (!lhs_addressable && !result.lhs.materialize) {
        result.lhs = materializedMatrixOperandLayout(plan, matrix.lhs, matrix.m, matrix.k);
        replan = true;
    }
    if (!rhs_addressable && !result.rhs.materialize) {
        result.rhs = materializedMatrixOperandLayout(plan, matrix.rhs, matrix.k, matrix.n);
        replan = true;
    }
    if (replan) {
        result.matmul =
            buildExpressionMatmulLayoutPlan(result.lhs, result.rhs, output_layout, result.swapped_orientation);
    }

    if (!result.matmul.canLowerWithoutMaterialization()) {
        return std::nullopt;
    }
    return result;
}

[[nodiscard]] Expression materializeMatrixOperandPermutation(const EinsumPlan& plan,
                                                              const EinsumMatrixOperandPlan& operand,
                                                              LogicalEinsumOperand source,
                                                              uint64_t logical_rows,
                                                              uint64_t logical_columns,
                                                              DataType output_dtype) {
    // Canonicalize only this operand into [batch..., rows..., columns...] order.
    // This path is used both for planner-known interleaving and for logical
    // diagonal views whose actual stride pattern cannot be represented by a
    // cuBLASLt matrix plane. Any one-sided reduction has already happened, so
    // the temporary is the smallest useful operand-local materialization.
    if (operand.permutation.size() != source.dimensions.size() || source.labels.size() != source.dimensions.size() ||
        source.strides.size() != source.dimensions.size()) {
        throw std::runtime_error("Einsum matrix materialization source metadata has inconsistent rank.");
    }

    std::vector<uint64_t> view_dimensions;
    std::vector<uint64_t> view_strides;
    view_dimensions.reserve(operand.permutation.size());
    view_strides.reserve(operand.permutation.size());
    for (uint32_t source_axis : operand.permutation) {
        if (source_axis >= source.dimensions.size()) {
            throw std::runtime_error("Einsum matrix materialization permutation contains an out-of-range source axis.");
        }
        view_dimensions.push_back(source.dimensions[source_axis]);
        view_strides.push_back(source.strides[source_axis]);
    }
    if (view_dimensions.empty()) {
        throw std::runtime_error("Einsum cannot materialize an empty-rank matrix operand permutation.");
    }
    source.expression = source.expression.stridedView(view_dimensions, view_strides);

    if (!operand.inserted_axes.empty()) {
        std::vector<uint64_t> inserted_axes;
        inserted_axes.reserve(operand.inserted_axes.size());
        for (uint32_t axis : operand.inserted_axes) {
            inserted_axes.push_back(axis);
        }
        source.expression = source.expression.unsqueeze(inserted_axes);
    }

    // A same-dtype cast is intentional here: it establishes the fused stage
    // boundary that writes dense canonical storage before Matmul.
    source.expression = source.expression.cast(output_dtype);
    return source.expression.reshape(matrixOperandStoredShape(plan, operand, logical_rows, logical_columns, false));
}

[[nodiscard]] Expression matrixOperandExpression(const EinsumPlan& plan,
                                                  LogicalEinsumOperand source,
                                                  const EinsumMatrixOperandPlan& operand,
                                                  const std::vector<int32_t>& pre_reduction_labels,
                                                  const std::vector<int32_t>& broadcast_elision_labels,
                                                  uint64_t logical_rows,
                                                  uint64_t logical_columns,
                                                  const MatrixOperandLayoutPlan& layout_plan,
                                                  DataType output_dtype) {
    source = preReduceLogicalOperand(std::move(source), pre_reduction_labels, output_dtype);
    source = elideLogicalSingletonLabels(std::move(source), broadcast_elision_labels);
    if (layout_plan.materialize) {
        return materializeMatrixOperandPermutation(
            plan, operand, std::move(source), logical_rows, logical_columns, output_dtype);
    }

    // Use an explicit storage alias rather than dense reshape. This preserves
    // padded/diagonal strides while collapsing flattenable logical label groups
    // to the [batch..., matrix-row, matrix-column] rank expected by Matmul.
    return source.expression.stridedView(
        layout_plan.layout.dimensions, layout_plan.layout.strides_elements, layout_plan.layout.storage_element_offset);
}

[[nodiscard]] std::vector<uint64_t> matrixCanonicalLogicalOutputDimensions(const EinsumPlan& plan) {
    if (!plan.matrix_multiply.has_value()) {
        throw std::runtime_error("Einsum canonical matrix output dimensions require a matrix plan.");
    }
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
    std::vector<uint64_t> dimensions;
    dimensions.reserve(matrix.canonical_output_labels.size());
    for (int32_t label : matrix.canonical_output_labels) {
        if (label < 0 || static_cast<size_t>(label) >= plan.equation.label_dimensions.size()) {
            throw std::runtime_error("Einsum matrix output contains an invalid canonical label.");
        }
        dimensions.push_back(plan.equation.label_dimensions[static_cast<size_t>(label)]);
    }
    return dimensions;
}

[[nodiscard]] Expression materializeMatrixOutputPermutation(const EinsumPlan& plan,
                                                              Expression result,
                                                              DataType output_dtype) {
    if (!plan.matrix_multiply.has_value()) {
        throw std::runtime_error("Einsum matrix output permutation requires a matrix plan.");
    }
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;
    if (!matrix.requires_output_permutation) {
        return result;
    }
    if (matrix.output_permutation.size() != matrix.canonical_output_labels.size() ||
        matrix.output_permutation.size() != plan.equation.output_labels.size()) {
        throw std::runtime_error("Einsum matrix output permutation metadata has inconsistent rank.");
    }

    // GEMM deliberately writes the smallest dense canonical result
    // [batch..., flattened(M), flattened(N)]. Restore the individual logical
    // einsum axes as a zero-copy dense reshape, then view those same bytes in
    // requested output-label order. Only the final output-sized tensor is
    // materialized; the full generic broadcast-product intermediate is never
    // formed.
    const std::vector<uint64_t> canonical_dimensions = matrixCanonicalLogicalOutputDimensions(plan);
    if (canonical_dimensions.size() != matrix.canonical_output_labels.size()) {
        throw std::runtime_error("Einsum matrix output permutation cannot operate on a scalar canonical result.");
    }
    result = result.reshape(canonical_dimensions);
    const std::vector<uint64_t> canonical_strides = denseStrides(canonical_dimensions);

    std::vector<uint64_t> requested_dimensions;
    std::vector<uint64_t> requested_strides;
    requested_dimensions.reserve(matrix.output_permutation.size());
    requested_strides.reserve(matrix.output_permutation.size());
    for (uint32_t canonical_axis : matrix.output_permutation) {
        if (canonical_axis >= canonical_dimensions.size()) {
            throw std::runtime_error("Einsum matrix output permutation contains an out-of-range canonical axis.");
        }
        requested_dimensions.push_back(canonical_dimensions[canonical_axis]);
        requested_strides.push_back(canonical_strides[canonical_axis]);
    }
    if (requested_dimensions != plan.equation.output_dimensions) {
        throw std::runtime_error("Einsum matrix output permutation produced unexpected requested dimensions.");
    }
    result = result.stridedView(requested_dimensions, requested_strides);

    // A same-dtype cast is intentional: as with operand-local permutation
    // materialization, it creates an Expression fused-stage boundary that
    // writes dense storage in the requested logical order. This copy is only
    // output-sized and can bind directly to a caller-provided output tensor.
    return result.cast(output_dtype);
}

[[nodiscard]] Expression matrixExpression(const EinsumPlan& pair_plan,
                                          LogicalEinsumOperand lhs_source,
                                          LogicalEinsumOperand rhs_source,
                                          const MatrixExpressionLayoutPlan& layout_plan,
                                          DataType output_dtype) {
    if (!pair_plan.matrix_multiply.has_value()) {
        throw std::runtime_error("Einsum matrix Expression lowering requires a matrix plan.");
    }
    const EinsumPlan& plan = pair_plan;
    const EinsumMatrixMultiplyPlan& matrix = *plan.matrix_multiply;

    Expression lhs = matrixOperandExpression(plan,
                                             std::move(lhs_source),
                                             matrix.lhs,
                                             matrix.lhs_reduction_labels,
                                             matrix.lhs_broadcast_elision_labels,
                                             matrix.m,
                                             matrix.k,
                                             layout_plan.lhs,
                                             output_dtype);
    Expression rhs = matrixOperandExpression(plan,
                                             std::move(rhs_source),
                                             matrix.rhs,
                                             matrix.rhs_reduction_labels,
                                             matrix.rhs_broadcast_elision_labels,
                                             matrix.k,
                                             matrix.n,
                                             layout_plan.rhs,
                                             output_dtype);
    // For [batch..., M, N] -> [batch..., N, M], do not materialize a GEMM
    // result and transpose it. Use (lhs @ rhs)^T = rhs^T @ lhs^T and have the
    // centralized matmul path produce the requested dense matrix orientation
    // directly. Operand-local materialization decisions are already reflected
    // in the effective transpose flags carried by layout_plan.
    Expression result = layout_plan.swapped_orientation
                            ? Expression::matmul(rhs,
                                                 lhs,
                                                 !layout_plan.rhs.transpose,
                                                 !layout_plan.lhs.transpose,
                                                 DataType::FP32,
                                                 output_dtype)
                            : Expression::matmul(lhs,
                                                 rhs,
                                                 layout_plan.lhs.transpose,
                                                 layout_plan.rhs.transpose,
                                                 DataType::FP32,
                                                 output_dtype);

    const std::vector<uint64_t> output_dimensions = physicalOutputDimensions(plan);
    if (matrix.requires_output_permutation && !layout_plan.swapped_orientation) {
        return materializeMatrixOutputPermutation(plan, std::move(result), output_dtype);
    }
    if (matrixOutputShape(plan, layout_plan.swapped_orientation) != output_dimensions) {
        result = result.reshape(output_dimensions);
    }
    return result;
}

[[nodiscard]] std::vector<int32_t> matrixPhysicalResultLabels(const EinsumMatrixMultiplyPlan& matrix,
                                                                 bool swapped_orientation) {
    std::vector<int32_t> labels;
    labels.reserve(matrix.canonical_output_labels.size());
    labels.insert(labels.end(), matrix.batch_labels.begin(), matrix.batch_labels.end());
    if (swapped_orientation) {
        labels.insert(labels.end(), matrix.rhs_free_labels.begin(), matrix.rhs_free_labels.end());
        labels.insert(labels.end(), matrix.lhs_free_labels.begin(), matrix.lhs_free_labels.end());
    } else {
        labels.insert(labels.end(), matrix.lhs_free_labels.begin(), matrix.lhs_free_labels.end());
        labels.insert(labels.end(), matrix.rhs_free_labels.begin(), matrix.rhs_free_labels.end());
    }
    return labels;
}

[[nodiscard]] std::vector<uint64_t> dimensionsForLabels(const EinsumPlan& plan,
                                                         const std::vector<int32_t>& labels) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(labels.size());
    for (int32_t label : labels) {
        if (label < 0 || static_cast<size_t>(label) >= plan.equation.label_dimensions.size()) {
            throw std::runtime_error("Einsum selected pair candidate contains an invalid result label.");
        }
        const uint64_t dimension = plan.equation.label_dimensions[static_cast<size_t>(label)];
        if (dimension == 0) {
            throw std::runtime_error("Einsum selected pair candidate references an unresolved result label.");
        }
        dimensions.push_back(dimension);
    }
    return dimensions;
}

[[nodiscard]] std::vector<uint64_t> logicalViewStridesFromPhysicalLabels(
    const std::vector<int32_t>& logical_labels,
    const std::vector<int32_t>& physical_labels,
    const std::vector<uint64_t>& physical_dimensions) {
    if (physical_labels.size() != physical_dimensions.size()) {
        throw std::runtime_error("Einsum selected pair result physical metadata has inconsistent rank.");
    }
    const std::vector<uint64_t> physical_strides = denseStrides(physical_dimensions);
    std::vector<uint64_t> logical_strides;
    logical_strides.reserve(logical_labels.size());
    for (int32_t logical_label : logical_labels) {
        const auto it = std::find(physical_labels.begin(), physical_labels.end(), logical_label);
        if (it == physical_labels.end()) {
            throw std::runtime_error("Einsum selected pair result label is absent from its physical backing.");
        }
        logical_strides.push_back(physical_strides[static_cast<size_t>(it - physical_labels.begin())]);
    }
    return logical_strides;
}

[[nodiscard]] Expression selectedMatrixCandidateExpression(
    const EinsumPlan& pair_plan,
    const EinsumPairPhysicalCandidate& candidate,
    LogicalEinsumOperand lhs_source,
    LogicalEinsumOperand rhs_source,
    const MatrixExpressionLayoutPlan& layout_plan,
    DataType output_dtype) {
    if (!pair_plan.matrix_multiply.has_value()) {
        throw std::runtime_error("Einsum selected matrix candidate requires a matrix plan.");
    }
    const EinsumMatrixMultiplyPlan& matrix = *pair_plan.matrix_multiply;
    if (layout_plan.swapped_orientation != candidate.swapped_gemm_orientation) {
        throw std::runtime_error("Einsum selected matrix candidate orientation does not match its runtime layout.");
    }
    if (layout_plan.lhs.materialize != candidate.lhs_materialized ||
        layout_plan.rhs.materialize != candidate.rhs_materialized) {
        throw std::runtime_error("Einsum selected matrix candidate materialization does not match runtime layout.");
    }

    Expression lhs = matrixOperandExpression(pair_plan,
                                             std::move(lhs_source),
                                             matrix.lhs,
                                             matrix.lhs_reduction_labels,
                                             matrix.lhs_broadcast_elision_labels,
                                             matrix.m,
                                             matrix.k,
                                             layout_plan.lhs,
                                             output_dtype);
    Expression rhs = matrixOperandExpression(pair_plan,
                                             std::move(rhs_source),
                                             matrix.rhs,
                                             matrix.rhs_reduction_labels,
                                             matrix.rhs_broadcast_elision_labels,
                                             matrix.k,
                                             matrix.n,
                                             layout_plan.rhs,
                                             output_dtype);

    Expression result = candidate.swapped_gemm_orientation
                            ? Expression::matmul(rhs,
                                                 lhs,
                                                 !layout_plan.rhs.transpose,
                                                 !layout_plan.lhs.transpose,
                                                 DataType::FP32,
                                                 output_dtype)
                            : Expression::matmul(lhs,
                                                 rhs,
                                                 layout_plan.lhs.transpose,
                                                 layout_plan.rhs.transpose,
                                                 DataType::FP32,
                                                 output_dtype);

    const std::vector<int32_t> physical_labels =
        matrixPhysicalResultLabels(matrix, candidate.swapped_gemm_orientation);
    const std::vector<uint64_t> physical_dimensions = dimensionsForLabels(pair_plan, physical_labels);
    const std::vector<uint64_t> physical_storage_dimensions =
        physical_dimensions.empty() ? std::vector<uint64_t>{1} : physical_dimensions;
    result = result.reshape(physical_storage_dimensions);

    if (physical_labels != candidate.result.labels) {
        if (candidate.result.labels.empty()) {
            throw std::runtime_error("Einsum selected matrix candidate cannot permute a non-scalar backing to scalar.");
        }
        const std::vector<uint64_t> view_strides = logicalViewStridesFromPhysicalLabels(
            candidate.result.labels, physical_labels, physical_dimensions);
        if (!candidate.output_materialized && view_strides != candidate.result.strides_elements) {
            throw std::runtime_error("Einsum selected matrix candidate runtime strides differ from planner metadata.");
        }
        result = result.stridedView(candidate.result.dimensions, view_strides);
    }

    if (candidate.output_materialized) {
        // Same-dtype cast intentionally materializes the selected logical
        // result order.  Intermediate candidates that do not request this copy
        // remain zero-copy strided views for their parent contraction.
        result = result.cast(output_dtype);
    }
    return result;
}

[[nodiscard]] EinsumPlan selectedPairExecutionPlan(const EinsumPlan& root_plan,
                                                   const EinsumPairPhysicalCandidate& candidate) {
    EinsumPlan pair_plan;
    pair_plan.kind = candidate.kind;
    pair_plan.equation.label_dimensions = root_plan.equation.label_dimensions;
    pair_plan.equation.output_labels = candidate.result.labels;
    pair_plan.equation.output_dimensions = candidate.result.dimensions;
    pair_plan.matrix_multiply = candidate.matrix_multiply;
    pair_plan.pair_product = candidate.pair_product;
    return pair_plan;
}

// Result of lowering one already-planned pair contraction. The pair lowering
// owns no scheduling state; it only builds the Expression subgraph and reports
// which existing execution family it selected. Keeping the logical operands as
// inputs to this primitive is the key seam needed by future contraction trees:
// leaves can supply original-input operands while internal nodes can supply
// intermediate Expressions with the same label/dimension/stride metadata.
struct LoweredPairContraction {
    Expression expression;
    EinsumExecutionPath execution_path = EinsumExecutionPath::GENERIC;
    bool uses_strided_batched_gemm = false;
};

[[nodiscard]] std::optional<LoweredPairContraction> lowerPairContraction(
    const EinsumPlan& pair_plan,
    LogicalEinsumOperand lhs,
    LogicalEinsumOperand rhs,
    DataType output_dtype) {
    if (pair_plan.matrix_multiply.has_value()) {
        const std::optional<MatrixExpressionLayoutPlan> matmul_layout =
            expressionMatmulLayoutPlan(pair_plan, lhs, rhs);
        if (!matmul_layout.has_value()) {
            return std::nullopt;
        }

        const EinsumExecutionPath execution_path = pair_plan.matrix_multiply->batch_labels.empty()
                                                       ? EinsumExecutionPath::GEMM
                                                       : EinsumExecutionPath::BATCHED_GEMM;
        const bool uses_strided_batched_gemm = execution_path == EinsumExecutionPath::BATCHED_GEMM
                                               && matmul_layout->matmul.grouping.batch_count > 1;
        return LoweredPairContraction{matrixExpression(pair_plan,
                                                       std::move(lhs),
                                                       std::move(rhs),
                                                       *matmul_layout,
                                                       output_dtype),
                                      execution_path,
                                      uses_strided_batched_gemm};
    }

    if (pair_plan.pair_product.has_value()) {
        return LoweredPairContraction{pairProductExpression(
                                          pair_plan, std::move(lhs), std::move(rhs), output_dtype),
                                      EinsumExecutionPath::PAIR_PRODUCT,
                                      false};
    }

    return std::nullopt;
}

[[nodiscard]] Expression selectedGenericPairExpression(
    const EinsumPlan& root_plan,
    const EinsumPairPhysicalCandidate& candidate,
    LogicalEinsumOperand lhs,
    LogicalEinsumOperand rhs,
    DataType output_dtype) {
    std::unordered_set<int32_t> present_labels(lhs.labels.begin(), lhs.labels.end());
    present_labels.insert(rhs.labels.begin(), rhs.labels.end());
    const std::unordered_set<int32_t> surviving_labels(candidate.result.labels.begin(), candidate.result.labels.end());

    std::vector<int32_t> reduction_labels;
    for (int32_t label : root_plan.iteration_labels) {
        if (present_labels.contains(label) && !surviving_labels.contains(label)) {
            reduction_labels.push_back(label);
        }
    }
    if (candidate.result.labels.size() + reduction_labels.size() != present_labels.size()) {
        throw std::runtime_error("Einsum selected generic pair could not order every participating label.");
    }

    std::vector<int32_t> iteration_labels = candidate.result.labels;
    iteration_labels.insert(iteration_labels.end(), reduction_labels.begin(), reduction_labels.end());
    Expression lhs_aligned = alignLogicalPairOperandToOutput(std::move(lhs), iteration_labels);
    Expression rhs_aligned = alignLogicalPairOperandToOutput(std::move(rhs), iteration_labels);
    Expression combined = lhs_aligned.cast(DataType::FP32) * rhs_aligned.cast(DataType::FP32);

    if (reduction_labels.empty()) {
        return output_dtype == DataType::FP32 ? combined : combined.cast(output_dtype);
    }

    std::vector<uint64_t> reduction_axes;
    reduction_axes.reserve(reduction_labels.size());
    for (size_t axis = candidate.result.labels.size(); axis < iteration_labels.size(); ++axis) {
        reduction_axes.push_back(static_cast<uint64_t>(axis));
    }
    const std::vector<uint64_t> squeeze_axes =
        candidate.result.labels.empty() ? std::vector<uint64_t>{UINT64_MAX} : reduction_axes;
    return combined.reduce_sum(reduction_axes, squeeze_axes, DataType::FP32).withOutputDType(output_dtype);
}

[[nodiscard]] std::optional<LoweredPairContraction> lowerSelectedPairContraction(
    const EinsumPlan& root_plan,
    const EinsumPairPhysicalCandidate& candidate,
    LogicalEinsumOperand lhs,
    LogicalEinsumOperand rhs,
    DataType output_dtype) {
    const EinsumPlan pair_plan = selectedPairExecutionPlan(root_plan, candidate);

    if (candidate.matrix_multiply.has_value()) {
        const std::optional<MatrixExpressionLayoutPlan> matmul_layout = expressionMatmulLayoutPlan(
            pair_plan, lhs, rhs, candidate.swapped_gemm_orientation);
        if (!matmul_layout.has_value()) {
            return std::nullopt;
        }
        if (matmul_layout->lhs.materialize != candidate.lhs_materialized ||
            matmul_layout->rhs.materialize != candidate.rhs_materialized) {
            return std::nullopt;
        }

        const EinsumExecutionPath execution_path = candidate.matrix_multiply->batch_labels.empty()
                                                       ? EinsumExecutionPath::GEMM
                                                       : EinsumExecutionPath::BATCHED_GEMM;
        const bool uses_strided_batched_gemm = execution_path == EinsumExecutionPath::BATCHED_GEMM &&
                                               matmul_layout->matmul.grouping.batch_count > 1;
        return LoweredPairContraction{selectedMatrixCandidateExpression(pair_plan,
                                                                        candidate,
                                                                        std::move(lhs),
                                                                        std::move(rhs),
                                                                        *matmul_layout,
                                                                        output_dtype),
                                      execution_path,
                                      uses_strided_batched_gemm};
    }

    if (candidate.pair_product.has_value()) {
        return LoweredPairContraction{pairProductExpression(
                                          pair_plan, std::move(lhs), std::move(rhs), output_dtype),
                                      EinsumExecutionPath::PAIR_PRODUCT,
                                      false};
    }

    if (candidate.kind == EinsumPlanKind::GENERAL) {
        return LoweredPairContraction{selectedGenericPairExpression(
                                          root_plan, candidate, std::move(lhs), std::move(rhs), output_dtype),
                                      EinsumExecutionPath::GENERIC,
                                      false};
    }

    return std::nullopt;
}

[[nodiscard]] LogicalEinsumOperand logicalIntermediateOperand(
    Expression expression,
    const EinsumLogicalOperandPlan& logical_plan) {
    if (logical_plan.labels.size() != logical_plan.dimensions.size() ||
        logical_plan.labels.size() != logical_plan.strides_elements.size()) {
        throw std::runtime_error("Einsum contraction-tree intermediate metadata has inconsistent rank.");
    }
    return LogicalEinsumOperand{std::move(expression),
                                logical_plan.labels,
                                logical_plan.dimensions,
                                logical_plan.strides_elements,
                                logical_plan.dense_storage};
}

struct LoweredContractionTree {
    Expression expression;
    bool uses_strided_batched_gemm = false;
};

[[nodiscard]] std::optional<LoweredContractionTree> lowerContractionTree(
    const EinsumPlan& plan,
    const std::vector<std::vector<uint64_t>>& input_dimensions,
    const std::vector<EinsumExactContractionStep>& steps,
    const EinsumLogicalOperandPlan& expected_result,
    DataType output_dtype) {
    const size_t operand_count = plan.logical_operands.size();
    if (operand_count < 3 || operand_count > EinsumPlanner::MAX_BEAM_SOURCE_OPERANDS) {
        return std::nullopt;
    }
    if (input_dimensions.size() != operand_count) {
        throw std::runtime_error(
            "Einsum contraction-tree input metadata count does not match its planner operands.");
    }
    if (steps.size() != operand_count - 1) {
        throw std::runtime_error(
            "Einsum contraction tree does not contain one binary step per operand merge.");
    }

    std::unordered_map<uint64_t, LogicalEinsumOperand> values;
    values.reserve(operand_count * 2);
    for (size_t operand_index = 0; operand_index < operand_count; ++operand_index) {
        values.emplace(uint64_t{1} << operand_index,
                       logicalInputOperand(operand_index, input_dimensions[operand_index], plan));
    }

    bool uses_strided_batched_gemm = false;
    for (const EinsumExactContractionStep& step : steps) {
        const auto lhs_it = values.find(step.lhs_source_mask);
        const auto rhs_it = values.find(step.rhs_source_mask);
        if (lhs_it == values.end() || rhs_it == values.end()) {
            throw std::runtime_error(
                "Einsum contraction-tree steps are not in executable postorder.");
        }

        std::optional<LoweredPairContraction> lowered = lowerSelectedPairContraction(
            plan, step.physical_candidate, lhs_it->second, rhs_it->second, output_dtype);
        if (!lowered.has_value()) {
            // Physical planning is intentionally speculative. If the selected
            // candidate cannot be reconstructed by the execution backend, keep
            // the established whole-equation generic expression as a correctness
            // fallback rather than making stamp-time planning a new failure mode.
            return std::nullopt;
        }
        uses_strided_batched_gemm = uses_strided_batched_gemm || lowered->uses_strided_batched_gemm;

        LogicalEinsumOperand intermediate = logicalIntermediateOperand(
            std::move(lowered->expression), step.physical_candidate.result);
        values.erase(lhs_it);
        values.erase(rhs_it);
        const auto [inserted_it, inserted] =
            values.emplace(step.result_source_mask, std::move(intermediate));
        if (!inserted) {
            throw std::runtime_error(
                "Einsum contraction tree produced a duplicate subset result.");
        }
        (void)inserted_it;
    }

    const uint64_t full_mask = (uint64_t{1} << operand_count) - 1;
    const auto root_it = values.find(full_mask);
    if (root_it == values.end()) {
        throw std::runtime_error(
            "Einsum contraction tree did not produce the complete operand subset.");
    }
    if (root_it->second.labels != expected_result.labels ||
        root_it->second.dimensions != expected_result.dimensions ||
        root_it->second.strides != expected_result.strides_elements ||
        root_it->second.dense_storage != expected_result.dense_storage) {
        throw std::runtime_error(
            "Einsum contraction-tree runtime result metadata differs from its planner result.");
    }
    return LoweredContractionTree{root_it->second.expression, uses_strided_batched_gemm};
}

[[nodiscard]] Expression genericExpression(const EinsumPlan& plan,
                                            const std::vector<Tensor>& inputs,
                                            DataType output_dtype) {
    std::vector<Expression> aligned_operands;
    aligned_operands.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        aligned_operands.push_back(alignedOperandExpression(i, inputs[i], plan));
    }

    Expression combined = aligned_operands.front();
    if (aligned_operands.size() > 1) {
        combined = combined.cast(DataType::FP32);
        for (size_t i = 1; i < aligned_operands.size(); ++i) {
            combined = combined * aligned_operands[i].cast(DataType::FP32);
        }
    }

    if (plan.reduction_axes.empty()) {
        if (aligned_operands.size() > 1 && output_dtype != DataType::FP32) {
            combined = combined.cast(output_dtype);
        }
        return combined;
    }

    std::vector<uint64_t> reduction_axes;
    reduction_axes.reserve(plan.reduction_axes.size());
    for (uint32_t axis : plan.reduction_axes) {
        reduction_axes.push_back(axis);
    }

    // Expression reductions retain reduced axes as size-one dimensions unless
    // they are explicitly squeezed. The einsum logical result removes every
    // contraction axis; a full scalar reduction uses the established physical
    // scalar representation {1}.
    const std::vector<uint64_t> squeeze_axes =
        plan.equation.output_dimensions.empty() ? std::vector<uint64_t>{UINT64_MAX} : reduction_axes;
    return combined.reduce_sum(reduction_axes, squeeze_axes, DataType::FP32).withOutputDType(output_dtype);
}

[[nodiscard]] std::shared_ptr<StampedEinsum> stampImpl(const std::string& equation,
                                                       const std::vector<Tensor>& inputs,
                                                       const std::optional<Tensor>& preallocated_output,
                                                       const Stream& stream,
                                                       bool force_generic_reference = false) {
    requireSupportedInputs(inputs, stream);

    std::vector<std::vector<uint64_t>> input_dimensions;
    input_dimensions.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        input_dimensions.push_back(input.getDimensions());
    }
    EinsumPlan plan = EinsumPlanner::parseAndPlan(equation, input_dimensions);
    const std::vector<uint64_t> output_dimensions = physicalOutputDimensions(plan);
    (void)checkedProduct(output_dimensions, "output");

    const DataType output_dtype = inputs.front().getDataType();
    if (preallocated_output.has_value()) {
        requireOutput(preallocated_output.value(), inputs, output_dimensions, output_dtype);
    }

    EinsumExecutionPath execution_path = EinsumExecutionPath::GENERIC;
    bool uses_strided_batched_gemm = false;
    Expression result = [&]() -> Expression {
        if (force_generic_reference) {
            return genericExpression(plan, inputs, output_dtype);
        }
        if (plan.exact_contraction.has_value()) {
            const EinsumExactContractionPlan& exact = *plan.exact_contraction;
            std::optional<LoweredContractionTree> lowered = lowerContractionTree(
                plan, input_dimensions, exact.steps, exact.result, output_dtype);
            if (lowered.has_value()) {
                execution_path = EinsumExecutionPath::EXACT_CONTRACTION;
                uses_strided_batched_gemm = lowered->uses_strided_batched_gemm;
                return std::move(lowered->expression);
            }
        }
        if (plan.beam_contraction.has_value()) {
            const EinsumBeamContractionPlan& beam = *plan.beam_contraction;
            std::optional<LoweredContractionTree> lowered = lowerContractionTree(
                plan, input_dimensions, beam.steps, beam.result, output_dtype);
            if (lowered.has_value()) {
                execution_path = EinsumExecutionPath::BEAM_CONTRACTION;
                uses_strided_batched_gemm = lowered->uses_strided_batched_gemm;
                return std::move(lowered->expression);
            }
        }
        if (inputs.size() == 2 && (plan.matrix_multiply.has_value() || plan.pair_product.has_value())) {
            LogicalEinsumOperand lhs = logicalInputOperand(0, input_dimensions[0], plan);
            LogicalEinsumOperand rhs = logicalInputOperand(1, input_dimensions[1], plan);
            std::optional<LoweredPairContraction> lowered =
                lowerPairContraction(plan, std::move(lhs), std::move(rhs), output_dtype);
            if (lowered.has_value()) {
                execution_path = lowered->execution_path;
                uses_strided_batched_gemm = lowered->uses_strided_batched_gemm;
                return std::move(lowered->expression);
            }
        }
        return genericExpression(plan, inputs, output_dtype);
    }();

    FusedEquation expression_equation = FusedEquation::compile(
        Expression::outputs({{kOutputName, result}}).physicalOutputs(), inputs.front().getPlacement().getDeviceNum());

    std::unordered_map<std::string, Tensor> preallocated_outputs;
    if (preallocated_output.has_value()) {
        preallocated_outputs.emplace(kOutputName, preallocated_output.value());
    } else if (execution_path == EinsumExecutionPath::GENERIC && inputs.size() == 1 && plan.reduction_axes.empty()) {
        // A unary permutation/identity can otherwise finish as a zero-copy alias
        // of its input. Einsum is an allocating operation, so force only this
        // view-only case into fresh dense output storage. Matrix and reduction
        // stages already produce their own dense output allocation.
        preallocated_outputs.emplace(
            kOutputName, Tensor(inputs.front().getPlacement(), TensorDescriptor(output_dtype, output_dimensions)));
    }
    StampedExecutionPlan stamped = expression_equation.stamp(expressionInputs(inputs), stream, {}, preallocated_outputs);
    Tensor output = stamped.output(kOutputName);
    if (output.getDimensions() != output_dimensions || output.getDataType() != output_dtype) {
        throw std::runtime_error("Einsum Expression lowering produced an unexpected output descriptor.");
    }

    return std::make_shared<StampedEinsum>(std::move(plan),
                                           output,
                                           stream,
                                           execution_path,
                                           uses_strided_batched_gemm,
                                           std::make_shared<StampedExecutionPlan>(std::move(stamped)));
}

}  // namespace

Einsum::Einsum(std::string equation) : equation(std::move(equation)) {
    if (this->equation.empty()) {
        throw std::invalid_argument("Einsum equation must not be empty.");
    }
}

std::shared_ptr<StampedEinsum> Einsum::stamp(const std::vector<Tensor>& inputs, const Stream& stream) const {
    return stampImpl(equation, inputs, std::nullopt, stream);
}

std::shared_ptr<StampedEinsum> Einsum::stamp(const std::vector<Tensor>& inputs,
                                             const Tensor& preallocated_output,
                                             const Stream& stream) const {
    return stampImpl(equation, inputs, preallocated_output, stream);
}

std::shared_ptr<StampedEinsum> Einsum::stampGenericReference(
    const std::vector<Tensor>& inputs, const Stream& stream) const {
    return stampImpl(equation, inputs, std::nullopt, stream, true);
}

StampedEinsum::StampedEinsum(EinsumPlan plan,
                             Tensor output,
                             const Stream& stream,
                             EinsumExecutionPath execution_path,
                             bool uses_strided_batched_gemm,
                             std::shared_ptr<StampedExecutionPlan> execution)
    : plan(std::move(plan)),
      output(std::move(output)),
      stream(stream),
      execution_path(execution_path),
      uses_strided_batched_gemm(uses_strided_batched_gemm),
      execution(std::move(execution)) {
    if (!this->execution) {
        throw std::invalid_argument("StampedEinsum requires a stamped Expression execution plan.");
    }
}

void StampedEinsum::run() { runOn(stream); }

void StampedEinsum::runOn(Stream& run_stream) const {
    if (!run_stream.isInitialized()) {
        throw std::invalid_argument("StampedEinsum::runOn requires an initialized stream.");
    }
    if (run_stream.getGpuNum() != output.getPlacement().getDeviceNum()) {
        throw std::invalid_argument("StampedEinsum::runOn stream must use the stamped output GPU.");
    }
    execution->runOn(run_stream);
}

bool StampedEinsum::usesStandaloneReduction() const { return !execution->reductionPaths().empty(); }

std::vector<CubReductionPath> StampedEinsum::getStandaloneReductionPaths() const {
    return execution->reductionPaths();
}

std::optional<CubReductionPath> StampedEinsum::getStandaloneReductionPath() const {
    const std::vector<CubReductionPath> paths = getStandaloneReductionPaths();
    if (paths.empty()) {
        return std::nullopt;
    }
    if (paths.size() != 1) {
        throw std::runtime_error(
            "StampedEinsum has multiple standalone Expression reduction stages; use getStandaloneReductionPaths().");
    }
    return paths.front();
}

std::vector<std::string> StampedEinsum::getExpressionStageKindNames() const { return execution->stageKindNames(); }

std::vector<StampedMatmulStageDiagnostic> StampedEinsum::getExpressionMatmulStageDiagnostics() const {
    return execution->matmulStageDiagnostics();
}

}  // namespace ThorImplementation
