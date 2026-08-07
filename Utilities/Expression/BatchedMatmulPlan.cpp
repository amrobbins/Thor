#include "Utilities/Expression/BatchedMatmulPlan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation {
namespace {

uint64_t checkedMultiply(uint64_t a, uint64_t b, const char* context) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string(context) + " dimension product overflowed.");
    }
    return a * b;
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const char* context) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::runtime_error(std::string(context) + " element offset overflowed.");
    }
    return a + b;
}

std::vector<uint64_t> denseStrides(const std::vector<uint64_t>& dimensions) {
    if (dimensions.empty()) {
        throw std::runtime_error("Matmul tensor layouts require rank at least two.");
    }
    std::vector<uint64_t> strides(dimensions.size(), 1);
    uint64_t running = 1;
    for (size_t i = dimensions.size(); i-- > 0;) {
        strides[i] = running;
        running = checkedMultiply(running, dimensions[i], "Dense matmul layout");
    }
    return strides;
}

void validateDimensions(const std::vector<uint64_t>& dimensions, const char* role) {
    if (dimensions.size() < 2) {
        throw std::runtime_error(std::string(role) + " matmul operand must have rank at least two.");
    }
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::runtime_error(std::string(role) + " matmul dimensions must be non-zero.");
        }
    }
}

void validateTensorLayout(const MatmulTensorLayout& layout, const char* role) {
    validateDimensions(layout.dimensions, role);
    if (layout.strides_elements.size() != layout.dimensions.size()) {
        throw std::runtime_error(std::string(role) + " matmul layout dimensions and strides must have equal rank.");
    }

    uint64_t max_relative_offset = 0;
    for (size_t axis = 0; axis < layout.dimensions.size(); ++axis) {
        const uint64_t extent_minus_one = layout.dimensions[axis] - 1;
        const uint64_t contribution = checkedMultiply(extent_minus_one, layout.strides_elements[axis], role);
        max_relative_offset = checkedAdd(max_relative_offset, contribution, role);
    }
    (void)checkedAdd(layout.storage_element_offset, max_relative_offset, role);
}

std::vector<uint64_t> alignBatchDimensions(const std::vector<uint64_t>& dimensions, size_t batch_rank) {
    const size_t source_batch_rank = dimensions.size() - 2;
    std::vector<uint64_t> aligned(batch_rank, 1);
    const size_t leading = batch_rank - source_batch_rank;
    for (size_t axis = 0; axis < source_batch_rank; ++axis) {
        aligned[leading + axis] = dimensions[axis];
    }
    return aligned;
}

std::vector<uint64_t> alignBatchStrides(const MatmulTensorLayout& layout,
                                        const std::vector<uint64_t>& output_batch_dimensions) {
    const size_t batch_rank = output_batch_dimensions.size();
    const size_t source_batch_rank = layout.dimensions.size() - 2;
    std::vector<uint64_t> aligned(batch_rank, 0);
    const size_t leading = batch_rank - source_batch_rank;
    for (size_t axis = 0; axis < source_batch_rank; ++axis) {
        const size_t out_axis = leading + axis;
        const uint64_t source_extent = layout.dimensions[axis];
        aligned[out_axis] = source_extent == 1 ? 0 : layout.strides_elements[axis];
    }
    return aligned;
}

MatmulMatrixPlanePlan planMatrixPlane(const MatmulTensorLayout& layout, bool logical_transpose) {
    const size_t rank = layout.dimensions.size();
    const uint64_t rows = layout.dimensions[rank - 2];
    const uint64_t cols = layout.dimensions[rank - 1];
    const uint64_t row_stride = layout.strides_elements[rank - 2];
    const uint64_t col_stride = layout.strides_elements[rank - 1];

    MatmulMatrixPlanePlan plan;
    plan.visible_rows = rows;
    plan.visible_cols = cols;
    plan.visible_row_stride = row_stride;
    plan.visible_col_stride = col_stride;
    plan.logical_transpose = logical_transpose;

    const bool row_major = (cols == 1 || col_stride == 1) && (rows == 1 || row_stride >= cols);
    if (row_major) {
        plan.storage_kind = MatmulMatrixStorageKind::RowMajor;
        plan.stored_rows = rows;
        plan.stored_cols = cols;
        plan.leading_dimension = rows == 1 ? cols : row_stride;
        plan.backend_transpose = logical_transpose;
    } else {
        const bool transposed_row_major = (rows == 1 || row_stride == 1) && (cols == 1 || col_stride >= rows);
        if (transposed_row_major) {
            plan.storage_kind = MatmulMatrixStorageKind::TransposedRowMajor;
            plan.stored_rows = cols;
            plan.stored_cols = rows;
            plan.leading_dimension = cols == 1 ? rows : col_stride;
            plan.backend_transpose = !logical_transpose;
        } else {
            plan.storage_kind = MatmulMatrixStorageKind::Unsupported;
            plan.stored_rows = rows;
            plan.stored_cols = cols;
            plan.leading_dimension = 0;
            plan.backend_transpose = logical_transpose;
        }
    }

    uint64_t max_relative_offset = 0;
    max_relative_offset = checkedAdd(max_relative_offset, checkedMultiply(rows - 1, row_stride, "Matmul matrix plane"), "Matmul matrix plane");
    max_relative_offset = checkedAdd(max_relative_offset, checkedMultiply(cols - 1, col_stride, "Matmul matrix plane"), "Matmul matrix plane");
    plan.storage_span_elements = checkedAdd(max_relative_offset, 1, "Matmul matrix plane");
    return plan;
}

bool strideCanAddressDistinctMatrices(uint64_t flat_stride, uint64_t matrix_span, bool allow_zero) {
    if (flat_stride == 0) {
        return allow_zero;
    }
    return flat_stride >= matrix_span;
}

bool canAppendBatchAxis(uint32_t previous_axis,
                        uint32_t next_axis,
                        const std::vector<uint64_t>& extents,
                        const std::vector<uint64_t>& lhs_strides,
                        const std::vector<uint64_t>& rhs_strides,
                        const std::vector<uint64_t>& output_strides) {
    const uint64_t next_extent = extents[next_axis];
    const auto stride_relation_holds = [&](const std::vector<uint64_t>& strides) {
        const uint64_t next_stride = strides[next_axis];
        if (next_stride != 0 && next_extent > std::numeric_limits<uint64_t>::max() / next_stride) {
            return false;
        }
        return strides[previous_axis] == next_extent * next_stride;
    };
    return stride_relation_holds(lhs_strides) && stride_relation_holds(rhs_strides) && stride_relation_holds(output_strides);
}

MatmulBatchGroupingPlan chooseGrouping(const std::vector<uint64_t>& extents,
                                       const std::vector<uint64_t>& lhs_strides,
                                       const std::vector<uint64_t>& rhs_strides,
                                       const std::vector<uint64_t>& output_strides,
                                       const MatmulMatrixPlanePlan& lhs_matrix,
                                       const MatmulMatrixPlanePlan& rhs_matrix,
                                       const MatmulMatrixPlanePlan& output_matrix) {
    MatmulBatchGroupingPlan best;
    uint64_t total_batch_count = 1;
    for (uint64_t extent : extents) {
        total_batch_count = checkedMultiply(total_batch_count, extent, "Matmul batch count");
    }
    best.group_count = total_batch_count;
    if (total_batch_count == 1 || extents.empty()) {
        return best;
    }

    const size_t rank = extents.size();
    std::vector<uint64_t> best_product_ending_at(rank, 0);
    std::vector<uint32_t> predecessor(rank, UINT32_MAX);
    std::vector<uint32_t> path_length(rank, 0);

    uint64_t best_batch_count = 1;
    uint32_t best_end_axis = UINT32_MAX;
    uint32_t best_path_length = 0;

    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (extents[axis] <= 1) {
            continue;
        }
        if (!strideCanAddressDistinctMatrices(lhs_strides[axis], lhs_matrix.storage_span_elements, true) ||
            !strideCanAddressDistinctMatrices(rhs_strides[axis], rhs_matrix.storage_span_elements, true) ||
            !strideCanAddressDistinctMatrices(output_strides[axis], output_matrix.storage_span_elements, false)) {
            continue;
        }

        best_product_ending_at[axis] = extents[axis];
        path_length[axis] = 1;
        for (uint32_t previous = 0; previous < axis; ++previous) {
            if (best_product_ending_at[previous] == 0 ||
                !canAppendBatchAxis(previous, axis, extents, lhs_strides, rhs_strides, output_strides)) {
                continue;
            }
            const uint64_t candidate_product =
                checkedMultiply(best_product_ending_at[previous], extents[axis], "Matmul grouped batch");
            const uint32_t candidate_length = path_length[previous] + 1;
            if (candidate_product > best_product_ending_at[axis] ||
                (candidate_product == best_product_ending_at[axis] && candidate_length > path_length[axis])) {
                best_product_ending_at[axis] = candidate_product;
                predecessor[axis] = previous;
                path_length[axis] = candidate_length;
            }
        }

        if (best_product_ending_at[axis] > best_batch_count ||
            (best_product_ending_at[axis] == best_batch_count && path_length[axis] > best_path_length)) {
            best_batch_count = best_product_ending_at[axis];
            best_end_axis = axis;
            best_path_length = path_length[axis];
        }
    }

    if (best_end_axis == UINT32_MAX) {
        return best;
    }

    for (uint32_t axis = best_end_axis; axis != UINT32_MAX; axis = predecessor[axis]) {
        best.varying_axes.push_back(axis);
    }
    std::reverse(best.varying_axes.begin(), best.varying_axes.end());
    best.batch_count = best_batch_count;
    best.group_count = total_batch_count / best_batch_count;
    best.lhs_batch_stride_elements = lhs_strides[best_end_axis];
    best.rhs_batch_stride_elements = rhs_strides[best_end_axis];
    best.output_batch_stride_elements = output_strides[best_end_axis];
    return best;
}

uint64_t fixedAxesOffsetForGroup(uint64_t group_index,
                                 const BatchedMatmulLayoutPlan& plan,
                                 bool lhs,
                                 bool rhs,
                                 bool output) {
    if (static_cast<int>(lhs) + static_cast<int>(rhs) + static_cast<int>(output) != 1) {
        throw std::runtime_error("Matmul group offset helper requires exactly one tensor role.");
    }
    const auto& grouping = plan.grouping;
    uint64_t offset = 0;
    uint64_t remaining = group_index;

    for (size_t axis = plan.batch_axes.size(); axis-- > 0;) {
        const bool varying = std::find(grouping.varying_axes.begin(), grouping.varying_axes.end(), static_cast<uint32_t>(axis)) !=
                             grouping.varying_axes.end();
        if (varying) {
            continue;
        }
        const uint64_t extent = plan.batch_axes[axis].extent;
        const uint64_t coordinate = remaining % extent;
        remaining /= extent;
        uint64_t stride = 0;
        if (lhs) {
            stride = plan.batch_axes[axis].lhs_stride_elements;
        } else if (rhs) {
            stride = plan.batch_axes[axis].rhs_stride_elements;
        } else {
            stride = plan.batch_axes[axis].output_stride_elements;
        }
        offset = checkedAdd(offset, checkedMultiply(coordinate, stride, "Matmul batch group"), "Matmul batch group");
    }
    if (remaining != 0) {
        throw std::runtime_error("Matmul batch group index exceeded the compact grouping extent.");
    }
    return offset;
}

}  // namespace

BatchedMatmulShapePlan planBatchedMatmulShape(const std::vector<uint64_t>& lhs_dimensions,
                                              const std::vector<uint64_t>& rhs_dimensions,
                                              bool transpose_lhs,
                                              bool transpose_rhs) {
    validateDimensions(lhs_dimensions, "LHS");
    validateDimensions(rhs_dimensions, "RHS");

    BatchedMatmulShapePlan plan;
    plan.transpose_lhs = transpose_lhs;
    plan.transpose_rhs = transpose_rhs;
    plan.lhs_dimensions = lhs_dimensions;
    plan.rhs_dimensions = rhs_dimensions;

    const uint64_t lhs_rows = lhs_dimensions[lhs_dimensions.size() - 2];
    const uint64_t lhs_cols = lhs_dimensions[lhs_dimensions.size() - 1];
    const uint64_t rhs_rows = rhs_dimensions[rhs_dimensions.size() - 2];
    const uint64_t rhs_cols = rhs_dimensions[rhs_dimensions.size() - 1];
    plan.m = transpose_lhs ? lhs_cols : lhs_rows;
    const uint64_t lhs_k = transpose_lhs ? lhs_rows : lhs_cols;
    const uint64_t rhs_k = transpose_rhs ? rhs_cols : rhs_rows;
    plan.n = transpose_rhs ? rhs_rows : rhs_cols;
    if (lhs_k != rhs_k) {
        throw std::runtime_error("Batched matmul found incompatible matrix dimensions.");
    }
    plan.k = lhs_k;

    const size_t batch_rank = std::max(lhs_dimensions.size(), rhs_dimensions.size()) - 2;
    plan.lhs_aligned_batch_dimensions = alignBatchDimensions(lhs_dimensions, batch_rank);
    plan.rhs_aligned_batch_dimensions = alignBatchDimensions(rhs_dimensions, batch_rank);
    plan.batch_dimensions.resize(batch_rank, 1);
    for (size_t axis = 0; axis < batch_rank; ++axis) {
        const uint64_t lhs_extent = plan.lhs_aligned_batch_dimensions[axis];
        const uint64_t rhs_extent = plan.rhs_aligned_batch_dimensions[axis];
        if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1) {
            throw std::runtime_error("Batched matmul found incompatible batch dimensions.");
        }
        plan.batch_dimensions[axis] = std::max(lhs_extent, rhs_extent);
        plan.batch_count = checkedMultiply(plan.batch_count, plan.batch_dimensions[axis], "Batched matmul");
    }

    plan.output_dimensions = plan.batch_dimensions;
    plan.output_dimensions.push_back(plan.m);
    plan.output_dimensions.push_back(plan.n);
    return plan;
}

MatmulTensorLayout denseMatmulTensorLayout(const std::vector<uint64_t>& dimensions, uint64_t storage_element_offset) {
    validateDimensions(dimensions, "Dense");
    return MatmulTensorLayout{dimensions, denseStrides(dimensions), storage_element_offset};
}

BatchedMatmulLayoutPlan planBatchedMatmulLayout(const MatmulTensorLayout& lhs,
                                                const MatmulTensorLayout& rhs,
                                                const MatmulTensorLayout& output,
                                                bool transpose_lhs,
                                                bool transpose_rhs) {
    validateTensorLayout(lhs, "LHS");
    validateTensorLayout(rhs, "RHS");
    validateTensorLayout(output, "Output");

    BatchedMatmulLayoutPlan plan;
    plan.shape = planBatchedMatmulShape(lhs.dimensions, rhs.dimensions, transpose_lhs, transpose_rhs);
    if (output.dimensions != plan.shape.output_dimensions) {
        throw std::runtime_error("Batched matmul output layout dimensions do not match the inferred output shape.");
    }
    plan.lhs_layout = lhs;
    plan.rhs_layout = rhs;
    plan.output_layout = output;
    plan.lhs_matrix = planMatrixPlane(lhs, transpose_lhs);
    plan.rhs_matrix = planMatrixPlane(rhs, transpose_rhs);
    plan.output_matrix = planMatrixPlane(output, false);

    const std::vector<uint64_t> lhs_batch_strides = alignBatchStrides(lhs, plan.shape.batch_dimensions);
    const std::vector<uint64_t> rhs_batch_strides = alignBatchStrides(rhs, plan.shape.batch_dimensions);
    const size_t output_batch_rank = output.dimensions.size() - 2;
    if (output_batch_rank != plan.shape.batch_dimensions.size()) {
        throw std::runtime_error("Batched matmul output layout has an unexpected batch rank.");
    }

    std::vector<uint64_t> output_batch_strides(plan.shape.batch_dimensions.size(), 0);
    for (size_t axis = 0; axis < output_batch_rank; ++axis) {
        output_batch_strides[axis] = output.strides_elements[axis];
    }

    plan.batch_axes.reserve(plan.shape.batch_dimensions.size());
    for (size_t axis = 0; axis < plan.shape.batch_dimensions.size(); ++axis) {
        MatmulBatchAxisPlan batch_axis;
        batch_axis.extent = plan.shape.batch_dimensions[axis];
        batch_axis.lhs_stride_elements = lhs_batch_strides[axis];
        batch_axis.rhs_stride_elements = rhs_batch_strides[axis];
        batch_axis.output_stride_elements = output_batch_strides[axis];
        batch_axis.lhs_broadcast = plan.shape.lhs_aligned_batch_dimensions[axis] == 1 && batch_axis.extent > 1;
        batch_axis.rhs_broadcast = plan.shape.rhs_aligned_batch_dimensions[axis] == 1 && batch_axis.extent > 1;
        plan.batch_axes.push_back(batch_axis);
    }

    plan.grouping = chooseGrouping(plan.shape.batch_dimensions,
                                   lhs_batch_strides,
                                   rhs_batch_strides,
                                   output_batch_strides,
                                   plan.lhs_matrix,
                                   plan.rhs_matrix,
                                   plan.output_matrix);
    return plan;
}

std::vector<MatmulBatchGroup> materializeBatchedMatmulGroups(const BatchedMatmulLayoutPlan& plan, uint64_t max_groups) {
    if (plan.grouping.group_count > max_groups) {
        throw std::runtime_error("Batched matmul grouping exceeds the requested materialization limit.");
    }
    if (plan.grouping.group_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("Batched matmul grouping cannot be represented by a host vector.");
    }

    std::vector<MatmulBatchGroup> groups;
    groups.reserve(static_cast<size_t>(plan.grouping.group_count));
    for (uint64_t group_index = 0; group_index < plan.grouping.group_count; ++group_index) {
        MatmulBatchGroup group;
        group.batch_count = plan.grouping.batch_count;
        group.lhs_relative_element_offset = fixedAxesOffsetForGroup(group_index, plan, true, false, false);
        group.rhs_relative_element_offset = fixedAxesOffsetForGroup(group_index, plan, false, true, false);
        group.output_relative_element_offset = fixedAxesOffsetForGroup(group_index, plan, false, false, true);
        group.lhs_batch_stride_elements = plan.grouping.lhs_batch_stride_elements;
        group.rhs_batch_stride_elements = plan.grouping.rhs_batch_stride_elements;
        group.output_batch_stride_elements = plan.grouping.output_batch_stride_elements;
        groups.push_back(group);
    }
    return groups;
}

}  // namespace ThorImplementation
