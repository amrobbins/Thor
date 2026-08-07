#pragma once

#include <cstdint>
#include <vector>

namespace ThorImplementation {

class Tensor;

struct BatchedMatmulShapePlan {
    bool transpose_lhs = false;
    bool transpose_rhs = false;
    std::vector<uint64_t> lhs_dimensions;
    std::vector<uint64_t> rhs_dimensions;
    std::vector<uint64_t> lhs_aligned_batch_dimensions;
    std::vector<uint64_t> rhs_aligned_batch_dimensions;
    std::vector<uint64_t> batch_dimensions;
    std::vector<uint64_t> output_dimensions;
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t batch_count = 1;

    bool operator==(const BatchedMatmulShapePlan& other) const = default;
};

struct MatmulTensorLayout {
    std::vector<uint64_t> dimensions;
    std::vector<uint64_t> strides_elements;
    uint64_t storage_element_offset = 0;

    bool operator==(const MatmulTensorLayout& other) const = default;
};

enum class MatmulMatrixStorageKind : uint8_t {
    RowMajor = 0,
    TransposedRowMajor,
    Unsupported,
};

struct MatmulMatrixPlanePlan {
    MatmulMatrixStorageKind storage_kind = MatmulMatrixStorageKind::Unsupported;
    uint64_t visible_rows = 0;
    uint64_t visible_cols = 0;
    uint64_t visible_row_stride = 0;
    uint64_t visible_col_stride = 0;
    uint64_t stored_rows = 0;
    uint64_t stored_cols = 0;
    uint64_t leading_dimension = 0;
    uint64_t storage_span_elements = 0;
    bool logical_transpose = false;
    bool backend_transpose = false;

    [[nodiscard]] bool isBlasAddressable() const { return storage_kind != MatmulMatrixStorageKind::Unsupported; }
    [[nodiscard]] bool storageIsTransposedRelativeToVisible() const {
        return storage_kind == MatmulMatrixStorageKind::TransposedRowMajor;
    }

    bool operator==(const MatmulMatrixPlanePlan& other) const = default;
};

struct MatmulBatchAxisPlan {
    uint64_t extent = 1;
    uint64_t lhs_stride_elements = 0;
    uint64_t rhs_stride_elements = 0;
    uint64_t output_stride_elements = 0;
    bool lhs_broadcast = false;
    bool rhs_broadcast = false;

    bool operator==(const MatmulBatchAxisPlan& other) const = default;
};

// A compact decomposition into identical strided-batched groups. The listed
// batch axes vary inside one group in logical axis order; all other batch axes
// are fixed for a group and enumerate the independent groups. The axes need
// not be contiguous: custom/permuted batch layouts can still form one regular
// strided batch when their physical strides describe the same flattened order.
// An empty varying_axes vector means one matrix multiply per group.
struct MatmulBatchGroupingPlan {
    std::vector<uint32_t> varying_axes;
    uint64_t batch_count = 1;
    uint64_t group_count = 1;
    uint64_t lhs_batch_stride_elements = 0;
    uint64_t rhs_batch_stride_elements = 0;
    uint64_t output_batch_stride_elements = 0;

    [[nodiscard]] bool hasVaryingBatchAxes() const { return !varying_axes.empty(); }
    [[nodiscard]] bool isSingleStridedBatch() const { return group_count == 1; }

    bool operator==(const MatmulBatchGroupingPlan& other) const = default;
};

struct BatchedMatmulLayoutPlan {
    BatchedMatmulShapePlan shape;
    MatmulTensorLayout lhs_layout;
    MatmulTensorLayout rhs_layout;
    MatmulTensorLayout output_layout;
    MatmulMatrixPlanePlan lhs_matrix;
    MatmulMatrixPlanePlan rhs_matrix;
    MatmulMatrixPlanePlan output_matrix;
    std::vector<MatmulBatchAxisPlan> batch_axes;
    MatmulBatchGroupingPlan grouping;

    [[nodiscard]] bool canAddressOperandsWithoutMaterialization() const {
        return lhs_matrix.isBlasAddressable() && rhs_matrix.isBlasAddressable();
    }
    [[nodiscard]] bool canWriteOutputWithoutPostprocess() const {
        return output_matrix.storage_kind == MatmulMatrixStorageKind::RowMajor;
    }
    [[nodiscard]] bool canLowerWithoutMaterialization() const {
        return canAddressOperandsWithoutMaterialization() && canWriteOutputWithoutPostprocess();
    }

    bool operator==(const BatchedMatmulLayoutPlan& other) const = default;
};

struct MatmulBatchGroup {
    uint64_t batch_count = 1;
    uint64_t lhs_relative_element_offset = 0;
    uint64_t rhs_relative_element_offset = 0;
    uint64_t output_relative_element_offset = 0;
    uint64_t lhs_batch_stride_elements = 0;
    uint64_t rhs_batch_stride_elements = 0;
    uint64_t output_batch_stride_elements = 0;

    bool operator==(const MatmulBatchGroup& other) const = default;
};

[[nodiscard]] BatchedMatmulShapePlan planBatchedMatmulShape(const std::vector<uint64_t>& lhs_dimensions,
                                                            const std::vector<uint64_t>& rhs_dimensions,
                                                            bool transpose_lhs = false,
                                                            bool transpose_rhs = false);

[[nodiscard]] MatmulTensorLayout denseMatmulTensorLayout(const std::vector<uint64_t>& dimensions,
                                                         uint64_t storage_element_offset = 0);
[[nodiscard]] MatmulTensorLayout matmulTensorLayout(const Tensor& tensor);

[[nodiscard]] BatchedMatmulLayoutPlan planBatchedMatmulLayout(const MatmulTensorLayout& lhs,
                                                              const MatmulTensorLayout& rhs,
                                                              const MatmulTensorLayout& output,
                                                              bool transpose_lhs = false,
                                                              bool transpose_rhs = false);
[[nodiscard]] BatchedMatmulLayoutPlan planBatchedMatmulLayout(const Tensor& lhs,
                                                              const Tensor& rhs,
                                                              const Tensor& output,
                                                              bool transpose_lhs = false,
                                                              bool transpose_rhs = false);

// Expands the compact grouping metadata into concrete groups. Intended for the
// compiler/lowering layer and tests, not runtime scheduling. Throws rather than
// accidentally allocating an enormous vector if max_groups would be exceeded.
[[nodiscard]] std::vector<MatmulBatchGroup> materializeBatchedMatmulGroups(const BatchedMatmulLayoutPlan& plan,
                                                                           uint64_t max_groups = 1'000'000);

}  // namespace ThorImplementation
