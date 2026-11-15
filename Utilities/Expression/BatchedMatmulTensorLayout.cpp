#include "Utilities/Expression/BatchedMatmulPlan.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation {

namespace {

uint64_t checkedGroupAdd(uint64_t a, uint64_t b, const std::string& what) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::runtime_error(what + " overflows uint64_t.");
    }
    return a + b;
}

StridedSpan matmulGroupOperand(const StridedSpan& operand,
                               uint64_t batch_count,
                               uint64_t batch_stride_elements,
                               uint64_t relative_element_offset,
                               const char* role) {
    const std::string what = std::string("Grouped MATMUL ") + role;
    const std::vector<uint64_t>& dims = operand.getDimensions();
    const std::vector<uint64_t>& strides = operand.getStridesElements();
    if (dims.size() < 2 || strides.size() != dims.size()) {
        throw std::runtime_error(what + " requires a rank >= 2 tensor with valid strides.");
    }
    if (batch_count == 0) {
        throw std::runtime_error(what + " requires a non-zero batch count.");
    }
    const uint64_t rows = dims[dims.size() - 2];
    const uint64_t cols = dims[dims.size() - 1];
    const uint64_t row_stride = strides[strides.size() - 2];
    const uint64_t col_stride = strides[strides.size() - 1];

    // The group starts relative_element_offset elements past the operand's first element; its span covers every one
    // of its matrices' elements, and the constructor throws if that reaches past the operand's storage.
    return StridedSpan(operand.getSourceTensor(),
                       checkedGroupAdd(operand.getStorage().offset, relative_element_offset, what + " offset"),
                       {batch_count, rows, cols},
                       {batch_stride_elements, row_stride, col_stride});
}

}  // namespace

MatmulTensorLayout matmulTensorLayout(const StridedSpan& operand) {
    return MatmulTensorLayout{operand.getDimensions(),
                              operand.getStridesElements(),
                              operand.getStorage().offset};
}

BatchedMatmulLayoutPlan planBatchedMatmulLayout(const StridedSpan& lhs,
                                                const StridedSpan& rhs,
                                                const StridedSpan& output,
                                                bool transpose_lhs,
                                                bool transpose_rhs) {
    return planBatchedMatmulLayout(matmulTensorLayout(lhs), matmulTensorLayout(rhs), matmulTensorLayout(output), transpose_lhs, transpose_rhs);
}

MatmulOperands matmulGroupOperands(const StridedSpan& lhs,
                                   const StridedSpan& rhs,
                                   const StridedSpan& output,
                                   const MatmulBatchGroup& group) {
    return MatmulOperands{
        matmulGroupOperand(lhs, group.batch_count, group.lhs_batch_stride_elements, group.lhs_relative_element_offset, "lhs"),
        matmulGroupOperand(rhs, group.batch_count, group.rhs_batch_stride_elements, group.rhs_relative_element_offset, "rhs"),
        matmulGroupOperand(output,
                           group.batch_count,
                           group.output_batch_stride_elements,
                           group.output_relative_element_offset,
                           "output"),
    };
}

}  // namespace ThorImplementation
