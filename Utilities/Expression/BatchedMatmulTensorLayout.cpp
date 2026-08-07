#include "Utilities/Expression/BatchedMatmulPlan.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

MatmulTensorLayout matmulTensorLayout(const Tensor& tensor) {
    return MatmulTensorLayout{tensor.getDimensions(), tensor.getStridesElements(), tensor.getStorageElementOffset()};
}

BatchedMatmulLayoutPlan planBatchedMatmulLayout(const Tensor& lhs,
                                                const Tensor& rhs,
                                                const Tensor& output,
                                                bool transpose_lhs,
                                                bool transpose_rhs) {
    return planBatchedMatmulLayout(matmulTensorLayout(lhs), matmulTensorLayout(rhs), matmulTensorLayout(output), transpose_lhs, transpose_rhs);
}

}  // namespace ThorImplementation
