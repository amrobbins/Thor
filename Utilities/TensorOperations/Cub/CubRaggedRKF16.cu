// 1295 (phase 4, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): retained ragged RK VALUE for FP16 and BF16 input, rows
// of D >= 8 elements (16 bytes; RaggedRKKernels.cuh, a lane's chunk 8 values of a row). Kernels: 2 formats x
// {additive, minimum}, 2 rows per worker; narrow rows (raggedRKRowsPerWorker 4) go to CubRaggedRKF16Narrow.cu's kernels.
#include "Utilities/TensorOperations/Cub/RaggedRKKernels.cuh"

#include <algorithm>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedRKLaunchMainF16(DataType input, int fixed_op, uint32_t rows, uint32_t grid, const RaggedRKParams& params,
                           cudaStream_t stream) {
    if (rows == RaggedMergePath::RAGGED_RK_ROWS_NARROW) {
        raggedRKLaunchMainF16Narrow(input, fixed_op, grid, params, stream);
        return;
    }
    if (rows != RaggedMergePath::RAGGED_RK_ROWS)
        throw std::invalid_argument("Ragged RK: rows per worker must be 2 or 4");
    if (input == DataType::FP16) {
        raggedRKLaunchMainFor<__half, RaggedMergePath::RAGGED_RK_ROWS>(fixed_op, grid, params, stream);
    } else if (input == DataType::BF16) {
        raggedRKLaunchMainFor<__nv_bfloat16, RaggedMergePath::RAGGED_RK_ROWS>(fixed_op, grid, params, stream);
    } else {
        throw std::invalid_argument("Ragged RK: the 16-bit object takes FP16 or BF16");
    }
}

int raggedRKResidentF16() {
    return std::min(raggedRKResidentFor<__half, RaggedMergePath::RAGGED_RK_ROWS>(), raggedRKResidentF16Narrow());
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
