// 1295 (phase 4, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): retained ragged RK VALUE for FP32 input, rows of
// D >= 4 elements (16 bytes; RaggedRKKernels.cuh, a lane's chunk 4 values of a row). Kernels: {additive, minimum},
// 2 rows per worker; narrow rows (raggedRKRowsPerWorker 4) go to CubRaggedRKFp32Narrow.cu's kernels.
#include "Utilities/TensorOperations/Cub/RaggedRKKernels.cuh"

#include <algorithm>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedRKLaunchMainFp32(int fixed_op, uint32_t rows, uint32_t grid, const RaggedRKParams& params,
                            cudaStream_t stream) {
    if (rows == RaggedMergePath::RAGGED_RK_ROWS_NARROW) {
        raggedRKLaunchMainFp32Narrow(fixed_op, grid, params, stream);
        return;
    }
    if (rows != RaggedMergePath::RAGGED_RK_ROWS)
        throw std::invalid_argument("Ragged RK: rows per worker must be 2 or 4");
    raggedRKLaunchMainFor<float, RaggedMergePath::RAGGED_RK_ROWS>(fixed_op, grid, params, stream);
}

int raggedRKResidentFp32() {
    return std::min(raggedRKResidentFor<float, RaggedMergePath::RAGGED_RK_ROWS>(), raggedRKResidentFp32Narrow());
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
