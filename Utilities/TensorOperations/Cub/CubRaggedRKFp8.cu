// 1285 (phase 4, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): retained ragged RK VALUE for FP8 input, rows of
// D >= 16 bytes (RaggedRKKernels.cuh). Kernels: 2 formats x {additive, minimum}, 2 rows per worker. 1288: narrow rows
// (raggedRKRowsPerWorker 4) go to CubRaggedRKFp8Narrow.cu's kernels.
#include "Utilities/TensorOperations/Cub/RaggedRKKernels.cuh"

#include <algorithm>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedRKLaunchMainFp8(DataType input, int fixed_op, uint32_t rows, uint32_t grid, const RaggedRKParams& params,
                           cudaStream_t stream) {
    if (rows == RaggedMergePath::RAGGED_RK_ROWS_NARROW) {
        raggedRKLaunchMainFp8Narrow(input, fixed_op, grid, params, stream);
        return;
    }
    if (rows != RaggedMergePath::RAGGED_RK_ROWS)
        throw std::invalid_argument("Ragged RK: rows per worker must be 2 or 4");
#if THOR_CUB_ENABLE_FP8_TYPES
    if (input == DataType::FP8_E4M3) {
        raggedRKLaunchMainFor<__nv_fp8_e4m3, RaggedMergePath::RAGGED_RK_ROWS>(fixed_op, grid, params, stream);
    } else if (input == DataType::FP8_E5M2) {
        raggedRKLaunchMainFor<__nv_fp8_e5m2, RaggedMergePath::RAGGED_RK_ROWS>(fixed_op, grid, params, stream);
    } else {
        throw std::invalid_argument("Ragged RK: the FP8 object takes FP8_E4M3 or FP8_E5M2");
    }
#else
    static_cast<void>(input);
    static_cast<void>(fixed_op);
    static_cast<void>(grid);
    static_cast<void>(params);
    static_cast<void>(stream);
    throw std::invalid_argument("Ragged RK: FP8 types are disabled in this build");
#endif
}

int raggedRKResidentFp8() {
#if THOR_CUB_ENABLE_FP8_TYPES
    return std::min(raggedRKResidentFor<__nv_fp8_e4m3, RaggedMergePath::RAGGED_RK_ROWS>(), raggedRKResidentFp8Narrow());
#else
    throw std::invalid_argument("Ragged RK: FP8 types are disabled in this build");
#endif
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
