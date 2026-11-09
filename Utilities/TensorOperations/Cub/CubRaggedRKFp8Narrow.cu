// 1288 (phase 4, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): retained ragged RK VALUE for FP8 input on narrow rows
// (one K tile, W >= 16 workers: R of 16 to 32 bytes), 4 rows per worker (RaggedRKKernels.cuh, raggedRKRowsPerWorker).
// Kernels: 2 formats x {additive, minimum}. Its own object, so it compiles in parallel with CubRaggedRKFp8.cu.
#include "Utilities/TensorOperations/Cub/RaggedRKKernels.cuh"

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedRKLaunchMainFp8Narrow(DataType input, int fixed_op, uint32_t grid, const RaggedRKParams& params,
                                 cudaStream_t stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (input == DataType::FP8_E4M3) {
        raggedRKLaunchMainFor<__nv_fp8_e4m3, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(fixed_op, grid, params, stream);
    } else if (input == DataType::FP8_E5M2) {
        raggedRKLaunchMainFor<__nv_fp8_e5m2, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(fixed_op, grid, params, stream);
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

int raggedRKResidentFp8Narrow() {
#if THOR_CUB_ENABLE_FP8_TYPES
    return raggedRKResidentFor<__nv_fp8_e4m3, RaggedMergePath::RAGGED_RK_ROWS_NARROW>();
#else
    throw std::invalid_argument("Ragged RK: FP8 types are disabled in this build");
#endif
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
