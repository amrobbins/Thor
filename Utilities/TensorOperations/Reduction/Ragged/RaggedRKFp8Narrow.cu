// Ragged RK VALUE for FP8 input on narrow rows (one K tile, W >= 16 workers: R of 16 to 32 bytes), 4 rows per worker
// (RaggedRKKernels.cuh, raggedRKRowsPerWorker).
// Kernels: 2 formats x {additive, minimum}. Its own object, so it compiles in parallel with RaggedRKFp8.cu.
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedRKKernels.cuh"

namespace ThorImplementation::ReductionInternal::RaggedKR {

void raggedRKLaunchMainFp8Narrow(DataType input, int fixed_op, uint32_t grid, const RaggedRKParams& params,
                                 cudaStream_t stream) {
    if (input == DataType::FP8_E4M3) {
        raggedRKLaunchMainFor<__nv_fp8_e4m3, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(fixed_op, grid, params, stream);
    } else if (input == DataType::FP8_E5M2) {
        raggedRKLaunchMainFor<__nv_fp8_e5m2, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(fixed_op, grid, params, stream);
    } else {
        throw std::invalid_argument("Ragged RK: the FP8 object takes FP8_E4M3 or FP8_E5M2");
    }
}

int raggedRKResidentFp8Narrow() {
    return raggedRKResidentFor<__nv_fp8_e4m3, RaggedMergePath::RAGGED_RK_ROWS_NARROW>();
}

}  // namespace ThorImplementation::ReductionInternal::RaggedKR
