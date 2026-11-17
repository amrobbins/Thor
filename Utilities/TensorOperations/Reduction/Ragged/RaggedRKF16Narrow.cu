// Ragged RK VALUE for FP16 and BF16 input on narrow rows (one K tile, W >= 16 workers: D of 8 to 16), 4 rows per worker
// (RaggedRKKernels.cuh, raggedRKRowsPerWorker). Kernels: 2 formats x {additive, minimum}. Its own object, as
// RaggedRKFp8Narrow.cu.
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedRKKernels.cuh"

namespace ThorImplementation::ReductionInternal::RaggedKR {

void raggedRKLaunchMainF16Narrow(DataType input, int fixed_op, uint32_t grid, const RaggedRKParams& params,
                                 cudaStream_t stream) {
    if (input == DataType::FP16) {
        raggedRKLaunchMainFor<__half, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(fixed_op, grid, params, stream);
    } else if (input == DataType::BF16) {
        raggedRKLaunchMainFor<__nv_bfloat16, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(fixed_op, grid, params, stream);
    } else {
        throw std::invalid_argument("Ragged RK: the 16-bit object takes FP16 or BF16");
    }
}

int raggedRKResidentF16Narrow() { return raggedRKResidentFor<__half, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(); }

}  // namespace ThorImplementation::ReductionInternal::RaggedKR
