// Ragged RK VALUE for FP32 input on narrow rows (one K tile, W >= 16 workers: D of 4 to 8), 4 rows per worker
// (RaggedRKKernels.cuh, raggedRKRowsPerWorker).
// Kernels: {additive, minimum}. Its own object, as RaggedRKFp8Narrow.cu.
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedRKKernels.cuh"

namespace ThorImplementation::ReductionInternal::RaggedKR {

void raggedRKLaunchMainFp32Narrow(int fixed_op, uint32_t grid, const RaggedRKParams& params, cudaStream_t stream) {
    raggedRKLaunchMainFor<float, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(fixed_op, grid, params, stream);
}

int raggedRKResidentFp32Narrow() { return raggedRKResidentFor<float, RaggedMergePath::RAGGED_RK_ROWS_NARROW>(); }

}  // namespace ThorImplementation::ReductionInternal::RaggedKR
