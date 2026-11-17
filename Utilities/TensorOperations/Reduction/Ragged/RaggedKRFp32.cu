// Ragged KR VALUE for FP32 input at D = 1..3 (rows under 16 bytes), the FP8 kernels of RaggedKRKernels.cuh at 4 rows
// per lane or worker and tile. Kernels: {additive, minimum} x {D = 1, D >= 2}.
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedKRKernels.cuh"

namespace ThorImplementation::ReductionInternal::RaggedKR {

void raggedKRLaunchMainFp32(int fixed_op, uint32_t grid, const RaggedKRParams& params, cudaStream_t stream) {
    raggedKRLaunchMainFor<float>(fixed_op, grid, params, stream);
}

int raggedKRResidentFp32(uint32_t components) { return raggedKRResidentFor<float>(components); }

}  // namespace ThorImplementation::ReductionInternal::RaggedKR
