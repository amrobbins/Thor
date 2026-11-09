// 1284 (phase 3b, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): retained ragged KR VALUE for FP16 and BF16 input at
// D = 1..7 (rows under 16 bytes), the FP8 kernels of RaggedKRKernels.cuh at 8 rows per lane or worker and tile.
// Kernels: 2 formats x {additive, minimum} x {D = 1, D >= 2}.
#include "Utilities/TensorOperations/Cub/RaggedKRKernels.cuh"

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedKRLaunchMainF16(DataType input, int fixed_op, uint32_t grid, const RaggedKRParams& params,
                           cudaStream_t stream) {
    if (input == DataType::FP16) {
        raggedKRLaunchMainFor<__half>(fixed_op, grid, params, stream);
    } else if (input == DataType::BF16) {
        raggedKRLaunchMainFor<__nv_bfloat16>(fixed_op, grid, params, stream);
    } else {
        throw std::invalid_argument("Ragged KR: the 16-bit object takes FP16 or BF16");
    }
}

int raggedKRResidentF16(uint32_t components) { return raggedKRResidentFor<__half>(components); }

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
