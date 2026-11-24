// Ragged KR VALUE for FP8 input at D = 1..15 (rows under 16 bytes): this object instantiates RaggedKRKernels.cuh's
// kernels for E4M3 and E5M2 (the host launcher is RaggedKR.cu). Kernels: 2 formats x {additive, minimum} x {D = 1,
// D >= 2}.
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedKRKernels.cuh"

namespace ThorImplementation::ReductionInternal::RaggedKR {

void raggedKRLaunchMainFp8(DataType input, int fixed_op, uint32_t grid, const RaggedKRParams& params,
                           cudaStream_t stream) {
    if (input == DataType::FP8_E4M3) {
        raggedKRLaunchMainFor<__nv_fp8_e4m3>(fixed_op, grid, params, stream);
    } else if (input == DataType::FP8_E5M2) {
        raggedKRLaunchMainFor<__nv_fp8_e5m2>(fixed_op, grid, params, stream);
    } else {
        throw std::invalid_argument("Ragged KR: the FP8 object takes FP8_E4M3 or FP8_E5M2");
    }
}

int raggedKRResidentFp8(uint32_t components) {
    return raggedKRResidentFor<__nv_fp8_e4m3>(components);
}

}  // namespace ThorImplementation::ReductionInternal::RaggedKR
