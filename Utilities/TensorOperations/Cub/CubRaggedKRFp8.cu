// 1265: retained ragged KR VALUE for original FP8 input (plans/Thor_Ragged_Reduction_Plan_2026-11-06.md, phase 2; 1280:
// D = 2..15). 1284 (phase 3b): the kernels moved to RaggedKRKernels.cuh (templated on the input element, with their
// history) and the host launcher to CubRaggedKR.cu; this object instantiates them for E4M3 and E5M2. Kernels: 2 formats
// x {additive, minimum} x {D = 1, D >= 2}.
#include "Utilities/TensorOperations/Cub/RaggedKRKernels.cuh"

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedKRLaunchMainFp8(DataType input, int fixed_op, uint32_t grid, const RaggedKRParams& params,
                           cudaStream_t stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (input == DataType::FP8_E4M3) {
        raggedKRLaunchMainFor<__nv_fp8_e4m3>(fixed_op, grid, params, stream);
    } else if (input == DataType::FP8_E5M2) {
        raggedKRLaunchMainFor<__nv_fp8_e5m2>(fixed_op, grid, params, stream);
    } else {
        throw std::invalid_argument("Ragged KR: the FP8 object takes FP8_E4M3 or FP8_E5M2");
    }
#else
    static_cast<void>(input);
    static_cast<void>(fixed_op);
    static_cast<void>(grid);
    static_cast<void>(params);
    static_cast<void>(stream);
    throw std::invalid_argument("Ragged KR: FP8 types are disabled in this build");
#endif
}

int raggedKRResidentFp8(uint32_t components) {
#if THOR_CUB_ENABLE_FP8_TYPES
    return raggedKRResidentFor<__nv_fp8_e4m3>(components);
#else
    static_cast<void>(components);
    throw std::invalid_argument("Ragged KR: FP8 types are disabled in this build");
#endif
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
