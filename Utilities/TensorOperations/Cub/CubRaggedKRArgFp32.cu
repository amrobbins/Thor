// 1313 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged KR ARG main kernels for FP32 input
// (RaggedKRArgKernels.cuh with RaggedArgKeyFp32): D = 1 and D = 2..3 (rows under 16 bytes), each one kernel for argmin
// and argmax (the keys are raw bit patterns; argmax is the fold's sign flip). Their tile candidates are 64-bit and their
// records pairs (RaggedArg.h), finished by CubRaggedFinishArgFp32.cu. The launcher is CubRaggedKRArg.cu.
#include "Utilities/TensorOperations/Cub/RaggedKRArgKernels.cuh"

namespace ThorImplementation::CubReductionInternal::RaggedKR {

namespace {
constexpr uint32_t FP32_MAX_COMPONENTS = 3;  // rows under 16 bytes
}

void raggedKRArgLaunchMainFp32(uint32_t grid, const RaggedKRArgParams& params, cudaStream_t stream) {
    if (params.components == 0 || params.components > FP32_MAX_COMPONENTS)
        throw std::invalid_argument("Ragged KR ARG: the FP32 kernels take D = 1 to 3");
    if (params.components == 1) {
        raggedKRArgKernel<RaggedArgKeyFp32><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else {
        raggedKRArgComponentsKernel<RaggedArgKeyFp32><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    }
    raggedKRCheck(cudaGetLastError(), "ARG main kernel launch");
}

int raggedKRArgResidentFp32(uint32_t components) {
    if (components == 0 || components > FP32_MAX_COMPONENTS)
        throw std::invalid_argument("Ragged KR ARG: the FP32 kernels take D = 1 to 3");
    int resident = 0;
    if (components == 1) {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, raggedKRArgKernel<RaggedArgKeyFp32>,
                                                                    RAGGED_KR_LANES, 0),
                      "ARG occupancy");
    } else {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &resident, raggedKRArgComponentsKernel<RaggedArgKeyFp32>, RAGGED_KR_LANES, 0),
                      "ARG occupancy");
    }
    return resident;
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
