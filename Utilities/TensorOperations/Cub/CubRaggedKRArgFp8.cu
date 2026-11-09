// 1298 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged KR ARG main kernels for FP8
// input (RaggedKRArgKernels.cuh): D = 1 and (1300) D = 2..15, each one kernel for argmin and argmax in both formats
// (the FP8 ARG keys need no decode, so this object does not need the FP8 CUDA types). The launcher is CubRaggedKRArg.cu.
#include "Utilities/TensorOperations/Cub/RaggedKRArgKernels.cuh"

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedKRArgLaunchMainFp8(uint32_t grid, const RaggedKRArgParams& params, cudaStream_t stream) {
    if (params.components == 0 || params.components > RaggedMergePath::RAGGED_KR_MAX_COMPONENTS)
        throw std::invalid_argument("Ragged KR ARG: the FP8 kernels take D = 1 to 15");
    if (params.components == 1) {
        raggedKRArgKernel<RaggedArgKeyFp8><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else {
        raggedKRArgComponentsKernel<RaggedArgKeyFp8><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);  // 1300
    }
    raggedKRCheck(cudaGetLastError(), "ARG main kernel launch");
}

int raggedKRArgResidentFp8(uint32_t components) {
    if (components == 0 || components > RaggedMergePath::RAGGED_KR_MAX_COMPONENTS)
        throw std::invalid_argument("Ragged KR ARG: the FP8 kernels take D = 1 to 15");
    int resident = 0;
    if (components == 1) {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, raggedKRArgKernel<RaggedArgKeyFp8>,
                                                                    RAGGED_KR_LANES, 0),
                      "ARG occupancy");
    } else {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &resident, raggedKRArgComponentsKernel<RaggedArgKeyFp8>, RAGGED_KR_LANES, 0),
                      "ARG occupancy");
    }
    return resident;
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
