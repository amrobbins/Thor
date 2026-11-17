// The ragged KR ARG main kernels for FP16 and BF16 input (RaggedKRArgKernels.cuh with RaggedArgKeyF16): D = 1 and
// D = 2..7 (rows under 16 bytes), each one kernel for argmin and argmax in both formats (the keys are raw 16-bit
// patterns; the NaN bias is the fold's). The launcher is RaggedKRArg.cu.
#include "Utilities/TensorOperations/Reduction/Ragged/RaggedKRArgKernels.cuh"

namespace ThorImplementation::ReductionInternal::RaggedKR {

namespace {
constexpr uint32_t F16_MAX_COMPONENTS = 7;  // rows under 16 bytes
}

void raggedKRArgLaunchMainF16(uint32_t grid, const RaggedKRArgParams& params, cudaStream_t stream) {
    if (params.components == 0 || params.components > F16_MAX_COMPONENTS)
        throw std::invalid_argument("Ragged KR ARG: the FP16 / BF16 kernels take D = 1 to 7");
    if (params.components == 1) {
        raggedKRArgKernel<RaggedArgKeyF16><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else {
        raggedKRArgComponentsKernel<RaggedArgKeyF16><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    }
    raggedKRCheck(cudaGetLastError(), "ARG main kernel launch");
}

int raggedKRArgResidentF16(uint32_t components) {
    if (components == 0 || components > F16_MAX_COMPONENTS)
        throw std::invalid_argument("Ragged KR ARG: the FP16 / BF16 kernels take D = 1 to 7");
    int resident = 0;
    if (components == 1) {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, raggedKRArgKernel<RaggedArgKeyF16>,
                                                                    RAGGED_KR_LANES, 0),
                      "ARG occupancy");
    } else {
        raggedKRCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &resident, raggedKRArgComponentsKernel<RaggedArgKeyF16>, RAGGED_KR_LANES, 0),
                      "ARG occupancy");
    }
    return resident;
}

}  // namespace ThorImplementation::ReductionInternal::RaggedKR
