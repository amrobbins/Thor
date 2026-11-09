// 1320 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged RK ARG main kernels for FP16 and
// BF16 input, rows of D >= 8 elements (16 bytes; RaggedRKArgKernels.cuh with RaggedArgKeyF16, 8 columns per lane): 2
// rows per worker, and 4 on narrow rows (raggedRKRowsPerWorker), each one kernel for argmin and argmax in both formats
// (the keys are raw 16-bit patterns; the NaN bias is the fold's). The launcher is CubRaggedRKArg.cu.
#include "Utilities/TensorOperations/Cub/RaggedRKArgKernels.cuh"

#include <algorithm>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedRKArgLaunchMainF16(uint32_t rows, uint32_t grid, const RaggedRKArgParams& params, cudaStream_t stream) {
    if (rows == RaggedMergePath::RAGGED_RK_ROWS_NARROW) {
        raggedRKArgKernel<RaggedArgKeyF16, RaggedMergePath::RAGGED_RK_ROWS_NARROW>
            <<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else if (rows == RaggedMergePath::RAGGED_RK_ROWS) {
        raggedRKArgKernel<RaggedArgKeyF16, RaggedMergePath::RAGGED_RK_ROWS><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else {
        throw std::invalid_argument("Ragged RK ARG: rows per worker must be 2 or 4");
    }
    raggedKRCheck(cudaGetLastError(), "RK ARG main kernel launch");
}

int raggedRKArgResidentF16() {
    return std::min(raggedRKArgResidentFor<RaggedArgKeyF16, RaggedMergePath::RAGGED_RK_ROWS>(),
                    raggedRKArgResidentFor<RaggedArgKeyF16, RaggedMergePath::RAGGED_RK_ROWS_NARROW>());
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
