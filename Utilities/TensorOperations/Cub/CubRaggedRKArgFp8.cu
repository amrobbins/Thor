// 1314 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged RK ARG main kernels for FP8 input,
// rows of D >= 16 bytes (RaggedRKArgKernels.cuh with RaggedArgKeyFp8): 2 rows per worker, and 4 on narrow rows
// (raggedRKRowsPerWorker), each one kernel for argmin and argmax in both formats (the keys need no decode, so this
// object does not need the FP8 CUDA types). The launcher is CubRaggedRKArg.cu.
#include "Utilities/TensorOperations/Cub/RaggedRKArgKernels.cuh"

#include <algorithm>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedRKArgLaunchMainFp8(uint32_t rows, uint32_t grid, const RaggedRKArgParams& params, cudaStream_t stream) {
    if (rows == RaggedMergePath::RAGGED_RK_ROWS_NARROW) {
        raggedRKArgKernel<RaggedArgKeyFp8, RaggedMergePath::RAGGED_RK_ROWS_NARROW>
            <<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else if (rows == RaggedMergePath::RAGGED_RK_ROWS) {
        raggedRKArgKernel<RaggedArgKeyFp8, RaggedMergePath::RAGGED_RK_ROWS><<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else {
        throw std::invalid_argument("Ragged RK ARG: rows per worker must be 2 or 4");
    }
    raggedKRCheck(cudaGetLastError(), "RK ARG main kernel launch");
}

int raggedRKArgResidentFp8() {
    return std::min(raggedRKArgResidentFor<RaggedArgKeyFp8, RaggedMergePath::RAGGED_RK_ROWS>(),
                    raggedRKArgResidentFor<RaggedArgKeyFp8, RaggedMergePath::RAGGED_RK_ROWS_NARROW>());
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
