// 1321 (phase 5, plans/Thor_Ragged_Reduction_Plan_2026-11-06.md): the retained ragged RK ARG main kernels for FP32
// input, rows of D >= 4 elements (16 bytes; RaggedRKArgKernels.cuh with RaggedArgKeyFp32: 4 columns per lane, 64-bit
// tile candidates key << 32 | field << 1, pair records): 2 rows per worker, and 4 on narrow rows
// (raggedRKRowsPerWorker), each one kernel for argmin and argmax (the fold's sign flip). The launcher is
// CubRaggedRKArg.cu; the finishers the pair finishers (CubRaggedFinishArgFp32.cu, CubRaggedFinishArgPairWide.cu).
#include "Utilities/TensorOperations/Cub/RaggedRKArgKernels.cuh"

#include <algorithm>

namespace ThorImplementation::CubReductionInternal::RaggedKR {

void raggedRKArgLaunchMainFp32(uint32_t rows, uint32_t grid, const RaggedRKArgParams& params, cudaStream_t stream) {
    if (rows == RaggedMergePath::RAGGED_RK_ROWS_NARROW) {
        raggedRKArgKernel<RaggedArgKeyFp32, RaggedMergePath::RAGGED_RK_ROWS_NARROW>
            <<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else if (rows == RaggedMergePath::RAGGED_RK_ROWS) {
        raggedRKArgKernel<RaggedArgKeyFp32, RaggedMergePath::RAGGED_RK_ROWS>
            <<<grid, RAGGED_KR_LANES, 0, stream>>>(params);
    } else {
        throw std::invalid_argument("Ragged RK ARG: rows per worker must be 2 or 4");
    }
    raggedKRCheck(cudaGetLastError(), "RK ARG main kernel launch");
}

int raggedRKArgResidentFp32() {
    return std::min(raggedRKArgResidentFor<RaggedArgKeyFp32, RaggedMergePath::RAGGED_RK_ROWS>(),
                    raggedRKArgResidentFor<RaggedArgKeyFp32, RaggedMergePath::RAGGED_RK_ROWS_NARROW>());
}

}  // namespace ThorImplementation::CubReductionInternal::RaggedKR
