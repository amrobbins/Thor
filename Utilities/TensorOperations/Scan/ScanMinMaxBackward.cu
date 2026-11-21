// The min / max scan backward's three kernels: dense rows, UINT32 offsets and UINT64 offsets
// (ScanMinMaxBackwardKernel.cuh). The direction and the formats of x, dy and dx are launch arguments.

#include "Utilities/TensorOperations/Scan/ScanMinMaxBackwardKernel.cuh"

#include <stdexcept>

namespace ThorImplementation::ScanMinMaxBackwardKernels {

void launch(const BackwardLaunch& args, bool reverse, uint32_t offset_bytes, cudaStream_t stream) {
    if (offset_bytes == 0) {
        launchKernel<void>(args, reverse, stream);
    } else if (offset_bytes == 4) {
        launchKernel<uint32_t>(args, reverse, stream);
    } else if (offset_bytes == 8) {
        launchKernel<uint64_t>(args, reverse, stream);
    } else {
        throw std::invalid_argument("ScanMinMaxBackward: offsets are UINT32 or UINT64.");
    }
}

}  // namespace ThorImplementation::ScanMinMaxBackwardKernels
