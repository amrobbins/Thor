#include "Utilities/TensorOperations/Scalar/SetScalar.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

__global__ void setInt64PairKernel(int64_t* first_d, int64_t first, int64_t* second_d, int64_t second) {
    *first_d = first;
    *second_d = second;
}

}  // namespace

void launchSetInt64Pair(int64_t* first_d, int64_t first, int64_t* second_d, int64_t second, Stream stream) {
    THOR_THROW_IF_FALSE(first_d != nullptr && second_d != nullptr);
    ScopedGpu scopedGpu(stream.getGpuNum());
    setInt64PairKernel<<<1, 1, 0, stream.getStream()>>>(first_d, first, second_d, second);
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace ThorImplementation
