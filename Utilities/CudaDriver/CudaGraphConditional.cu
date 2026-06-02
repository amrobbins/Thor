#include "Utilities/CudaDriver/CudaGraphConditional.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions()) {
        numel *= dim;
    }
    return numel;
}

__global__ void setCudaGraphConditionalFromBoolKernel(cudaGraphConditionalHandle handle, const bool* predicate) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (predicate == nullptr) {
        asm("trap;");
    }
    cudaGraphSetConditional(handle, predicate[0] ? 1U : 0U);
}

}  // namespace

void launchSetCudaGraphConditionalFromBool(cudaGraphConditionalHandle handle, const Tensor& predicate, Stream stream) {
    if (!predicate.isInitialized()) {
        throw std::invalid_argument("Graph-level conditional predicate tensor is not initialized.");
    }
    if (predicate.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Graph-level conditional predicate tensor must live on GPU.");
    }
    if (predicate.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::invalid_argument("Graph-level conditional predicate tensor GPU must match the launch stream GPU.");
    }
    if (predicate.getDataType() != DataType::BOOLEAN) {
        throw std::invalid_argument("Graph-level conditional predicate output must have BOOLEAN dtype.");
    }
    if (tensorNumel(predicate) != 1) {
        throw std::invalid_argument("Graph-level conditional predicate output must contain exactly one element.");
    }

    ScopedGpu scoped(stream.getGpuNum());
    setCudaGraphConditionalFromBoolKernel<<<1, 1, 0, stream.getStream()>>>(handle, predicate.getMemPtr<bool>());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ThorImplementation
