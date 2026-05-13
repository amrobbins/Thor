#include "Utilities/TensorOperations/DeepLearning/BatchNormFrontendHelpers.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

namespace ThorImplementation {
namespace {

__global__ void computeBatchNormInvVarianceFp32Kernel(const float* variance, float* inv_variance, float epsilon, uint64_t num_channels) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_channels) {
        inv_variance[idx] = rsqrtf(variance[idx] + epsilon);
    }
}

__global__ void accumulateBatchNormGradientFp32Kernel(float* dest, const float* addend, uint64_t num_channels) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_channels) {
        dest[idx] += addend[idx];
    }
}

}  // namespace

void launchComputeBatchNormInvVarianceFp32(const float* variance_d, float* inv_variance_d, float epsilon, uint64_t num_channels, Stream stream) {
    if (num_channels == 0) {
        return;
    }
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((num_channels + threads_per_block - 1) / threads_per_block);
    computeBatchNormInvVarianceFp32Kernel<<<blocks, threads_per_block, 0, stream>>>(variance_d, inv_variance_d, epsilon, num_channels);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchAccumulateBatchNormGradientFp32(float* dest_d, const float* addend_d, uint64_t num_channels, Stream stream) {
    if (num_channels == 0) {
        return;
    }
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((num_channels + threads_per_block - 1) / threads_per_block);
    accumulateBatchNormGradientFp32Kernel<<<blocks, threads_per_block, 0, stream>>>(dest_d, addend_d, num_channels);
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace ThorImplementation
