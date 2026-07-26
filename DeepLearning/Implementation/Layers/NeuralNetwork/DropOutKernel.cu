#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOutKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstdint>

namespace ThorImplementation {
namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

__global__ void bfloat16DropOutForwardKernel(const __nv_bfloat16 *input,
                                             __nv_bfloat16 *output,
                                             uint8_t *keepMask,
                                             uint64_t numElements,
                                             float probabilityOfDroppingOut,
                                             float keptValueScale,
                                             uint64_t randomSeed,
                                             uint64_t forwardSequence) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    if (probabilityOfDroppingOut >= 1.0f) {
        for (uint64_t index = first; index < numElements; index += stride) {
            keepMask[index] = 0;
            output[index] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    curandStatePhilox4_32_10_t randomState;
    // Give every layer invocation and CUDA thread a disjoint Philox
    // subsequence, independent of the batch-dependent grid size.
    const uint64_t subsequence = (forwardSequence << 32U) | first;
    curand_init(randomSeed, subsequence, 0, &randomState);

    for (uint64_t index = first; index < numElements; index += stride) {
        const bool keep = curand_uniform(&randomState) > probabilityOfDroppingOut;
        keepMask[index] = static_cast<uint8_t>(keep);
        const float value = keep ? __bfloat162float(input[index]) * keptValueScale : 0.0f;
        output[index] = __float2bfloat16_rn(value);
    }
}

__global__ void bfloat16DropOutBackwardKernel(const __nv_bfloat16 *errorInput,
                                              __nv_bfloat16 *errorOutput,
                                              const uint8_t *keepMask,
                                              uint64_t numElements,
                                              float keptValueScale) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t index = first; index < numElements; index += stride) {
        const float value =
            keepMask[index] != 0 ? __bfloat162float(errorInput[index]) * keptValueScale : 0.0f;
        errorOutput[index] = __float2bfloat16_rn(value);
    }
}

uint32_t blockCount(uint64_t numElements) {
    const uint64_t requestedBlocks = (numElements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(requestedBlocks, 65535)));
}

}  // namespace

void launchBfloat16DropOutForward(const void *input,
                                  void *output,
                                  uint8_t *keepMask,
                                  uint64_t numElements,
                                  float probabilityOfDroppingOut,
                                  uint64_t randomSeed,
                                  uint64_t forwardSequence,
                                  Stream stream) {
    THOR_THROW_IF_FALSE(input != nullptr);
    THOR_THROW_IF_FALSE(output != nullptr);
    THOR_THROW_IF_FALSE(keepMask != nullptr);
    THOR_THROW_IF_FALSE(numElements > 0);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut > 0.0f);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut <= 1.0f);

    const float keptValueScale =
        probabilityOfDroppingOut < 1.0f ? 1.0f / (1.0f - probabilityOfDroppingOut) : 0.0f;
    bfloat16DropOutForwardKernel<<<blockCount(numElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        static_cast<const __nv_bfloat16 *>(input),
        static_cast<__nv_bfloat16 *>(output),
        keepMask,
        numElements,
        probabilityOfDroppingOut,
        keptValueScale,
        randomSeed,
        forwardSequence);
    CUDA_CHECK(cudaGetLastError());
}

void launchBfloat16DropOutBackward(const void *errorInput,
                                   void *errorOutput,
                                   const uint8_t *keepMask,
                                   uint64_t numElements,
                                   float probabilityOfDroppingOut,
                                   Stream stream) {
    THOR_THROW_IF_FALSE(errorInput != nullptr);
    THOR_THROW_IF_FALSE(errorOutput != nullptr);
    THOR_THROW_IF_FALSE(keepMask != nullptr);
    THOR_THROW_IF_FALSE(numElements > 0);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut > 0.0f);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut <= 1.0f);

    const float keptValueScale =
        probabilityOfDroppingOut < 1.0f ? 1.0f / (1.0f - probabilityOfDroppingOut) : 0.0f;
    bfloat16DropOutBackwardKernel<<<blockCount(numElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        static_cast<const __nv_bfloat16 *>(errorInput),
        static_cast<__nv_bfloat16 *>(errorOutput),
        keepMask,
        numElements,
        keptValueScale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ThorImplementation
