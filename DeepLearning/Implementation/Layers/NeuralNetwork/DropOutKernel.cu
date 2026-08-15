#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOutKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ThorImplementation {
namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

template <typename T>
__device__ float loadAsFloat(T value);

template <>
__device__ float loadAsFloat<float>(float value) {
    return value;
}

template <>
__device__ float loadAsFloat<half>(half value) {
    return __half2float(value);
}

template <>
__device__ float loadAsFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T storeFromFloat(float value);

template <>
__device__ float storeFromFloat<float>(float value) {
    return value;
}

template <>
__device__ half storeFromFloat<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 storeFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void dropOutForwardKernel(const T *input,
                                     T *output,
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
            output[index] = storeFromFloat<T>(0.0f);
        }
        return;
    }

    curandStatePhilox4_32_10_t randomState;
    // Give every layer invocation and CUDA thread a disjoint Philox
    // subsequence, independent of the packed capacity beyond numElements.
    const uint64_t subsequence = (forwardSequence << 32U) | first;
    curand_init(randomSeed, subsequence, 0, &randomState);

    for (uint64_t index = first; index < numElements; index += stride) {
        const bool keep = curand_uniform(&randomState) > probabilityOfDroppingOut;
        keepMask[index] = static_cast<uint8_t>(keep);
        const float value = keep ? loadAsFloat<T>(input[index]) * keptValueScale : 0.0f;
        output[index] = storeFromFloat<T>(value);
    }
}

template <typename T>
__global__ void dropOutBackwardKernel(const T *errorInput,
                                      T *errorOutput,
                                      const uint8_t *keepMask,
                                      uint64_t numElements,
                                      float keptValueScale) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t index = first; index < numElements; index += stride) {
        const float value = keepMask[index] != 0 ? loadAsFloat<T>(errorInput[index]) * keptValueScale : 0.0f;
        errorOutput[index] = storeFromFloat<T>(value);
    }
}

uint32_t blockCount(uint64_t numElements) {
    const uint64_t requestedBlocks = (numElements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(requestedBlocks, 65535)));
}

template <typename T>
void launchForwardTyped(const void *input,
                        void *output,
                        uint8_t *keepMask,
                        uint64_t numElements,
                        float probabilityOfDroppingOut,
                        uint64_t randomSeed,
                        uint64_t forwardSequence,
                        Stream stream) {
    const float keptValueScale =
        probabilityOfDroppingOut < 1.0f ? 1.0f / (1.0f - probabilityOfDroppingOut) : 0.0f;
    dropOutForwardKernel<T><<<blockCount(numElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        static_cast<const T *>(input),
        static_cast<T *>(output),
        keepMask,
        numElements,
        probabilityOfDroppingOut,
        keptValueScale,
        randomSeed,
        forwardSequence);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launchBackwardTyped(const void *errorInput,
                         void *errorOutput,
                         const uint8_t *keepMask,
                         uint64_t numElements,
                         float probabilityOfDroppingOut,
                         Stream stream) {
    const float keptValueScale =
        probabilityOfDroppingOut < 1.0f ? 1.0f / (1.0f - probabilityOfDroppingOut) : 0.0f;
    dropOutBackwardKernel<T><<<blockCount(numElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        static_cast<const T *>(errorInput),
        static_cast<T *>(errorOutput),
        keepMask,
        numElements,
        keptValueScale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchDropOutForward(const void *input,
                          void *output,
                          uint8_t *keepMask,
                          DataType dataType,
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

    switch (dataType) {
        case DataType::FP16:
            launchForwardTyped<half>(input, output, keepMask, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::FP32:
            launchForwardTyped<float>(input, output, keepMask, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::BF16:
            launchForwardTyped<__nv_bfloat16>(input, output, keepMask, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        default:
            throw std::invalid_argument("Native DropOut kernel supports only FP16, FP32, and BF16 tensors.");
    }
}

void launchDropOutBackward(const void *errorInput,
                           void *errorOutput,
                           const uint8_t *keepMask,
                           DataType dataType,
                           uint64_t numElements,
                           float probabilityOfDroppingOut,
                           Stream stream) {
    THOR_THROW_IF_FALSE(errorInput != nullptr);
    THOR_THROW_IF_FALSE(errorOutput != nullptr);
    THOR_THROW_IF_FALSE(keepMask != nullptr);
    THOR_THROW_IF_FALSE(numElements > 0);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut > 0.0f);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut <= 1.0f);

    switch (dataType) {
        case DataType::FP16:
            launchBackwardTyped<half>(errorInput, errorOutput, keepMask, numElements, probabilityOfDroppingOut, stream);
            return;
        case DataType::FP32:
            launchBackwardTyped<float>(errorInput, errorOutput, keepMask, numElements, probabilityOfDroppingOut, stream);
            return;
        case DataType::BF16:
            launchBackwardTyped<__nv_bfloat16>(errorInput, errorOutput, keepMask, numElements, probabilityOfDroppingOut, stream);
            return;
        default:
            throw std::invalid_argument("Native DropOut kernel supports only FP16, FP32, and BF16 tensors.");
    }
}

void launchBfloat16DropOutForward(const void *input,
                                  void *output,
                                  uint8_t *keepMask,
                                  uint64_t numElements,
                                  float probabilityOfDroppingOut,
                                  uint64_t randomSeed,
                                  uint64_t forwardSequence,
                                  Stream stream) {
    launchDropOutForward(input,
                         output,
                         keepMask,
                         DataType::BF16,
                         numElements,
                         probabilityOfDroppingOut,
                         randomSeed,
                         forwardSequence,
                         stream);
}

void launchBfloat16DropOutBackward(const void *errorInput,
                                   void *errorOutput,
                                   const uint8_t *keepMask,
                                   uint64_t numElements,
                                   float probabilityOfDroppingOut,
                                   Stream stream) {
    launchDropOutBackward(errorInput,
                          errorOutput,
                          keepMask,
                          DataType::BF16,
                          numElements,
                          probabilityOfDroppingOut,
                          stream);
}

}  // namespace ThorImplementation
