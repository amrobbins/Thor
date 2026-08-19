#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOutKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ThorImplementation {
namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr uint32_t ELEMENTS_PER_THREAD = 4;

// Philox4x32-10.  Dropout is bandwidth-bound, so generating four counter-based
// random values per thread avoids curand state setup/storage and makes each
// element's mask independent of launch geometry and inactive ragged capacity.
__device__ __forceinline__ uint4 philoxRound(uint4 counter, uint2 key) {
    constexpr uint32_t M0 = 0xD2511F53U;
    constexpr uint32_t M1 = 0xCD9E8D57U;
    const uint32_t hi0 = __umulhi(M0, counter.x);
    const uint32_t lo0 = M0 * counter.x;
    const uint32_t hi1 = __umulhi(M1, counter.z);
    const uint32_t lo1 = M1 * counter.z;
    return make_uint4(hi1 ^ counter.y ^ key.x, lo1, hi0 ^ counter.w ^ key.y, lo0);
}

__device__ __forceinline__ uint4 philox4x32_10(uint64_t groupIndex, uint64_t randomSeed, uint64_t forwardSequence) {
    constexpr uint32_t W0 = 0x9E3779B9U;
    constexpr uint32_t W1 = 0xBB67AE85U;
    uint4 counter = make_uint4(static_cast<uint32_t>(groupIndex),
                               static_cast<uint32_t>(groupIndex >> 32U),
                               static_cast<uint32_t>(forwardSequence),
                               static_cast<uint32_t>(forwardSequence >> 32U));
    uint2 key = make_uint2(static_cast<uint32_t>(randomSeed), static_cast<uint32_t>(randomSeed >> 32U));
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        counter = philoxRound(counter, key);
        if (round != 9) {
            key.x += W0;
            key.y += W1;
        }
    }
    return counter;
}

template <typename T>
struct alignas(sizeof(T) * ELEMENTS_PER_THREAD) DropOutVector {
    T values[ELEMENTS_PER_THREAD];
};

template <typename T>
__device__ __forceinline__ float loadAsFloat(T value);

template <>
__device__ __forceinline__ float loadAsFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float loadAsFloat<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float loadAsFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T storeFromFloat(float value);

template <>
__device__ __forceinline__ float storeFromFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ half storeFromFloat<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 storeFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ bool keepForRandom(uint32_t randomBits, float probabilityOfDroppingOut) {
    // Mapping the 32 random bits to (0, 1] matches the old curand_uniform
    // comparison semantics closely while keeping the hot path stateless.
    constexpr float UINT32_TO_UNIT = 2.3283064365386963e-10f;  // 2^-32
    const float uniform = (static_cast<float>(randomBits) + 1.0f) * UINT32_TO_UNIT;
    return uniform > probabilityOfDroppingOut;
}


__global__ void dropOutForwardDoubleKernel(const double *input,
                                           double *output,
                                           uint64_t numElements,
                                           float probabilityOfDroppingOut,
                                           double keptValueScale,
                                           uint64_t randomSeed,
                                           uint64_t forwardSequence) {
    const uint64_t firstGroup = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t groupStride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t numGroups = (numElements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    if (probabilityOfDroppingOut >= 1.0f) {
        for (uint64_t group = firstGroup; group < numGroups; group += groupStride) {
#pragma unroll
            for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
                const uint64_t index = group * ELEMENTS_PER_THREAD + lane;
                if (index < numElements) output[index] = 0.0;
            }
        }
        return;
    }
    for (uint64_t group = firstGroup; group < numGroups; group += groupStride) {
        const uint4 random = philox4x32_10(group, randomSeed, forwardSequence);
        const uint32_t randomBits[ELEMENTS_PER_THREAD] = {random.x, random.y, random.z, random.w};
        const uint64_t baseIndex = group * ELEMENTS_PER_THREAD;
#pragma unroll
        for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
            const uint64_t index = baseIndex + lane;
            if (index >= numElements) continue;
            output[index] = keepForRandom(randomBits[lane], probabilityOfDroppingOut) ? input[index] * keptValueScale : 0.0;
        }
    }
}

__global__ void dropOutBackwardDoubleKernel(const double *errorInput,
                                            double *errorOutput,
                                            uint64_t numElements,
                                            float probabilityOfDroppingOut,
                                            double keptValueScale,
                                            uint64_t randomSeed,
                                            uint64_t forwardSequence) {
    const uint64_t firstGroup = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t groupStride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t numGroups = (numElements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    for (uint64_t group = firstGroup; group < numGroups; group += groupStride) {
        const uint4 random = philox4x32_10(group, randomSeed, forwardSequence);
        const uint32_t randomBits[ELEMENTS_PER_THREAD] = {random.x, random.y, random.z, random.w};
        const uint64_t baseIndex = group * ELEMENTS_PER_THREAD;
#pragma unroll
        for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
            const uint64_t index = baseIndex + lane;
            if (index >= numElements) continue;
            errorOutput[index] = keepForRandom(randomBits[lane], probabilityOfDroppingOut)
                ? errorInput[index] * keptValueScale
                : 0.0;
        }
    }
}

template <typename T>
__global__ void dropOutForwardKernel(const T *input,
                                     T *output,
                                     uint64_t numElements,
                                     float probabilityOfDroppingOut,
                                     float keptValueScale,
                                     uint64_t randomSeed,
                                     uint64_t forwardSequence) {
    const uint64_t firstGroup = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t groupStride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t numGroups = (numElements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;

    if (probabilityOfDroppingOut >= 1.0f) {
        for (uint64_t group = firstGroup; group < numGroups; group += groupStride) {
#pragma unroll
            for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
                const uint64_t index = group * ELEMENTS_PER_THREAD + lane;
                if (index < numElements) {
                    output[index] = storeFromFloat<T>(0.0f);
                }
            }
        }
        return;
    }

    for (uint64_t group = firstGroup; group < numGroups; group += groupStride) {
        const uint4 random = philox4x32_10(group, randomSeed, forwardSequence);
        const uint32_t randomBits[ELEMENTS_PER_THREAD] = {random.x, random.y, random.z, random.w};
        const uint64_t baseIndex = group * ELEMENTS_PER_THREAD;
        if (baseIndex + ELEMENTS_PER_THREAD <= numElements) {
            const DropOutVector<T> packedInput = reinterpret_cast<const DropOutVector<T> *>(input)[group];
            DropOutVector<T> packedOutput;
#pragma unroll
            for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
                const bool keep = keepForRandom(randomBits[lane], probabilityOfDroppingOut);
                const float value = keep ? loadAsFloat<T>(packedInput.values[lane]) * keptValueScale : 0.0f;
                packedOutput.values[lane] = storeFromFloat<T>(value);
            }
            reinterpret_cast<DropOutVector<T> *>(output)[group] = packedOutput;
        } else {
#pragma unroll
            for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
                const uint64_t index = baseIndex + lane;
                if (index >= numElements) continue;
                const bool keep = keepForRandom(randomBits[lane], probabilityOfDroppingOut);
                const float value = keep ? loadAsFloat<T>(input[index]) * keptValueScale : 0.0f;
                output[index] = storeFromFloat<T>(value);
            }
        }
    }
}

template <typename T>
__global__ void dropOutBackwardKernel(const T *errorInput,
                                      T *errorOutput,
                                      uint64_t numElements,
                                      float probabilityOfDroppingOut,
                                      float keptValueScale,
                                      uint64_t randomSeed,
                                      uint64_t forwardSequence) {
    const uint64_t firstGroup = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t groupStride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t numGroups = (numElements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    for (uint64_t group = firstGroup; group < numGroups; group += groupStride) {
        const uint4 random = philox4x32_10(group, randomSeed, forwardSequence);
        const uint32_t randomBits[ELEMENTS_PER_THREAD] = {random.x, random.y, random.z, random.w};
        const uint64_t baseIndex = group * ELEMENTS_PER_THREAD;
        if (baseIndex + ELEMENTS_PER_THREAD <= numElements) {
            const DropOutVector<T> packedInput = reinterpret_cast<const DropOutVector<T> *>(errorInput)[group];
            DropOutVector<T> packedOutput;
#pragma unroll
            for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
                const bool keep = keepForRandom(randomBits[lane], probabilityOfDroppingOut);
                const float value = keep ? loadAsFloat<T>(packedInput.values[lane]) * keptValueScale : 0.0f;
                packedOutput.values[lane] = storeFromFloat<T>(value);
            }
            reinterpret_cast<DropOutVector<T> *>(errorOutput)[group] = packedOutput;
        } else {
#pragma unroll
            for (uint32_t lane = 0; lane < ELEMENTS_PER_THREAD; ++lane) {
                const uint64_t index = baseIndex + lane;
                if (index >= numElements) continue;
                const bool keep = keepForRandom(randomBits[lane], probabilityOfDroppingOut);
                const float value = keep ? loadAsFloat<T>(errorInput[index]) * keptValueScale : 0.0f;
                errorOutput[index] = storeFromFloat<T>(value);
            }
        }
    }
}

uint32_t blockCount(uint64_t numElements) {
    const uint64_t numGroups = (numElements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    const uint64_t requestedBlocks = (numGroups + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(requestedBlocks, 65535)));
}

template <typename T>
void launchForwardTyped(const void *input,
                        void *output,
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
                         uint64_t numElements,
                         float probabilityOfDroppingOut,
                         uint64_t randomSeed,
                         uint64_t forwardSequence,
                         Stream stream) {
    const float keptValueScale =
        probabilityOfDroppingOut < 1.0f ? 1.0f / (1.0f - probabilityOfDroppingOut) : 0.0f;
    dropOutBackwardKernel<T><<<blockCount(numElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        static_cast<const T *>(errorInput),
        static_cast<T *>(errorOutput),
        numElements,
        probabilityOfDroppingOut,
        keptValueScale,
        randomSeed,
        forwardSequence);
    CUDA_CHECK(cudaGetLastError());
}

void launchForwardDouble(const void *input,
                         void *output,
                         uint64_t numElements,
                         float probabilityOfDroppingOut,
                         uint64_t randomSeed,
                         uint64_t forwardSequence,
                         Stream stream) {
    const double keptValueScale =
        probabilityOfDroppingOut < 1.0f ? 1.0 / (1.0 - static_cast<double>(probabilityOfDroppingOut)) : 0.0;
    dropOutForwardDoubleKernel<<<blockCount(numElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        static_cast<const double *>(input),
        static_cast<double *>(output),
        numElements,
        probabilityOfDroppingOut,
        keptValueScale,
        randomSeed,
        forwardSequence);
    CUDA_CHECK(cudaGetLastError());
}

void launchBackwardDouble(const void *errorInput,
                          void *errorOutput,
                          uint64_t numElements,
                          float probabilityOfDroppingOut,
                          uint64_t randomSeed,
                          uint64_t forwardSequence,
                          Stream stream) {
    const double keptValueScale =
        probabilityOfDroppingOut < 1.0f ? 1.0 / (1.0 - static_cast<double>(probabilityOfDroppingOut)) : 0.0;
    dropOutBackwardDoubleKernel<<<blockCount(numElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        static_cast<const double *>(errorInput),
        static_cast<double *>(errorOutput),
        numElements,
        probabilityOfDroppingOut,
        keptValueScale,
        randomSeed,
        forwardSequence);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchDropOutForward(const void *input,
                          void *output,
                          DataType dataType,
                          uint64_t numElements,
                          float probabilityOfDroppingOut,
                          uint64_t randomSeed,
                          uint64_t forwardSequence,
                          Stream stream) {
    THOR_THROW_IF_FALSE(input != nullptr);
    THOR_THROW_IF_FALSE(output != nullptr);
    THOR_THROW_IF_FALSE(numElements > 0);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut > 0.0f);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut <= 1.0f);

    switch (dataType) {
        case DataType::FP64:
            launchForwardDouble(input, output, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::FP16:
            launchForwardTyped<half>(input, output, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::FP32:
            launchForwardTyped<float>(input, output, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::BF16:
            launchForwardTyped<__nv_bfloat16>(input, output, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        default:
            throw std::invalid_argument("Native DropOut kernel supports only FP16, FP32, FP64, and BF16 tensors.");
    }
}

void launchDropOutBackward(const void *errorInput,
                           void *errorOutput,
                           DataType dataType,
                           uint64_t numElements,
                           float probabilityOfDroppingOut,
                           uint64_t randomSeed,
                           uint64_t forwardSequence,
                           Stream stream) {
    THOR_THROW_IF_FALSE(errorInput != nullptr);
    THOR_THROW_IF_FALSE(errorOutput != nullptr);
    THOR_THROW_IF_FALSE(numElements > 0);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut > 0.0f);
    THOR_THROW_IF_FALSE(probabilityOfDroppingOut <= 1.0f);

    switch (dataType) {
        case DataType::FP64:
            launchBackwardDouble(errorInput, errorOutput, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::FP16:
            launchBackwardTyped<half>(errorInput, errorOutput, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::FP32:
            launchBackwardTyped<float>(errorInput, errorOutput, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        case DataType::BF16:
            launchBackwardTyped<__nv_bfloat16>(errorInput, errorOutput, numElements, probabilityOfDroppingOut, randomSeed, forwardSequence, stream);
            return;
        default:
            throw std::invalid_argument("Native DropOut kernel supports only FP16, FP32, FP64, and BF16 tensors.");
    }
}

void launchBfloat16DropOutForward(const void *input,
                                  void *output,
                                  uint64_t numElements,
                                  float probabilityOfDroppingOut,
                                  uint64_t randomSeed,
                                  uint64_t forwardSequence,
                                  Stream stream) {
    launchDropOutForward(input,
                         output,
                         DataType::BF16,
                         numElements,
                         probabilityOfDroppingOut,
                         randomSeed,
                         forwardSequence,
                         stream);
}

void launchBfloat16DropOutBackward(const void *errorInput,
                                   void *errorOutput,
                                   uint64_t numElements,
                                   float probabilityOfDroppingOut,
                                   uint64_t randomSeed,
                                   uint64_t forwardSequence,
                                   Stream stream) {
    launchDropOutBackward(errorInput,
                          errorOutput,
                          DataType::BF16,
                          numElements,
                          probabilityOfDroppingOut,
                          randomSeed,
                          forwardSequence,
                          stream);
}

}  // namespace ThorImplementation
