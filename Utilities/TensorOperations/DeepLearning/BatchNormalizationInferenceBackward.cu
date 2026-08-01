#include "Utilities/TensorOperations/DeepLearning/BatchNormalizationInferenceBackward.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ThorImplementation {
namespace {

template <typename T>
__device__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    } else {
        return __bfloat162float(value);
    }
}

template <typename T>
__device__ T fromFloat(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(value);
    } else {
        return __float2bfloat16(value);
    }
}

template <typename T>
__global__ void batchNormalizationInferenceBackwardKernel(
    const T* __restrict__ errorInput,
    T* __restrict__ errorOutput,
    const float* __restrict__ scale,
    const float* __restrict__ runningVariance,
    float epsilon,
    uint32_t numChannels,
    uint64_t spatialElements,
    uint64_t totalElements) {
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < totalElements;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint32_t channel =
            static_cast<uint32_t>((index / spatialElements) % numChannels);
        const float inverseStandardDeviation =
            rsqrtf(runningVariance[channel] + epsilon);
        const float gradient =
            toFloat(errorInput[index]) * scale[channel] * inverseStandardDeviation;
        errorOutput[index] = fromFloat<T>(gradient);
    }
}

void validate(const Tensor& errorInput,
              const Tensor& errorOutput,
              const Tensor& scale,
              const Tensor& runningVariance,
              double epsilon,
              uint32_t numChannels,
              Stream stream) {
    THOR_THROW_IF_FALSE(errorInput.isInitialized());
    THOR_THROW_IF_FALSE(errorOutput.isInitialized());
    THOR_THROW_IF_FALSE(scale.isInitialized());
    THOR_THROW_IF_FALSE(runningVariance.isInitialized());
    THOR_THROW_IF_FALSE(stream.isInitialized());
    THOR_THROW_IF_FALSE(errorInput.getPlacement().getMemDevice() ==
                        TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(errorOutput.getPlacement() == errorInput.getPlacement());
    THOR_THROW_IF_FALSE(scale.getPlacement() == errorInput.getPlacement());
    THOR_THROW_IF_FALSE(runningVariance.getPlacement() == errorInput.getPlacement());
    THOR_THROW_IF_FALSE(stream.getGpuNum() == errorInput.getPlacement().getDeviceNum());
    THOR_THROW_IF_FALSE(errorOutput.getDescriptor() == errorInput.getDescriptor());
    THOR_THROW_IF_FALSE(!errorInput.hasCustomStrides());
    THOR_THROW_IF_FALSE(!errorOutput.hasCustomStrides());
    THOR_THROW_IF_FALSE(errorInput.isDenseContiguous());
    THOR_THROW_IF_FALSE(errorOutput.isDenseContiguous());
    THOR_THROW_IF_FALSE(scale.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(runningVariance.getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(numChannels >= 1);
    THOR_THROW_IF_FALSE(scale.getDimensions() == std::vector<uint64_t>{numChannels});
    THOR_THROW_IF_FALSE(runningVariance.getDimensions() ==
                        std::vector<uint64_t>{numChannels});
    THOR_THROW_IF_FALSE(epsilon > 0.0);

    const std::vector<uint64_t> dimensions = errorInput.getDimensions();
    THOR_THROW_IF_FALSE(dimensions.size() == 2 || dimensions.size() == 4 ||
                        dimensions.size() == 5);
    THOR_THROW_IF_FALSE(dimensions.at(1) == numChannels);
    THOR_THROW_IF_FALSE(errorInput.getTotalNumElements() >= 1);
}

template <typename T>
void launchTyped(const Tensor& errorInput,
                 Tensor& errorOutput,
                 const Tensor& scale,
                 const Tensor& runningVariance,
                 float epsilon,
                 uint32_t numChannels,
                 uint64_t spatialElements,
                 Stream stream) {
    constexpr uint32_t blockSize = 256;
    const uint64_t totalElements = errorInput.getTotalNumElements();
    const uint64_t blocks = (totalElements + blockSize - 1) / blockSize;
    const uint32_t gridSize =
        static_cast<uint32_t>(std::min<uint64_t>(blocks, 65535));
    batchNormalizationInferenceBackwardKernel<T>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(
            errorInput.getMemPtr<T>(),
            errorOutput.getMemPtr<T>(),
            scale.getMemPtr<float>(),
            runningVariance.getMemPtr<float>(),
            epsilon,
            numChannels,
            spatialElements,
            totalElements);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchBatchNormalizationInferenceBackward(const Tensor& errorInput,
                                                Tensor& errorOutput,
                                                const Tensor& scale,
                                                const Tensor& runningVariance,
                                                double epsilon,
                                                uint32_t numChannels,
                                                Stream stream) {
    validate(errorInput,
             errorOutput,
             scale,
             runningVariance,
             epsilon,
             numChannels,
             stream);
    const std::vector<uint64_t> dimensions = errorInput.getDimensions();
    uint64_t spatialElements = 1;
    for (uint64_t dimensionIndex = 2; dimensionIndex < dimensions.size();
         ++dimensionIndex) {
        spatialElements *= dimensions.at(dimensionIndex);
    }

    ScopedGpu scopedGpu(errorInput.getPlacement().getDeviceNum());
    switch (errorInput.getDataType()) {
        case DataType::FP32:
            launchTyped<float>(errorInput,
                               errorOutput,
                               scale,
                               runningVariance,
                               static_cast<float>(epsilon),
                               numChannels,
                               spatialElements,
                               stream);
            return;
        case DataType::FP16:
            launchTyped<__half>(errorInput,
                                errorOutput,
                                scale,
                                runningVariance,
                                static_cast<float>(epsilon),
                                numChannels,
                                spatialElements,
                                stream);
            return;
        case DataType::BF16:
            launchTyped<__nv_bfloat16>(errorInput,
                                       errorOutput,
                                       scale,
                                       runningVariance,
                                       static_cast<float>(epsilon),
                                       numChannels,
                                       spatialElements,
                                       stream);
            return;
        default:
            throw std::runtime_error(
                "BatchNormalization inference backward supports FP32, FP16, and BF16 tensors.");
    }
}

}  // namespace ThorImplementation
