#include "DeepLearning/Implementation/Layers/Utility/ScaleGradientKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ThorImplementation {
namespace {

template <typename T>
__global__ void scaleGradientKernel(const T *source, T *destination, float scale, uint64_t numElements) {
    uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element < numElements)
        destination[element] = T(static_cast<float>(source[element]) * scale);
}

template <>
__global__ void scaleGradientKernel<double>(const double *source, double *destination, float scale, uint64_t numElements) {
    uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element < numElements)
        destination[element] = source[element] * static_cast<double>(scale);
}

template <typename T>
void launchTyped(const void *source, void *destination, float scale, uint64_t numElements, Stream stream) {
    constexpr uint32_t blockSize = 256;
    const uint32_t gridSize = static_cast<uint32_t>((numElements + blockSize - 1) / blockSize);
    ScopedGpu scopedGpu(stream.getGpuNum());
    scaleGradientKernel<T><<<gridSize, blockSize, 0, stream.getStream()>>>(
        static_cast<const T *>(source), static_cast<T *>(destination), scale, numElements);
}

}  // namespace

void launchScaleGradient(const void *source,
                         void *destination,
                         DataType dataType,
                         float scale,
                         uint64_t numElements,
                         Stream stream) {
    if (numElements == 0)
        return;
    THOR_THROW_IF_FALSE(source != nullptr);
    THOR_THROW_IF_FALSE(destination != nullptr);

    switch (dataType) {
        case DataType::FP16:
            launchTyped<half>(source, destination, scale, numElements, stream);
            return;
        case DataType::BF16:
            launchTyped<__nv_bfloat16>(source, destination, scale, numElements, stream);
            return;
        case DataType::FP32:
            launchTyped<float>(source, destination, scale, numElements, stream);
            return;
        case DataType::FP64:
            launchTyped<double>(source, destination, scale, numElements, stream);
            return;
        case DataType::FP8_E4M3:
            launchTyped<__nv_fp8_e4m3>(source, destination, scale, numElements, stream);
            return;
        case DataType::FP8_E5M2:
            launchTyped<__nv_fp8_e5m2>(source, destination, scale, numElements, stream);
            return;
        default:
            THOR_THROW_LOGIC_ERROR("ScaleGradient requires a floating-point tensor storage type.");
    }
}

}  // namespace ThorImplementation
