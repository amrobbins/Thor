#include "DeepLearning/Implementation/Parameter/ParameterConstraint.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace ThorImplementation {

namespace {

template <typename T>
__device__ double scalarValue(T value) {
    return static_cast<double>(value);
}

template <typename T>
__device__ T scalarToStorage(double value) {
    return static_cast<T>(value);
}

template <>
__device__ double scalarValue<half>(half value) {
    return static_cast<double>(__half2float(value));
}

template <>
__device__ half scalarToStorage<half>(double value) {
    return __float2half(static_cast<float>(value));
}

template <>
__device__ double scalarValue<__nv_bfloat16>(__nv_bfloat16 value) {
    return static_cast<double>(__bfloat162float(value));
}

template <>
__device__ __nv_bfloat16 scalarToStorage<__nv_bfloat16>(double value) {
    return __float2bfloat16(static_cast<float>(value));
}

template <>
__device__ double scalarValue<__nv_fp8_e4m3>(__nv_fp8_e4m3 value) {
    return static_cast<double>(static_cast<float>(value));
}

template <>
__device__ __nv_fp8_e4m3 scalarToStorage<__nv_fp8_e4m3>(double value) {
    return __nv_fp8_e4m3(static_cast<float>(value));
}

template <>
__device__ double scalarValue<__nv_fp8_e5m2>(__nv_fp8_e5m2 value) {
    return static_cast<double>(static_cast<float>(value));
}

template <>
__device__ __nv_fp8_e5m2 scalarToStorage<__nv_fp8_e5m2>(double value) {
    return __nv_fp8_e5m2(static_cast<float>(value));
}

template <typename T>
__global__ void clampParameterKernel(T* values,
                                     uint64_t numElements,
                                     bool hasMinValue,
                                     double minValue,
                                     bool hasMaxValue,
                                     double maxValue) {
    uint64_t element = static_cast<uint64_t>(blockIdx.x) * 1024ULL + static_cast<uint64_t>(threadIdx.x);

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        if (element < numElements) {
            double value = scalarValue<T>(values[element]);
            if (hasMinValue && value < minValue) {
                value = minValue;
            }
            if (hasMaxValue && value > maxValue) {
                value = maxValue;
            }
            values[element] = scalarToStorage<T>(value);
        }
        element += 256ULL;
    }
}

template <typename T>
void launchClampParameterTyped(void* values,
                               uint64_t numElements,
                               bool hasMinValue,
                               double minValue,
                               bool hasMaxValue,
                               double maxValue,
                               Stream stream) {
    if (numElements == 0) {
        return;
    }
    dim3 blockSize(256);
    dim3 gridSize(static_cast<uint32_t>((numElements + 1023ULL) / 1024ULL));
    clampParameterKernel<T><<<gridSize, blockSize, 0, stream.getStream()>>>(
        static_cast<T*>(values), numElements, hasMinValue, minValue, hasMaxValue, maxValue);
}

}  // namespace

void launchClampParameter(void* values,
                          DataType dataType,
                          uint64_t numElements,
                          bool hasMinValue,
                          double minValue,
                          bool hasMaxValue,
                          double maxValue,
                          int gpuNum,
                          Stream stream) {
    ScopedGpu scopedGpu(gpuNum);
    switch (dataType) {
        case DataType::FP16:
            launchClampParameterTyped<half>(values, numElements, hasMinValue, minValue, hasMaxValue, maxValue, stream);
            return;
        case DataType::BF16:
            launchClampParameterTyped<__nv_bfloat16>(values, numElements, hasMinValue, minValue, hasMaxValue, maxValue, stream);
            return;
        case DataType::FP32:
            launchClampParameterTyped<float>(values, numElements, hasMinValue, minValue, hasMaxValue, maxValue, stream);
            return;
        case DataType::FP64:
            launchClampParameterTyped<double>(values, numElements, hasMinValue, minValue, hasMaxValue, maxValue, stream);
            return;
        case DataType::FP8_E4M3:
            launchClampParameterTyped<__nv_fp8_e4m3>(values, numElements, hasMinValue, minValue, hasMaxValue, maxValue, stream);
            return;
        case DataType::FP8_E5M2:
            launchClampParameterTyped<__nv_fp8_e5m2>(values, numElements, hasMinValue, minValue, hasMaxValue, maxValue, stream);
            return;
        default:
            THOR_UNREACHABLE();
    }
}

}  // namespace ThorImplementation
