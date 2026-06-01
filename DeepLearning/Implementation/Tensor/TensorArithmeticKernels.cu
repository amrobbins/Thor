// FIXME: Any math ops here are deprecated. Fill and non-math ops stay, should be re-homed and this file deleted.

#include <curand.h>
#include <curand_kernel.h>
#include <optional>
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;
using namespace std;

template <typename DATA_TYPE, typename VALUE_TYPE>
__device__ inline DATA_TYPE castGpuFillValue(VALUE_TYPE value) {
    return DATA_TYPE(value);
}

template <>
__device__ inline half castGpuFillValue<half, float>(float value) {
    return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 castGpuFillValue<__nv_bfloat16, float>(float value) {
    return __float2bfloat16(value);
}

// CUDA kernel to set random values in GPU device memory
template <typename DATA_TYPE, typename SCALE_TYPE>
__global__ void setRandomValues(DATA_TYPE *mem, uint64_t numElements, SCALE_TYPE minValue, SCALE_TYPE range, uint64_t seed) {
    uint64_t offset = 4 * blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= numElements)
        return;

    curandState_t state;
    curand_init(seed, offset, 0, &state);
    SCALE_TYPE randomValue = curand_uniform(&state) * range + minValue;
    mem[offset] = castGpuFillValue<DATA_TYPE>(randomValue);

    offset += blockDim.x;
    if (offset >= numElements)
        return;
    randomValue = curand_uniform(&state) * range + minValue;
    mem[offset] = castGpuFillValue<DATA_TYPE>(randomValue);

    offset += blockDim.x;
    if (offset >= numElements)
        return;
    randomValue = curand_uniform(&state) * range + minValue;
    mem[offset] = castGpuFillValue<DATA_TYPE>(randomValue);

    offset += blockDim.x;
    if (offset >= numElements)
        return;
    randomValue = curand_uniform(&state) * range + minValue;
    mem[offset] = castGpuFillValue<DATA_TYPE>(randomValue);
}

// Function to set random values in GPU device memory
template <typename DATA_TYPE>
void Tensor::launchGpuFillRandom(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream) {
    random_device rd;
    uint64_t seed = Tensor::Tensor::getThreadIdHash64(rd());
    int blockSize = 256;
    int gridSize = (numElements + (4 * blockSize) - 1) / (4 * blockSize);

    double range = maxValue - minValue;
    const bool useDoubleScale = abs(maxValue) > 1000000 || abs(minValue) > 1000000 || abs(range) > 1000000 || sizeof(DATA_TYPE) >= 8;

    if constexpr (is_same<DATA_TYPE, half>::value) {
        setRandomValues<half, float><<<gridSize, blockSize, 0, stream>>>((half *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, __nv_bfloat16>::value) {
        setRandomValues<__nv_bfloat16, float><<<gridSize, blockSize, 0, stream>>>((__nv_bfloat16 *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, __nv_fp8_e4m3>::value) {
        setRandomValues<__nv_fp8_e4m3, float><<<gridSize, blockSize, 0, stream>>>((__nv_fp8_e4m3 *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, __nv_fp8_e5m2>::value) {
        setRandomValues<__nv_fp8_e5m2, float><<<gridSize, blockSize, 0, stream>>>((__nv_fp8_e5m2 *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, double>::value) {
        setRandomValues<double, double><<<gridSize, blockSize, 0, stream>>>((double *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, int64_t>::value) {
        setRandomValues<int64_t, double><<<gridSize, blockSize, 0, stream>>>((int64_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, uint64_t>::value) {
        setRandomValues<uint64_t, double><<<gridSize, blockSize, 0, stream>>>((uint64_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, int32_t>::value) {
        if (useDoubleScale)
            setRandomValues<int32_t, double><<<gridSize, blockSize, 0, stream>>>((int32_t *)mem, numElements, minValue, range, seed);
        else
            setRandomValues<int32_t, float><<<gridSize, blockSize, 0, stream>>>((int32_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, uint32_t>::value) {
        if (useDoubleScale)
            setRandomValues<uint32_t, double><<<gridSize, blockSize, 0, stream>>>((uint32_t *)mem, numElements, minValue, range, seed);
        else
            setRandomValues<uint32_t, float><<<gridSize, blockSize, 0, stream>>>((uint32_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, float>::value) {
        setRandomValues<float, float><<<gridSize, blockSize, 0, stream>>>((float *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, int8_t>::value) {
        setRandomValues<int8_t, float><<<gridSize, blockSize, 0, stream>>>((int8_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, int16_t>::value) {
        setRandomValues<int16_t, float><<<gridSize, blockSize, 0, stream>>>((int16_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, uint8_t>::value) {
        setRandomValues<uint8_t, float><<<gridSize, blockSize, 0, stream>>>((uint8_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, uint16_t>::value) {
        setRandomValues<uint16_t, float><<<gridSize, blockSize, 0, stream>>>((uint16_t *)mem, numElements, minValue, range, seed);
    } else if constexpr (is_same<DATA_TYPE, bool>::value) {
        setRandomValues<bool, float><<<gridSize, blockSize, 0, stream>>>((bool *)mem, numElements, minValue, range, seed);
    } else {
        THOR_UNREACHABLE();
    }
}

template void Tensor::launchGpuFillRandom<half>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<__nv_bfloat16>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<__nv_fp8_e4m3>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<__nv_fp8_e5m2>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<float>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<double>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<int8_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<int16_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<int32_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<int64_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<uint8_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<uint16_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<uint32_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<uint64_t>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);
template void Tensor::launchGpuFillRandom<bool>(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);

template <typename DATA_TYPE>
__global__ void fillValue1B(DATA_TYPE value, DATA_TYPE *mem, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + threadIdx.x * 16;
    if (offset >= numElements)
        return;
    uint64_t offset16Elements = offset >> 4;

    DATA_TYPE buffer[16];
    buffer[0] = value;
    buffer[1] = value;
    buffer[2] = value;
    buffer[3] = value;
    buffer[4] = value;
    buffer[5] = value;
    buffer[6] = value;
    buffer[7] = value;
    buffer[8] = value;
    buffer[9] = value;
    buffer[10] = value;
    buffer[11] = value;
    buffer[12] = value;
    buffer[13] = value;
    buffer[14] = value;
    buffer[15] = value;

    ((float4 *)mem)[offset16Elements] = ((float4 *)buffer)[0];
}

template <typename DATA_TYPE>
__global__ void fillValue2B(DATA_TYPE value, DATA_TYPE *mem, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    DATA_TYPE buffer[8];
    buffer[0] = value;
    buffer[1] = value;
    buffer[2] = value;
    buffer[3] = value;
    buffer[4] = value;
    buffer[5] = value;
    buffer[6] = value;
    buffer[7] = value;

    ((float4 *)mem)[offset8Elements] = ((float4 *)buffer)[0];
}

template <typename DATA_TYPE>
__global__ void fillValue4B(DATA_TYPE value, DATA_TYPE *mem, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    DATA_TYPE buffer[4];
    buffer[0] = value;
    buffer[1] = value;
    buffer[2] = value;
    buffer[3] = value;

    ((float4 *)mem)[offset4Elements] = ((float4 *)buffer)[0];
}

template <typename DATA_TYPE>
__global__ void fillValue8B(DATA_TYPE value, DATA_TYPE *mem, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    DATA_TYPE buffer[2];
    buffer[0] = value;
    buffer[1] = value;

    ((float4 *)mem)[offset2Elements] = ((float4 *)buffer)[0];
}

template <typename T>
void Tensor::launchFillValueGpuKernel(T value, T *mem, uint64_t numElements, uint32_t deviceNum, Stream stream) {
    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    dim3 blockSize(256);
    if constexpr (is_same<T, uint8_t>::value || is_same<T, int8_t>::value || is_same<T, bool>::value || is_same<T, __nv_fp8_e4m3>::value ||
                  is_same<T, __nv_fp8_e5m2>::value) {
        dim3 gridSize((numElements + 4095) / 4096);
        fillValue1B<T><<<gridSize, blockSize, 0, stream>>>(value, mem, numElements);
    } else if constexpr (is_same<T, half>::value || is_same<T, __nv_bfloat16>::value || is_same<T, uint16_t>::value ||
                         is_same<T, int16_t>::value) {
        dim3 gridSize((numElements + 2047) / 2048);
        fillValue2B<T><<<gridSize, blockSize, 0, stream>>>(value, mem, numElements);
    } else if constexpr (is_same<T, float>::value || is_same<T, uint32_t>::value || is_same<T, int32_t>::value) {
        dim3 gridSize((numElements + 1023) / 1024);
        fillValue4B<T><<<gridSize, blockSize, 0, stream>>>(value, mem, numElements);
    } else if constexpr (is_same<T, double>::value || is_same<T, uint64_t>::value || is_same<T, int64_t>::value) {
        dim3 gridSize((numElements + 511) / 512);
        fillValue8B<T><<<gridSize, blockSize, 0, stream>>>(value, mem, numElements);
    } else {
        THOR_UNREACHABLE();
    }
}

template void Tensor::launchFillValueGpuKernel<half>(half value, half *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<__nv_bfloat16>(
    __nv_bfloat16 value, __nv_bfloat16 *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<__nv_fp8_e4m3>(
    __nv_fp8_e4m3 value, __nv_fp8_e4m3 *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<__nv_fp8_e5m2>(
    __nv_fp8_e5m2 value, __nv_fp8_e5m2 *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<float>(float value, float *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<double>(double value, double *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<uint8_t>(
    uint8_t value, uint8_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<uint16_t>(
    uint16_t value, uint16_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<uint32_t>(
    uint32_t value, uint32_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<uint64_t>(
    uint64_t value, uint64_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<int8_t>(int8_t value, int8_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<int16_t>(
    int16_t value, int16_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<int32_t>(
    int32_t value, int32_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<int64_t>(
    int64_t value, int64_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<bool>(bool value, bool *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);

template <typename DATA_TYPE>
__global__ void fillIdentityOnes(DATA_TYPE *mem, uint32_t N) {
    uint32_t index = blockIdx.x * 256 + threadIdx.x;
    if (index >= N)
        return;

    mem[index * N + index] = castGpuFillValue<DATA_TYPE>(1.0f);
}

void Tensor::fillGpuIdentityMatrixOnes(Stream stream) {
    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    DataType dataType = getDataType();
    THOR_THROW_IF_FALSE(dataType != DataType::PACKED_BOOLEAN);
    uint32_t N = getDimensions()[0];

    dim3 blockSize(256);
    dim3 gridSize((N + 255) / 256);
    if (dataType == DataType::FP16) {
        fillIdentityOnes<half><<<gridSize, blockSize, 0, stream>>>(getMemPtr<half>(), N);
    } else if (dataType == DataType::BF16) {
        fillIdentityOnes<__nv_bfloat16><<<gridSize, blockSize, 0, stream>>>(getMemPtr<__nv_bfloat16>(), N);
    } else if (dataType == DataType::FP8_E4M3) {
        fillIdentityOnes<__nv_fp8_e4m3><<<gridSize, blockSize, 0, stream>>>(getMemPtr<__nv_fp8_e4m3>(), N);
    } else if (dataType == DataType::FP8_E5M2) {
        fillIdentityOnes<__nv_fp8_e5m2><<<gridSize, blockSize, 0, stream>>>(getMemPtr<__nv_fp8_e5m2>(), N);
    } else if (dataType == DataType::FP32) {
        fillIdentityOnes<float><<<gridSize, blockSize, 0, stream>>>(getMemPtr<float>(), N);
    } else if (dataType == DataType::FP64) {
        fillIdentityOnes<double><<<gridSize, blockSize, 0, stream>>>(getMemPtr<double>(), N);
    } else if (dataType == DataType::INT8) {
        fillIdentityOnes<int8_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<int8_t>(), N);
    } else if (dataType == DataType::INT16) {
        fillIdentityOnes<int16_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<int16_t>(), N);
    } else if (dataType == DataType::INT32) {
        fillIdentityOnes<int32_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<int32_t>(), N);
    } else if (dataType == DataType::INT64) {
        fillIdentityOnes<int64_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<int64_t>(), N);
    } else if (dataType == DataType::UINT8) {
        fillIdentityOnes<uint8_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<uint8_t>(), N);
    } else if (dataType == DataType::UINT16) {
        fillIdentityOnes<uint16_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<uint16_t>(), N);
    } else if (dataType == DataType::UINT32) {
        fillIdentityOnes<uint32_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<uint32_t>(), N);
    } else if (dataType == DataType::UINT64) {
        fillIdentityOnes<uint64_t><<<gridSize, blockSize, 0, stream>>>(getMemPtr<uint64_t>(), N);
    } else if (dataType == DataType::BOOLEAN) {
        fillIdentityOnes<bool><<<gridSize, blockSize, 0, stream>>>(getMemPtr<bool>(), N);
    } else {
        THOR_UNREACHABLE();
    }
}
