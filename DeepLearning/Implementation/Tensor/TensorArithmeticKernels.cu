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

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar1BUnsignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[8];
    DATA_TYPE addendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;
    addendBuffer[2] = addend;
    addendBuffer[3] = addend;

    ((float2 *)buffer)[0] = ((float2 *)augend)[offset8Elements];
    ((uint32_t *)buffer)[0] = __vadd4(((uint32_t *)buffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)buffer)[1] = __vadd4(((uint32_t *)buffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar1BSignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[8];
    DATA_TYPE addendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;
    addendBuffer[2] = addend;
    addendBuffer[3] = addend;

    ((float2 *)buffer)[0] = ((float2 *)augend)[offset8Elements];
    ((uint32_t *)buffer)[0] = __vaddss4(((uint32_t *)buffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)buffer)[1] = __vaddss4(((uint32_t *)buffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar2BUnsignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vadd2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[2] = __vadd2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[3] = __vadd2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[0]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar2BSignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vaddss2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[2] = __vaddss2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[3] = __vaddss2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[0]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void addScalarHalf(half *augend, half *dest, half addend, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 buffer[4];
    half2 addendHalf2;
    addendHalf2.x = addend;
    addendHalf2.y = addend;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)augend)[offset8Elements];
    buffer[0] = __hadd2(buffer[0], addendHalf2);
    buffer[1] = __hadd2(buffer[1], addendHalf2);
    buffer[2] = __hadd2(buffer[2], addendHalf2);
    buffer[3] = __hadd2(buffer[3], addendHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)buffer)[0] = ((float4 *)augend)[offset8Elements];
    buffer[0] = __hadd2(buffer[0], addendHalf2);
    buffer[1] = __hadd2(buffer[1], addendHalf2);
    buffer[2] = __hadd2(buffer[2], addendHalf2);
    buffer[3] = __hadd2(buffer[3], addendHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar4B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)buffer)[0] = ((double2 *)augend)[offset4Elements];
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    buffer[2] = buffer[2] + addend;
    buffer[3] = buffer[3] + addend;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)buffer)[0] = ((double2 *)augend)[offset4Elements];
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    buffer[2] = buffer[2] + addend;
    buffer[3] = buffer[3] + addend;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise1BUnsignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vadd4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vadd4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise1BSignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vaddss4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vaddss4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise2BUnsignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vadd2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vadd2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vadd2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vadd2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vadd2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vadd2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 32 elements : 8192 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise2BSignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vaddss2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vaddss2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vaddss2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    ((uint32_t *)augendBuffer)[0] = __vaddss2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vaddss2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vaddss2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void addElementwiseHalf(half *augend, half *dest, half *addend, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 augendBuffer[4];
    half2 addendBuffer[4];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)augendBuffer)[0] = ((float4 *)augend)[offset8Elements];
    ((float4 *)addendBuffer)[0] = ((float4 *)addend)[offset8Elements];
    augendBuffer[0] = __hadd2(augendBuffer[0], addendBuffer[0]);
    augendBuffer[1] = __hadd2(augendBuffer[1], addendBuffer[1]);
    augendBuffer[2] = __hadd2(augendBuffer[2], addendBuffer[2]);
    augendBuffer[3] = __hadd2(augendBuffer[3], addendBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)augendBuffer)[0] = ((float4 *)augend)[offset8Elements];
    ((float4 *)addendBuffer)[0] = ((float4 *)addend)[offset8Elements];
    augendBuffer[0] = __hadd2(augendBuffer[0], addendBuffer[0]);
    augendBuffer[1] = __hadd2(augendBuffer[1], addendBuffer[1]);
    augendBuffer[2] = __hadd2(augendBuffer[2], addendBuffer[2]);
    augendBuffer[3] = __hadd2(augendBuffer[3], addendBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise4B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[4];
    DATA_TYPE addendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset4Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset4Elements];
    augendBuffer[0] = augendBuffer[0] + addendBuffer[0];
    augendBuffer[1] = augendBuffer[1] + addendBuffer[1];
    augendBuffer[2] = augendBuffer[2] + addendBuffer[2];
    augendBuffer[3] = augendBuffer[3] + addendBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)augendBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset4Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset4Elements];
    augendBuffer[0] = augendBuffer[0] + addendBuffer[0];
    augendBuffer[1] = augendBuffer[1] + addendBuffer[1];
    augendBuffer[2] = augendBuffer[2] + addendBuffer[2];
    augendBuffer[3] = augendBuffer[3] + addendBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar1BUnsignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, float alpha, uint64_t numElements) {
    DATA_TYPE buffer[8];
    DATA_TYPE addendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;
    addendBuffer[2] = addend;
    addendBuffer[3] = addend;

    ((float2 *)buffer)[0] = ((float2 *)augend)[offset8Elements];
    buffer[0] *= alpha;
    buffer[1] *= alpha;
    buffer[2] *= alpha;
    buffer[3] *= alpha;
    buffer[4] *= alpha;
    buffer[5] *= alpha;
    buffer[6] *= alpha;
    buffer[7] *= alpha;
    ((uint32_t *)buffer)[0] = __vadd4(((uint32_t *)buffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)buffer)[1] = __vadd4(((uint32_t *)buffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

template <typename DATA_TYPE>
__global__ void addScalar1BSignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, float alpha, uint64_t numElements) {
    DATA_TYPE buffer[8];
    DATA_TYPE addendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;
    addendBuffer[2] = addend;
    addendBuffer[3] = addend;

    ((float2 *)buffer)[0] = ((float2 *)augend)[offset8Elements];
    buffer[0] *= alpha;
    buffer[1] *= alpha;
    buffer[2] *= alpha;
    buffer[3] *= alpha;
    buffer[4] *= alpha;
    buffer[5] *= alpha;
    buffer[6] *= alpha;
    buffer[7] *= alpha;
    ((uint32_t *)buffer)[0] = __vaddss4(((uint32_t *)buffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)buffer)[1] = __vaddss4(((uint32_t *)buffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar2BUnsignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, float alpha, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    ((uint32_t *)augendBuffer)[0] = __vadd2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[2] = __vadd2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[3] = __vadd2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[0]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

template <typename DATA_TYPE>
__global__ void addScalar2BSignedInt(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, float alpha, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    addendBuffer[0] = addend;
    addendBuffer[1] = addend;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    ((uint32_t *)augendBuffer)[0] = __vaddss2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[2] = __vaddss2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[3] = __vaddss2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[0]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void addScalarHalf(half *augend, half *dest, half addend, half alpha, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 buffer[4];
    half2 addendHalf2;
    addendHalf2.x = addend;
    addendHalf2.y = addend;
    half2 alphaHalf2;
    alphaHalf2.x = alpha;
    alphaHalf2.y = alpha;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)augend)[offset8Elements];
    buffer[0] = __hmul2(buffer[0], alphaHalf2);
    buffer[1] = __hmul2(buffer[1], alphaHalf2);
    buffer[2] = __hmul2(buffer[2], alphaHalf2);
    buffer[3] = __hmul2(buffer[3], alphaHalf2);
    buffer[0] = __hadd2(buffer[0], addendHalf2);
    buffer[1] = __hadd2(buffer[1], addendHalf2);
    buffer[2] = __hadd2(buffer[2], addendHalf2);
    buffer[3] = __hadd2(buffer[3], addendHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)buffer)[0] = ((float4 *)augend)[offset8Elements];
    buffer[0] = __hmul2(buffer[0], alphaHalf2);
    buffer[1] = __hmul2(buffer[1], alphaHalf2);
    buffer[2] = __hmul2(buffer[2], alphaHalf2);
    buffer[3] = __hmul2(buffer[3], alphaHalf2);
    buffer[0] = __hadd2(buffer[0], addendHalf2);
    buffer[1] = __hadd2(buffer[1], addendHalf2);
    buffer[2] = __hadd2(buffer[2], addendHalf2);
    buffer[3] = __hadd2(buffer[3], addendHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar4B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, float alpha, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)buffer)[0] = ((double2 *)augend)[offset4Elements];
    buffer[0] *= alpha;
    buffer[1] *= alpha;
    buffer[2] *= alpha;
    buffer[3] *= alpha;
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    buffer[2] = buffer[2] + addend;
    buffer[3] = buffer[3] + addend;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)buffer)[0] = ((double2 *)augend)[offset4Elements];
    buffer[0] *= alpha;
    buffer[1] *= alpha;
    buffer[2] *= alpha;
    buffer[3] *= alpha;
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    buffer[2] = buffer[2] + addend;
    buffer[3] = buffer[3] + addend;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise1BUnsignedInt(
    DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, float alpha, float beta, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vadd4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vadd4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise1BSignedInt(
    DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, float alpha, float beta, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vaddss4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vaddss4(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss4(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise2BUnsignedInt(
    DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, float alpha, float beta, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vadd2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vadd2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vadd2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vadd2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vadd2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vadd2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vadd2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise2BSignedInt(
    DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, float alpha, float beta, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vaddss2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vaddss2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vaddss2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset8Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset8Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    augendBuffer[4] *= alpha;
    augendBuffer[5] *= alpha;
    augendBuffer[6] *= alpha;
    augendBuffer[7] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    addendBuffer[4] *= beta;
    addendBuffer[5] *= beta;
    addendBuffer[6] *= beta;
    addendBuffer[7] *= beta;
    ((uint32_t *)augendBuffer)[0] = __vaddss2(((uint32_t *)augendBuffer)[0], ((uint32_t *)addendBuffer)[0]);
    ((uint32_t *)augendBuffer)[1] = __vaddss2(((uint32_t *)augendBuffer)[1], ((uint32_t *)addendBuffer)[1]);
    ((uint32_t *)augendBuffer)[2] = __vaddss2(((uint32_t *)augendBuffer)[2], ((uint32_t *)addendBuffer)[2]);
    ((uint32_t *)augendBuffer)[3] = __vaddss2(((uint32_t *)augendBuffer)[3], ((uint32_t *)addendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void addElementwiseHalf(half *augend, half *dest, half *addend, float alpha, float beta, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 augendBuffer[4];
    half2 addendBuffer[4];
    half2 alphaHalf2;
    half2 betaHalf2;
    alphaHalf2.x = alpha;
    alphaHalf2.y = alpha;
    betaHalf2.x = beta;
    betaHalf2.y = beta;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)augendBuffer)[0] = ((float4 *)augend)[offset8Elements];
    ((float4 *)addendBuffer)[0] = ((float4 *)addend)[offset8Elements];
    augendBuffer[0] = __hmul2(augendBuffer[0], alphaHalf2);
    augendBuffer[1] = __hmul2(augendBuffer[1], alphaHalf2);
    augendBuffer[2] = __hmul2(augendBuffer[2], alphaHalf2);
    augendBuffer[3] = __hmul2(augendBuffer[3], alphaHalf2);
    addendBuffer[0] = __hmul2(addendBuffer[0], betaHalf2);
    addendBuffer[1] = __hmul2(addendBuffer[1], betaHalf2);
    addendBuffer[2] = __hmul2(addendBuffer[2], betaHalf2);
    addendBuffer[3] = __hmul2(addendBuffer[3], betaHalf2);
    augendBuffer[0] = __hadd2(augendBuffer[0], addendBuffer[0]);
    augendBuffer[1] = __hadd2(augendBuffer[1], addendBuffer[1]);
    augendBuffer[2] = __hadd2(augendBuffer[2], addendBuffer[2]);
    augendBuffer[3] = __hadd2(augendBuffer[3], addendBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)augendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)augendBuffer)[0] = ((float4 *)augend)[offset8Elements];
    ((float4 *)addendBuffer)[0] = ((float4 *)addend)[offset8Elements];
    augendBuffer[0] = __hmul2(augendBuffer[0], alphaHalf2);
    augendBuffer[1] = __hmul2(augendBuffer[1], alphaHalf2);
    augendBuffer[2] = __hmul2(augendBuffer[2], alphaHalf2);
    augendBuffer[3] = __hmul2(augendBuffer[3], alphaHalf2);
    addendBuffer[0] = __hmul2(addendBuffer[0], betaHalf2);
    addendBuffer[1] = __hmul2(addendBuffer[1], betaHalf2);
    addendBuffer[2] = __hmul2(addendBuffer[2], betaHalf2);
    addendBuffer[3] = __hmul2(addendBuffer[3], betaHalf2);
    augendBuffer[0] = __hadd2(augendBuffer[0], addendBuffer[0]);
    augendBuffer[1] = __hadd2(augendBuffer[1], addendBuffer[1]);
    augendBuffer[2] = __hadd2(augendBuffer[2], addendBuffer[2]);
    augendBuffer[3] = __hadd2(augendBuffer[3], addendBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise4B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, float alpha, float beta, uint64_t numElements) {
    DATA_TYPE augendBuffer[4];
    DATA_TYPE addendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset4Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset4Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    augendBuffer[0] = augendBuffer[0] + addendBuffer[0];
    augendBuffer[1] = augendBuffer[1] + addendBuffer[1];
    augendBuffer[2] = augendBuffer[2] + addendBuffer[2];
    augendBuffer[3] = augendBuffer[3] + addendBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)augendBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)augendBuffer)[0] = ((double2 *)augend)[offset4Elements];
    ((double2 *)addendBuffer)[0] = ((double2 *)addend)[offset4Elements];
    augendBuffer[0] *= alpha;
    augendBuffer[1] *= alpha;
    augendBuffer[2] *= alpha;
    augendBuffer[3] *= alpha;
    addendBuffer[0] *= beta;
    addendBuffer[1] *= beta;
    addendBuffer[2] *= beta;
    addendBuffer[3] *= beta;
    augendBuffer[0] = augendBuffer[0] + addendBuffer[0];
    augendBuffer[1] = augendBuffer[1] + addendBuffer[1];
    augendBuffer[2] = augendBuffer[2] + addendBuffer[2];
    augendBuffer[3] = augendBuffer[3] + addendBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarDenominator1B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE denominator, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)numerator)[offset8Elements];
    buffer[0] = buffer[0] / denominator;
    buffer[1] = buffer[1] / denominator;
    buffer[2] = buffer[2] / denominator;
    buffer[3] = buffer[3] / denominator;
    buffer[4] = buffer[4] / denominator;
    buffer[5] = buffer[5] / denominator;
    buffer[6] = buffer[6] / denominator;
    buffer[7] = buffer[7] / denominator;
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarDenominator2B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE denominator, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)buffer)[0] = ((double2 *)numerator)[offset8Elements];
    buffer[0] = buffer[0] / denominator;
    buffer[1] = buffer[1] / denominator;
    buffer[2] = buffer[2] / denominator;
    buffer[3] = buffer[3] / denominator;
    buffer[4] = buffer[4] / denominator;
    buffer[5] = buffer[5] / denominator;
    buffer[6] = buffer[6] / denominator;
    buffer[7] = buffer[7] / denominator;
    ((double2 *)dest)[offset8Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void divideScalarDenominatorHalf(half *numerator, half *dest, half denominator, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 buffer[4];
    half2 denominatorHalf2;
    denominatorHalf2.x = denominator;
    denominatorHalf2.y = denominator;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)numerator)[offset8Elements];
    buffer[0] = __h2div(buffer[0], denominatorHalf2);
    buffer[1] = __h2div(buffer[1], denominatorHalf2);
    buffer[2] = __h2div(buffer[2], denominatorHalf2);
    buffer[3] = __h2div(buffer[3], denominatorHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)buffer)[0] = ((float4 *)numerator)[offset8Elements];
    buffer[0] = __h2div(buffer[0], denominatorHalf2);
    buffer[1] = __h2div(buffer[1], denominatorHalf2);
    buffer[2] = __h2div(buffer[2], denominatorHalf2);
    buffer[3] = __h2div(buffer[3], denominatorHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarDenominator4B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE denominator, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)buffer)[0] = ((double2 *)numerator)[offset4Elements];
    buffer[0] = buffer[0] / denominator;
    buffer[1] = buffer[1] / denominator;
    buffer[2] = buffer[2] / denominator;
    buffer[3] = buffer[3] / denominator;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)buffer)[0] = ((double2 *)numerator)[offset4Elements];
    buffer[0] = buffer[0] / denominator;
    buffer[1] = buffer[1] / denominator;
    buffer[2] = buffer[2] / denominator;
    buffer[3] = buffer[3] / denominator;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarNumerator1B(DATA_TYPE *denominator, DATA_TYPE *dest, DATA_TYPE numerator, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)denominator)[offset8Elements];
    buffer[0] = numerator / buffer[0];
    buffer[1] = numerator / buffer[1];
    buffer[2] = numerator / buffer[2];
    buffer[3] = numerator / buffer[3];
    buffer[4] = numerator / buffer[4];
    buffer[5] = numerator / buffer[5];
    buffer[6] = numerator / buffer[6];
    buffer[7] = numerator / buffer[7];
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarNumerator2B(DATA_TYPE *denominator, DATA_TYPE *dest, DATA_TYPE numerator, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)buffer)[0] = ((double2 *)denominator)[offset8Elements];
    buffer[0] = numerator / buffer[0];
    buffer[1] = numerator / buffer[1];
    buffer[2] = numerator / buffer[2];
    buffer[3] = numerator / buffer[3];
    buffer[4] = numerator / buffer[4];
    buffer[5] = numerator / buffer[5];
    buffer[6] = numerator / buffer[6];
    buffer[7] = numerator / buffer[7];
    ((double2 *)dest)[offset8Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void divideScalarNumeratorHalf(half *denominator, half *dest, half numerator, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 buffer[4];
    half2 numeratorHalf2;
    numeratorHalf2.x = numerator;
    numeratorHalf2.y = numerator;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)denominator)[offset8Elements];
    buffer[0] = __h2div(numeratorHalf2, buffer[0]);
    buffer[1] = __h2div(numeratorHalf2, buffer[1]);
    buffer[2] = __h2div(numeratorHalf2, buffer[2]);
    buffer[3] = __h2div(numeratorHalf2, buffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)buffer)[0] = ((float4 *)denominator)[offset8Elements];
    buffer[0] = __h2div(numeratorHalf2, buffer[0]);
    buffer[1] = __h2div(numeratorHalf2, buffer[1]);
    buffer[2] = __h2div(numeratorHalf2, buffer[2]);
    buffer[3] = __h2div(numeratorHalf2, buffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarNumerator4B(DATA_TYPE *denominator, DATA_TYPE *dest, DATA_TYPE numerator, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)buffer)[0] = ((double2 *)denominator)[offset4Elements];
    buffer[0] = numerator / buffer[0];
    buffer[1] = numerator / buffer[1];
    buffer[2] = numerator / buffer[2];
    buffer[3] = numerator / buffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)buffer)[0] = ((double2 *)denominator)[offset4Elements];
    buffer[0] = numerator / buffer[0];
    buffer[1] = numerator / buffer[1];
    buffer[2] = numerator / buffer[2];
    buffer[3] = numerator / buffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideElementwise1B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE *denominator, uint64_t numElements) {
    DATA_TYPE numeratorBuffer[8];
    DATA_TYPE denominatorBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)numeratorBuffer)[0] = ((float2 *)numerator)[offset8Elements];
    ((float2 *)denominatorBuffer)[0] = ((float2 *)denominator)[offset8Elements];
    numeratorBuffer[0] = numeratorBuffer[0] / denominatorBuffer[0];
    numeratorBuffer[1] = numeratorBuffer[1] / denominatorBuffer[1];
    numeratorBuffer[2] = numeratorBuffer[2] / denominatorBuffer[2];
    numeratorBuffer[3] = numeratorBuffer[3] / denominatorBuffer[3];
    numeratorBuffer[4] = numeratorBuffer[4] / denominatorBuffer[4];
    numeratorBuffer[5] = numeratorBuffer[5] / denominatorBuffer[5];
    numeratorBuffer[6] = numeratorBuffer[6] / denominatorBuffer[6];
    numeratorBuffer[7] = numeratorBuffer[7] / denominatorBuffer[7];
    ((float2 *)dest)[offset8Elements] = ((float2 *)numeratorBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideElementwise2B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE *denominator, uint64_t numElements) {
    DATA_TYPE numeratorBuffer[8];
    DATA_TYPE denominatorBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)numeratorBuffer)[0] = ((double2 *)numerator)[offset8Elements];
    ((double2 *)denominatorBuffer)[0] = ((double2 *)denominator)[offset8Elements];
    numeratorBuffer[0] = numeratorBuffer[0] / denominatorBuffer[0];
    numeratorBuffer[1] = numeratorBuffer[1] / denominatorBuffer[1];
    numeratorBuffer[2] = numeratorBuffer[2] / denominatorBuffer[2];
    numeratorBuffer[3] = numeratorBuffer[3] / denominatorBuffer[3];
    numeratorBuffer[4] = numeratorBuffer[4] / denominatorBuffer[4];
    numeratorBuffer[5] = numeratorBuffer[5] / denominatorBuffer[5];
    numeratorBuffer[6] = numeratorBuffer[6] / denominatorBuffer[6];
    numeratorBuffer[7] = numeratorBuffer[7] / denominatorBuffer[7];
    ((double2 *)dest)[offset8Elements] = ((double2 *)numeratorBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void divideElementwiseHalf(half *numerator, half *dest, half *denominator, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 numeratorBuffer[4];
    half2 denominatorBuffer[4];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)numeratorBuffer)[0] = ((float4 *)numerator)[offset8Elements];
    ((float4 *)denominatorBuffer)[0] = ((float4 *)denominator)[offset8Elements];
    numeratorBuffer[0] = __h2div(numeratorBuffer[0], denominatorBuffer[0]);
    numeratorBuffer[1] = __h2div(numeratorBuffer[1], denominatorBuffer[1]);
    numeratorBuffer[2] = __h2div(numeratorBuffer[2], denominatorBuffer[2]);
    numeratorBuffer[3] = __h2div(numeratorBuffer[3], denominatorBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)numeratorBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)numeratorBuffer)[0] = ((float4 *)numerator)[offset8Elements];
    ((float4 *)denominatorBuffer)[0] = ((float4 *)denominator)[offset8Elements];
    numeratorBuffer[0] = __h2div(numeratorBuffer[0], denominatorBuffer[0]);
    numeratorBuffer[1] = __h2div(numeratorBuffer[1], denominatorBuffer[1]);
    numeratorBuffer[2] = __h2div(numeratorBuffer[2], denominatorBuffer[2]);
    numeratorBuffer[3] = __h2div(numeratorBuffer[3], denominatorBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)numeratorBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideElementwise4B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE *denominator, uint64_t numElements) {
    DATA_TYPE numeratorBuffer[4];
    DATA_TYPE denominatorBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)numeratorBuffer)[0] = ((double2 *)numerator)[offset4Elements];
    ((double2 *)denominatorBuffer)[0] = ((double2 *)denominator)[offset4Elements];
    numeratorBuffer[0] = numeratorBuffer[0] / denominatorBuffer[0];
    numeratorBuffer[1] = numeratorBuffer[1] / denominatorBuffer[1];
    numeratorBuffer[2] = numeratorBuffer[2] / denominatorBuffer[2];
    numeratorBuffer[3] = numeratorBuffer[3] / denominatorBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)numeratorBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)numeratorBuffer)[0] = ((double2 *)numerator)[offset4Elements];
    ((double2 *)denominatorBuffer)[0] = ((double2 *)denominator)[offset4Elements];
    numeratorBuffer[0] = numeratorBuffer[0] / denominatorBuffer[0];
    numeratorBuffer[1] = numeratorBuffer[1] / denominatorBuffer[1];
    numeratorBuffer[2] = numeratorBuffer[2] / denominatorBuffer[2];
    numeratorBuffer[3] = numeratorBuffer[3] / denominatorBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)numeratorBuffer)[0];
}

void Tensor::add(Tensor augend, Tensor addend, Stream stream) {
    THOR_THROW_IF_FALSE(augend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(augend.getDataType() == addend.getDataType());
    THOR_THROW_IF_FALSE(augend.getDataType() == getDataType());
    THOR_THROW_IF_FALSE(augend.getTotalNumElements() == addend.getTotalNumElements());
    THOR_THROW_IF_FALSE(augend.getTotalNumElements() == getTotalNumElements());

    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    DataType dataType = augend.getDataType();
    uint64_t numElements = augend.getTotalNumElements();
    void *augendMem = augend.getMemPtr();
    void *addendMem = addend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwiseHalf<<<gridSize, blockSize, 0, stream>>>((half *)augendMem, (half *)destMem, (half *)addendMem, numElements);
    } else if (dataType == DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<float><<<gridSize, blockSize, 0, stream>>>((float *)augendMem, (float *)destMem, (float *)addendMem, numElements);
    } else if (dataType == DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise1BUnsignedInt<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)augendMem, (uint8_t *)destMem, (uint8_t *)addendMem, numElements);
    } else if (dataType == DataType::UINT16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise2BUnsignedInt<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)augendMem, (uint16_t *)destMem, (uint16_t *)addendMem, numElements);
    } else if (dataType == DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)augendMem, (uint32_t *)destMem, (uint32_t *)addendMem, numElements);
    } else if (dataType == DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise1BSignedInt<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)augendMem, (int8_t *)destMem, (int8_t *)addendMem, numElements);
    } else if (dataType == DataType::INT16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise2BSignedInt<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)augendMem, (int16_t *)destMem, (int16_t *)addendMem, numElements);
    } else if (dataType == DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)augendMem, (int32_t *)destMem, (int32_t *)addendMem, numElements);
    } else {
        THOR_UNREACHABLE();
    }
}

void Tensor::divide(Tensor numerator, double denominator, Stream stream) {
    THOR_THROW_IF_FALSE(numerator.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(numerator.getDataType() == getDataType());

    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    DataType dataType = numerator.getDataType();
    uint64_t numElements = numerator.getTotalNumElements();
    void *numeratorMem = numerator.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        divideScalarDenominatorHalf<<<gridSize, blockSize, 0, stream>>>((half *)numeratorMem, (half *)destMem, denominator, numElements);
    } else if (dataType == DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)numeratorMem, (float *)destMem, denominator, numElements);
    } else if (dataType == DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)numeratorMem, (uint8_t *)destMem, denominator, numElements);
    } else if (dataType == DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)numeratorMem, (uint16_t *)destMem, denominator, numElements);
    } else if (dataType == DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)numeratorMem, (uint32_t *)destMem, denominator, numElements);
    } else if (dataType == DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)numeratorMem, (int8_t *)destMem, denominator, numElements);
    } else if (dataType == DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)numeratorMem, (int16_t *)destMem, denominator, numElements);
    } else if (dataType == DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)numeratorMem, (int32_t *)destMem, denominator, numElements);
    } else {
        THOR_UNREACHABLE();
    }
}

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
