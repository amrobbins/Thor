#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;
/*
// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyScalarMultiplier1B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)multiplicand)[offset8Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    buffer[2] = buffer[2] * multiplier;
    buffer[3] = buffer[3] * multiplier;
    buffer[4] = buffer[4] * multiplier;
    buffer[5] = buffer[5] * multiplier;
    buffer[6] = buffer[6] * multiplier;
    buffer[7] = buffer[7] * multiplier;
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyScalarMultiplier2B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)multiplicand)[offset4Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    buffer[2] = buffer[2] * multiplier;
    buffer[3] = buffer[3] * multiplier;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void multiplyScalarMultiplierHalf(half *multiplicand, half *dest, half multiplier, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 buffer[4];
    half2 multiplierHalf2;
    multiplierHalf2.x = multiplier;
    multiplierHalf2.y = multiplier;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)multiplicand)[offset8Elements];
    buffer[0] = __hmul2(buffer[0], multiplierHalf2);
    buffer[1] = __hmul2(buffer[1], multiplierHalf2);
    buffer[2] = __hmul2(buffer[2], multiplierHalf2);
    buffer[3] = __hmul2(buffer[3], multiplierHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)buffer)[0] = ((float4 *)multiplicand)[offset8Elements];
    buffer[0] = __hmul2(buffer[0], multiplierHalf2);
    buffer[1] = __hmul2(buffer[1], multiplierHalf2);
    buffer[2] = __hmul2(buffer[2], multiplierHalf2);
    buffer[3] = __hmul2(buffer[3], multiplierHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyScalarMultiplier4B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)multiplicand)[offset2Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyElementwise1B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE *multiplier, uint64_t numElements) {
    DATA_TYPE multiplicandBuffer[8];
    DATA_TYPE multiplierBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)multiplicandBuffer)[0] = ((float2 *)multiplicand)[offset8Elements];
    ((float2 *)multiplierBuffer)[0] = ((float2 *)multiplier)[offset8Elements];
    multiplicandBuffer[0] = multiplicandBuffer[0] * multiplierBuffer[0];
    multiplicandBuffer[1] = multiplicandBuffer[1] * multiplierBuffer[1];
    multiplicandBuffer[2] = multiplicandBuffer[2] * multiplierBuffer[2];
    multiplicandBuffer[3] = multiplicandBuffer[3] * multiplierBuffer[3];
    multiplicandBuffer[4] = multiplicandBuffer[4] * multiplierBuffer[4];
    multiplicandBuffer[5] = multiplicandBuffer[5] * multiplierBuffer[5];
    multiplicandBuffer[6] = multiplicandBuffer[6] * multiplierBuffer[6];
    multiplicandBuffer[7] = multiplicandBuffer[7] * multiplierBuffer[7];
    ((float2 *)dest)[offset8Elements] = ((float2 *)multiplicandBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyElementwise2B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE *multiplier, uint64_t numElements) {
    DATA_TYPE multiplicandBuffer[4];
    DATA_TYPE multiplierBuffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)multiplicandBuffer)[0] = ((float2 *)multiplicand)[offset4Elements];
    ((float2 *)multiplierBuffer)[0] = ((float2 *)multiplier)[offset4Elements];
    multiplicandBuffer[0] = multiplicandBuffer[0] * multiplierBuffer[0];
    multiplicandBuffer[1] = multiplicandBuffer[1] * multiplierBuffer[1];
    multiplicandBuffer[2] = multiplicandBuffer[2] * multiplierBuffer[2];
    multiplicandBuffer[3] = multiplicandBuffer[3] * multiplierBuffer[3];
    ((float2 *)dest)[offset4Elements] = ((float2 *)multiplicandBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void multiplyElementwiseHalf(half *multiplicand, half *dest, half *multiplier, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 multiplicandBuffer[4];
    half2 multiplierBuffer[4];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)multiplicandBuffer)[0] = ((float4 *)multiplicand)[offset8Elements];
    ((float4 *)multiplierBuffer)[0] = ((float4 *)multiplier)[offset8Elements];
    multiplicandBuffer[0] = __hmul2(multiplicandBuffer[0], multiplierBuffer[0]);
    multiplicandBuffer[1] = __hmul2(multiplicandBuffer[1], multiplierBuffer[1]);
    multiplicandBuffer[2] = __hmul2(multiplicandBuffer[2], multiplierBuffer[2]);
    multiplicandBuffer[3] = __hmul2(multiplicandBuffer[3], multiplierBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)multiplicandBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)multiplicandBuffer)[0] = ((float4 *)multiplicand)[offset8Elements];
    ((float4 *)multiplierBuffer)[0] = ((float4 *)multiplier)[offset8Elements];
    multiplicandBuffer[0] = __hmul2(multiplicandBuffer[0], multiplierBuffer[0]);
    multiplicandBuffer[1] = __hmul2(multiplicandBuffer[1], multiplierBuffer[1]);
    multiplicandBuffer[2] = __hmul2(multiplicandBuffer[2], multiplierBuffer[2]);
    multiplicandBuffer[3] = __hmul2(multiplicandBuffer[3], multiplierBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)multiplicandBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyElementwise4B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE *multiplier, uint64_t numElements) {
    DATA_TYPE multiplicandBuffer[2];
    DATA_TYPE multiplierBuffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)multiplicandBuffer)[0] = ((float2 *)multiplicand)[offset2Elements];
    ((float2 *)multiplierBuffer)[0] = ((float2 *)multiplier)[offset2Elements];
    multiplicandBuffer[0] = multiplicandBuffer[0] * multiplierBuffer[0];
    multiplicandBuffer[1] = multiplicandBuffer[1] * multiplierBuffer[1];
    ((float2 *)dest)[offset2Elements] = ((float2 *)multiplicandBuffer)[0];
}


void Tensor::multiply(Tensor multiplicand, double multiplier, Stream stream) {
    assert(multiplicand.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(multiplicand.getDataType() == getDataType());

    TensorDescriptor::DataType dataType = multiplicand.getDataType();
    uint64_t numElements = multiplicand.getTotalNumElements();
    void *multiplicandMem = multiplicand.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        multiplyScalarMultiplierHalf<<<gridSize, blockSize, 0, stream>>>((half *)multiplicandMem, (half *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyScalarMultiplier4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)multiplicandMem, (float *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)multiplicandMem, (uint8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        multiplyScalarMultiplier2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)multiplicandMem, (uint16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyScalarMultiplier4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)multiplicandMem, (uint32_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)multiplicandMem, (int8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        multiplyScalarMultiplier2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)multiplicandMem, (int16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyScalarMultiplier4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)multiplicandMem, (int32_t *)destMem, multiplier, numElements);
    } else {
        assert(false);
    }
}

void Tensor::multiply(double multiplicand, Tensor multiplier, Stream stream) { multiply(multiplier, multiplicand, stream); }

void Tensor::multiply(Tensor multiplicand, Tensor multiplier, Stream stream) {
    assert(multiplicand.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(multiplicand.getDataType() == multiplier.getDataType());
    assert(multiplicand.getDataType() == getDataType());
    assert(multiplicand.getTotalNumElements() == multiplier.getTotalNumElements());
    assert(multiplicand.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType dataType = multiplicand.getDataType();
    uint64_t numElements = multiplicand.getTotalNumElements();
    void *multiplicandMem = multiplicand.getMemPtr();
    void *multiplierMem = multiplier.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        multiplyElementwiseHalf<<<gridSize, blockSize, 0, stream>>>(
            (half *)multiplicandMem, (half *)destMem, (half *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyElementwise4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)multiplicandMem, (float *)destMem, (float *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)multiplicandMem, (uint8_t *)destMem, (uint8_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        multiplyElementwise2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)multiplicandMem, (uint16_t *)destMem, (uint16_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)multiplicandMem, (uint32_t *)destMem, (uint32_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)multiplicandMem, (int8_t *)destMem, (int8_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        multiplyElementwise2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)multiplicandMem, (int16_t *)destMem, (int16_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)multiplicandMem, (int32_t *)destMem, (int32_t *)multiplierMem, numElements);
    } else {
        assert(false);
    }
}
*/