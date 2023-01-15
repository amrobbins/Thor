#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void scalarMultiply1B(DATA_TYPE *source, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)source)[offset8Elements];
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
__global__ void scalarMultiply2B(DATA_TYPE *source, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)source)[offset4Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    buffer[2] = buffer[2] * multiplier;
    buffer[3] = buffer[3] * multiplier;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void scalarMultiplyHalf(half *source, half *dest, half multiplier, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    /* FIXME: Dont know why I am getting: error: identifier "__hmul2" is undefined
    half2 buffer[2];
    half2 multiplierHalf2;
    multiplierHalf2.x = multiplier;
    multiplierHalf2.y = multiplier;
    buffer[0] = __hmul2(buffer[0], multiplierHalf2);
    buffer[1] = __hmul2(buffer[1], multiplierHalf2);
     */

    half buffer[4];
    ((float2 *)buffer)[0] = ((float2 *)source)[offset4Elements];
    buffer[0] = (float)buffer[0] * (float)multiplier;
    buffer[1] = (float)buffer[1] * (float)multiplier;
    buffer[2] = (float)buffer[2] * (float)multiplier;
    buffer[3] = (float)buffer[3] * (float)multiplier;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void scalarMultiply4B(DATA_TYPE *source, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)source)[offset2Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void scalarAdd1B(DATA_TYPE *source, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)source)[offset8Elements];
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    buffer[2] = buffer[2] + addend;
    buffer[3] = buffer[3] + addend;
    buffer[4] = buffer[4] + addend;
    buffer[5] = buffer[5] + addend;
    buffer[6] = buffer[6] + addend;
    buffer[7] = buffer[7] + addend;
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void scalarAdd2B(DATA_TYPE *source, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)source)[offset4Elements];
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    buffer[2] = buffer[2] + addend;
    buffer[3] = buffer[3] + addend;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void scalarAddHalf(half *source, half *dest, half addend, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    /* FIXME: Dont know why im getting this error: error: identifier "__hadd2" is undefined
    half2 buffer[2];
    half2 addendHalf2;
    addendHalf2.x = addend;
    addendHalf2.y = addend;

    ((float2*)buffer)[0] = ((float2*)source)[offset4Elements];
    buffer[0] = __hadd2(buffer[0], addendHalf2);
    buffer[1] = __hadd2(buffer[1], addendHalf2);
    ((float2*)dest)[offset4Elements] = ((float2*)buffer)[0];
     */

    half buffer[4];
    ((float2 *)buffer)[0] = ((float2 *)source)[offset4Elements];
    buffer[0] = (float)buffer[0] + (float)addend;
    buffer[1] = (float)buffer[1] + (float)addend;
    buffer[2] = (float)buffer[2] + (float)addend;
    buffer[3] = (float)buffer[3] + (float)addend;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void scalarAdd4B(DATA_TYPE *source, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)source)[offset2Elements];
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
}

void Tensor::add(Tensor source, double addend, Stream stream) {
    assert(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    TensorDescriptor::DataType dataType = source.getDataType();
    uint64_t numElements = source.getTotalNumElements();
    void *sourceMem = source.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 1023) / 1024);
        scalarAddHalf<<<gridSize, blockSize, 0, stream>>>((half *)sourceMem, (half *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        scalarAdd4B<float><<<gridSize, blockSize, 0, stream>>>((float *)sourceMem, (float *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        scalarAdd1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)sourceMem, (uint8_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        scalarAdd2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)sourceMem, (uint16_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        scalarAdd4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)sourceMem, (uint32_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        scalarAdd1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)sourceMem, (int8_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        scalarAdd2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)sourceMem, (int16_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        scalarAdd4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)sourceMem, (int32_t *)destMem, addend, numElements);
    } else {
        assert(false);
    }
}

void Tensor::subtract(Tensor source, double subtrahend, Stream stream) {
    assert(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    add(source, -subtrahend, stream);
}

void Tensor::multiply(Tensor source, double multiplier, Stream stream) {
    assert(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    TensorDescriptor::DataType dataType = source.getDataType();
    uint64_t numElements = source.getTotalNumElements();
    void *sourceMem = source.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 1023) / 1024);
        scalarMultiplyHalf<<<gridSize, blockSize, 0, stream>>>((half *)sourceMem, (half *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        scalarMultiply4B<float><<<gridSize, blockSize, 0, stream>>>((float *)sourceMem, (float *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        scalarMultiply1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)sourceMem, (uint8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        scalarMultiply2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)sourceMem, (uint16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        scalarMultiply4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)sourceMem, (uint32_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        scalarMultiply1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)sourceMem, (int8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        scalarMultiply2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)sourceMem, (int16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        scalarMultiply4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)sourceMem, (int32_t *)destMem, multiplier, numElements);
    } else {
        assert(false);
    }
}

void Tensor::divide(Tensor source, double divisor, Stream stream) {
    assert(source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    assert(divisor != 0);
    multiply(source, 1.0 / divisor, stream);
}

void Tensor::add(double addend, Stream stream) { add(*this, addend, stream); }

void Tensor::subtract(double subtrahend, Stream stream) { subtract(*this, subtrahend, stream); }

void Tensor::multiply(double multiplier, Stream stream) { multiply(*this, multiplier, stream); }

void Tensor::divide(double divisor, Stream stream) { divide(*this, divisor, stream); }
