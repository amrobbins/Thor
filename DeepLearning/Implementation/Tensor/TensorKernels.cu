#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyScalar1B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
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
__global__ void multiplyScalar2B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
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
__global__ void multiplyScalarHalf(half *multiplicand, half *dest, half multiplier, uint64_t numElements) {
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
__global__ void multiplyScalar4B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
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
__global__ void addScalar1B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)augend)[offset8Elements];
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
__global__ void addScalar2B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)augend)[offset4Elements];
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    buffer[2] = buffer[2] + addend;
    buffer[3] = buffer[3] + addend;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
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
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addScalar4B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE addend, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)augend)[offset2Elements];
    buffer[0] = buffer[0] + addend;
    buffer[1] = buffer[1] + addend;
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarMinuend1B(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    buffer[0] = minuend - buffer[0];
    buffer[1] = minuend - buffer[1];
    buffer[2] = minuend - buffer[2];
    buffer[3] = minuend - buffer[3];
    buffer[4] = minuend - buffer[4];
    buffer[5] = minuend - buffer[5];
    buffer[6] = minuend - buffer[6];
    buffer[7] = minuend - buffer[7];
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarMinuend2B(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)subtrahend)[offset4Elements];
    buffer[0] = minuend - buffer[0];
    buffer[1] = minuend - buffer[1];
    buffer[2] = minuend - buffer[2];
    buffer[3] = minuend - buffer[3];
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void subtractScalarMinuendHalf(half *subtrahend, half *dest, half minuend, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 buffer[4];
    half2 minuendHalf2;
    minuendHalf2.x = minuend;
    minuendHalf2.y = minuend;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)subtrahend)[offset8Elements];
    buffer[0] = __hsub2(minuendHalf2, buffer[0]);
    buffer[1] = __hsub2(minuendHalf2, buffer[1]);
    buffer[2] = __hsub2(minuendHalf2, buffer[2]);
    buffer[3] = __hsub2(minuendHalf2, buffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)buffer)[0] = ((float4 *)subtrahend)[offset8Elements];
    buffer[0] = __hsub2(minuendHalf2, buffer[0]);
    buffer[1] = __hsub2(minuendHalf2, buffer[1]);
    buffer[2] = __hsub2(minuendHalf2, buffer[2]);
    buffer[3] = __hsub2(minuendHalf2, buffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarMinuend4B(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)subtrahend)[offset2Elements];
    buffer[0] = minuend - buffer[0];
    buffer[1] = minuend - buffer[1];
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
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
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarDenominator2B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE denominator, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)numerator)[offset4Elements];
    buffer[0] = buffer[0] / denominator;
    buffer[1] = buffer[1] / denominator;
    buffer[2] = buffer[2] / denominator;
    buffer[3] = buffer[3] / denominator;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
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
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarDenominator4B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE denominator, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)numerator)[offset2Elements];
    buffer[0] = buffer[0] / denominator;
    buffer[1] = buffer[1] / denominator;
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
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
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarNumerator2B(DATA_TYPE *denominator, DATA_TYPE *dest, DATA_TYPE numerator, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)denominator)[offset4Elements];
    buffer[0] = numerator / buffer[0];
    buffer[1] = numerator / buffer[1];
    buffer[2] = numerator / buffer[2];
    buffer[3] = numerator / buffer[3];
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
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
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideScalarNumerator4B(DATA_TYPE *denominator, DATA_TYPE *dest, DATA_TYPE numerator, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)denominator)[offset2Elements];
    buffer[0] = numerator / buffer[0];
    buffer[1] = numerator / buffer[1];
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
}

void Tensor::add(Tensor augend, double addend, Stream stream) {
    assert(augend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    TensorDescriptor::DataType dataType = augend.getDataType();
    uint64_t numElements = augend.getTotalNumElements();
    void *augendMem = augend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addScalarHalf<<<gridSize, blockSize, 0, stream>>>((half *)augendMem, (half *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        addScalar4B<float><<<gridSize, blockSize, 0, stream>>>((float *)augendMem, (float *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)augendMem, (uint8_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        addScalar2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)augendMem, (uint16_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        addScalar4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)augendMem, (uint32_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)augendMem, (int8_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        addScalar2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)augendMem, (int16_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        addScalar4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)augendMem, (int32_t *)destMem, addend, numElements);
    } else {
        assert(false);
    }
}

void Tensor::subtract(double minuend, Tensor subtrahend, Stream stream) {
    assert(subtrahend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    TensorDescriptor::DataType dataType = subtrahend.getDataType();
    uint64_t numElements = subtrahend.getTotalNumElements();
    void *subtrahendMem = subtrahend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        subtractScalarMinuendHalf<<<gridSize, blockSize, 0, stream>>>((half *)subtrahendMem, (half *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractScalarMinuend4B<float><<<gridSize, blockSize, 0, stream>>>((float *)subtrahendMem, (float *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)subtrahendMem, (uint8_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        subtractScalarMinuend2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)subtrahendMem, (uint16_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractScalarMinuend4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)subtrahendMem, (uint32_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)subtrahendMem, (int8_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        subtractScalarMinuend2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)subtrahendMem, (int16_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractScalarMinuend4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)subtrahendMem, (int32_t *)destMem, minuend, numElements);
    } else {
        assert(false);
    }
}

void Tensor::subtract(Tensor minuend, double subtrahend, Stream stream) {
    assert(minuend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    add(minuend, -subtrahend, stream);
}

void Tensor::multiply(Tensor multiplicand, double multiplier, Stream stream) {
    assert(multiplicand.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    TensorDescriptor::DataType dataType = multiplicand.getDataType();
    uint64_t numElements = multiplicand.getTotalNumElements();
    void *multiplicandMem = multiplicand.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        multiplyScalarHalf<<<gridSize, blockSize, 0, stream>>>((half *)multiplicandMem, (half *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyScalar4B<float><<<gridSize, blockSize, 0, stream>>>((float *)multiplicandMem, (float *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalar1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)multiplicandMem, (uint8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        multiplyScalar2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)multiplicandMem, (uint16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyScalar4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)multiplicandMem, (uint32_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalar1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)multiplicandMem, (int8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        multiplyScalar2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)multiplicandMem, (int16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        multiplyScalar4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)multiplicandMem, (int32_t *)destMem, multiplier, numElements);
    } else {
        assert(false);
    }
}

void Tensor::divide(Tensor numerator, double denominator, Stream stream) {
    assert(numerator.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    TensorDescriptor::DataType dataType = numerator.getDataType();
    uint64_t numElements = numerator.getTotalNumElements();
    void *numeratorMem = numerator.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        divideScalarDenominatorHalf<<<gridSize, blockSize, 0, stream>>>((half *)numeratorMem, (half *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        divideScalarDenominator4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)numeratorMem, (float *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)numeratorMem, (uint8_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        divideScalarDenominator2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)numeratorMem, (uint16_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        divideScalarDenominator4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)numeratorMem, (uint32_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)numeratorMem, (int8_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        divideScalarDenominator2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)numeratorMem, (int16_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        divideScalarDenominator4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)numeratorMem, (int32_t *)destMem, denominator, numElements);
    } else {
        assert(false);
    }
}

void Tensor::divide(double numerator, Tensor denominator, Stream stream) {
    assert(denominator.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    TensorDescriptor::DataType dataType = denominator.getDataType();
    uint64_t numElements = denominator.getTotalNumElements();
    void *denominatorMem = denominator.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        divideScalarNumeratorHalf<<<gridSize, blockSize, 0, stream>>>((half *)denominatorMem, (half *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        divideScalarNumerator4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)denominatorMem, (float *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)denominatorMem, (uint8_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        divideScalarNumerator2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)denominatorMem, (uint16_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        divideScalarNumerator4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)denominatorMem, (uint32_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)denominatorMem, (int8_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        divideScalarNumerator2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)denominatorMem, (int16_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        divideScalarNumerator4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)denominatorMem, (int32_t *)destMem, numerator, numElements);
    } else {
        assert(false);
    }
}