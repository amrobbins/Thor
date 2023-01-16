#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;

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
__global__ void addElementwise1B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[8];
    DATA_TYPE addendBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset8Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset8Elements];
    augendBuffer[0] = augendBuffer[0] + addendBuffer[0];
    augendBuffer[1] = augendBuffer[1] + addendBuffer[1];
    augendBuffer[2] = augendBuffer[2] + addendBuffer[2];
    augendBuffer[3] = augendBuffer[3] + addendBuffer[3];
    augendBuffer[4] = augendBuffer[4] + addendBuffer[4];
    augendBuffer[5] = augendBuffer[5] + addendBuffer[5];
    augendBuffer[6] = augendBuffer[6] + addendBuffer[6];
    augendBuffer[7] = augendBuffer[7] + addendBuffer[7];
    ((float2 *)dest)[offset8Elements] = ((float2 *)augendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise2B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[4];
    DATA_TYPE addendBuffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset4Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset4Elements];
    augendBuffer[0] = augendBuffer[0] + addendBuffer[0];
    augendBuffer[1] = augendBuffer[1] + addendBuffer[1];
    augendBuffer[2] = augendBuffer[2] + addendBuffer[2];
    augendBuffer[3] = augendBuffer[3] + addendBuffer[3];
    ((float2 *)dest)[offset4Elements] = ((float2 *)augendBuffer)[0];
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
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void addElementwise4B(DATA_TYPE *augend, DATA_TYPE *dest, DATA_TYPE *addend, uint64_t numElements) {
    DATA_TYPE augendBuffer[2];
    DATA_TYPE addendBuffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)augendBuffer)[0] = ((float2 *)augend)[offset2Elements];
    ((float2 *)addendBuffer)[0] = ((float2 *)addend)[offset2Elements];
    augendBuffer[0] = augendBuffer[0] + addendBuffer[0];
    augendBuffer[1] = augendBuffer[1] + addendBuffer[1];
    ((float2 *)dest)[offset2Elements] = ((float2 *)augendBuffer)[0];
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
__global__ void subtractScalarSubtrahend1B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)buffer)[0] = ((float2 *)minuend)[offset8Elements];
    buffer[0] = buffer[0] - subtrahend;
    buffer[1] = buffer[1] - subtrahend;
    buffer[2] = buffer[2] - subtrahend;
    buffer[3] = buffer[3] - subtrahend;
    buffer[4] = buffer[4] - subtrahend;
    buffer[5] = buffer[5] - subtrahend;
    buffer[6] = buffer[6] - subtrahend;
    buffer[7] = buffer[7] - subtrahend;
    ((float2 *)dest)[offset8Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarSubtrahend2B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)buffer)[0] = ((float2 *)minuend)[offset4Elements];
    buffer[0] = buffer[0] - subtrahend;
    buffer[1] = buffer[1] - subtrahend;
    buffer[2] = buffer[2] - subtrahend;
    buffer[3] = buffer[3] - subtrahend;
    ((float2 *)dest)[offset4Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void subtractScalarSubtrahendHalf(half *minuend, half *dest, half subtrahend, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 buffer[4];
    half2 subtrahendHalf2;
    subtrahendHalf2.x = subtrahend;
    subtrahendHalf2.y = subtrahend;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)minuend)[offset8Elements];
    buffer[0] = __hsub2(buffer[0], subtrahendHalf2);
    buffer[1] = __hsub2(buffer[1], subtrahendHalf2);
    buffer[2] = __hsub2(buffer[2], subtrahendHalf2);
    buffer[3] = __hsub2(buffer[3], subtrahendHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)buffer)[0] = ((float4 *)minuend)[offset8Elements];
    buffer[0] = __hsub2(buffer[0], subtrahendHalf2);
    buffer[1] = __hsub2(buffer[1], subtrahendHalf2);
    buffer[2] = __hsub2(buffer[2], subtrahendHalf2);
    buffer[3] = __hsub2(buffer[3], subtrahendHalf2);
    ((float4 *)dest)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarSubtrahend4B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE buffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)buffer)[0] = ((float2 *)minuend)[offset2Elements];
    buffer[0] = buffer[0] - subtrahend;
    buffer[1] = buffer[1] - subtrahend;
    ((float2 *)dest)[offset2Elements] = ((float2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise1B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset8Elements];
    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    minuendBuffer[0] = minuendBuffer[0] - subtrahendBuffer[0];
    minuendBuffer[1] = minuendBuffer[1] - subtrahendBuffer[1];
    minuendBuffer[2] = minuendBuffer[2] - subtrahendBuffer[2];
    minuendBuffer[3] = minuendBuffer[3] - subtrahendBuffer[3];
    minuendBuffer[4] = minuendBuffer[4] - subtrahendBuffer[4];
    minuendBuffer[5] = minuendBuffer[5] - subtrahendBuffer[5];
    minuendBuffer[6] = minuendBuffer[6] - subtrahendBuffer[6];
    minuendBuffer[7] = minuendBuffer[7] - subtrahendBuffer[7];
    ((float2 *)dest)[offset8Elements] = ((float2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise2B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[4];
    DATA_TYPE subtrahendBuffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset4Elements];
    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset4Elements];
    minuendBuffer[0] = minuendBuffer[0] - subtrahendBuffer[0];
    minuendBuffer[1] = minuendBuffer[1] - subtrahendBuffer[1];
    minuendBuffer[2] = minuendBuffer[2] - subtrahendBuffer[2];
    minuendBuffer[3] = minuendBuffer[3] - subtrahendBuffer[3];
    ((float2 *)dest)[offset4Elements] = ((float2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
__global__ void subtractElementwiseHalf(half *minuend, half *dest, half *subtrahend, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 minuendBuffer[4];
    half2 subtrahendBuffer[4];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)minuendBuffer)[0] = ((float4 *)minuend)[offset8Elements];
    ((float4 *)subtrahendBuffer)[0] = ((float4 *)subtrahend)[offset8Elements];
    minuendBuffer[0] = __hsub2(minuendBuffer[0], subtrahendBuffer[0]);
    minuendBuffer[1] = __hsub2(minuendBuffer[1], subtrahendBuffer[1]);
    minuendBuffer[2] = __hsub2(minuendBuffer[2], subtrahendBuffer[2]);
    minuendBuffer[3] = __hsub2(minuendBuffer[3], subtrahendBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)minuendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float4 *)minuendBuffer)[0] = ((float4 *)minuend)[offset8Elements];
    ((float4 *)subtrahendBuffer)[0] = ((float4 *)subtrahend)[offset8Elements];
    minuendBuffer[0] = __hsub2(minuendBuffer[0], subtrahendBuffer[0]);
    minuendBuffer[1] = __hsub2(minuendBuffer[1], subtrahendBuffer[1]);
    minuendBuffer[2] = __hsub2(minuendBuffer[2], subtrahendBuffer[2]);
    minuendBuffer[3] = __hsub2(minuendBuffer[3], subtrahendBuffer[3]);
    ((float4 *)dest)[offset8Elements] = ((float4 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise4B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[2];
    DATA_TYPE subtrahendBuffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset2Elements];
    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset2Elements];
    minuendBuffer[0] = minuendBuffer[0] - subtrahendBuffer[0];
    minuendBuffer[1] = minuendBuffer[1] - subtrahendBuffer[1];
    ((float2 *)dest)[offset2Elements] = ((float2 *)minuendBuffer)[0];
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
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideElementwise2B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE *denominator, uint64_t numElements) {
    DATA_TYPE numeratorBuffer[4];
    DATA_TYPE denominatorBuffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((float2 *)numeratorBuffer)[0] = ((float2 *)numerator)[offset4Elements];
    ((float2 *)denominatorBuffer)[0] = ((float2 *)denominator)[offset4Elements];
    numeratorBuffer[0] = numeratorBuffer[0] / denominatorBuffer[0];
    numeratorBuffer[1] = numeratorBuffer[1] / denominatorBuffer[1];
    numeratorBuffer[2] = numeratorBuffer[2] / denominatorBuffer[2];
    numeratorBuffer[3] = numeratorBuffer[3] / denominatorBuffer[3];
    ((float2 *)dest)[offset4Elements] = ((float2 *)numeratorBuffer)[0];
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
// each thread reads 2 elements : 512 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void divideElementwise4B(DATA_TYPE *numerator, DATA_TYPE *dest, DATA_TYPE *denominator, uint64_t numElements) {
    DATA_TYPE numeratorBuffer[2];
    DATA_TYPE denominatorBuffer[2];

    uint64_t offset = blockIdx.x * 512 + threadIdx.x * 2;
    if (offset >= numElements)
        return;
    uint64_t offset2Elements = offset >> 1;

    ((float2 *)numeratorBuffer)[0] = ((float2 *)numerator)[offset2Elements];
    ((float2 *)denominatorBuffer)[0] = ((float2 *)denominator)[offset2Elements];
    numeratorBuffer[0] = numeratorBuffer[0] / denominatorBuffer[0];
    numeratorBuffer[1] = numeratorBuffer[1] / denominatorBuffer[1];
    ((float2 *)dest)[offset2Elements] = ((float2 *)numeratorBuffer)[0];
}

void Tensor::add(Tensor augend, double addend, Stream stream) {
    assert(augend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(augend.getDataType() == getDataType());

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

void Tensor::add(double augend, Tensor addend, Stream stream) { add(addend, augend, stream); }

void Tensor::add(Tensor augend, Tensor addend, Stream stream) {
    assert(augend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(augend.getDataType() == addend.getDataType());
    assert(augend.getDataType() == getDataType());
    assert(augend.getTotalNumElements() == addend.getTotalNumElements());
    assert(augend.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType dataType = augend.getDataType();
    uint64_t numElements = augend.getTotalNumElements();
    void *augendMem = augend.getMemPtr();
    void *addendMem = addend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwiseHalf<<<gridSize, blockSize, 0, stream>>>((half *)augendMem, (half *)destMem, (half *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        addElementwise4B<float><<<gridSize, blockSize, 0, stream>>>((float *)augendMem, (float *)destMem, (float *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)augendMem, (uint8_t *)destMem, (uint8_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        addElementwise2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)augendMem, (uint16_t *)destMem, (uint16_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        addElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)augendMem, (uint32_t *)destMem, (uint32_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)augendMem, (int8_t *)destMem, (int8_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        addElementwise2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)augendMem, (int16_t *)destMem, (int16_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        addElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)augendMem, (int32_t *)destMem, (int32_t *)addendMem, numElements);
    } else {
        assert(false);
    }
}

void Tensor::subtract(double minuend, Tensor subtrahend, Stream stream) {
    assert(subtrahend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(subtrahend.getDataType() == getDataType());

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
    assert(minuend.getDataType() == getDataType());

    TensorDescriptor::DataType dataType = minuend.getDataType();
    uint64_t numElements = minuend.getTotalNumElements();
    void *minuendMem = minuend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        subtractScalarSubtrahendHalf<<<gridSize, blockSize, 0, stream>>>((half *)minuendMem, (half *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractScalarSubtrahend4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)minuendMem, (float *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)minuendMem, (uint8_t *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        subtractScalarSubtrahend2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)minuendMem, (uint16_t *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractScalarSubtrahend4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)minuendMem, (uint32_t *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)minuendMem, (int8_t *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        subtractScalarSubtrahend2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)minuendMem, (int16_t *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractScalarSubtrahend4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)minuendMem, (int32_t *)destMem, subtrahend, numElements);
    } else {
        assert(false);
    }
}

void Tensor::subtract(Tensor minuend, Tensor subtrahend, Stream stream) {
    assert(minuend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(minuend.getDataType() == subtrahend.getDataType());
    assert(minuend.getDataType() == getDataType());
    assert(minuend.getTotalNumElements() == subtrahend.getTotalNumElements());
    assert(minuend.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType dataType = minuend.getDataType();
    uint64_t numElements = minuend.getTotalNumElements();
    void *minuendMem = minuend.getMemPtr();
    void *subtrahendMem = subtrahend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        subtractElementwiseHalf<<<gridSize, blockSize, 0, stream>>>(
            (half *)minuendMem, (half *)destMem, (half *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractElementwise4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)minuendMem, (float *)destMem, (float *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractElementwise1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)minuendMem, (uint8_t *)destMem, (uint8_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        subtractElementwise2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)minuendMem, (uint16_t *)destMem, (uint16_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)minuendMem, (uint32_t *)destMem, (uint32_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractElementwise1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)minuendMem, (int8_t *)destMem, (int8_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        subtractElementwise2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)minuendMem, (int16_t *)destMem, (int16_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        subtractElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)minuendMem, (int32_t *)destMem, (int32_t *)subtrahendMem, numElements);
    } else {
        assert(false);
    }
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

void Tensor::divide(Tensor numerator, double denominator, Stream stream) {
    assert(numerator.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(numerator.getDataType() == getDataType());

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
    assert(denominator.getDataType() == getDataType());

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

void Tensor::divide(Tensor numerator, Tensor denominator, Stream stream) {
    assert(numerator.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(numerator.getDataType() == denominator.getDataType());
    assert(numerator.getDataType() == getDataType());
    assert(numerator.getTotalNumElements() == denominator.getTotalNumElements());
    assert(numerator.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType dataType = numerator.getDataType();
    uint64_t numElements = numerator.getTotalNumElements();
    void *numeratorMem = numerator.getMemPtr();
    void *denominatorMem = denominator.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        divideElementwiseHalf<<<gridSize, blockSize, 0, stream>>>(
            (half *)numeratorMem, (half *)destMem, (half *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 511) / 512);
        divideElementwise4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)numeratorMem, (float *)destMem, (float *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)numeratorMem, (uint8_t *)destMem, (uint8_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        divideElementwise2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)numeratorMem, (uint16_t *)destMem, (uint16_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 511) / 512);
        divideElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)numeratorMem, (uint32_t *)destMem, (uint32_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)numeratorMem, (int8_t *)destMem, (int8_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 1023) / 1024);
        divideElementwise2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)numeratorMem, (int16_t *)destMem, (int16_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 511) / 512);
        divideElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)numeratorMem, (int32_t *)destMem, (int32_t *)denominatorMem, numElements);
    } else {
        assert(false);
    }
}