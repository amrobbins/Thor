#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;
using namespace std;

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
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyScalarMultiplier2B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)buffer)[0] = ((double2 *)multiplicand)[offset8Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    buffer[2] = buffer[2] * multiplier;
    buffer[3] = buffer[3] * multiplier;
    buffer[4] = buffer[4] * multiplier;
    buffer[5] = buffer[5] * multiplier;
    buffer[6] = buffer[6] * multiplier;
    buffer[7] = buffer[7] * multiplier;
    ((double2 *)dest)[offset8Elements] = ((double2 *)buffer)[0];
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
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyScalarMultiplier4B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE multiplier, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)buffer)[0] = ((double2 *)multiplicand)[offset4Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    buffer[2] = buffer[2] * multiplier;
    buffer[3] = buffer[3] * multiplier;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)buffer)[0] = ((double2 *)multiplicand)[offset4Elements];
    buffer[0] = buffer[0] * multiplier;
    buffer[1] = buffer[1] * multiplier;
    buffer[2] = buffer[2] * multiplier;
    buffer[3] = buffer[3] * multiplier;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];
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
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyElementwise2B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE *multiplier, uint64_t numElements) {
    DATA_TYPE multiplicandBuffer[8];
    DATA_TYPE multiplierBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)multiplicandBuffer)[0] = ((double2 *)multiplicand)[offset8Elements];
    ((double2 *)multiplierBuffer)[0] = ((double2 *)multiplier)[offset8Elements];
    multiplicandBuffer[0] = multiplicandBuffer[0] * multiplierBuffer[0];
    multiplicandBuffer[1] = multiplicandBuffer[1] * multiplierBuffer[1];
    multiplicandBuffer[2] = multiplicandBuffer[2] * multiplierBuffer[2];
    multiplicandBuffer[3] = multiplicandBuffer[3] * multiplierBuffer[3];
    multiplicandBuffer[4] = multiplicandBuffer[4] * multiplierBuffer[4];
    multiplicandBuffer[5] = multiplicandBuffer[5] * multiplierBuffer[5];
    multiplicandBuffer[6] = multiplicandBuffer[6] * multiplierBuffer[6];
    multiplicandBuffer[7] = multiplicandBuffer[7] * multiplierBuffer[7];
    ((double2 *)dest)[offset8Elements] = ((double2 *)multiplicandBuffer)[0];
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
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void multiplyElementwise4B(DATA_TYPE *multiplicand, DATA_TYPE *dest, DATA_TYPE *multiplier, uint64_t numElements) {
    DATA_TYPE multiplicandBuffer[4];
    DATA_TYPE multiplierBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)multiplicandBuffer)[0] = ((double2 *)multiplicand)[offset4Elements];
    ((double2 *)multiplierBuffer)[0] = ((double2 *)multiplier)[offset4Elements];
    multiplicandBuffer[0] = multiplicandBuffer[0] * multiplierBuffer[0];
    multiplicandBuffer[1] = multiplicandBuffer[1] * multiplierBuffer[1];
    multiplicandBuffer[2] = multiplicandBuffer[2] * multiplierBuffer[2];
    multiplicandBuffer[3] = multiplicandBuffer[3] * multiplierBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)multiplicandBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)multiplicandBuffer)[0] = ((double2 *)multiplicand)[offset4Elements];
    ((double2 *)multiplierBuffer)[0] = ((double2 *)multiplier)[offset4Elements];
    multiplicandBuffer[0] = multiplicandBuffer[0] * multiplierBuffer[0];
    multiplicandBuffer[1] = multiplicandBuffer[1] * multiplierBuffer[1];
    multiplicandBuffer[2] = multiplicandBuffer[2] * multiplierBuffer[2];
    multiplicandBuffer[3] = multiplicandBuffer[3] * multiplierBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)multiplicandBuffer)[0];
}

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
__global__ void subtractScalarMinuend1BUnsignedInt(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE subtrahendBuffer[8];
    DATA_TYPE minuendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    minuendBuffer[0] = minuend;
    minuendBuffer[1] = minuend;
    minuendBuffer[2] = minuend;
    minuendBuffer[3] = minuend;

    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    ((uint32_t *)subtrahendBuffer)[0] = __vsub4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)subtrahendBuffer)[1] = __vsub4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)subtrahendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarMinuend1BSignedInt(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE subtrahendBuffer[8];
    DATA_TYPE minuendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    minuendBuffer[0] = minuend;
    minuendBuffer[1] = minuend;
    minuendBuffer[2] = minuend;
    minuendBuffer[3] = minuend;

    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    ((uint32_t *)subtrahendBuffer)[0] = __vsubss4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)subtrahendBuffer)[1] = __vsubss4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)subtrahendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarMinuend2BUnsignedInt(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE subtrahendBuffer[8];
    DATA_TYPE minuendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    minuendBuffer[0] = minuend;
    minuendBuffer[1] = minuend;

    ((double2 *)subtrahendBuffer)[0] = ((double2 *)subtrahend)[offset8Elements];
    ((uint32_t *)subtrahendBuffer)[0] = __vsub2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)subtrahendBuffer)[1] = __vsub2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[1]);
    ((uint32_t *)subtrahendBuffer)[2] = __vsub2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[2]);
    ((uint32_t *)subtrahendBuffer)[3] = __vsub2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)subtrahendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarMinuend2BSignedInt(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE subtrahendBuffer[8];
    DATA_TYPE minuendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    minuendBuffer[0] = minuend;
    minuendBuffer[1] = minuend;

    ((double2 *)subtrahendBuffer)[0] = ((double2 *)subtrahend)[offset8Elements];
    ((uint32_t *)subtrahendBuffer)[0] = __vsubss2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)subtrahendBuffer)[1] = __vsubss2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[1]);
    ((uint32_t *)subtrahendBuffer)[2] = __vsubss2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[2]);
    ((uint32_t *)subtrahendBuffer)[3] = __vsubss2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)subtrahendBuffer)[0];
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
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarMinuend4B(DATA_TYPE *subtrahend, DATA_TYPE *dest, DATA_TYPE minuend, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)buffer)[0] = ((double2 *)subtrahend)[offset4Elements];
    buffer[0] = minuend - buffer[0];
    buffer[1] = minuend - buffer[1];
    buffer[2] = minuend - buffer[2];
    buffer[3] = minuend - buffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)buffer)[0] = ((double2 *)subtrahend)[offset4Elements];
    buffer[0] = minuend - buffer[0];
    buffer[1] = minuend - buffer[1];
    buffer[2] = minuend - buffer[2];
    buffer[3] = minuend - buffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarSubtrahend1BUnsignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    subtrahendBuffer[0] = subtrahend;
    subtrahendBuffer[1] = subtrahend;
    subtrahendBuffer[2] = subtrahend;
    subtrahendBuffer[3] = subtrahend;

    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsub4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsub4(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[0]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarSubtrahend1BSignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    subtrahendBuffer[0] = subtrahend;
    subtrahendBuffer[1] = subtrahend;
    subtrahendBuffer[2] = subtrahend;
    subtrahendBuffer[3] = subtrahend;

    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsubss4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsubss4(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[0]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarSubtrahend2BUnsignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    subtrahendBuffer[0] = subtrahend;
    subtrahendBuffer[1] = subtrahend;

    ((double2 *)minuendBuffer)[0] = ((double2 *)minuend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsub2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsub2(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[2] = __vsub2(((uint32_t *)minuendBuffer)[2], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[3] = __vsub2(((uint32_t *)minuendBuffer)[3], ((uint32_t *)subtrahendBuffer)[0]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarSubtrahend2BSignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[2];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    subtrahendBuffer[0] = subtrahend;
    subtrahendBuffer[1] = subtrahend;

    ((double2 *)minuendBuffer)[0] = ((double2 *)minuend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsubss2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsubss2(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[2] = __vsubss2(((uint32_t *)minuendBuffer)[2], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[3] = __vsubss2(((uint32_t *)minuendBuffer)[3], ((uint32_t *)subtrahendBuffer)[0]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)minuendBuffer)[0];
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
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractScalarSubtrahend4B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE subtrahend, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)buffer)[0] = ((double2 *)minuend)[offset4Elements];
    buffer[0] = buffer[0] - subtrahend;
    buffer[1] = buffer[1] - subtrahend;
    buffer[2] = buffer[2] - subtrahend;
    buffer[3] = buffer[3] - subtrahend;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)buffer)[0] = ((double2 *)minuend)[offset4Elements];
    buffer[0] = buffer[0] - subtrahend;
    buffer[1] = buffer[1] - subtrahend;
    buffer[2] = buffer[2] - subtrahend;
    buffer[3] = buffer[3] - subtrahend;
    ((double2 *)dest)[offset4Elements] = ((double2 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise1BUnsignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset8Elements];
    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsub4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsub4(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)minuendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset8Elements];
    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsub4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsub4(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise1BSignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[8];

    uint64_t offset = blockIdx.x * 4096 + 512 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset8Elements];
    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsubss4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsubss4(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)minuendBuffer)[0];

    offset += 256;
    if (offset >= numElements)
        return;
    offset8Elements = offset >> 3;
    ((float2 *)minuendBuffer)[0] = ((float2 *)minuend)[offset8Elements];
    ((float2 *)subtrahendBuffer)[0] = ((float2 *)subtrahend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsubss4(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsubss4(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[1]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise2BUnsignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)minuendBuffer)[0] = ((double2 *)minuend)[offset8Elements];
    ((double2 *)subtrahendBuffer)[0] = ((double2 *)subtrahend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsub2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsub2(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[1]);
    ((uint32_t *)minuendBuffer)[2] = __vsub2(((uint32_t *)minuendBuffer)[2], ((uint32_t *)subtrahendBuffer)[2]);
    ((uint32_t *)minuendBuffer)[3] = __vsub2(((uint32_t *)minuendBuffer)[3], ((uint32_t *)subtrahendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)minuendBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise2BSignedInt(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[8];
    DATA_TYPE subtrahendBuffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    ((double2 *)minuendBuffer)[0] = ((double2 *)minuend)[offset8Elements];
    ((double2 *)subtrahendBuffer)[0] = ((double2 *)subtrahend)[offset8Elements];
    ((uint32_t *)minuendBuffer)[0] = __vsubss2(((uint32_t *)minuendBuffer)[0], ((uint32_t *)subtrahendBuffer)[0]);
    ((uint32_t *)minuendBuffer)[1] = __vsubss2(((uint32_t *)minuendBuffer)[1], ((uint32_t *)subtrahendBuffer)[1]);
    ((uint32_t *)minuendBuffer)[2] = __vsubss2(((uint32_t *)minuendBuffer)[2], ((uint32_t *)subtrahendBuffer)[2]);
    ((uint32_t *)minuendBuffer)[3] = __vsubss2(((uint32_t *)minuendBuffer)[3], ((uint32_t *)subtrahendBuffer)[3]);
    ((double2 *)dest)[offset8Elements] = ((double2 *)minuendBuffer)[0];
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
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void subtractElementwise4B(DATA_TYPE *minuend, DATA_TYPE *dest, DATA_TYPE *subtrahend, uint64_t numElements) {
    DATA_TYPE minuendBuffer[4];
    DATA_TYPE subtrahendBuffer[4];

    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    ((double2 *)minuendBuffer)[0] = ((double2 *)minuend)[offset4Elements];
    ((double2 *)subtrahendBuffer)[0] = ((double2 *)subtrahend)[offset4Elements];
    minuendBuffer[0] = minuendBuffer[0] - subtrahendBuffer[0];
    minuendBuffer[1] = minuendBuffer[1] - subtrahendBuffer[1];
    minuendBuffer[2] = minuendBuffer[2] - subtrahendBuffer[2];
    minuendBuffer[3] = minuendBuffer[3] - subtrahendBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)minuendBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((double2 *)minuendBuffer)[0] = ((double2 *)minuend)[offset4Elements];
    ((double2 *)subtrahendBuffer)[0] = ((double2 *)subtrahend)[offset4Elements];
    minuendBuffer[0] = minuendBuffer[0] - subtrahendBuffer[0];
    minuendBuffer[1] = minuendBuffer[1] - subtrahendBuffer[1];
    minuendBuffer[2] = minuendBuffer[2] - subtrahendBuffer[2];
    minuendBuffer[3] = minuendBuffer[3] - subtrahendBuffer[3];
    ((double2 *)dest)[offset4Elements] = ((double2 *)minuendBuffer)[0];
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

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void multiplyAccumulateDest1B(DEST_DATA_TYPE *dest, float *a, float *b, float *c, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float aBuffer[8];
    float bBuffer[8];
    float cBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)aBuffer)[0] = ((double4 *)a)[offset8Elements];
    ((double4 *)bBuffer)[0] = ((double4 *)b)[offset8Elements];
    ((double4 *)cBuffer)[0] = ((double4 *)c)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)fmaf(aBuffer[0], bBuffer[0], cBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)fmaf(aBuffer[1], bBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)fmaf(aBuffer[2], bBuffer[2], cBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)fmaf(aBuffer[3], bBuffer[3], cBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)fmaf(aBuffer[4], bBuffer[4], cBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)fmaf(aBuffer[5], bBuffer[5], cBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)fmaf(aBuffer[6], bBuffer[6], cBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)fmaf(aBuffer[7], bBuffer[7], cBuffer[7]);
    ((float2 *)dest)[offset8Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void multiplyAccumulateDest2B(DEST_DATA_TYPE *dest, float *a, float *b, float *c, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float aBuffer[4];
    float bBuffer[4];
    float cBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)aBuffer)[0] = ((float4 *)a)[offset4Elements];
    ((float4 *)bBuffer)[0] = ((float4 *)b)[offset4Elements];
    ((float4 *)cBuffer)[0] = ((float4 *)c)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)fmaf(aBuffer[0], bBuffer[0], cBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)fmaf(aBuffer[1], bBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)fmaf(aBuffer[2], bBuffer[2], cBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)fmaf(aBuffer[3], bBuffer[3], cBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)aBuffer)[0] = ((float4 *)a)[offset4Elements];
    ((float4 *)bBuffer)[0] = ((float4 *)b)[offset4Elements];
    ((float4 *)cBuffer)[0] = ((float4 *)c)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)fmaf(aBuffer[0], bBuffer[0], cBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)fmaf(aBuffer[1], bBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)fmaf(aBuffer[2], bBuffer[2], cBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)fmaf(aBuffer[3], bBuffer[3], cBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void multiplyAccumulateDest4B(DEST_DATA_TYPE *dest, float *a, float *b, float *c, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float aBuffer[4];
    float bBuffer[4];
    float cBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)aBuffer)[0] = ((float4 *)a)[offset4Elements];
    ((float4 *)bBuffer)[0] = ((float4 *)b)[offset4Elements];
    ((float4 *)cBuffer)[0] = ((float4 *)c)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)fmaf(aBuffer[0], bBuffer[0], cBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)fmaf(aBuffer[1], bBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)fmaf(aBuffer[2], bBuffer[2], cBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)fmaf(aBuffer[3], bBuffer[3], cBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)aBuffer)[0] = ((float4 *)a)[offset4Elements];
    ((float4 *)bBuffer)[0] = ((float4 *)b)[offset4Elements];
    ((float4 *)cBuffer)[0] = ((float4 *)c)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)fmaf(aBuffer[0], bBuffer[0], cBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)fmaf(aBuffer[1], bBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)fmaf(aBuffer[2], bBuffer[2], cBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)fmaf(aBuffer[3], bBuffer[3], cBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void multiplyAccumulateDest1B(DEST_DATA_TYPE *dest, half *a, half *b, half *c, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 aBuffer[4];
    half2 bBuffer[4];
    half2 cBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)aBuffer)[0] = ((float4 *)a)[offset8Elements];
    ((float4 *)bBuffer)[0] = ((float4 *)b)[offset8Elements];
    ((float4 *)cBuffer)[0] = ((float4 *)c)[offset8Elements];
    aBuffer[0] = __hmul2(aBuffer[0], bBuffer[0]);
    aBuffer[0] = __hadd2(aBuffer[0], cBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)aBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)aBuffer[0].y;
    aBuffer[1] = __hmul2(aBuffer[1], bBuffer[1]);
    aBuffer[1] = __hadd2(aBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)aBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)aBuffer[1].y;
    aBuffer[2] = __hmul2(aBuffer[2], bBuffer[2]);
    aBuffer[2] = __hadd2(aBuffer[2], cBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)aBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)aBuffer[2].y;
    aBuffer[3] = __hmul2(aBuffer[3], bBuffer[3]);
    aBuffer[3] = __hadd2(aBuffer[3], cBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)aBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)aBuffer[3].y;
    ((float2 *)dest)[offset8Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void multiplyAccumulateDest2B(DEST_DATA_TYPE *dest, half *a, half *b, half *c, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 aBuffer[2];
    half2 bBuffer[2];
    half2 cBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)aBuffer)[0] = ((float2 *)a)[offset4Elements];
    ((float2 *)bBuffer)[0] = ((float2 *)b)[offset4Elements];
    ((float2 *)cBuffer)[0] = ((float2 *)c)[offset4Elements];
    aBuffer[0] = __hmul2(aBuffer[0], bBuffer[0]);
    aBuffer[0] = __hadd2(aBuffer[0], cBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)aBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)aBuffer[0].y;
    aBuffer[1] = __hmul2(aBuffer[1], bBuffer[1]);
    aBuffer[1] = __hadd2(aBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)aBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)aBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)aBuffer)[0] = ((float2 *)a)[offset4Elements];
    ((float2 *)bBuffer)[0] = ((float2 *)b)[offset4Elements];
    ((float2 *)cBuffer)[0] = ((float2 *)c)[offset4Elements];
    aBuffer[0] = __hmul2(aBuffer[0], bBuffer[0]);
    aBuffer[0] = __hadd2(aBuffer[0], cBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)aBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)aBuffer[0].y;
    aBuffer[1] = __hmul2(aBuffer[1], bBuffer[1]);
    aBuffer[1] = __hadd2(aBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)aBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)aBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void multiplyAccumulateDest4B(DEST_DATA_TYPE *dest, half *a, half *b, half *c, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 aBuffer[2];
    half2 bBuffer[2];
    half2 cBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)aBuffer)[0] = ((float2 *)a)[offset4Elements];
    ((float2 *)bBuffer)[0] = ((float2 *)b)[offset4Elements];
    ((float2 *)cBuffer)[0] = ((float2 *)c)[offset4Elements];
    aBuffer[0] = __hmul2(aBuffer[0], bBuffer[0]);
    aBuffer[0] = __hadd2(aBuffer[0], cBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)aBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)aBuffer[0].y;
    aBuffer[1] = __hmul2(aBuffer[1], bBuffer[1]);
    aBuffer[1] = __hadd2(aBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)aBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)aBuffer[1].y;

    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)aBuffer)[0] = ((float2 *)a)[offset4Elements];
    ((float2 *)bBuffer)[0] = ((float2 *)b)[offset4Elements];
    ((float2 *)cBuffer)[0] = ((float2 *)c)[offset4Elements];
    aBuffer[0] = __hmul2(aBuffer[0], bBuffer[0]);
    aBuffer[0] = __hadd2(aBuffer[0], cBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)aBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)aBuffer[0].y;
    aBuffer[1] = __hmul2(aBuffer[1], bBuffer[1]);
    aBuffer[1] = __hadd2(aBuffer[1], cBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)aBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)aBuffer[1].y;

    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

void Tensor::add(Tensor augend, double addend, Stream stream) {
    assert(augend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(augend.getDataType() == getDataType());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = augend.getDataType();
    uint64_t numElements = augend.getTotalNumElements();
    void *augendMem = augend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addScalarHalf<<<gridSize, blockSize, 0, stream>>>((half *)augendMem, (half *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar4B<float><<<gridSize, blockSize, 0, stream>>>((float *)augendMem, (float *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar1BUnsignedInt<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)augendMem, (uint8_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar2BUnsignedInt<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)augendMem, (uint16_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)augendMem, (uint32_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar1BSignedInt<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)augendMem, (int8_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar2BSignedInt<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)augendMem, (int16_t *)destMem, addend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
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

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

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
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<float><<<gridSize, blockSize, 0, stream>>>((float *)augendMem, (float *)destMem, (float *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise1BUnsignedInt<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)augendMem, (uint8_t *)destMem, (uint8_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise2BUnsignedInt<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)augendMem, (uint16_t *)destMem, (uint16_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)augendMem, (uint32_t *)destMem, (uint32_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise1BSignedInt<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)augendMem, (int8_t *)destMem, (int8_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise2BSignedInt<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)augendMem, (int16_t *)destMem, (int16_t *)addendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)augendMem, (int32_t *)destMem, (int32_t *)addendMem, numElements);
    } else {
        assert(false);
    }
}

void Tensor::add(Tensor augend, double addend, float alpha, Stream stream) {
    assert(augend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(augend.getDataType() == getDataType());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = augend.getDataType();
    uint64_t numElements = augend.getTotalNumElements();
    void *augendMem = augend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addScalarHalf<<<gridSize, blockSize, 0, stream>>>((half *)augendMem, (half *)destMem, addend, alpha, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar4B<float><<<gridSize, blockSize, 0, stream>>>((float *)augendMem, (float *)destMem, addend, alpha, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        assert(alpha >= 0.0f);
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar1BUnsignedInt<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)augendMem, (uint8_t *)destMem, addend, alpha, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        assert(alpha >= 0.0f);
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar2BUnsignedInt<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)augendMem, (uint16_t *)destMem, addend, alpha, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        assert(alpha >= 0.0f);
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)augendMem, (uint32_t *)destMem, addend, alpha, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar1BSignedInt<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)augendMem, (int8_t *)destMem, addend, alpha, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar2BSignedInt<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)augendMem, (int16_t *)destMem, addend, alpha, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addScalar4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)augendMem, (int32_t *)destMem, addend, alpha, numElements);
    } else {
        assert(false);
    }
}

void Tensor::add(double augend, Tensor addend, float beta, Stream stream) { add(addend, augend, beta, stream); }

void Tensor::add(Tensor augend, Tensor addend, float alpha, float beta, Stream stream) {
    assert(augend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(augend.getDataType() == addend.getDataType());
    assert(augend.getDataType() == getDataType());
    assert(augend.getTotalNumElements() == addend.getTotalNumElements());
    assert(augend.getTotalNumElements() == getTotalNumElements());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = augend.getDataType();
    uint64_t numElements = augend.getTotalNumElements();
    void *augendMem = augend.getMemPtr();
    void *addendMem = addend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwiseHalf<<<gridSize, blockSize, 0, stream>>>(
            (half *)augendMem, (half *)destMem, (half *)addendMem, alpha, beta, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)augendMem, (float *)destMem, (float *)addendMem, alpha, beta, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        assert(alpha >= 0.0f);
        assert(beta >= 0.0f);
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise1BUnsignedInt<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)augendMem, (uint8_t *)destMem, (uint8_t *)addendMem, alpha, beta, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        assert(alpha >= 0.0f);
        assert(beta >= 0.0f);
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise2BUnsignedInt<uint16_t><<<gridSize, blockSize, 0, stream>>>(
            (uint16_t *)augendMem, (uint16_t *)destMem, (uint16_t *)addendMem, alpha, beta, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        assert(alpha >= 0.0f);
        assert(beta >= 0.0f);
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<uint32_t><<<gridSize, blockSize, 0, stream>>>(
            (uint32_t *)augendMem, (uint32_t *)destMem, (uint32_t *)addendMem, alpha, beta, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise1BSignedInt<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)augendMem, (int8_t *)destMem, (int8_t *)addendMem, alpha, beta, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 4095) / 4096);
        addElementwise2BSignedInt<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)augendMem, (int16_t *)destMem, (int16_t *)addendMem, alpha, beta, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        addElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)augendMem, (int32_t *)destMem, (int32_t *)addendMem, alpha, beta, numElements);
    } else {
        assert(false);
    }
}

void Tensor::subtract(double minuend, Tensor subtrahend, Stream stream) {
    assert(subtrahend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(subtrahend.getDataType() == getDataType());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = subtrahend.getDataType();
    uint64_t numElements = subtrahend.getTotalNumElements();
    void *subtrahendMem = subtrahend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        subtractScalarMinuendHalf<<<gridSize, blockSize, 0, stream>>>((half *)subtrahendMem, (half *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend4B<float><<<gridSize, blockSize, 0, stream>>>((float *)subtrahendMem, (float *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend1BUnsignedInt<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)subtrahendMem, (uint8_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend2BUnsignedInt<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)subtrahendMem, (uint16_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)subtrahendMem, (uint32_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend1BSignedInt<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)subtrahendMem, (int8_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend2BSignedInt<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)subtrahendMem, (int16_t *)destMem, minuend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarMinuend4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)subtrahendMem, (int32_t *)destMem, minuend, numElements);
    } else {
        assert(false);
    }
}

void Tensor::subtract(Tensor minuend, double subtrahend, Stream stream) {
    assert(minuend.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(minuend.getDataType() == getDataType());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = minuend.getDataType();
    uint64_t numElements = minuend.getTotalNumElements();
    void *minuendMem = minuend.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        subtractScalarSubtrahendHalf<<<gridSize, blockSize, 0, stream>>>((half *)minuendMem, (half *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)minuendMem, (float *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend1BUnsignedInt<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)minuendMem, (uint8_t *)destMem, ((uint8_t)subtrahend), numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend2BUnsignedInt<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)minuendMem, (uint16_t *)destMem, ((uint16_t)subtrahend), numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)minuendMem, (uint32_t *)destMem, subtrahend, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend1BSignedInt<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)minuendMem, (int8_t *)destMem, ((int8_t)subtrahend), numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractScalarSubtrahend2BSignedInt<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)minuendMem, (int16_t *)destMem, ((int16_t)subtrahend), numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
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

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

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
        dim3 gridSize((numElements + 2047) / 2048);
        subtractElementwise4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)minuendMem, (float *)destMem, (float *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        subtractElementwise1BUnsignedInt<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)minuendMem, (uint8_t *)destMem, (uint8_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractElementwise2BUnsignedInt<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)minuendMem, (uint16_t *)destMem, (uint16_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)minuendMem, (uint32_t *)destMem, (uint32_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        subtractElementwise1BSignedInt<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)minuendMem, (int8_t *)destMem, (int8_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractElementwise2BSignedInt<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)minuendMem, (int16_t *)destMem, (int16_t *)subtrahendMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        subtractElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)minuendMem, (int32_t *)destMem, (int32_t *)subtrahendMem, numElements);
    } else {
        assert(false);
    }
}

void Tensor::multiply(Tensor multiplicand, double multiplier, Stream stream) {
    assert(multiplicand.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(multiplicand.getDataType() == getDataType());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = multiplicand.getDataType();
    uint64_t numElements = multiplicand.getTotalNumElements();
    void *multiplicandMem = multiplicand.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        multiplyScalarMultiplierHalf<<<gridSize, blockSize, 0, stream>>>((half *)multiplicandMem, (half *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)multiplicandMem, (float *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)multiplicandMem, (uint8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)multiplicandMem, (uint16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)multiplicandMem, (uint32_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)multiplicandMem, (int8_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyScalarMultiplier2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)multiplicandMem, (int16_t *)destMem, multiplier, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
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

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

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
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)multiplicandMem, (float *)destMem, (float *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)multiplicandMem, (uint8_t *)destMem, (uint8_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)multiplicandMem, (uint16_t *)destMem, (uint16_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)multiplicandMem, (uint32_t *)destMem, (uint32_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)multiplicandMem, (int8_t *)destMem, (int8_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)multiplicandMem, (int16_t *)destMem, (int16_t *)multiplierMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        multiplyElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)multiplicandMem, (int32_t *)destMem, (int32_t *)multiplierMem, numElements);
    } else {
        assert(false);
    }
}

void Tensor::divide(Tensor numerator, double denominator, Stream stream) {
    assert(numerator.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(numerator.getDataType() == getDataType());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = numerator.getDataType();
    uint64_t numElements = numerator.getTotalNumElements();
    void *numeratorMem = numerator.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        divideScalarDenominatorHalf<<<gridSize, blockSize, 0, stream>>>((half *)numeratorMem, (half *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)numeratorMem, (float *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)numeratorMem, (uint8_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)numeratorMem, (uint16_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)numeratorMem, (uint32_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)numeratorMem, (int8_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)numeratorMem, (int16_t *)destMem, denominator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarDenominator4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)numeratorMem, (int32_t *)destMem, denominator, numElements);
    } else {
        assert(false);
    }
}

void Tensor::divide(double numerator, Tensor denominator, Stream stream) {
    assert(denominator.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(denominator.getDataType() == getDataType());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = denominator.getDataType();
    uint64_t numElements = denominator.getTotalNumElements();
    void *denominatorMem = denominator.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 4095) / 4096);
        divideScalarNumeratorHalf<<<gridSize, blockSize, 0, stream>>>((half *)denominatorMem, (half *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)denominatorMem, (float *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)denominatorMem, (uint8_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)denominatorMem, (uint16_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)denominatorMem, (uint32_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)denominatorMem, (int8_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideScalarNumerator2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)denominatorMem, (int16_t *)destMem, numerator, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
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

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

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
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise4B<float>
            <<<gridSize, blockSize, 0, stream>>>((float *)numeratorMem, (float *)destMem, (float *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise1B<uint8_t>
            <<<gridSize, blockSize, 0, stream>>>((uint8_t *)numeratorMem, (uint8_t *)destMem, (uint8_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise2B<uint16_t>
            <<<gridSize, blockSize, 0, stream>>>((uint16_t *)numeratorMem, (uint16_t *)destMem, (uint16_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise4B<uint32_t>
            <<<gridSize, blockSize, 0, stream>>>((uint32_t *)numeratorMem, (uint32_t *)destMem, (uint32_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise1B<int8_t>
            <<<gridSize, blockSize, 0, stream>>>((int8_t *)numeratorMem, (int8_t *)destMem, (int8_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise2B<int16_t>
            <<<gridSize, blockSize, 0, stream>>>((int16_t *)numeratorMem, (int16_t *)destMem, (int16_t *)denominatorMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        divideElementwise4B<int32_t>
            <<<gridSize, blockSize, 0, stream>>>((int32_t *)numeratorMem, (int32_t *)destMem, (int32_t *)denominatorMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = [a] * [b] + [c], elementwise
 * <div/>
 * argument must be float or half.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::multiplyAccumulate(Tensor a, Tensor b, Tensor c, Stream stream) {
    assert(a.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(b.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(c.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((a.getDataType() == TensorDescriptor::DataType::FP32 || a.getDataType() == TensorDescriptor::DataType::FP16));
    assert((b.getDataType() == TensorDescriptor::DataType::FP32 || b.getDataType() == TensorDescriptor::DataType::FP16));
    assert((c.getDataType() == TensorDescriptor::DataType::FP32 || c.getDataType() == TensorDescriptor::DataType::FP16));
    assert(a.getDataType() == b.getDataType());
    assert(a.getDataType() == c.getDataType());
    assert(a.getTotalNumElements() == getTotalNumElements());
    assert(b.getTotalNumElements() == getTotalNumElements());
    assert(c.getTotalNumElements() == getTotalNumElements());

    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = a.getTotalNumElements();
    void *aMem = a.getMemPtr();
    void *bMem = b.getMemPtr();
    void *cMem = c.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (a.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            multiplyAccumulateDest2B<<<gridSize, blockSize, 0, stream>>>(
                (half *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            multiplyAccumulateDest4B<<<gridSize, blockSize, 0, stream>>>(
                (float *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            multiplyAccumulateDest1B<<<gridSize, blockSize, 0, stream>>>(
                (uint8_t *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            multiplyAccumulateDest2B<<<gridSize, blockSize, 0, stream>>>(
                (uint16_t *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            multiplyAccumulateDest4B<<<gridSize, blockSize, 0, stream>>>(
                (uint32_t *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            multiplyAccumulateDest1B<<<gridSize, blockSize, 0, stream>>>(
                (int8_t *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            multiplyAccumulateDest2B<<<gridSize, blockSize, 0, stream>>>(
                (int16_t *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            multiplyAccumulateDest4B<<<gridSize, blockSize, 0, stream>>>(
                (int32_t *)destMem, (half *)aMem, (half *)bMem, (half *)cMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            multiplyAccumulateDest2B<<<gridSize, blockSize, 0, stream>>>(
                (half *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            multiplyAccumulateDest4B<<<gridSize, blockSize, 0, stream>>>(
                (float *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            multiplyAccumulateDest1B<<<gridSize, blockSize, 0, stream>>>(
                (uint8_t *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            multiplyAccumulateDest2B<<<gridSize, blockSize, 0, stream>>>(
                (uint16_t *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            multiplyAccumulateDest4B<<<gridSize, blockSize, 0, stream>>>(
                (uint32_t *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            multiplyAccumulateDest1B<<<gridSize, blockSize, 0, stream>>>(
                (int8_t *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            multiplyAccumulateDest2B<<<gridSize, blockSize, 0, stream>>>(
                (int16_t *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            multiplyAccumulateDest4B<<<gridSize, blockSize, 0, stream>>>(
                (int32_t *)destMem, (float *)aMem, (float *)bMem, (float *)cMem, numElements);
        } else {
            assert(false);
        }
    }
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void max1B(DATA_TYPE *mem, DATA_TYPE minValue, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + threadIdx.x * 16;
    if (offset >= numElements)
        return;
    uint64_t offset16Elements = offset >> 4;

    DATA_TYPE buffer[16];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset16Elements];
    buffer[0] = buffer[0] > minValue ? buffer[0] : minValue;
    buffer[1] = buffer[1] > minValue ? buffer[1] : minValue;
    buffer[2] = buffer[2] > minValue ? buffer[2] : minValue;
    buffer[3] = buffer[3] > minValue ? buffer[3] : minValue;
    buffer[4] = buffer[4] > minValue ? buffer[4] : minValue;
    buffer[5] = buffer[5] > minValue ? buffer[5] : minValue;
    buffer[6] = buffer[6] > minValue ? buffer[6] : minValue;
    buffer[7] = buffer[7] > minValue ? buffer[7] : minValue;
    buffer[8] = buffer[8] > minValue ? buffer[8] : minValue;
    buffer[9] = buffer[9] > minValue ? buffer[9] : minValue;
    buffer[10] = buffer[10] > minValue ? buffer[10] : minValue;
    buffer[11] = buffer[11] > minValue ? buffer[11] : minValue;
    buffer[12] = buffer[12] > minValue ? buffer[12] : minValue;
    buffer[13] = buffer[13] > minValue ? buffer[13] : minValue;
    buffer[14] = buffer[14] > minValue ? buffer[14] : minValue;
    buffer[15] = buffer[15] > minValue ? buffer[15] : minValue;
    ((float4 *)mem)[offset16Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void max2B(DATA_TYPE *mem, DATA_TYPE minValue, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset8Elements];
    buffer[0] = buffer[0] > minValue ? buffer[0] : minValue;
    buffer[1] = buffer[1] > minValue ? buffer[1] : minValue;
    buffer[2] = buffer[2] > minValue ? buffer[2] : minValue;
    buffer[3] = buffer[3] > minValue ? buffer[3] : minValue;
    buffer[4] = buffer[4] > minValue ? buffer[4] : minValue;
    buffer[5] = buffer[5] > minValue ? buffer[5] : minValue;
    buffer[6] = buffer[6] > minValue ? buffer[6] : minValue;
    buffer[7] = buffer[7] > minValue ? buffer[7] : minValue;
    ((float4 *)mem)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void max4B(DATA_TYPE *mem, DATA_TYPE minValue, uint64_t numElements) {
    DATA_TYPE buffer[4];

    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset4Elements];
    buffer[0] = buffer[0] > minValue ? buffer[0] : minValue;
    buffer[1] = buffer[1] > minValue ? buffer[1] : minValue;
    buffer[2] = buffer[2] > minValue ? buffer[2] : minValue;
    buffer[3] = buffer[3] > minValue ? buffer[3] : minValue;
    ((float4 *)mem)[offset4Elements] = ((float4 *)buffer)[0];
}

void Tensor::max(double minValue, Stream stream) {
    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = getDataType();
    uint64_t numElements = getTotalNumElements();
    void *mem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 2047) / 2048);
        max2B<half><<<gridSize, blockSize, 0, stream>>>((half *)mem, minValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 1023) / 1024);
        max4B<float><<<gridSize, blockSize, 0, stream>>>((float *)mem, minValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        max1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)mem, minValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        max2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)mem, minValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        max4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)mem, minValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        max1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)mem, minValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        max2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)mem, minValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        max4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)mem, minValue, numElements);
    } else {
        assert(false);
    }
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void min1B(DATA_TYPE *mem, DATA_TYPE maxValue, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + threadIdx.x * 16;
    if (offset >= numElements)
        return;
    uint64_t offset16Elements = offset >> 4;

    DATA_TYPE buffer[16];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset16Elements];
    buffer[0] = buffer[0] < maxValue ? buffer[0] : maxValue;
    buffer[1] = buffer[1] < maxValue ? buffer[1] : maxValue;
    buffer[2] = buffer[2] < maxValue ? buffer[2] : maxValue;
    buffer[3] = buffer[3] < maxValue ? buffer[3] : maxValue;
    buffer[4] = buffer[4] < maxValue ? buffer[4] : maxValue;
    buffer[5] = buffer[5] < maxValue ? buffer[5] : maxValue;
    buffer[6] = buffer[6] < maxValue ? buffer[6] : maxValue;
    buffer[7] = buffer[7] < maxValue ? buffer[7] : maxValue;
    buffer[8] = buffer[8] < maxValue ? buffer[8] : maxValue;
    buffer[9] = buffer[9] < maxValue ? buffer[9] : maxValue;
    buffer[10] = buffer[10] < maxValue ? buffer[10] : maxValue;
    buffer[11] = buffer[11] < maxValue ? buffer[11] : maxValue;
    buffer[12] = buffer[12] < maxValue ? buffer[12] : maxValue;
    buffer[13] = buffer[13] < maxValue ? buffer[13] : maxValue;
    buffer[14] = buffer[14] < maxValue ? buffer[14] : maxValue;
    buffer[15] = buffer[15] < maxValue ? buffer[15] : maxValue;
    ((float4 *)mem)[offset16Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void min2B(DATA_TYPE *mem, DATA_TYPE maxValue, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset8Elements];
    buffer[0] = buffer[0] < maxValue ? buffer[0] : maxValue;
    buffer[1] = buffer[1] < maxValue ? buffer[1] : maxValue;
    buffer[2] = buffer[2] < maxValue ? buffer[2] : maxValue;
    buffer[3] = buffer[3] < maxValue ? buffer[3] : maxValue;
    buffer[4] = buffer[4] < maxValue ? buffer[4] : maxValue;
    buffer[5] = buffer[5] < maxValue ? buffer[5] : maxValue;
    buffer[6] = buffer[6] < maxValue ? buffer[6] : maxValue;
    buffer[7] = buffer[7] < maxValue ? buffer[7] : maxValue;
    ((float4 *)mem)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void min4B(DATA_TYPE *mem, DATA_TYPE maxValue, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    DATA_TYPE buffer[4];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset4Elements];
    buffer[0] = buffer[0] < maxValue ? buffer[0] : maxValue;
    buffer[1] = buffer[1] < maxValue ? buffer[1] : maxValue;
    buffer[2] = buffer[2] < maxValue ? buffer[2] : maxValue;
    buffer[3] = buffer[3] < maxValue ? buffer[3] : maxValue;
    ((float4 *)mem)[offset4Elements] = ((float4 *)buffer)[0];
}

void Tensor::min(double maxValue, Stream stream) {
    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = getDataType();
    uint64_t numElements = getTotalNumElements();
    void *mem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 2047) / 2048);
        min2B<half><<<gridSize, blockSize, 0, stream>>>((half *)mem, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 1023) / 1024);
        min4B<float><<<gridSize, blockSize, 0, stream>>>((float *)mem, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        min1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)mem, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        min2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)mem, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        min4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)mem, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        min1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)mem, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        min2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)mem, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        min4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)mem, maxValue, numElements);
    } else {
        assert(false);
    }
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void bound1B(DATA_TYPE *mem, DATA_TYPE minValue, DATA_TYPE maxValue, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + threadIdx.x * 16;
    if (offset >= numElements)
        return;
    uint64_t offset16Elements = offset >> 4;

    DATA_TYPE buffer[16];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset16Elements];
    buffer[0] = buffer[0] > minValue ? (buffer[0] < maxValue ? buffer[0] : maxValue) : minValue;
    buffer[1] = buffer[1] > minValue ? (buffer[1] < maxValue ? buffer[1] : maxValue) : minValue;
    buffer[2] = buffer[2] > minValue ? (buffer[2] < maxValue ? buffer[2] : maxValue) : minValue;
    buffer[3] = buffer[3] > minValue ? (buffer[3] < maxValue ? buffer[3] : maxValue) : minValue;
    buffer[4] = buffer[4] > minValue ? (buffer[4] < maxValue ? buffer[4] : maxValue) : minValue;
    buffer[5] = buffer[5] > minValue ? (buffer[5] < maxValue ? buffer[5] : maxValue) : minValue;
    buffer[6] = buffer[6] > minValue ? (buffer[6] < maxValue ? buffer[6] : maxValue) : minValue;
    buffer[7] = buffer[7] > minValue ? (buffer[7] < maxValue ? buffer[7] : maxValue) : minValue;
    buffer[8] = buffer[8] > minValue ? (buffer[8] < maxValue ? buffer[8] : maxValue) : minValue;
    buffer[9] = buffer[9] > minValue ? (buffer[9] < maxValue ? buffer[9] : maxValue) : minValue;
    buffer[10] = buffer[10] > minValue ? (buffer[10] < maxValue ? buffer[10] : maxValue) : minValue;
    buffer[11] = buffer[11] > minValue ? (buffer[11] < maxValue ? buffer[11] : maxValue) : minValue;
    buffer[12] = buffer[12] > minValue ? (buffer[12] < maxValue ? buffer[12] : maxValue) : minValue;
    buffer[13] = buffer[13] > minValue ? (buffer[13] < maxValue ? buffer[13] : maxValue) : minValue;
    buffer[14] = buffer[14] > minValue ? (buffer[14] < maxValue ? buffer[14] : maxValue) : minValue;
    buffer[15] = buffer[15] > minValue ? (buffer[15] < maxValue ? buffer[15] : maxValue) : minValue;
    ((float4 *)mem)[offset16Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void bound2B(DATA_TYPE *mem, DATA_TYPE minValue, DATA_TYPE maxValue, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset8Elements];
    buffer[0] = buffer[0] > minValue ? (buffer[0] < maxValue ? buffer[0] : maxValue) : minValue;
    buffer[1] = buffer[1] > minValue ? (buffer[1] < maxValue ? buffer[1] : maxValue) : minValue;
    buffer[2] = buffer[2] > minValue ? (buffer[2] < maxValue ? buffer[2] : maxValue) : minValue;
    buffer[3] = buffer[3] > minValue ? (buffer[3] < maxValue ? buffer[3] : maxValue) : minValue;
    buffer[4] = buffer[4] > minValue ? (buffer[4] < maxValue ? buffer[4] : maxValue) : minValue;
    buffer[5] = buffer[5] > minValue ? (buffer[5] < maxValue ? buffer[5] : maxValue) : minValue;
    buffer[6] = buffer[6] > minValue ? (buffer[6] < maxValue ? buffer[6] : maxValue) : minValue;
    buffer[7] = buffer[7] > minValue ? (buffer[7] < maxValue ? buffer[7] : maxValue) : minValue;
    ((float4 *)mem)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void bound4B(DATA_TYPE *mem, DATA_TYPE minValue, DATA_TYPE maxValue, uint64_t numElements) {
    DATA_TYPE buffer[8];

    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    // Note: all tensors end on 16 byte boundary
    ((double4 *)buffer)[0] = ((double4 *)mem)[offset8Elements];
    buffer[0] = buffer[0] > minValue ? (buffer[0] < maxValue ? buffer[0] : maxValue) : minValue;
    buffer[1] = buffer[1] > minValue ? (buffer[1] < maxValue ? buffer[1] : maxValue) : minValue;
    buffer[2] = buffer[2] > minValue ? (buffer[2] < maxValue ? buffer[2] : maxValue) : minValue;
    buffer[3] = buffer[3] > minValue ? (buffer[3] < maxValue ? buffer[3] : maxValue) : minValue;
    buffer[4] = buffer[4] > minValue ? (buffer[4] < maxValue ? buffer[4] : maxValue) : minValue;
    buffer[5] = buffer[5] > minValue ? (buffer[5] < maxValue ? buffer[5] : maxValue) : minValue;
    buffer[6] = buffer[6] > minValue ? (buffer[6] < maxValue ? buffer[6] : maxValue) : minValue;
    buffer[7] = buffer[7] > minValue ? (buffer[7] < maxValue ? buffer[7] : maxValue) : minValue;
    ((double4 *)mem)[offset8Elements] = ((double4 *)buffer)[0];
}

void Tensor::bound(double minValue, double maxValue, Stream stream) {
    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = getDataType();
    uint64_t numElements = getTotalNumElements();
    void *mem = getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 2047) / 2048);
        bound2B<half><<<gridSize, blockSize, 0, stream>>>((half *)mem, minValue, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 2047) / 2048);
        bound4B<float><<<gridSize, blockSize, 0, stream>>>((float *)mem, minValue, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        bound1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)mem, minValue, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        bound2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)mem, minValue, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        bound4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)mem, minValue, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        bound1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)mem, minValue, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        bound2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)mem, minValue, maxValue, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 2047) / 2048);
        bound4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)mem, minValue, maxValue, numElements);
    } else {
        assert(false);
    }
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void min1B(DATA_TYPE *mem, DATA_TYPE *other, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + threadIdx.x * 16;
    if (offset >= numElements)
        return;
    uint64_t offset16Elements = offset >> 4;

    DATA_TYPE buffer[16];
    DATA_TYPE otherBuffer[16];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset16Elements];
    ((float4 *)otherBuffer)[0] = ((float4 *)other)[offset16Elements];

    buffer[0] = buffer[0] < otherBuffer[0] ? buffer[0] : otherBuffer[0];
    buffer[1] = buffer[1] < otherBuffer[1] ? buffer[1] : otherBuffer[1];
    buffer[2] = buffer[2] < otherBuffer[2] ? buffer[2] : otherBuffer[2];
    buffer[3] = buffer[3] < otherBuffer[3] ? buffer[3] : otherBuffer[3];
    buffer[4] = buffer[4] < otherBuffer[4] ? buffer[4] : otherBuffer[4];
    buffer[5] = buffer[5] < otherBuffer[5] ? buffer[5] : otherBuffer[5];
    buffer[6] = buffer[6] < otherBuffer[6] ? buffer[6] : otherBuffer[6];
    buffer[7] = buffer[7] < otherBuffer[7] ? buffer[7] : otherBuffer[7];
    buffer[8] = buffer[8] < otherBuffer[8] ? buffer[8] : otherBuffer[8];
    buffer[9] = buffer[9] < otherBuffer[9] ? buffer[9] : otherBuffer[9];
    buffer[10] = buffer[10] < otherBuffer[10] ? buffer[10] : otherBuffer[10];
    buffer[11] = buffer[11] < otherBuffer[11] ? buffer[11] : otherBuffer[11];
    buffer[12] = buffer[12] < otherBuffer[12] ? buffer[12] : otherBuffer[12];
    buffer[13] = buffer[13] < otherBuffer[13] ? buffer[13] : otherBuffer[13];
    buffer[14] = buffer[14] < otherBuffer[14] ? buffer[14] : otherBuffer[14];
    buffer[15] = buffer[15] < otherBuffer[15] ? buffer[15] : otherBuffer[15];

    ((float4 *)mem)[offset16Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void min2B(DATA_TYPE *mem, DATA_TYPE *other, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    DATA_TYPE buffer[8];
    DATA_TYPE otherBuffer[8];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset8Elements];
    ((float4 *)otherBuffer)[0] = ((float4 *)other)[offset8Elements];

    buffer[0] = buffer[0] < otherBuffer[0] ? buffer[0] : otherBuffer[0];
    buffer[1] = buffer[1] < otherBuffer[1] ? buffer[1] : otherBuffer[1];
    buffer[2] = buffer[2] < otherBuffer[2] ? buffer[2] : otherBuffer[2];
    buffer[3] = buffer[3] < otherBuffer[3] ? buffer[3] : otherBuffer[3];
    buffer[4] = buffer[4] < otherBuffer[4] ? buffer[4] : otherBuffer[4];
    buffer[5] = buffer[5] < otherBuffer[5] ? buffer[5] : otherBuffer[5];
    buffer[6] = buffer[6] < otherBuffer[6] ? buffer[6] : otherBuffer[6];
    buffer[7] = buffer[7] < otherBuffer[7] ? buffer[7] : otherBuffer[7];

    ((float4 *)mem)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void min4B(DATA_TYPE *mem, DATA_TYPE *other, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    DATA_TYPE buffer[4];
    DATA_TYPE otherBuffer[4];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset4Elements];
    ((float4 *)otherBuffer)[0] = ((float4 *)other)[offset4Elements];

    buffer[0] = buffer[0] < otherBuffer[0] ? buffer[0] : otherBuffer[0];
    buffer[1] = buffer[1] < otherBuffer[1] ? buffer[1] : otherBuffer[1];
    buffer[2] = buffer[2] < otherBuffer[2] ? buffer[2] : otherBuffer[2];
    buffer[3] = buffer[3] < otherBuffer[3] ? buffer[3] : otherBuffer[3];

    ((float4 *)mem)[offset4Elements] = ((float4 *)buffer)[0];
}

void Tensor::min(Tensor other, Stream stream) {
    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = getDataType();
    uint64_t numElements = getTotalNumElements();
    void *mem = getMemPtr();
    void *otherMem = other.getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 2047) / 2048);
        min2B<half><<<gridSize, blockSize, 0, stream>>>((half *)mem, (half *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 1023) / 1024);
        min4B<float><<<gridSize, blockSize, 0, stream>>>((float *)mem, (float *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        min1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)mem, (uint8_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        min2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)mem, (uint16_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        min4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)mem, (uint32_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        min1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)mem, (int8_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        min2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)mem, (int16_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        min4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)mem, (int32_t *)otherMem, numElements);
    } else {
        assert(false);
    }
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void max1B(DATA_TYPE *mem, DATA_TYPE *other, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 4096 + threadIdx.x * 16;
    if (offset >= numElements)
        return;
    uint64_t offset16Elements = offset >> 4;

    DATA_TYPE buffer[16];
    DATA_TYPE otherBuffer[16];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset16Elements];
    ((float4 *)otherBuffer)[0] = ((float4 *)other)[offset16Elements];

    buffer[0] = buffer[0] > otherBuffer[0] ? buffer[0] : otherBuffer[0];
    buffer[1] = buffer[1] > otherBuffer[1] ? buffer[1] : otherBuffer[1];
    buffer[2] = buffer[2] > otherBuffer[2] ? buffer[2] : otherBuffer[2];
    buffer[3] = buffer[3] > otherBuffer[3] ? buffer[3] : otherBuffer[3];
    buffer[4] = buffer[4] > otherBuffer[4] ? buffer[4] : otherBuffer[4];
    buffer[5] = buffer[5] > otherBuffer[5] ? buffer[5] : otherBuffer[5];
    buffer[6] = buffer[6] > otherBuffer[6] ? buffer[6] : otherBuffer[6];
    buffer[7] = buffer[7] > otherBuffer[7] ? buffer[7] : otherBuffer[7];
    buffer[8] = buffer[8] > otherBuffer[8] ? buffer[8] : otherBuffer[8];
    buffer[9] = buffer[9] > otherBuffer[9] ? buffer[9] : otherBuffer[9];
    buffer[10] = buffer[10] > otherBuffer[10] ? buffer[10] : otherBuffer[10];
    buffer[11] = buffer[11] > otherBuffer[11] ? buffer[11] : otherBuffer[11];
    buffer[12] = buffer[12] > otherBuffer[12] ? buffer[12] : otherBuffer[12];
    buffer[13] = buffer[13] > otherBuffer[13] ? buffer[13] : otherBuffer[13];
    buffer[14] = buffer[14] > otherBuffer[14] ? buffer[14] : otherBuffer[14];
    buffer[15] = buffer[15] > otherBuffer[15] ? buffer[15] : otherBuffer[15];

    ((float4 *)mem)[offset16Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void max2B(DATA_TYPE *mem, DATA_TYPE *other, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    DATA_TYPE buffer[8];
    DATA_TYPE otherBuffer[8];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset8Elements];
    ((float4 *)otherBuffer)[0] = ((float4 *)other)[offset8Elements];

    buffer[0] = buffer[0] > otherBuffer[0] ? buffer[0] : otherBuffer[0];
    buffer[1] = buffer[1] > otherBuffer[1] ? buffer[1] : otherBuffer[1];
    buffer[2] = buffer[2] > otherBuffer[2] ? buffer[2] : otherBuffer[2];
    buffer[3] = buffer[3] > otherBuffer[3] ? buffer[3] : otherBuffer[3];
    buffer[4] = buffer[4] > otherBuffer[4] ? buffer[4] : otherBuffer[4];
    buffer[5] = buffer[5] > otherBuffer[5] ? buffer[5] : otherBuffer[5];
    buffer[6] = buffer[6] > otherBuffer[6] ? buffer[6] : otherBuffer[6];
    buffer[7] = buffer[7] > otherBuffer[7] ? buffer[7] : otherBuffer[7];

    ((float4 *)mem)[offset8Elements] = ((float4 *)buffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DATA_TYPE>
__global__ void max4B(DATA_TYPE *mem, DATA_TYPE *other, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 2;

    DATA_TYPE buffer[4];
    DATA_TYPE otherBuffer[4];

    // Note: all tensors end on 16 byte boundary
    ((float4 *)buffer)[0] = ((float4 *)mem)[offset8Elements];
    ((float4 *)otherBuffer)[0] = ((float4 *)other)[offset8Elements];

    buffer[0] = buffer[0] > otherBuffer[0] ? buffer[0] : otherBuffer[0];
    buffer[1] = buffer[1] > otherBuffer[1] ? buffer[1] : otherBuffer[1];
    buffer[2] = buffer[2] > otherBuffer[2] ? buffer[2] : otherBuffer[2];
    buffer[3] = buffer[3] > otherBuffer[3] ? buffer[3] : otherBuffer[3];

    ((float4 *)mem)[offset8Elements] = ((float4 *)buffer)[0];
}

void Tensor::max(Tensor other, Stream stream) {
    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    TensorDescriptor::DataType dataType = getDataType();
    uint64_t numElements = getTotalNumElements();
    void *mem = getMemPtr();
    void *otherMem = other.getMemPtr();

    dim3 blockSize(256);
    if (dataType == TensorDescriptor::DataType::FP16) {
        dim3 gridSize((numElements + 2047) / 2048);
        max2B<half><<<gridSize, blockSize, 0, stream>>>((half *)mem, (half *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        dim3 gridSize((numElements + 1023) / 1024);
        max4B<float><<<gridSize, blockSize, 0, stream>>>((float *)mem, (float *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        max1B<uint8_t><<<gridSize, blockSize, 0, stream>>>((uint8_t *)mem, (uint8_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        max2B<uint16_t><<<gridSize, blockSize, 0, stream>>>((uint16_t *)mem, (uint16_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        max4B<uint32_t><<<gridSize, blockSize, 0, stream>>>((uint32_t *)mem, (uint32_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        dim3 gridSize((numElements + 4095) / 4096);
        max1B<int8_t><<<gridSize, blockSize, 0, stream>>>((int8_t *)mem, (int8_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        dim3 gridSize((numElements + 2047) / 2048);
        max2B<int16_t><<<gridSize, blockSize, 0, stream>>>((int16_t *)mem, (int16_t *)otherMem, numElements);
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        dim3 gridSize((numElements + 1023) / 1024);
        max4B<int32_t><<<gridSize, blockSize, 0, stream>>>((int32_t *)mem, (int32_t *)otherMem, numElements);
    } else {
        assert(false);
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

template <typename T>
void Tensor::launchFillValueGpuKernel(T value, T *mem, uint64_t numElements, uint32_t deviceNum, Stream stream) {
    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    uint32_t gpuNum = getPlacement().getDeviceNum();
    ScopedGpu scopedGpu(gpuNum);

    dim3 blockSize(256);
    if (is_same<T, half>::value || is_same<T, uint16_t>::value || is_same<T, int16_t>::value) {
        dim3 gridSize((numElements + 2047) / 2048);
        fillValue2B<T><<<gridSize, blockSize, 0, stream>>>(value, mem, numElements);
    } else if (is_same<T, float>::value || is_same<T, uint32_t>::value || is_same<T, int32_t>::value) {
        dim3 gridSize((numElements + 1023) / 1024);
        fillValue4B<T><<<gridSize, blockSize, 0, stream>>>(value, mem, numElements);
    } else if (is_same<T, uint8_t>::value || is_same<T, int8_t>::value) {
        dim3 gridSize((numElements + 4095) / 4096);
        fillValue1B<T><<<gridSize, blockSize, 0, stream>>>(value, mem, numElements);
    } else {
        assert(false);
    }
}

template void Tensor::launchFillValueGpuKernel<half>(half value, half *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<float>(float value, float *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<uint8_t>(
    uint8_t value, uint8_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<uint16_t>(
    uint16_t value, uint16_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<uint32_t>(
    uint32_t value, uint32_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<int8_t>(int8_t value, int8_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<int16_t>(
    int16_t value, int16_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
template void Tensor::launchFillValueGpuKernel<int32_t>(
    int32_t value, int32_t *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
