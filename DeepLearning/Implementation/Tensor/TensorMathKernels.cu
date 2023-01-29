#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;
using namespace std;

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powElementwiseDest1B(float *base, DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float baseBuffer[8];
    float exponentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((double4 *)baseBuffer)[0] = ((double4 *)base)[offset8Elements];
    ((double4 *)exponentBuffer)[0] = ((double4 *)exponent)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)powf(baseBuffer[4], exponentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)powf(baseBuffer[5], exponentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)powf(baseBuffer[6], exponentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)powf(baseBuffer[7], exponentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powElementwiseDest2B(float *base, DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float baseBuffer[4];
    float exponentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powElementwiseDest4B(float *base, DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float baseBuffer[4];
    float exponentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powScalarExponentDest1B(float *base, DEST_DATA_TYPE *dest, float exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float baseBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((double4 *)baseBuffer)[0] = ((double4 *)base)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponent);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponent);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponent);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponent);
    destBuffer[4] = (DEST_DATA_TYPE)powf(baseBuffer[4], exponent);
    destBuffer[5] = (DEST_DATA_TYPE)powf(baseBuffer[5], exponent);
    destBuffer[6] = (DEST_DATA_TYPE)powf(baseBuffer[6], exponent);
    destBuffer[7] = (DEST_DATA_TYPE)powf(baseBuffer[7], exponent);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powScalarExponentDest2B(float *base, DEST_DATA_TYPE *dest, float exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float baseBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponent);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponent);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponent);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponent);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponent);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponent);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponent);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponent);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powScalarExponentDest4B(float *base, DEST_DATA_TYPE *dest, float exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float baseBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponent);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponent);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponent);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponent);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)baseBuffer)[0] = ((float4 *)base)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(baseBuffer[0], exponent);
    destBuffer[1] = (DEST_DATA_TYPE)powf(baseBuffer[1], exponent);
    destBuffer[2] = (DEST_DATA_TYPE)powf(baseBuffer[2], exponent);
    destBuffer[3] = (DEST_DATA_TYPE)powf(baseBuffer[3], exponent);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powScalarBaseDest1B(float base, DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float exponentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((double4 *)exponentBuffer)[0] = ((double4 *)exponent)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(base, exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(base, exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(base, exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(base, exponentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)powf(base, exponentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)powf(base, exponentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)powf(base, exponentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)powf(base, exponentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powScalarBaseDest2B(float base, DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float exponentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(base, exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(base, exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(base, exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(base, exponentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(base, exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(base, exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(base, exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(base, exponentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void powScalarBaseDest4B(float base, DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float exponentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(base, exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(base, exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(base, exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(base, exponentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)powf(base, exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)powf(base, exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)powf(base, exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)powf(base, exponentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void expDest1B(DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float exponentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((double4 *)exponentBuffer)[0] = ((double4 *)exponent)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)expf(exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)expf(exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)expf(exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)expf(exponentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)expf(exponentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)expf(exponentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)expf(exponentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)expf(exponentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void expDest2B(DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float exponentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)expf(exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)expf(exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)expf(exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)expf(exponentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)expf(exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)expf(exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)expf(exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)expf(exponentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void expDest4B(DEST_DATA_TYPE *dest, float *exponent, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float exponentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and exponent
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)expf(exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)expf(exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)expf(exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)expf(exponentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)exponentBuffer)[0] = ((float4 *)exponent)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)expf(exponentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)expf(exponentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)expf(exponentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)expf(exponentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logDest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)logf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)logf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)logf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)logf(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)logf(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)logf(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)logf(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)logf(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)logf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)logf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)logf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)logf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)logf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)logf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)logf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)logf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)logf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)logf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)logf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)logf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)logf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)logf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)logf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)logf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2log(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2log(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2log(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logXDest1B(DEST_DATA_TYPE *dest, float *argument, float log2ConversionFactor, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(log2f(argumentBuffer[0]) * log2ConversionFactor);
    destBuffer[1] = (DEST_DATA_TYPE)(log2f(argumentBuffer[1]) * log2ConversionFactor);
    destBuffer[2] = (DEST_DATA_TYPE)(log2f(argumentBuffer[2]) * log2ConversionFactor);
    destBuffer[3] = (DEST_DATA_TYPE)(log2f(argumentBuffer[3]) * log2ConversionFactor);
    destBuffer[4] = (DEST_DATA_TYPE)(log2f(argumentBuffer[4]) * log2ConversionFactor);
    destBuffer[5] = (DEST_DATA_TYPE)(log2f(argumentBuffer[5]) * log2ConversionFactor);
    destBuffer[6] = (DEST_DATA_TYPE)(log2f(argumentBuffer[6]) * log2ConversionFactor);
    destBuffer[7] = (DEST_DATA_TYPE)(log2f(argumentBuffer[7]) * log2ConversionFactor);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logXDest2B(DEST_DATA_TYPE *dest, float *argument, float log2ConversionFactor, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(log2f(argumentBuffer[0]) * log2ConversionFactor);
    destBuffer[1] = (DEST_DATA_TYPE)(log2f(argumentBuffer[1]) * log2ConversionFactor);
    destBuffer[2] = (DEST_DATA_TYPE)(log2f(argumentBuffer[2]) * log2ConversionFactor);
    destBuffer[3] = (DEST_DATA_TYPE)(log2f(argumentBuffer[3]) * log2ConversionFactor);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(log2f(argumentBuffer[0]) * log2ConversionFactor);
    destBuffer[1] = (DEST_DATA_TYPE)(log2f(argumentBuffer[1]) * log2ConversionFactor);
    destBuffer[2] = (DEST_DATA_TYPE)(log2f(argumentBuffer[2]) * log2ConversionFactor);
    destBuffer[3] = (DEST_DATA_TYPE)(log2f(argumentBuffer[3]) * log2ConversionFactor);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logXDest4B(DEST_DATA_TYPE *dest, float *argument, float log2ConversionFactor, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(log2f(argumentBuffer[0]) * log2ConversionFactor);
    destBuffer[1] = (DEST_DATA_TYPE)(log2f(argumentBuffer[1]) * log2ConversionFactor);
    destBuffer[2] = (DEST_DATA_TYPE)(log2f(argumentBuffer[2]) * log2ConversionFactor);
    destBuffer[3] = (DEST_DATA_TYPE)(log2f(argumentBuffer[3]) * log2ConversionFactor);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(log2f(argumentBuffer[0]) * log2ConversionFactor);
    destBuffer[1] = (DEST_DATA_TYPE)(log2f(argumentBuffer[1]) * log2ConversionFactor);
    destBuffer[2] = (DEST_DATA_TYPE)(log2f(argumentBuffer[2]) * log2ConversionFactor);
    destBuffer[3] = (DEST_DATA_TYPE)(log2f(argumentBuffer[3]) * log2ConversionFactor);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logXDest1B(DEST_DATA_TYPE *dest, half *argument, half log2ConversionFactor, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    half2 conversionBuffer;
    DEST_DATA_TYPE destBuffer[8];

    conversionBuffer.x = log2ConversionFactor;
    conversionBuffer.y = log2ConversionFactor;

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    argumentBuffer[0] = __hmul2(argumentBuffer[0], conversionBuffer);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    argumentBuffer[1] = __hmul2(argumentBuffer[1], conversionBuffer);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2log2(argumentBuffer[2]);
    argumentBuffer[2] = __hmul2(argumentBuffer[2], conversionBuffer);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2log2(argumentBuffer[3]);
    argumentBuffer[3] = __hmul2(argumentBuffer[3], conversionBuffer);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logXDest2B(DEST_DATA_TYPE *dest, half *argument, half log2ConversionFactor, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    half2 conversionBuffer;
    DEST_DATA_TYPE destBuffer[4];

    conversionBuffer.x = log2ConversionFactor;
    conversionBuffer.y = log2ConversionFactor;

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    argumentBuffer[0] = __hmul2(argumentBuffer[0], conversionBuffer);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    argumentBuffer[1] = __hmul2(argumentBuffer[1], conversionBuffer);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    argumentBuffer[0] = __hmul2(argumentBuffer[0], conversionBuffer);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    argumentBuffer[1] = __hmul2(argumentBuffer[1], conversionBuffer);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void logXDest4B(DEST_DATA_TYPE *dest, half *argument, half log2ConversionFactor, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    half2 conversionBuffer;
    DEST_DATA_TYPE destBuffer[4];

    conversionBuffer.x = log2ConversionFactor;
    conversionBuffer.y = log2ConversionFactor;

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    argumentBuffer[0] = __hmul2(argumentBuffer[0], conversionBuffer);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    argumentBuffer[1] = __hmul2(argumentBuffer[1], conversionBuffer);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    argumentBuffer[0] = __hmul2(argumentBuffer[0], conversionBuffer);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    argumentBuffer[1] = __hmul2(argumentBuffer[1], conversionBuffer);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log2Dest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log2f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log2f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log2f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log2f(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)log2f(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)log2f(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)log2f(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)log2f(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log2Dest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log2f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log2f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log2f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log2f(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log2f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log2f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log2f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log2f(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log2Dest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log2f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log2f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log2f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log2f(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log2f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log2f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log2f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log2f(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log2Dest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2log2(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2log2(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log2Dest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log2Dest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log2(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log2(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log10Dest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log10f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log10f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log10f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log10f(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)log10f(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)log10f(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)log10f(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)log10f(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log10Dest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log10f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log10f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log10f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log10f(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log10f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log10f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log10f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log10f(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log10Dest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log10f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log10f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log10f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log10f(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)log10f(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)log10f(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)log10f(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)log10f(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log10Dest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2log10(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log10(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2log10(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2log10(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log10Dest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log10(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log10(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log10(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log10(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void log10Dest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log10(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log10(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2log10(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2log10(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

/**
 * [thisTensor] = [base] ^ [exponent], elementwise
 * <div/>
 * exponent and base must both have data type FP32
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::pow(Tensor base, Tensor exponent, Stream stream) {
    assert(base.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(base.getPlacement() == exponent.getPlacement());
    assert(exponent.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((base.getDataType() == TensorDescriptor::DataType::FP32 && exponent.getDataType() == TensorDescriptor::DataType::FP32));
    assert(base.getTotalNumElements() == exponent.getTotalNumElements());
    assert(base.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = base.getTotalNumElements();
    void *baseMem = base.getMemPtr();
    void *exponentMem = exponent.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        powElementwiseDest2B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (half *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        powElementwiseDest4B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (float *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT8) {
        powElementwiseDest1B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (uint8_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT16) {
        powElementwiseDest2B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (uint16_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT32) {
        powElementwiseDest4B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (uint32_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT8) {
        powElementwiseDest1B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (int8_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT16) {
        powElementwiseDest2B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (int16_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT32) {
        powElementwiseDest4B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (int32_t *)destMem, (float *)exponentMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = [base] ^ exponent, elementwise
 * <div/>
 * base must have data type FP32.
 */
void Tensor::pow(Tensor base, float exponent, Stream stream) {
    assert(base.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(base.getDataType() == TensorDescriptor::DataType::FP32);
    assert(base.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = base.getTotalNumElements();
    void *baseMem = base.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        powScalarExponentDest2B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (half *)destMem, exponent, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        powScalarExponentDest4B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (float *)destMem, exponent, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT8) {
        powScalarExponentDest1B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (uint8_t *)destMem, exponent, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT16) {
        powScalarExponentDest2B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (uint16_t *)destMem, exponent, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT32) {
        powScalarExponentDest4B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (uint32_t *)destMem, exponent, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT8) {
        powScalarExponentDest1B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (int8_t *)destMem, exponent, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT16) {
        powScalarExponentDest2B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (int16_t *)destMem, exponent, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT32) {
        powScalarExponentDest4B<<<gridSize, blockSize, 0, stream>>>((float *)baseMem, (int32_t *)destMem, exponent, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = base ^ [exponent], elementwise
 * <div/>
 * exponent must have data type FP32.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::pow(float base, Tensor exponent, Stream stream) {
    assert(exponent.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(exponent.getDataType() == TensorDescriptor::DataType::FP32);
    assert(exponent.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = exponent.getTotalNumElements();
    void *exponentMem = exponent.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        powScalarBaseDest2B<<<gridSize, blockSize, 0, stream>>>(base, (half *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        powScalarBaseDest4B<<<gridSize, blockSize, 0, stream>>>(base, (float *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT8) {
        powScalarBaseDest1B<<<gridSize, blockSize, 0, stream>>>(base, (uint8_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT16) {
        powScalarBaseDest2B<<<gridSize, blockSize, 0, stream>>>(base, (uint16_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT32) {
        powScalarBaseDest4B<<<gridSize, blockSize, 0, stream>>>(base, (uint32_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT8) {
        powScalarBaseDest1B<<<gridSize, blockSize, 0, stream>>>(base, (int8_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT16) {
        powScalarBaseDest2B<<<gridSize, blockSize, 0, stream>>>(base, (int16_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT32) {
        powScalarBaseDest4B<<<gridSize, blockSize, 0, stream>>>(base, (int32_t *)destMem, (float *)exponentMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = e ^ [exponent], elementwise
 * <div/>
 * where e is euler's constant.
 * exponent must have data type FP32,
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::exp(Tensor exponent, Stream stream) {
    assert(exponent.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(exponent.getDataType() == TensorDescriptor::DataType::FP32);
    assert(exponent.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = exponent.getTotalNumElements();
    void *exponentMem = exponent.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        expDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        expDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT8) {
        expDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT16) {
        expDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT32) {
        expDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT8) {
        expDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT16) {
        expDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)exponentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT32) {
        expDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)exponentMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = ln([argument]), elementwise
 * <div/>
 * Compute the natural log of the argument tensor
 * argument must be float or half
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::log(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            logDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            logDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            logDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            logDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            logDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            logDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            logDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            logDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            logDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            logDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            logDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            logDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            logDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            logDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            logDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            logDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
        } else {
            assert(false);
        }
    }
}

float getLog2ConversionFactor(float base) { return 1.0f / log2f(base); }

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void sqrtDest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void sqrtDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void sqrtDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sqrtf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void sqrtDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2sqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2sqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2sqrt(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2sqrt(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void sqrtDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2sqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2sqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2sqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2sqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void sqrtDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2sqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2sqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2sqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2sqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void ceilDest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)ceilf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)ceilf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)ceilf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)ceilf(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)ceilf(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)ceilf(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)ceilf(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)ceilf(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void ceilDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)ceilf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)ceilf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)ceilf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)ceilf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)ceilf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)ceilf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)ceilf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)ceilf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void ceilDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)ceilf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)ceilf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)ceilf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)ceilf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)ceilf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)ceilf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)ceilf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)ceilf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void ceilDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2ceil(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2ceil(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2ceil(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2ceil(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void ceilDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2ceil(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2ceil(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2ceil(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2ceil(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void ceilDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2ceil(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2ceil(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2ceil(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2ceil(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void floorDest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)floorf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)floorf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)floorf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)floorf(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)floorf(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)floorf(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)floorf(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)floorf(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void floorDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)floorf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)floorf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)floorf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)floorf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)floorf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)floorf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)floorf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)floorf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void floorDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)floorf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)floorf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)floorf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)floorf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)floorf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)floorf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)floorf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)floorf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void floorDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2floor(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2floor(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2floor(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2floor(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void floorDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2floor(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2floor(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2floor(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2floor(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void floorDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2floor(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2floor(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2floor(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2floor(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void roundDest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rintf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rintf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rintf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rintf(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)rintf(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)rintf(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)rintf(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)rintf(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void roundDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rintf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rintf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rintf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rintf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rintf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rintf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rintf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rintf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void roundDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rintf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rintf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rintf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rintf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rintf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rintf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rintf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rintf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void roundDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2rint(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rint(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2rint(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2rint(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void roundDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rint(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rint(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rint(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rint(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void roundDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rint(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rint(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rint(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rint(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void truncateFloatingPointDest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)truncf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)truncf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)truncf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)truncf(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)truncf(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)truncf(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)truncf(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)truncf(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void truncateFloatingPointDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)truncf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)truncf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)truncf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)truncf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)truncf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)truncf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)truncf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)truncf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void truncateFloatingPointDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)truncf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)truncf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)truncf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)truncf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)truncf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)truncf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)truncf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)truncf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void truncateFloatingPointDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void truncateFloatingPointDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void truncateFloatingPointDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)(float)htrunc(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2rcp(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rcp(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2rcp(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2rcp(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rcp(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rcp(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rcp(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rcp(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rcp(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rcp(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rcp(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rcp(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalSqrtDest1B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    float argumentBuffer[8];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((double4 *)argumentBuffer)[0] = ((double4 *)argument)[offset8Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[3]);
    destBuffer[4] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[4]);
    destBuffer[5] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[5]);
    destBuffer[6] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[6]);
    destBuffer[7] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[7]);
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalSqrtDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalSqrtDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)rsqrtf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 16 elements : 4096 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalSqrtDest1B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 8;
    if (offset >= numElements)
        return;
    uint64_t offset8Elements = offset >> 3;

    half2 argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[8];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset8Elements];
    argumentBuffer[0] = h2rsqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rsqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    argumentBuffer[2] = h2rsqrt(argumentBuffer[2]);
    destBuffer[4] = (DEST_DATA_TYPE)(float)argumentBuffer[2].x;
    destBuffer[5] = (DEST_DATA_TYPE)(float)argumentBuffer[2].y;
    argumentBuffer[3] = h2rsqrt(argumentBuffer[3]);
    destBuffer[6] = (DEST_DATA_TYPE)(float)argumentBuffer[3].x;
    destBuffer[7] = (DEST_DATA_TYPE)(float)argumentBuffer[3].y;
    ((double *)dest)[offset8Elements] = ((double *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalSqrtDest2B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rsqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rsqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rsqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rsqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void reciprocalSqrtDest4B(DEST_DATA_TYPE *dest, half *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    half2 argumentBuffer[2];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rsqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rsqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float2 *)argumentBuffer)[0] = ((float2 *)argument)[offset4Elements];
    argumentBuffer[0] = h2rsqrt(argumentBuffer[0]);
    destBuffer[0] = (DEST_DATA_TYPE)(float)argumentBuffer[0].x;
    destBuffer[1] = (DEST_DATA_TYPE)(float)argumentBuffer[0].y;
    argumentBuffer[1] = h2rsqrt(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)(float)argumentBuffer[1].x;
    destBuffer[3] = (DEST_DATA_TYPE)(float)argumentBuffer[1].y;
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erff(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erff(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erff(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erff(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erff(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erff(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erff(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erff(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erff(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erff(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erff(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erff(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erff(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erff(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erff(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erff(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfinvDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfinvDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfinvf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfcDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfcDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfcinvDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfcinvDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcinvf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfcxDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void erfcxDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)erfcxf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void tgammaDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void tgammaDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tgammaf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void lgammaDest2B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE>
__global__ void lgammaDest4B(DEST_DATA_TYPE *dest, float *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)lgammaf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

/**
 * [thisTensor] = log<base>([argument]), elementwise
 * <div/>
 * Compute the log with the specified base of the argument tensor.
 * base must be positive and not equal to 1.
 * argument must be float or half.
 * base will be converted into the type of argument.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::log(Tensor argument, float base, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    assert(base > 0.0f && base != 1.0f);

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);

    if (base == 2.0f) {
        if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
            if (destDataType == TensorDescriptor::DataType::FP16) {
                log2Dest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::FP32) {
                log2Dest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT8) {
                log2Dest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT16) {
                log2Dest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT32) {
                log2Dest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT8) {
                log2Dest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT16) {
                log2Dest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT32) {
                log2Dest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
            } else {
                assert(false);
            }
        } else {
            if (destDataType == TensorDescriptor::DataType::FP16) {
                log2Dest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::FP32) {
                log2Dest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT8) {
                log2Dest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT16) {
                log2Dest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT32) {
                log2Dest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT8) {
                log2Dest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT16) {
                log2Dest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT32) {
                log2Dest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
            } else {
                assert(false);
            }
        }
    } else if (base == 10.0f) {
        if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
            if (destDataType == TensorDescriptor::DataType::FP16) {
                log10Dest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::FP32) {
                log10Dest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT8) {
                log10Dest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT16) {
                log10Dest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT32) {
                log10Dest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT8) {
                log10Dest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT16) {
                log10Dest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT32) {
                log10Dest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
            } else {
                assert(false);
            }
        } else {
            if (destDataType == TensorDescriptor::DataType::FP16) {
                log10Dest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::FP32) {
                log10Dest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT8) {
                log10Dest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT16) {
                log10Dest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT32) {
                log10Dest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT8) {
                log10Dest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT16) {
                log10Dest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT32) {
                log10Dest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
            } else {
                assert(false);
            }
        }
    } else {
        float log2ConversionFactor = getLog2ConversionFactor(base);

        if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
            if (destDataType == TensorDescriptor::DataType::FP16) {
                logXDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::FP32) {
                logXDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT8) {
                logXDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT16) {
                logXDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT32) {
                logXDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT8) {
                logXDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT16) {
                logXDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT32) {
                logXDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, log2ConversionFactor, numElements);
            } else {
                assert(false);
            }
        } else {
            if (destDataType == TensorDescriptor::DataType::FP16) {
                logXDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::FP32) {
                logXDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT8) {
                logXDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT16) {
                logXDest2B<<<gridSize, blockSize, 0, stream>>>(
                    (uint16_t *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::UINT32) {
                logXDest4B<<<gridSize, blockSize, 0, stream>>>(
                    (uint32_t *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT8) {
                logXDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT16) {
                logXDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else if (destDataType == TensorDescriptor::DataType::INT32) {
                logXDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, log2ConversionFactor, numElements);
            } else {
                assert(false);
            }
        }
    }
}

/**
 * [thisTensor] = ⌈ [argument] ⌉, elementwise
 * <div/>
 * Compute the ceil of each element in the argument tensor
 * argument must be float or half.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::ceil(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            ceilDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            ceilDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            ceilDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            ceilDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            ceilDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            ceilDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            ceilDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            ceilDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            ceilDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            ceilDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            ceilDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            ceilDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            ceilDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            ceilDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            ceilDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            ceilDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
        } else {
            assert(false);
        }
    }
}

/**
 * [thisTensor] = ⌊ [argument] ⌋, elementwise
 * <div/>
 * Compute the floor of each element in the argument tensor
 * argument must be float or half.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::floor(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            floorDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            floorDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            floorDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            floorDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            floorDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            floorDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            floorDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            floorDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            floorDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            floorDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            floorDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            floorDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            floorDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            floorDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            floorDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            floorDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
        } else {
            assert(false);
        }
    }
}

/**
 * [thisTensor] = round([argument]), elementwise
 * <div/>
 * Round to nearest integer, 0.5 rounds up.
 * argument must be float or half.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::round(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            roundDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            roundDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            roundDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            roundDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            roundDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            roundDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            roundDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            roundDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            roundDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            roundDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            roundDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            roundDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            roundDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            roundDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            roundDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            roundDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
        } else {
            assert(false);
        }
    }
}

/**
 * [thisTensor] = the integer componet of [argument], elementwise
 * <div/>
 * argument must be float or half.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::truncateFloatingPoint(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            truncateFloatingPointDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            truncateFloatingPointDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            truncateFloatingPointDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            truncateFloatingPointDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            truncateFloatingPointDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            truncateFloatingPointDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            truncateFloatingPointDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            truncateFloatingPointDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            truncateFloatingPointDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            truncateFloatingPointDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            truncateFloatingPointDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            truncateFloatingPointDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            truncateFloatingPointDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            truncateFloatingPointDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            truncateFloatingPointDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            truncateFloatingPointDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
        } else {
            assert(false);
        }
    }
}

/**
 * [thisTensor] = 1 / [argument], elementwise
 * <div/>
 * Compute the reciprocal of each element in the argument tensor
 * argument must be half. Use divide for other data types.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::reciprocal(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(argument.getDataType() == TensorDescriptor::DataType::FP16);
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        reciprocalDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        reciprocalDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT8) {
        reciprocalDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT16) {
        reciprocalDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::UINT32) {
        reciprocalDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT8) {
        reciprocalDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT16) {
        reciprocalDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::INT32) {
        reciprocalDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = √([argument]), elementwise
 * <div/>
 * Compute the square root of each element in the argument tensor
 * argument must be float or half.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::sqrt(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            sqrtDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            sqrtDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            sqrtDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            sqrtDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            sqrtDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            sqrtDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            sqrtDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            sqrtDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            sqrtDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            sqrtDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            sqrtDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            sqrtDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            sqrtDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            sqrtDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            sqrtDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            sqrtDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
        } else {
            assert(false);
        }
    }
}

/**
 * [thisTensor] = 1 / sqrt([argument]), elementwise
 * <div/>
 * Compute the reciprocal of the square root of each element in the argument tensor
 * argument must be float or half.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::reciprocalSqrt(Tensor argument, Stream stream) {
    assert(argument.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert((argument.getDataType() == TensorDescriptor::DataType::FP32 || argument.getDataType() == TensorDescriptor::DataType::FP16));
    assert(argument.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = argument.getTotalNumElements();
    void *argumentMem = argument.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (argument.getDataType() == TensorDescriptor::DataType::FP16) {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            reciprocalSqrtDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            reciprocalSqrtDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            reciprocalSqrtDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            reciprocalSqrtDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            reciprocalSqrtDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            reciprocalSqrtDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            reciprocalSqrtDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (half *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            reciprocalSqrtDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (half *)argumentMem, numElements);
        } else {
            assert(false);
        }
    } else {
        if (destDataType == TensorDescriptor::DataType::FP16) {
            reciprocalSqrtDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::FP32) {
            reciprocalSqrtDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT8) {
            reciprocalSqrtDest1B<<<gridSize, blockSize, 0, stream>>>((uint8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT16) {
            reciprocalSqrtDest2B<<<gridSize, blockSize, 0, stream>>>((uint16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::UINT32) {
            reciprocalSqrtDest4B<<<gridSize, blockSize, 0, stream>>>((uint32_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT8) {
            reciprocalSqrtDest1B<<<gridSize, blockSize, 0, stream>>>((int8_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT16) {
            reciprocalSqrtDest2B<<<gridSize, blockSize, 0, stream>>>((int16_t *)destMem, (float *)argumentMem, numElements);
        } else if (destDataType == TensorDescriptor::DataType::INT32) {
            reciprocalSqrtDest4B<<<gridSize, blockSize, 0, stream>>>((int32_t *)destMem, (float *)argumentMem, numElements);
        } else {
            assert(false);
        }
    }
}

/**
 * [thisTensor] = erf(x), elementwise
 * <div/>
 * Compute the error function: https://mathworld.wolfram.com/Erf.html
 * x must be float.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::erf(Tensor x, Stream stream) {
    assert(x.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(x.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(x.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = x.getTotalNumElements();
    void *xMem = x.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        erfDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)xMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        erfDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)xMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = erfinv(x), elementwise
 * <div/>
 * Compute the inverse error function defined as erfinv(erf(x))=x : https://www.mathworks.com/help/symbolic/erfinv.html
 * x must be float.
 * The return type may be float or half
 */
void Tensor::erfinv(Tensor x, Stream stream) {
    assert(x.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(x.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(x.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = x.getTotalNumElements();
    void *xMem = x.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        erfinvDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)xMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        erfinvDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)xMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = erfc(x), elementwise
 * <div/>
 * Compute the complementary error function: https://mathworld.wolfram.com/Erfc.html
 * x must be float.
 * there is no restriction on the data type of this destination tensor.
 */
void Tensor::erfc(Tensor x, Stream stream) {
    assert(x.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(x.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(x.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = x.getTotalNumElements();
    void *xMem = x.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        erfcDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)xMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        erfcDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)xMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = erfcinv(x), elementwise
 * <div/>
 * Compute the inverse complementary error function defined as erfcinv(erfc(x))=x :
 * https://www.mathworks.com/help/matlab/ref/erfcinv.html#bup512o-2 x must be float. The return type may be float or half
 */
void Tensor::erfcinv(Tensor x, Stream stream) {
    assert(x.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(x.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(x.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = x.getTotalNumElements();
    void *xMem = x.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        erfcinvDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)xMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        erfcinvDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)xMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = erfcx(x), elementwise
 * <div/>
 * Compute the scaled complementary error function that is equal to exp(x^2)*erfc(x): https://www.mathworks.com/help/matlab/ref/erfcx.html
 * x must be float.
 * The return type may be float or half
 */
void Tensor::erfcx(Tensor x, Stream stream) {
    assert(x.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(x.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(x.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = x.getTotalNumElements();
    void *xMem = x.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        erfcxDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)xMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        erfcxDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)xMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = gamma(x), elementwise
 * <div/>
 * Compute the gamma(x): https://mathworld.wolfram.com/GammaFunction.html
 * x must be float. The return type may be float or half
 */
void Tensor::tgamma(Tensor x, Stream stream) {
    assert(x.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(x.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(x.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = x.getTotalNumElements();
    void *xMem = x.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        tgammaDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)xMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        tgammaDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)xMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = ln(gamma(x)), elementwise
 * <div/>
 * gamma(x): https://mathworld.wolfram.com/GammaFunction.html
 * x must be float. The return type may be float or half
 */
void Tensor::lgamma(Tensor x, Stream stream) {
    assert(x.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(x.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(x.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = x.getTotalNumElements();
    void *xMem = x.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        lgammaDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)xMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        lgammaDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)xMem, numElements);
    } else {
        assert(false);
    }
}