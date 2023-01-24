#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;
using namespace std;

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void sinDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void sinDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cosDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)cosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)cosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)cosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)cosf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)cosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)cosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)cosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)cosf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cosDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)cosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)cosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)cosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)cosf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)cosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)cosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)cosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)cosf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void tanDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void tanDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cscDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cscDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void secDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void secDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / cosf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cotDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cotDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asinDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asinDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acosDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acosDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void atanDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void atanDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acscDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acscDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asecDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asecDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acosf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acotDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acotDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

/**
 * [thisTensor] = sin(r), elementwise
 * <div/>
 * Compute the sine of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::sin(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        sinDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        sinDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = cos(r), elementwise
 * <div/>
 * Compute the cosine of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::cos(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        cosDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        cosDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = tan(r), elementwise
 * <div/>
 * Compute the tangent of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::tan(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        tanDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        tanDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = sec(r), elementwise
 * <div/>
 * Compute the secant of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::csc(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        cscDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        cscDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = csc(r), elementwise
 * <div/>
 * Compute the cosecant of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::sec(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        secDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        secDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = cot(r), elementwise
 * <div/>
 * Compute the cotangent of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::cot(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        cotDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        cotDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = asin(r), elementwise
 * <div/>
 * Compute the arcsine of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::asin(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        asinDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        asinDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = acos(r), elementwise
 * <div/>
 * Compute the arccosine of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::acos(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        acosDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        acosDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = atan(r), elementwise
 * <div/>
 * Compute the arctangent of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::atan(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        atanDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        atanDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = acsc(r), elementwise
 * <div/>
 * Compute the arccosecant of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::acsc(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        acscDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        acscDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = asec(r), elementwise
 * <div/>
 * Compute the arcsecant of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::asec(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        asecDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        asecDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = acot(r), elementwise
 * <div/>
 * Compute the arccotangent of r, where r is in radians
 * r must be float. The return type may be float or half
 */
void Tensor::acot(Tensor r, Stream stream) {
    assert(r.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(r.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(r.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = r.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = r.getTotalNumElements();
    void *rMem = r.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        acotDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        acotDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}