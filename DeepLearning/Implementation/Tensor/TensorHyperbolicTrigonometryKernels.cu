#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;
using namespace std;

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void sinhDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void sinhDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)sinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)sinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)sinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)sinhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void coshDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)coshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)coshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)coshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)coshf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)coshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)coshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)coshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)coshf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void coshDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)coshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)coshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)coshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)coshf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)coshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)coshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)coshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)coshf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void tanhDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void tanhDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)tanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)tanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)tanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)tanhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cschDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cschDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / sinhf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void sechDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void sechDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / coshf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cothDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[3]));
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void cothDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[0]));
    destBuffer[1] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[1]));
    destBuffer[2] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[2]));
    destBuffer[3] = (DEST_DATA_TYPE)(1.0f / tanhf(argumentBuffer[3]));
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asinhDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asinhDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acoshDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acoshDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void atanhDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void atanhDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acschDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acschDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)asinhf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asechDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void asechDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)acoshf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acothDest2B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[3]);
    ((float2 *)dest)[offset4Elements] = ((float2 *)destBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
// Note that this kernel is memory bandwidth bound
template <typename DEST_DATA_TYPE, typename SOURCE_DATA_TYPE>
__global__ void acothDest4B(DEST_DATA_TYPE *dest, SOURCE_DATA_TYPE *argument, uint64_t numElements) {
    uint64_t offset = blockIdx.x * 2048 + 256 * (threadIdx.x / 32) + (threadIdx.x % 32) * 4;
    if (offset >= numElements)
        return;
    uint64_t offset4Elements = offset >> 2;

    float argumentBuffer[4];
    DEST_DATA_TYPE destBuffer[4];

    // Note: all tensors end on 16 byte boundary, here I don't want to read past the end of base and argument
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];

    offset += 128;
    if (offset >= numElements)
        return;
    offset4Elements = offset >> 2;
    ((float4 *)argumentBuffer)[0] = ((float4 *)argument)[offset4Elements];
    destBuffer[0] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[0]);
    destBuffer[1] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[1]);
    destBuffer[2] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[2]);
    destBuffer[3] = (DEST_DATA_TYPE)atanhf(1.0f / argumentBuffer[3]);
    ((float4 *)dest)[offset4Elements] = ((float4 *)destBuffer)[0];
}

/**
 * [thisTensor] = sinh([radians]), elementwise
 * <div/>
 * Compute the hyperbolic sine of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::sinh(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        sinhDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        sinhDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = cosh([radians]), elementwise
 * <div/>
 * Compute the hyperbolic cosine of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::cosh(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        coshDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        coshDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = tanh([radians]), elementwise
 * <div/>
 * Compute the hyperbolic tangent of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::tanh(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        tanhDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        tanhDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = csch([radians]), elementwise
 * <div/>
 * Compute the hyperbolic cosecant of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::csch(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        cschDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        cschDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = sec([radians]), elementwise
 * <div/>
 * Compute the hyperbolic secant of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::sech(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        sechDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        sechDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = cot([radians]), elementwise
 * <div/>
 * Compute the hyperbolic cotangent of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::coth(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        cothDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        cothDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = asin([radians]), elementwise
 * <div/>
 * Compute the arcsine of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::asinh(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        asinhDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        asinhDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = acos([radians]), elementwise
 * <div/>
 * Compute the arccosine of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::acosh(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        acoshDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        acoshDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = atan([radians]), elementwise
 * <div/>
 * Compute the arctangent of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::atanh(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        atanhDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        atanhDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = acsc([radians]), elementwise
 * <div/>
 * Compute the arccosecant of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::acsch(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        acschDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        acschDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = asec([radians]), elementwise
 * <div/>
 * Compute the arcsecant of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::asech(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        asechDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        asechDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}

/**
 * [thisTensor] = acot([radians]), elementwise
 * <div/>
 * Compute the arccotangent of radians
 * radians must be float. The return type may be float or half
 */
void Tensor::acoth(Tensor radians, Stream stream) {
    assert(radians.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(radians.getDataType() == TensorDescriptor::DataType::FP32);
    assert(getDataType() == TensorDescriptor::DataType::FP32 || getDataType() == TensorDescriptor::DataType::FP16);
    assert(radians.getTotalNumElements() == getTotalNumElements());

    TensorDescriptor::DataType sourceDataType = radians.getDataType();
    TensorDescriptor::DataType destDataType = getDataType();
    uint64_t numElements = radians.getTotalNumElements();
    void *rMem = radians.getMemPtr();
    void *destMem = getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((numElements + 2047) / 2048);
    if (destDataType == TensorDescriptor::DataType::FP16) {
        acothDest2B<<<gridSize, blockSize, 0, stream>>>((half *)destMem, (float *)rMem, numElements);
    } else if (destDataType == TensorDescriptor::DataType::FP32) {
        acothDest4B<<<gridSize, blockSize, 0, stream>>>((float *)destMem, (float *)rMem, numElements);
    } else {
        assert(false);
    }
}