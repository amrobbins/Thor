#include "SumManyToOne.h"

// 1. A block is 256 threads, each thread accumulates one element at a time, untill all elements have been accumulated for one item in a
// batch
// 2. A reduction is performed across these 256 threads, resulting in one sum per batch, which is the summation of every element in that
// batch. batchSize blocks are launched to handle the whole batch

__device__ __forceinline__ float smtn_warpReduce32(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0x0000ffff, val, 8);
    val += __shfl_down_sync(0x000000ff, val, 4);
    val += __shfl_down_sync(0x0000000f, val, 2);
    return val + __shfl_down_sync(0x00003, val, 1);
}

__device__ __forceinline__ float smtn_warpReduceBottom8(float val) {
    val += __shfl_down_sync(0x000000ff, val, 4);
    val += __shfl_down_sync(0x0000000f, val, 2);
    return val + __shfl_down_sync(0x00003, val, 1);
}

template <typename SOURCE_TYPE, typename DEST_TYPE>
__global__ void sumManyToOne(SOURCE_TYPE *source, DEST_TYPE *dest, uint32_t numElementsPerBatch, uint32_t batchSize, bool invert) {
    __shared__ float sharedBuffer[8];

    // Create 256 partial sums
    float sum = 0.0f;
    uint32_t offset = blockIdx.x * numElementsPerBatch;
    for (uint32_t element = threadIdx.x; element < numElementsPerBatch; element += 256) {
        sum += (float)source[offset + element];
    }

    // Reduce to the final single sum and write the result
    sum = smtn_warpReduce32(sum);
    if (threadIdx.x % 32 == 0)
        sharedBuffer[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32)
        sum = smtn_warpReduceBottom8(sharedBuffer[threadIdx.x]);
    if (threadIdx.x == 0) {
        if (invert)
            sum = 1.0f / sum;
        dest[blockIdx.x] = (DEST_TYPE)sum;
    }
}

template <typename SOURCE_TYPE, typename DEST_TYPE>
void launchSumManyToOne(
    SOURCE_TYPE *source_d, DEST_TYPE *dest_d, uint32_t numElementsPerBatch, uint32_t batchSize, bool invert, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize(batchSize);
    assert(numElementsPerBatch > 0);
    sumManyToOne<SOURCE_TYPE, DEST_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElementsPerBatch, batchSize, invert);
}

template void launchSumManyToOne<half, float>(
    half *source_d, float *dest, uint32_t numElementsPerBatch, uint32_t batchSize, bool invert, Stream stream);
template void launchSumManyToOne<float, float>(
    float *source_d, float *dest, uint32_t numElementsPerBatch, uint32_t batchSize, bool invert, Stream stream);
template void launchSumManyToOne<float, half>(
    float *source_d, half *dest, uint32_t numElementsPerBatch, uint32_t batchSize, bool invert, Stream stream);
