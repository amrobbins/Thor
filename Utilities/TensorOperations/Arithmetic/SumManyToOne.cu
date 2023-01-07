#include "SumManyToOne.h"

// 1. A block is 256 threads, each thread accumulates one element at a time, until all elements have been accumulated for one item in a
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
__global__ void sumManyToOne(
    SOURCE_TYPE *source, DEST_TYPE *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool invert, bool accumulate) {
    __shared__ float sharedBuffer[8];

    // Create 256 partial sums
    float sum = 0.0f;
    if (threadIdx.x == 0 && accumulate)
        sum = (float)dest[blockIdx.x];

    uint32_t offset = blockIdx.x * numElementsPerBatchItem;
    for (uint32_t element = threadIdx.x; element < numElementsPerBatchItem; element += 256) {
        sum += (float)source[offset + element];
    }

    // Reduce to the final single sum and write the result
    sum = smtn_warpReduce32(sum);
    if (threadIdx.x % 32 == 0)
        sharedBuffer[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float val = 0.0f;
        if (threadIdx.x < 8)
            val = sharedBuffer[threadIdx.x];
        sum = smtn_warpReduceBottom8(val);
    }
    if (threadIdx.x == 0) {
        if (invert)
            sum = 1.0f / sum;
        dest[blockIdx.x] = (DEST_TYPE)sum;
    }
}

template <typename SOURCE_TYPE, typename DEST_TYPE>
void launchSumManyToOne(SOURCE_TYPE *source_d,
                        DEST_TYPE *dest_d,
                        uint32_t numElementsPerBatchItem,
                        uint32_t batchSize,
                        bool invert,
                        bool accumulate,
                        Stream stream) {
    assert(numElementsPerBatchItem > 0);
    assert(batchSize > 0);

    dim3 blockSize(256);
    dim3 gridSize(batchSize);
    sumManyToOne<SOURCE_TYPE, DEST_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElementsPerBatchItem, batchSize, invert, accumulate);
}

template void launchSumManyToOne<half, half>(
    half *source_d, half *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool invert, bool accumulate, Stream stream);
template void launchSumManyToOne<half, float>(
    half *source_d, float *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool invert, bool accumulate, Stream stream);
template void launchSumManyToOne<float, float>(
    float *source_d, float *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool invert, bool accumulate, Stream stream);
template void launchSumManyToOne<float, half>(
    float *source_d, half *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool invert, bool accumulate, Stream stream);

template <typename SOURCE_TYPE, typename DEST_TYPE>
__global__ void sumBatch(SOURCE_TYPE *source, DEST_TYPE *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool accumulate) {
    const uint32_t elementOffset = blockIdx.x * 256 + threadIdx.x;
    if (elementOffset >= numElementsPerBatchItem)
        return;

    float buff = 0.0f;
    for (uint32_t i = 0; i < batchSize; ++i) {
        buff += (float)source[i * numElementsPerBatchItem + elementOffset];
    }
    dest[elementOffset] = (DEST_TYPE)buff;
}

template <typename SOURCE_TYPE, typename DEST_TYPE>
void launchSumBatch(
    SOURCE_TYPE *source_d, DEST_TYPE *dest_d, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool accumulate, Stream stream) {
    assert(numElementsPerBatchItem > 0);
    assert(batchSize > 0);

    dim3 blockSize(256);
    dim3 gridSize((numElementsPerBatchItem + 255) / 256);
    sumBatch<SOURCE_TYPE, DEST_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElementsPerBatchItem, batchSize, accumulate);
}

template void launchSumBatch<half, half>(
    half *source_d, half *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool accumulate, Stream stream);
template void launchSumBatch<half, float>(
    half *source_d, float *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool accumulate, Stream stream);
template void launchSumBatch<float, float>(
    float *source_d, float *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool accumulate, Stream stream);
template void launchSumBatch<float, half>(
    float *source_d, half *dest, uint32_t numElementsPerBatchItem, uint32_t batchSize, bool accumulate, Stream stream);
