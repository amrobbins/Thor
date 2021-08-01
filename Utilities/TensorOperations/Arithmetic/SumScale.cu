#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

// Note: this kernel ends up being very important to performance, it should be optimized as much as possible.

template <typename DEST_TYPE, typename NO_SCALE_SOURCE_TYPE, typename SCALE_SOURCE_TYPE, typename SCALE_TYPE>
__global__ void sumScale(DEST_TYPE *dest,
                         NO_SCALE_SOURCE_TYPE *noScaleSource,
                         SCALE_SOURCE_TYPE *scaleSource,
                         SCALE_TYPE scale,
                         uint64_t numElements,
                         uint32_t iterationsPerBlock) {
    uint64_t element = blockIdx.x * (256 * iterationsPerBlock) + threadIdx.x;
#pragma unroll 2
    for (int i = 0; i < iterationsPerBlock; ++i) {
        if (element >= numElements)
            return;
        dest[element] = __fmaf_ieee_rn((float)scale, (float)scaleSource[element], (float)noScaleSource[element]);

        element += 256;
    }
}

__global__ void sumScaleHalfSourceDest(
    half *dest, half *noScaleSource, float *scaleSource, float scale, uint64_t numElements, uint32_t iterationsPerBlock) {
    half2 noScaleSourceBuffer;
    float2 scaleSourceBuffer;
    half2 destBuffer;

    uint64_t element = blockIdx.x * (512 * iterationsPerBlock) + 2 * threadIdx.x;
#pragma unroll 2
    for (int i = 0; i < iterationsPerBlock; ++i) {
        if (element >= numElements)
            return;
        noScaleSourceBuffer = ((half2 *)noScaleSource)[element / 2];
        scaleSourceBuffer = ((float2 *)scaleSource)[element / 2];
        destBuffer.x = (half)__fmaf_ieee_rn(scale, scaleSourceBuffer.x, (float)noScaleSourceBuffer.x);
        destBuffer.y = (half)__fmaf_ieee_rn(scale, scaleSourceBuffer.y, (float)noScaleSourceBuffer.y);

        if (element == numElements - 1)
            dest[element] = destBuffer.x;
        else
            ((half2 *)dest)[element / 2] = destBuffer;

        element += 512;
    }
}

__global__ void sumScaleHalfSourceDestScaleSource(
    half *dest, half *noScaleSource, half *scaleSource, float scale, uint64_t numElements, uint32_t iterationsPerBlock) {
    half2 noScaleSourceBuffer;
    half2 scaleSourceBuffer;
    half2 destBuffer;

    uint64_t element = blockIdx.x * (512 * iterationsPerBlock) + 2 * threadIdx.x;
#pragma unroll 2
    for (int i = 0; i < iterationsPerBlock; ++i) {
        if (element >= numElements)
            return;
        noScaleSourceBuffer = ((half2 *)noScaleSource)[element / 2];
        scaleSourceBuffer = ((half2 *)scaleSource)[element / 2];
        destBuffer.x = (half)__fmaf_ieee_rn(scale, (float)scaleSourceBuffer.x, (float)noScaleSourceBuffer.x);
        destBuffer.y = (half)__fmaf_ieee_rn(scale, (float)scaleSourceBuffer.y, (float)noScaleSourceBuffer.y);

        if (element == numElements - 1)
            dest[element] = destBuffer.x;
        else
            ((half2 *)dest)[element / 2] = destBuffer;

        element += 512;
    }
}

__global__ void sumScaleHalfAll(
    half *dest, half *noScaleSource, half *scaleSource, half scale, uint64_t numElements, uint32_t iterationsPerBlock) {
    half2 noScaleSourceBuffer;
    half2 scaleSourceBuffer;
    half2 destBuffer;
    half2 scaleH2 = __half2half2(scale);

    uint64_t element = blockIdx.x * (512 * iterationsPerBlock) + 2 * threadIdx.x;
#pragma unroll 2
    for (int i = 0; i < iterationsPerBlock; ++i) {
        if (element >= numElements)
            return;
        noScaleSourceBuffer = ((half2 *)noScaleSource)[element / 2];
        scaleSourceBuffer = ((half2 *)scaleSource)[element / 2];
        destBuffer = __hfma2(scaleH2, scaleSourceBuffer, noScaleSourceBuffer);

        if (element == numElements - 1)
            dest[element] = destBuffer.x;
        else
            ((half2 *)dest)[element / 2] = destBuffer;

        element += 512;
    }
}

template <typename DEST_TYPE, typename NO_SCALE_SOURCE_TYPE, typename SCALE_SOURCE_TYPE, typename SCALE_TYPE>
void launchSumScale(DEST_TYPE result_d[],
                    NO_SCALE_SOURCE_TYPE noScaleSource[],
                    SCALE_SOURCE_TYPE scaleSource[],
                    SCALE_TYPE scale,
                    uint64_t numElements,
                    Stream stream) {
    uint64_t elementsPerBlock = (numElements + 2047) / 2048;
    if (elementsPerBlock < 256)
        elementsPerBlock = 256;
    uint32_t iterationsPerBlock = (elementsPerBlock + 255) / 256;
    uint32_t numBlocks = (numElements + 255) / elementsPerBlock;

    dim3 blockSize(256);
    dim3 gridSize(numBlocks);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sumScale<<<gridSize, blockSize, 0, stream.getStream()>>>(result_d, noScaleSource, scaleSource, scale, numElements, iterationsPerBlock);
}

void launchSumScaleHalfSourceDest(
    half result_d[], half noScaleSource[], float scaleSource[], float scale, uint64_t numElements, Stream stream) {
    uint64_t elementsPerBlock = (numElements + 2047) / 2048;
    if (elementsPerBlock < 512)
        elementsPerBlock = 512;
    uint32_t iterationsPerBlock = (elementsPerBlock + 511) / 512;
    uint32_t numBlocks = (numElements + 511) / elementsPerBlock;

    dim3 blockSize(256);
    dim3 gridSize(numBlocks);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sumScaleHalfSourceDest<<<gridSize, blockSize, 0, stream.getStream()>>>(
        result_d, noScaleSource, scaleSource, scale, numElements, iterationsPerBlock);
}

void launchSumScaleHalfSourceDestScaleSource(
    half result_d[], half noScaleSource[], half scaleSource[], float scale, uint64_t numElements, Stream stream) {
    uint64_t elementsPerBlock = (numElements + 2047) / 2048;
    if (elementsPerBlock < 512)
        elementsPerBlock = 512;
    uint32_t iterationsPerBlock = (elementsPerBlock + 511) / 512;
    uint32_t numBlocks = (numElements + 511) / elementsPerBlock;

    dim3 blockSize(256);
    dim3 gridSize(numBlocks);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sumScaleHalfSourceDestScaleSource<<<gridSize, blockSize, 0, stream.getStream()>>>(
        result_d, noScaleSource, scaleSource, scale, numElements, iterationsPerBlock);
}

void launchSumScaleHalfAll(half result_d[], half noScaleSource[], half scaleSource[], half scale, uint64_t numElements, Stream stream) {
    uint64_t elementsPerBlock = (numElements + 2047) / 2048;
    if (elementsPerBlock < 512)
        elementsPerBlock = 512;
    uint32_t iterationsPerBlock = (elementsPerBlock + 511) / 512;
    uint32_t numBlocks = (numElements + 511) / elementsPerBlock;

    dim3 blockSize(256);
    dim3 gridSize(numBlocks);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sumScaleHalfAll<<<gridSize, blockSize, 0, stream.getStream()>>>(
        result_d, noScaleSource, scaleSource, scale, numElements, iterationsPerBlock);
}

// float math, half result
template void launchSumScale<half, float, half, float>(
    half result_d[], float noScaleSource[], half scaleSource[], float scale, uint64_t numElements, Stream stream);
template void launchSumScale<half, float, float, float>(
    half result_d[], float noScaleSource[], float scaleSource[], float scale, uint64_t numElements, Stream stream);

// float math, float result
template void launchSumScale<float, half, half, float>(
    float result_d[], half noScaleSource[], half scaleSource[], float scale, uint64_t numElements, Stream stream);
template void launchSumScale<float, half, float, float>(
    float result_d[], half noScaleSource[], float scaleSource[], float scale, uint64_t numElements, Stream stream);
template void launchSumScale<float, float, half, float>(
    float result_d[], float noScaleSource[], half scaleSource[], float scale, uint64_t numElements, Stream stream);
template void launchSumScale<float, float, float, float>(
    float result_d[], float noScaleSource[], float scaleSource[], float scale, uint64_t numElements, Stream stream);

// The other supported variants are the non-template case optimized kernels.
