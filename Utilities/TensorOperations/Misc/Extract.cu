#include "Extract.h"

__global__ void extract(half *dest,
                        half *source,
                        unsigned long numDestElements,
                        unsigned int numDimensions,
                        unsigned long stridePerPaddedDimension[],
                        unsigned long stridePerUnpaddedDimension[],
                        unsigned int padBefore[],
                        unsigned int padAfter[]) {
    extern __shared__ unsigned long dynamicShared[];

    unsigned long *stridePerPaddedDimensionShared = dynamicShared;
    unsigned long *stridePerUnpaddedDimensionShared = &(dynamicShared[numDimensions]);
    unsigned int *padBeforeShared = (unsigned int *)&(dynamicShared[2 * numDimensions]);
    unsigned int *padAfterShared = (unsigned int *)&(dynamicShared[3 * numDimensions]);

    if (threadIdx.x < 32) {
        for (int d = threadIdx.x; d < numDimensions; d += 32)
            stridePerPaddedDimensionShared[d] = stridePerPaddedDimension[d];
    } else if (threadIdx.x < 64) {
        for (int d = threadIdx.x % 32; d < numDimensions; d += 32)
            stridePerUnpaddedDimensionShared[d] = stridePerUnpaddedDimension[d];
    } else if (threadIdx.x < 96) {
        for (int d = threadIdx.x % 32; d < numDimensions; d += 32)
            padBeforeShared[d] = padBefore[d];
    } else if (threadIdx.x < 128) {
        for (int d = threadIdx.x % 32; d < numDimensions; d += 32)
            padAfterShared[d] = padAfter[d];
    }
    __syncthreads();

    unsigned long destIndex = blockIdx.x * 256 * 8 + threadIdx.x;
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        if (destIndex >= numDestElements)
            return;
        unsigned long runningDestIndex = destIndex;
        unsigned long sourceIndex = 0;
        for (int d = 0; d < numDimensions; ++d) {
            unsigned long strideForThisUnpaddedDimension = stridePerUnpaddedDimensionShared[d];
            unsigned long dimensionIndex = runningDestIndex / strideForThisUnpaddedDimension;
            runningDestIndex -= dimensionIndex * strideForThisUnpaddedDimension;

            sourceIndex += (dimensionIndex + padBeforeShared[d]) * stridePerPaddedDimensionShared[d];
        }
        dest[destIndex] = source[sourceIndex];

        destIndex += 256;
    }
}

void launchExtract(half *dest_d,
                   half *source_d,
                   unsigned long numDestElements,
                   unsigned int numDimensions,
                   unsigned long stridePerPaddedDimension_d[],
                   unsigned long stridePerUnpaddedDimension_d[],
                   unsigned int padBefore_d[],
                   unsigned int padAfter_d[],
                   Stream stream) {
    // in place is not supported
    assert(dest_d != source_d);

    dim3 blockSize(256);
    dim3 gridSize((numDestElements + 2047) / 2048);

    int sharedRequirement = 4 * numDimensions * sizeof(unsigned long);
    extract<<<gridSize, blockSize, sharedRequirement, stream>>>(dest_d,
                                                                source_d,
                                                                numDestElements,
                                                                numDimensions,
                                                                stridePerPaddedDimension_d,
                                                                stridePerUnpaddedDimension_d,
                                                                padBefore_d,
                                                                padAfter_d);
}
