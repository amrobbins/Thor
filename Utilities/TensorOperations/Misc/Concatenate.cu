#include "Concatenate.h"

__device__ __forceinline__ void computeDestIndex(long destFlatIndex, long destIndex[], int numDimensions, long stridePerDestDimension[]) {
    for (int i = 0; i < numDimensions - 1; ++i) {
        int dimensionIndex = destFlatIndex / stridePerDestDimension[i];
        destFlatIndex -= dimensionIndex * stridePerDestDimension[i];
        destIndex[i] = dimensionIndex;
    }
    destIndex[numDimensions - 1] = destFlatIndex;
}

__device__ __forceinline__ void computeSourceArray(
    long axisElementIndex, long axisElementsPerSourceArray[], int numSourceArrays, int &sourceArray, long &sourceAxisElementIndex) {
    for (int i = 0; i < numSourceArrays; ++i) {
        if (axisElementIndex < axisElementsPerSourceArray[i]) {
            sourceArray = i;
            sourceAxisElementIndex = axisElementIndex;
            return;
        }
        axisElementIndex -= axisElementsPerSourceArray[i];
    }
}

__device__ __forceinline__ long computeFlatIndexFromIndex(long index[], long stridePerDimension[], int numDimensions) {
    long flatIndex = 0;
    for (int i = 0; i < numDimensions; ++i)
        flatIndex += index[i] * stridePerDimension[i];
    return flatIndex;
}

__global__ void concatenate(half *dest,
                            half *source[],
                            long numElements,
                            int numDimensions,
                            int numSourceArrays,
                            int axisDimension,
                            long axisElementsPerSourceArray[],
                            long stridePerDestDimension[],
                            long stridePerSourceDimension[]) {
    extern __shared__ long indexShared[];
    long *destIndex = &(indexShared[threadIdx.x * numDimensions]);
    long *axisElementsPerSourceArrayShared = &(indexShared[256 * numDimensions]);
    long *stridePerDestDimensionShared = &(indexShared[256 * numDimensions + numSourceArrays]);
    long *stridePerSourceDimensionShared = &(indexShared[256 * numDimensions + numSourceArrays + numDimensions]);

    if (threadIdx.x < 32) {
        for (int sourceArray = threadIdx.x; sourceArray < numSourceArrays; sourceArray += 32)
            axisElementsPerSourceArrayShared[sourceArray] = axisElementsPerSourceArray[sourceArray];
    } else if (threadIdx.x < 64) {
        for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32)
            stridePerDestDimensionShared[dimension] = stridePerDestDimension[dimension];
    } else if (threadIdx.x < 96) {
        for (int sourceArray = 0; sourceArray < numSourceArrays; ++sourceArray) {
            for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32) {
                stridePerSourceDimensionShared[sourceArray * numDimensions + dimension] =
                    stridePerSourceDimension[sourceArray * numDimensions + dimension];
            }
        }
    }
    __syncthreads();

    long destFlatIndex = blockIdx.x * (256 * 16) + threadIdx.x;

#pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        if (destFlatIndex >= numElements)
            return;

        computeDestIndex(destFlatIndex, destIndex, numDimensions, stridePerDestDimensionShared);
        int sourceArray;
        long sourceAxisElementIndex;
        computeSourceArray(
            destIndex[axisDimension], axisElementsPerSourceArrayShared, numSourceArrays, sourceArray, sourceAxisElementIndex);

        destIndex[axisDimension] = sourceAxisElementIndex;
        long sourceFlatIndex =
            computeFlatIndexFromIndex(destIndex, &(stridePerSourceDimensionShared[sourceArray * numDimensions]), numDimensions);

        dest[destFlatIndex] = source[sourceArray][sourceFlatIndex];

        destFlatIndex += 256;
    }
}

void launchConcatenate(half *dest,
                       half *source[],
                       long numElements,
                       int numDimensions,
                       int numSourceArrays,
                       int axisDimension,
                       long axisElementsPerSourceArray[],
                       long stridePerDestDimension[],
                       long stridePerSourceDimension[],
                       Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 4095) / 4096);
    int sharedRequirement = (256 * numDimensions + numSourceArrays + numDimensions + numSourceArrays * numDimensions) * sizeof(long);
    concatenate<<<gridSize, blockSize, sharedRequirement, stream.getStream()>>>(dest,
                                                                                source,
                                                                                numElements,
                                                                                numDimensions,
                                                                                numSourceArrays,
                                                                                axisDimension,
                                                                                axisElementsPerSourceArray,
                                                                                stridePerDestDimension,
                                                                                stridePerSourceDimension);
}
