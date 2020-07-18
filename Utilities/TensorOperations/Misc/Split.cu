#include "Split.h"

__device__ __forceinline__ void computeSourceIndex(long sourceFlatIndex,
                                                   long sourceIndex[],
                                                   int numDimensions,
                                                   long stridePerSourceDimension[]) {
    for (int i = 0; i < numDimensions - 1; ++i) {
        int dimensionIndex = sourceFlatIndex / stridePerSourceDimension[i];
        sourceFlatIndex -= dimensionIndex * stridePerSourceDimension[i];
        sourceIndex[i] = dimensionIndex;
    }
    sourceIndex[numDimensions - 1] = sourceFlatIndex;
}

__device__ __forceinline__ void computeDestArray(
    long axisElementIndex, long axisElementsPerDestArray[], int numDestArrays, int &destArray, long &destAxisElementIndex) {
    for (int i = 0; i < numDestArrays; ++i) {
        if (axisElementIndex < axisElementsPerDestArray[i]) {
            destArray = i;
            destAxisElementIndex = axisElementIndex;
            return;
        }
        axisElementIndex -= axisElementsPerDestArray[i];
    }
}

__device__ __forceinline__ long computeFlatIndexFromIndex(long index[], long stridePerDimension[], int numDimensions) {
    long flatIndex = 0;
    for (int i = 0; i < numDimensions; ++i)
        flatIndex += index[i] * stridePerDimension[i];
    return flatIndex;
}

__global__ void split(half *dest[],
                      half *source,
                      long numElements,
                      int numDimensions,
                      int numDestArrays,
                      int axisDimension,
                      long axisElementsPerDestArray[],
                      long stridePerSourceDimension[],
                      long stridePerDestDimension[]) {
    extern __shared__ long indexShared[];
    long *sourceIndex = &(indexShared[threadIdx.x * numDimensions]);
    long *axisElementsPerDestArrayShared = &(indexShared[256 * numDimensions]);
    long *stridePerSourceDimensionShared = &(indexShared[256 * numDimensions + numDestArrays]);
    long *stridePerDestDimensionShared = &(indexShared[256 * numDimensions + numDestArrays + numDimensions]);

    if (threadIdx.x < 32) {
        for (int destArray = threadIdx.x; destArray < numDestArrays; destArray += 32)
            axisElementsPerDestArrayShared[destArray] = axisElementsPerDestArray[destArray];
    } else if (threadIdx.x < 64) {
        for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32)
            stridePerSourceDimensionShared[dimension] = stridePerSourceDimension[dimension];
    } else if (threadIdx.x < 96) {
        for (int destArray = 0; destArray < numDestArrays; ++destArray) {
            for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32) {
                stridePerDestDimensionShared[destArray * numDimensions + dimension] =
                    stridePerDestDimension[destArray * numDimensions + dimension];
            }
        }
    }
    __syncthreads();

    long sourceFlatIndex = blockIdx.x * (256 * 16) + threadIdx.x;

#pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        if (sourceFlatIndex >= numElements)
            return;

        computeSourceIndex(sourceFlatIndex, sourceIndex, numDimensions, stridePerSourceDimensionShared);
        int destArray;
        long destAxisElementIndex;
        computeDestArray(sourceIndex[axisDimension], axisElementsPerDestArrayShared, numDestArrays, destArray, destAxisElementIndex);

        sourceIndex[axisDimension] = destAxisElementIndex;
        long destFlatIndex =
            computeFlatIndexFromIndex(sourceIndex, &(stridePerDestDimensionShared[destArray * numDimensions]), numDimensions);

        dest[destArray][destFlatIndex] = source[sourceFlatIndex];

        sourceFlatIndex += 256;
    }
}

void launchSplit(half *dest[],
                 half *source,
                 long numElements,
                 int numDimensions,
                 int numDestArrays,
                 int axisDimension,
                 long axisElementsPerDestArray[],
                 long stridePerSourceDimension[],
                 long stridePerDestDimension[],
                 Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 4095) / 4096);
    int sharedRequirement = (256 * numDimensions + numDestArrays + numDimensions + numDestArrays * numDimensions) * sizeof(long);
    split<<<gridSize, blockSize, sharedRequirement, stream.getStream()>>>(dest,
                                                                          source,
                                                                          numElements,
                                                                          numDimensions,
                                                                          numDestArrays,
                                                                          axisDimension,
                                                                          axisElementsPerDestArray,
                                                                          stridePerSourceDimension,
                                                                          stridePerDestDimension);
}
