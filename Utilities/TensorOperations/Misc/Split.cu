#include "Split.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

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

__global__ void split(unsigned char *dest[],
                      unsigned char *source,
                      unsigned long elementSizeBytes,
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

        unsigned char *destElement = dest[destArray] + static_cast<unsigned long>(destFlatIndex) * elementSizeBytes;
        unsigned char *sourceElement = source + static_cast<unsigned long>(sourceFlatIndex) * elementSizeBytes;
        for (unsigned long byte = 0; byte < elementSizeBytes; ++byte)
            destElement[byte] = sourceElement[byte];

        sourceFlatIndex += 256;
    }
}

void launchSplit(void *dest[],
                 void *source,
                 std::size_t elementSizeBytes,
                 long numElements,
                 int numDimensions,
                 int numDestArrays,
                 int axisDimension,
                 long axisElementsPerDestArray[],
                 long stridePerSourceDimension[],
                 long stridePerDestDimension[],
                 Stream stream) {
    ScopedGpu scopedGpu(stream.getGpuNum());

    dim3 blockSize(256);
    dim3 gridSize((numElements + 4095) / 4096);
    int sharedRequirement = (256 * numDimensions + numDestArrays + numDimensions + numDestArrays * numDimensions) * sizeof(long);
    split<<<gridSize, blockSize, sharedRequirement, stream.getStream()>>>(reinterpret_cast<unsigned char **>(dest),
                                                                          static_cast<unsigned char *>(source),
                                                                          static_cast<unsigned long>(elementSizeBytes),
                                                                          numElements,
                                                                          numDimensions,
                                                                          numDestArrays,
                                                                          axisDimension,
                                                                          axisElementsPerDestArray,
                                                                          stridePerSourceDimension,
                                                                          stridePerDestDimension);
    CUDA_CHECK(cudaGetLastError());
}
