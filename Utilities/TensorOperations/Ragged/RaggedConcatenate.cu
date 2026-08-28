#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cstdint>
#include <stdexcept>

namespace {

__device__ __forceinline__ uint64_t activePackedRows(
    const void *offsets, unsigned long offsetsElementSizeBytes, uint64_t batchSize) {
    if (offsetsElementSizeBytes == sizeof(uint32_t)) {
        return static_cast<uint64_t>(reinterpret_cast<const uint32_t *>(offsets)[batchSize]);
    }
    return reinterpret_cast<const uint64_t *>(offsets)[batchSize];
}

__device__ __forceinline__ void computeIndex(
    long flatIndex, long index[], int numDimensions, const long stridePerDimension[]) {
    for (int i = 0; i < numDimensions - 1; ++i) {
        const long dimensionIndex = flatIndex / stridePerDimension[i];
        flatIndex -= dimensionIndex * stridePerDimension[i];
        index[i] = dimensionIndex;
    }
    index[numDimensions - 1] = flatIndex;
}

__device__ __forceinline__ void selectArray(long axisElementIndex,
                                             const long axisElementsPerArray[],
                                             int numArrays,
                                             int &arrayIndex,
                                             long &arrayAxisElementIndex) {
    for (int i = 0; i < numArrays; ++i) {
        if (axisElementIndex < axisElementsPerArray[i]) {
            arrayIndex = i;
            arrayAxisElementIndex = axisElementIndex;
            return;
        }
        axisElementIndex -= axisElementsPerArray[i];
    }
}

__device__ __forceinline__ long computeFlatIndex(
    const long index[], const long stridePerDimension[], int numDimensions) {
    long flatIndex = 0;
    for (int i = 0; i < numDimensions; ++i) flatIndex += index[i] * stridePerDimension[i];
    return flatIndex;
}

__global__ void raggedConcatenate(unsigned char *dest,
                                  unsigned char *source[],
                                  unsigned long elementSizeBytes,
                                  long fullCapacityNumElements,
                                  uint64_t elementsPerOutputValue,
                                  int numDimensions,
                                  int numSourceArrays,
                                  int axisDimension,
                                  long axisElementsPerSourceArray[],
                                  long stridePerDestDimension[],
                                  long stridePerSourceDimension[],
                                  const void *offsets,
                                  unsigned long offsetsElementSizeBytes,
                                  uint64_t batchSize) {
    extern __shared__ long shared[];
    long *destIndex = &(shared[threadIdx.x * numDimensions]);
    long *axisElementsShared = &(shared[256 * numDimensions]);
    long *destStridesShared = &(shared[256 * numDimensions + numSourceArrays]);
    long *sourceStridesShared = &(shared[256 * numDimensions + numSourceArrays + numDimensions]);

    if (threadIdx.x < 32) {
        for (int sourceArray = threadIdx.x; sourceArray < numSourceArrays; sourceArray += 32)
            axisElementsShared[sourceArray] = axisElementsPerSourceArray[sourceArray];
    } else if (threadIdx.x < 64) {
        for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32)
            destStridesShared[dimension] = stridePerDestDimension[dimension];
    } else if (threadIdx.x < 96) {
        for (int sourceArray = 0; sourceArray < numSourceArrays; ++sourceArray) {
            for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32) {
                sourceStridesShared[sourceArray * numDimensions + dimension] =
                    stridePerSourceDimension[sourceArray * numDimensions + dimension];
            }
        }
    }
    __syncthreads();

    const uint64_t activeRows = activePackedRows(offsets, offsetsElementSizeBytes, batchSize);
    const uint64_t activeNumElements = activeRows * elementsPerOutputValue;
    long destFlatIndex = blockIdx.x * (256 * 16) + threadIdx.x;

#pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        if (destFlatIndex >= fullCapacityNumElements || static_cast<uint64_t>(destFlatIndex) >= activeNumElements) return;

        computeIndex(destFlatIndex, destIndex, numDimensions, destStridesShared);
        int sourceArray = 0;
        long sourceAxisElementIndex = 0;
        selectArray(destIndex[axisDimension], axisElementsShared, numSourceArrays, sourceArray, sourceAxisElementIndex);
        destIndex[axisDimension] = sourceAxisElementIndex;
        const long sourceFlatIndex = computeFlatIndex(
            destIndex, &(sourceStridesShared[sourceArray * numDimensions]), numDimensions);

        unsigned char *destElement = dest + static_cast<unsigned long>(destFlatIndex) * elementSizeBytes;
        unsigned char *sourceElement = source[sourceArray] + static_cast<unsigned long>(sourceFlatIndex) * elementSizeBytes;
        for (unsigned long byte = 0; byte < elementSizeBytes; ++byte) destElement[byte] = sourceElement[byte];
        destFlatIndex += 256;
    }
}

__global__ void raggedSplit(unsigned char *dest[],
                            unsigned char *source,
                            unsigned long elementSizeBytes,
                            long fullCapacityNumElements,
                            uint64_t elementsPerSourceValue,
                            int numDimensions,
                            int numDestArrays,
                            int axisDimension,
                            long axisElementsPerDestArray[],
                            long stridePerSourceDimension[],
                            long stridePerDestDimension[],
                            const void *offsets,
                            unsigned long offsetsElementSizeBytes,
                            uint64_t batchSize) {
    extern __shared__ long shared[];
    long *sourceIndex = &(shared[threadIdx.x * numDimensions]);
    long *axisElementsShared = &(shared[256 * numDimensions]);
    long *sourceStridesShared = &(shared[256 * numDimensions + numDestArrays]);
    long *destStridesShared = &(shared[256 * numDimensions + numDestArrays + numDimensions]);

    if (threadIdx.x < 32) {
        for (int destArray = threadIdx.x; destArray < numDestArrays; destArray += 32)
            axisElementsShared[destArray] = axisElementsPerDestArray[destArray];
    } else if (threadIdx.x < 64) {
        for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32)
            sourceStridesShared[dimension] = stridePerSourceDimension[dimension];
    } else if (threadIdx.x < 96) {
        for (int destArray = 0; destArray < numDestArrays; ++destArray) {
            for (int dimension = threadIdx.x % 32; dimension < numDimensions; dimension += 32) {
                destStridesShared[destArray * numDimensions + dimension] =
                    stridePerDestDimension[destArray * numDimensions + dimension];
            }
        }
    }
    __syncthreads();

    const uint64_t activeRows = activePackedRows(offsets, offsetsElementSizeBytes, batchSize);
    const uint64_t activeNumElements = activeRows * elementsPerSourceValue;
    long sourceFlatIndex = blockIdx.x * (256 * 16) + threadIdx.x;

#pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        if (sourceFlatIndex >= fullCapacityNumElements || static_cast<uint64_t>(sourceFlatIndex) >= activeNumElements) return;

        computeIndex(sourceFlatIndex, sourceIndex, numDimensions, sourceStridesShared);
        int destArray = 0;
        long destAxisElementIndex = 0;
        selectArray(sourceIndex[axisDimension], axisElementsShared, numDestArrays, destArray, destAxisElementIndex);
        sourceIndex[axisDimension] = destAxisElementIndex;
        const long destFlatIndex = computeFlatIndex(
            sourceIndex, &(destStridesShared[destArray * numDimensions]), numDimensions);

        unsigned char *destElement = dest[destArray] + static_cast<unsigned long>(destFlatIndex) * elementSizeBytes;
        unsigned char *sourceElement = source + static_cast<unsigned long>(sourceFlatIndex) * elementSizeBytes;
        for (unsigned long byte = 0; byte < elementSizeBytes; ++byte) destElement[byte] = sourceElement[byte];
        sourceFlatIndex += 256;
    }
}

void validateOffsetsElementSize(std::size_t offsetsElementSizeBytes) {
    if (offsetsElementSizeBytes != sizeof(uint32_t) && offsetsElementSizeBytes != sizeof(uint64_t)) {
        throw std::invalid_argument("Ragged concatenate requires UINT32 or UINT64 offsets storage.");
    }
}

}  // namespace

void launchRaggedConcatenate(void *dest,
                             void *source[],
                             std::size_t elementSizeBytes,
                             long fullCapacityNumElements,
                             uint64_t elementsPerOutputValue,
                             int numDimensions,
                             int numSourceArrays,
                             int axisDimension,
                             long axisElementsPerSourceArray[],
                             long stridePerDestDimension[],
                             long stridePerSourceDimension[],
                             const void *offsets,
                             std::size_t offsetsElementSizeBytes,
                             uint64_t batchSize,
                             Stream stream) {
    validateOffsetsElementSize(offsetsElementSizeBytes);
    ScopedGpu scopedGpu(stream.getGpuNum());
    dim3 blockSize(256);
    dim3 gridSize((fullCapacityNumElements + 4095) / 4096);
    const int sharedRequirement =
        (256 * numDimensions + numSourceArrays + numDimensions + numSourceArrays * numDimensions) * sizeof(long);
    raggedConcatenate<<<gridSize, blockSize, sharedRequirement, stream.getStream()>>>(
        static_cast<unsigned char *>(dest),
        reinterpret_cast<unsigned char **>(source),
        static_cast<unsigned long>(elementSizeBytes),
        fullCapacityNumElements,
        elementsPerOutputValue,
        numDimensions,
        numSourceArrays,
        axisDimension,
        axisElementsPerSourceArray,
        stridePerDestDimension,
        stridePerSourceDimension,
        offsets,
        static_cast<unsigned long>(offsetsElementSizeBytes),
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}

void launchRaggedSplit(void *dest[],
                       void *source,
                       std::size_t elementSizeBytes,
                       long fullCapacityNumElements,
                       uint64_t elementsPerSourceValue,
                       int numDimensions,
                       int numDestArrays,
                       int axisDimension,
                       long axisElementsPerDestArray[],
                       long stridePerSourceDimension[],
                       long stridePerDestDimension[],
                       const void *offsets,
                       std::size_t offsetsElementSizeBytes,
                       uint64_t batchSize,
                       Stream stream) {
    validateOffsetsElementSize(offsetsElementSizeBytes);
    ScopedGpu scopedGpu(stream.getGpuNum());
    dim3 blockSize(256);
    dim3 gridSize((fullCapacityNumElements + 4095) / 4096);
    const int sharedRequirement =
        (256 * numDimensions + numDestArrays + numDimensions + numDestArrays * numDimensions) * sizeof(long);
    raggedSplit<<<gridSize, blockSize, sharedRequirement, stream.getStream()>>>(
        reinterpret_cast<unsigned char **>(dest),
        static_cast<unsigned char *>(source),
        static_cast<unsigned long>(elementSizeBytes),
        fullCapacityNumElements,
        elementsPerSourceValue,
        numDimensions,
        numDestArrays,
        axisDimension,
        axisElementsPerDestArray,
        stridePerSourceDimension,
        stridePerDestDimension,
        offsets,
        static_cast<unsigned long>(offsetsElementSizeBytes),
        batchSize);
    CUDA_CHECK(cudaGetLastError());
}
