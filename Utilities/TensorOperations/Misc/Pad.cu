#include "Pad.h"
#include "DataTypeDispatch.h"
#include "DeepLearning/Implementation/ThorError.h"

namespace {

template <typename ELEMENT_TYPE>
__device__ ELEMENT_TYPE zeroValue() {
    return ELEMENT_TYPE(0.0f);
}

template <typename ELEMENT_TYPE>
__global__ void padKernel(ELEMENT_TYPE *dest,
                          ELEMENT_TYPE *source,
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
        bool isPadding = false;
        unsigned long runningDestIndex = destIndex;
        unsigned long sourceIndex = 0;
        for (int d = 0; d < numDimensions; ++d) {
            unsigned long strideForThisPaddedDimension = stridePerPaddedDimensionShared[d];
            unsigned long dimensionIndex = runningDestIndex / strideForThisPaddedDimension;
            runningDestIndex -= dimensionIndex * strideForThisPaddedDimension;

            if (dimensionIndex < padBeforeShared[d] || dimensionIndex > padAfterShared[d]) {
                isPadding = true;
                break;
            }

            sourceIndex += (dimensionIndex - padBeforeShared[d]) * stridePerUnpaddedDimensionShared[d];
        }
        ELEMENT_TYPE val = isPadding ? zeroValue<ELEMENT_TYPE>() : source[sourceIndex];
        dest[destIndex] = val;

        destIndex += 256;
    }
}

template <typename ELEMENT_TYPE>
void launchPadTyped(void *dest_d,
                    void *source_d,
                    unsigned long numDestElements,
                    unsigned int numDimensions,
                    unsigned long stridePerPaddedDimension_d[],
                    unsigned long stridePerUnpaddedDimension_d[],
                    unsigned int padBefore_d[],
                    unsigned int padAfter_d[],
                    Stream stream) {
    // in place is not supported
    THOR_THROW_IF_FALSE(dest_d != source_d);

    dim3 blockSize(256);
    dim3 gridSize((numDestElements + 2047) / 2048);

    int sharedRequirement = 4 * numDimensions * sizeof(unsigned long);
    padKernel<ELEMENT_TYPE><<<gridSize, blockSize, sharedRequirement, stream>>>(reinterpret_cast<ELEMENT_TYPE *>(dest_d),
                                                                                reinterpret_cast<ELEMENT_TYPE *>(source_d),
                                                                                numDestElements,
                                                                                numDimensions,
                                                                                stridePerPaddedDimension_d,
                                                                                stridePerUnpaddedDimension_d,
                                                                                padBefore_d,
                                                                                padAfter_d);
}

struct LaunchPadFunctor {
    void *dest_d;
    void *source_d;
    unsigned long numDestElements;
    unsigned int numDimensions;
    unsigned long *stridePerPaddedDimension_d;
    unsigned long *stridePerUnpaddedDimension_d;
    unsigned int *padBefore_d;
    unsigned int *padAfter_d;
    Stream stream;

    template <typename ELEMENT_TYPE>
    void operator()() const {
        launchPadTyped<ELEMENT_TYPE>(dest_d,
                                     source_d,
                                     numDestElements,
                                     numDimensions,
                                     stridePerPaddedDimension_d,
                                     stridePerUnpaddedDimension_d,
                                     padBefore_d,
                                     padAfter_d,
                                     stream);
    }
};

}  // namespace

void launchPad(void *dest_d,
               void *source_d,
               unsigned long numDestElements,
               unsigned int numDimensions,
               unsigned long stridePerPaddedDimension_d[],
               unsigned long stridePerUnpaddedDimension_d[],
               unsigned int padBefore_d[],
               unsigned int padAfter_d[],
               ThorImplementation::DataType dataType,
               Stream stream) {
    ThorImplementation::MiscTensorOperationSupport::dispatchTensorDataType(dataType,
                                                                           LaunchPadFunctor{dest_d,
                                                                                            source_d,
                                                                                            numDestElements,
                                                                                            numDimensions,
                                                                                            stridePerPaddedDimension_d,
                                                                                            stridePerUnpaddedDimension_d,
                                                                                            padBefore_d,
                                                                                            padAfter_d,
                                                                                            stream});
}
