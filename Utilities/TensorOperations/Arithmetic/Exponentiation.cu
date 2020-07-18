#include "Exponentiation.h"

template <typename SOURCE_TYPE, typename DEST_TYPE>
__global__ void exponentiation(SOURCE_TYPE *source, DEST_TYPE *dest, uint64_t numElements) {
    int element = blockIdx.x * 1024 + threadIdx.x;

    if (element >= numElements)
        return;
    dest[element] = (DEST_TYPE)expf((float)source[element]);
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = (DEST_TYPE)expf((float)source[element]);
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = (DEST_TYPE)expf((float)source[element]);
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = (DEST_TYPE)expf((float)source[element]);
}

template <typename SOURCE_TYPE, typename DEST_TYPE>
void launchExponentiation(SOURCE_TYPE *source_d, DEST_TYPE *dest_d, uint64_t numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    exponentiation<SOURCE_TYPE, DEST_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, dest_d, numElements);
}

template void launchExponentiation<half, float>(half *source_d, float *dest_d, uint64_t numElements, Stream stream);
template void launchExponentiation<float, float>(float *source_d, float *dest_d, uint64_t numElements, Stream stream);
template void launchExponentiation<float, half>(float *source_d, half *dest_d, uint64_t numElements, Stream stream);
