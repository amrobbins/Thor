#include "Scale.h"

template <typename SOURCE_TYPE, typename SCALAR_TYPE, typename DEST_TYPE>
__global__ void scale(SOURCE_TYPE *source, SCALAR_TYPE scalar, DEST_TYPE *dest, uint64_t numElements) {
    int element = blockIdx.x * 1024 + threadIdx.x;
    if (element < numElements)
        dest[element] = ((DEST_TYPE)source[element]) * (DEST_TYPE)scalar;

    element += 256;
    if (element < numElements)
        dest[element] = ((DEST_TYPE)source[element]) * (DEST_TYPE)scalar;

    element += 256;
    if (element < numElements)
        dest[element] = ((DEST_TYPE)source[element]) * (DEST_TYPE)scalar;

    element += 256;
    if (element < numElements)
        dest[element] = ((DEST_TYPE)source[element]) * (DEST_TYPE)scalar;
}

template <typename SOURCE_TYPE, typename SCALAR_TYPE, typename DEST_TYPE>
void launchScale(SOURCE_TYPE *source_d, SCALAR_TYPE scalar_h, DEST_TYPE *dest_d, uint64_t numElements, Stream stream) {
    dim3 blockSize(1024);
    dim3 gridSize((numElements + 1023) / 1024);
    scale<SOURCE_TYPE, SCALAR_TYPE, DEST_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, scalar_h, dest_d, numElements);
}

template void launchScale<half, half, float>(half *source_d, half scalar_h, float *dest_d, uint64_t numElements, Stream stream);
template void launchScale<float, half, float>(float *source_d, half scalar_h, float *dest_d, uint64_t numElements, Stream stream);
template void launchScale<half, float, float>(half *source_d, float scalar_h, float *dest_d, uint64_t numElements, Stream stream);
template void launchScale<float, float, float>(float *source_d, float scalar_h, float *dest_d, uint64_t numElements, Stream stream);

template void launchScale<half, half, half>(half *source_d, half scalar_h, half *dest_d, uint64_t numElements, Stream stream);
template void launchScale<float, half, half>(float *source_d, half scalar_h, half *dest_d, uint64_t numElements, Stream stream);
template void launchScale<half, float, half>(half *source_d, float scalar_h, half *dest_d, uint64_t numElements, Stream stream);
template void launchScale<float, float, half>(float *source_d, float scalar_h, half *dest_d, uint64_t numElements, Stream stream);
