#include "ElementwiseSubtract.h"

template <typename SOURCE0_TYPE, typename SOURCE1_TYPE, typename DEST_TYPE>
__global__ void elementwiseSubtract(SOURCE0_TYPE *source0, SOURCE1_TYPE *source1, DEST_TYPE *dest, uint64_t numElements) {
    int element = blockIdx.x * 1024 + threadIdx.x;

    if (element >= numElements)
        return;
    dest[element] = ((DEST_TYPE)source0[element]) - ((DEST_TYPE)source1[element]);
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = ((DEST_TYPE)source0[element]) - ((DEST_TYPE)source1[element]);
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = ((DEST_TYPE)source0[element]) - ((DEST_TYPE)source1[element]);
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = ((DEST_TYPE)source0[element]) - ((DEST_TYPE)source1[element]);
}

template <typename SOURCE0_TYPE, typename SOURCE1_TYPE, typename DEST_TYPE>
void launchElementwiseSubtract(SOURCE0_TYPE *source0_d, SOURCE1_TYPE *source1_d, DEST_TYPE *dest_d, uint64_t numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    elementwiseSubtract<SOURCE0_TYPE, SOURCE1_TYPE, DEST_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source0_d, source1_d, dest_d, numElements);
}

template void launchElementwiseSubtract<half, half, float>(
    half *source0_d, half *source1_d, float *dest_d, uint64_t numElements, Stream stream);
template void launchElementwiseSubtract<float, half, float>(
    float *source0_d, half *source1_d, float *dest_d, uint64_t numElements, Stream stream);
template void launchElementwiseSubtract<half, float, float>(
    half *source0_d, float *source1_d, float *dest_d, uint64_t numElements, Stream stream);
template void launchElementwiseSubtract<float, float, float>(
    float *source0_d, float *source1_d, float *dest_d, uint64_t numElements, Stream stream);

template void launchElementwiseSubtract<half, half, half>(
    half *source0_d, half *source1_d, half *dest_d, uint64_t numElements, Stream stream);
template void launchElementwiseSubtract<float, half, half>(
    float *source0_d, half *source1_d, half *dest_d, uint64_t numElements, Stream stream);
template void launchElementwiseSubtract<half, float, half>(
    half *source0_d, float *source1_d, half *dest_d, uint64_t numElements, Stream stream);
template void launchElementwiseSubtract<float, float, half>(
    float *source0_d, float *source1_d, half *dest_d, uint64_t numElements, Stream stream);
