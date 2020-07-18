#include "Tanh.h"

__global__ void tanh(half *dest, half *source, int numElements) {
    int element = blockIdx.x * 1024 + threadIdx.x;

    if (element >= numElements)
        return;
    dest[element] = (half)(tanhf(source[element]));
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = (half)(tanhf(source[element]));
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = (half)(tanhf(source[element]));
    element += 256;

    if (element >= numElements)
        return;
    dest[element] = (half)(tanhf(source[element]));
}

void launchTanh(half *dest_d, half *source_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    tanh<<<gridSize, blockSize, 0, stream.getStream()>>>(dest_d, source_d, numElements);
}
