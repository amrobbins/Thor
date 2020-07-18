#include "Relu.h"

__global__ void relu(half *dest, half *source, int numElements) {
    half h;
    int element = blockIdx.x * 1024 + threadIdx.x;

    if (element >= numElements)
        return;
    half zero = half(0.0f);
    h = source[element];
    dest[element] = h > zero ? h : zero;
    element += 256;

    if (element >= numElements)
        return;
    h = source[element];
    dest[element] = h > zero ? h : zero;
    element += 256;

    if (element >= numElements)
        return;
    h = source[element];
    dest[element] = h > zero ? h : zero;
    element += 256;

    if (element >= numElements)
        return;
    h = source[element];
    dest[element] = h > zero ? h : zero;
}

void launchRelu(half *dest_d, half *source_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    relu<<<gridSize, blockSize, 0, stream.getStream()>>>(dest_d, source_d, numElements);
}
