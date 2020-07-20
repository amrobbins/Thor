#include "Relu.h"

__global__ void relu(half *dest, half *source, int numElements) {
    half fout;
    half zero = half(0.0f);
    int element = blockIdx.x * 1024 + threadIdx.x;

    if (element >= numElements)
        return;
    fout = source[element];
    if (fout <= zero)
        fout = zero;
    dest[element] = fout;
    element += 256;

    if (element >= numElements)
        return;
    fout = source[element];
    if (fout <= zero)
        fout = zero;
    dest[element] = fout;
    element += 256;

    if (element >= numElements)
        return;
    fout = source[element];
    if (fout <= zero)
        fout = zero;
    dest[element] = fout;
    element += 256;

    if (element >= numElements)
        return;
    fout = source[element];
    if (fout <= zero)
        fout = zero;
    dest[element] = fout;
}

__global__ void reluBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    half fin;
    half eout;
    half zero = half(0.0f);

    int element = blockIdx.x * 1024 + threadIdx.x;
    if (element >= numElements)
        return;
    fin = featureIn[element];
    if (fin > zero)
        eout = errorIn[element];
    else
        eout = zero;
    errorOut[element] = eout;

    element += 256;
    if (element >= numElements)
        return;
    fin = featureIn[element];
    if (fin > zero)
        eout = errorIn[element];
    else
        eout = zero;
    errorOut[element] = eout;

    element += 256;
    if (element >= numElements)
        return;
    fin = featureIn[element];
    if (fin > zero)
        eout = errorIn[element];
    else
        eout = zero;
    errorOut[element] = eout;

    element += 256;
    if (element >= numElements)
        return;
    fin = featureIn[element];
    if (fin > zero)
        eout = errorIn[element];
    else
        eout = zero;
    errorOut[element] = eout;
}

void launchRelu(half *dest_d, half *source_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    relu<<<gridSize, blockSize, 0, stream>>>(dest_d, source_d, numElements);
}

void launchReluBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    reluBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
