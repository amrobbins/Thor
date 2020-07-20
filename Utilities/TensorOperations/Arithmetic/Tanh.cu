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

__global__ void tanhBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    float tanx;
    float result;
    int element;

    element = blockIdx.x * 1024 + threadIdx.x;
    if (element >= numElements)
        return;
    tanx = tanhf(featureIn[element]);
    result = (float)errorIn[element] * (1.0f - tanx * tanx);
    errorOut[element] = (half)result;

    element += 256;
    if (element >= numElements)
        return;
    tanx = tanhf(featureIn[element]);
    result = (float)errorIn[element] * (1.0f - tanx * tanx);
    errorOut[element] = (half)result;

    element += 256;
    if (element >= numElements)
        return;
    tanx = tanhf(featureIn[element]);
    result = (float)errorIn[element] * (1.0f - tanx * tanx);
    errorOut[element] = (half)result;

    element += 256;
    if (element >= numElements)
        return;
    tanx = tanhf(featureIn[element]);
    result = (float)errorIn[element] * (1.0f - tanx * tanx);
    errorOut[element] = (half)result;
}

void launchTanh(half *dest_d, half *source_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    tanh<<<gridSize, blockSize, 0, stream>>>(dest_d, source_d, numElements);
}

void launchTanhBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    tanhBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
