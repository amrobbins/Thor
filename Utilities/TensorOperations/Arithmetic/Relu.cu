#include "Relu.h"

__global__ void relu(half *dest, half *source, int numElements) {
    half zero = half(0.0f);
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *source_half_4 = (double *)source;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = source_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    half fin = finBuffer[0];
    if (fin < zero)
        fin = zero;
    foutBuffer[0] = fin;

    element += 1;
    if (element < numElements) {
        fin = finBuffer[1];
        if (fin < zero)
            fin = zero;
        foutBuffer[1] = fin;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[2];
        if (fin < zero)
            fin = zero;
        foutBuffer[2] = fin;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[3];
        if (fin < zero)
            fin = zero;
        foutBuffer[3] = fin;
    }

    double *fout_half_4 = (double *)foutBuffer;
    double *dest_half_4 = (double *)dest;
    dest_half_4[element / 4] = fout_half_4[0];
}

__global__ void reluBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    half zero = half(0.0f);
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;
    half errorOutBuffer[4];

    half eOut;
    half fin = featureInBuffer[0];
    if (fin > zero)
        eOut = errorIn[element];
    else
        eOut = zero;
    errorOutBuffer[0] = eOut;

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[1];
        if (fin > zero)
            eOut = errorIn[element];
        else
            eOut = zero;
        errorOutBuffer[1] = eOut;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[2];
        if (fin > zero)
            eOut = errorIn[element];
        else
            eOut = zero;
        errorOutBuffer[2] = eOut;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[3];
        if (fin > zero)
            eOut = errorIn[element];
        else
            eOut = zero;
        errorOutBuffer[3] = eOut;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
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
