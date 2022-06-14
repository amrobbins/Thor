#include "Relu.h"

__global__ void relu(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    half zero = half(0.0f);

    double *featureIn_half_4 = (double *)featureIn;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    half fin = finBuffer[0];
    if (fin < zero)
        fin = zero;
    foutBuffer[0] = fin;

    if (element + 1 < numElements) {
        fin = finBuffer[1];
        if (fin < zero)
            fin = zero;
        foutBuffer[1] = fin;
    }

    if (element + 2 < numElements) {
        fin = finBuffer[2];
        if (fin < zero)
            fin = zero;
        foutBuffer[2] = fin;
    }

    if (element + 3 < numElements) {
        fin = finBuffer[3];
        if (fin < zero)
            fin = zero;
        foutBuffer[3] = fin;
    }

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

__global__ void reluBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    half zero = half(0.0f);

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;
    half errorOutBuffer[4];

    double *errorIn_half_4 = (double *)errorIn;
    double errorInBuffer_half_4[1];
    errorInBuffer_half_4[0] = errorIn_half_4[element / 4];
    half *errorInBuffer = (half *)errorInBuffer_half_4;

    half eOut;
    half fin = featureInBuffer[0];
    if (fin > zero)
        eOut = errorInBuffer[0];
    else
        eOut = zero;
    errorOutBuffer[0] = eOut;

    if (element + 1 < numElements) {
        fin = featureInBuffer[1];
        if (fin > zero)
            eOut = errorInBuffer[1];
        else
            eOut = zero;
        errorOutBuffer[1] = eOut;
    }

    if (element + 2 < numElements) {
        fin = featureInBuffer[2];
        if (fin > zero)
            eOut = errorInBuffer[2];
        else
            eOut = zero;
        errorOutBuffer[2] = eOut;
    }

    if (element + 3 < numElements) {
        fin = featureInBuffer[3];
        if (fin > zero)
            eOut = errorInBuffer[3];
        else
            eOut = zero;
        errorOutBuffer[3] = eOut;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchRelu(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    relu<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchReluBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    reluBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
