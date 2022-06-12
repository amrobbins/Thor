#include "SoftPlus.h"

/**
 * ln(e(x) + 1)
 */
__global__ void softPlus(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *featureIn_half_4 = (double *)featureIn;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    float fin;
    float fout;

    fin = (float)finBuffer[0];
    fout = logf(expf(fin) + 1.0f);
    foutBuffer[0] = (half)fout;

    element += 1;
    if (element < numElements) {
        fin = finBuffer[1];
        fout = logf(expf(fin) + 1.0f);
        foutBuffer[1] = (half)fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[2];
        fout = logf(expf(fin) + 1.0f);
        foutBuffer[2] = (half)fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[3];
        fout = logf(expf(fin) + 1.0f);
        foutBuffer[3] = (half)fout;
    }

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(ln(exp(x) + 1)) = e^x/(e^x + 1)
 */
__global__ void softPlusBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    const half zero = half(0.0f);
    float fin;
    float ein;
    float eout;

    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;

    double *errorIn_half_4 = (double *)errorIn;
    double errorInBuffer_half_4[1];
    errorInBuffer_half_4[0] = errorIn_half_4[element / 4];
    half *errorInBuffer = (half *)errorInBuffer_half_4;
    half errorOutBuffer[4];

    float expX;

    fin = featureInBuffer[0];
    ein = errorInBuffer[0];
    expX = expf(fin);
    fout = ein * (expX / (expX + 1));
    errorOutBuffer[0] = (half)eout;

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[1];
        ein = errorInBuffer[1];
        expX = expf(fin);
        fout = ein * (expX / (expX + 1));
        errorOutBuffer[1] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[2];
        ein = errorInBuffer[2];
        expX = expf(fin);
        fout = ein * (expX / (expX + 1));
        errorOutBuffer[2] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[3];
        ein = errorInBuffer[3];
        expX = expf(fin);
        fout = ein * (expX / (expX + 1));
        errorOutBuffer[3] = (half)eout;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchSoftPlus(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    softPlus<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchSoftPlusBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    softPlusBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
