#include "Elu.h"

/**
 * x when x >= 0
 * alpha * (exp(x) - 1) when x < 0
 * where alpha is a scalar parameter that defaults to 1.0 and must be >= 0.0
 */
__global__ void elu(half *featureOut, half *featureIn, int numElements, float alpha) {
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
    if (fin >= 0.0f)
        fout = fin;
    else
        fout = alpha * (expf(fin) - 1.0f);
    fout = 1.0f / (1.0f + expf(-fin));
    foutBuffer[0] = (half)fout;

    element += 1;
    if (element < numElements) {
        fin = finBuffer[1];
        if (fin >= 0.0f)
            fout = fin;
        else
            fout = alpha * (expf(fin) - 1.0f);
        foutBuffer[1] = (half)fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[2];
        if (fin >= 0.0f)
            fout = fin;
        else
            fout = alpha * (expf(fin) - 1.0f);
        foutBuffer[2] = (half)fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[3];
        if (fin >= 0.0f)
            fout = fin;
        else
            fout = alpha * (expf(fin) - 1.0f);
        foutBuffer[3] = (half)fout;
    }

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(x) = 1 when x >= 0
 * d/dx(alpha * (exp(x) - 1)) = alpha * exp(x) when x < 0
 * where alpha is a scalar parameter that defaults to 1.0 and must be >= 0.0
 */
__global__ void eluBackward(half *errorOut, half *featureIn, half *errorIn, int numElements, float alpha) {
    const half zero = half(0.0f);
    float fin;
    float ein;
    float e_x;
    float e_x_1;
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

    fin = featureInBuffer[0];
    ein = errorInBuffer[0];
    if (fin >= 0.0f)
        eout = ein;
    else
        eout = ein * alpha * expf(fin);
    e_x = expf(fin);
    e_x_1 = e_x + 1.0f;
    eout = (ein * e_x) / (e_x_1 * e_x_1);
    errorOutBuffer[0] = (half)eout;

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[1];
        ein = errorInBuffer[1];
        if (fin >= 0.0f)
            eout = ein;
        else
            eout = ein * alpha * expf(fin);
        errorOutBuffer[1] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[2];
        ein = errorInBuffer[2];
        if (fin >= 0.0f)
            eout = ein;
        else
            eout = ein * alpha * expf(fin);
        errorOutBuffer[2] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[3];
        ein = errorInBuffer[3];
        if (fin >= 0.0f)
            eout = ein;
        else
            eout = ein * alpha * expf(fin);
        errorOutBuffer[3] = (half)eout;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchElu(half *featureOut_d, half *featureIn_d, int numElements, float alpha, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    elu<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements, alpha);
}

void launchEluBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, float alpha, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    eluBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements, alpha);
}
