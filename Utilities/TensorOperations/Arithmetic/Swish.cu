#include "Swish.h"

__global__ void swish(half *dest, half *source, int numElements) {
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
    half fout = (float)fin / (1.0f + expf(-fin));
    foutBuffer[0] = fout;

    element += 1;
    if (element < numElements) {
        fin = finBuffer[1];
        half fout = (float)fin / (1.0f + expf(-fin));
        foutBuffer[1] = fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[2];
        half fout = (float)fin / (1.0f + expf(-fin));
        foutBuffer[2] = fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[3];
        half fout = (float)fin / (1.0f + expf(-fin));
        foutBuffer[3] = fout;
    }

    double *fout_half_4 = (double *)foutBuffer;
    double *dest_half_4 = (double *)dest;
    dest_half_4[element / 4] = fout_half_4[0];
}

// d/dx(x/(1 + exp(-x))) = (e^x (x + e^x + 1))/(e^x + 1)^2
__global__ void swishBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
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
    e_x = expf(fin);
    e_x_1 = e_x + 1.0f;
    eout = ein * (e_x * ((float)fin + e_x_1)) / (e_x_1 * e_x_1);
    errorOutBuffer[0] = (half)eout;

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[1];
        ein = errorInBuffer[1];
        e_x = expf(fin);
        e_x_1 = e_x + 1.0f;
        eout = ein * (e_x * ((float)fin + e_x_1)) / (e_x_1 * e_x_1);
        errorOutBuffer[1] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[2];
        ein = errorInBuffer[2];
        e_x = expf(fin);
        e_x_1 = e_x + 1.0f;
        eout = ein * (e_x * ((float)fin + e_x_1)) / (e_x_1 * e_x_1);
        errorOutBuffer[2] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[3];
        ein = errorInBuffer[3];
        e_x = expf(fin);
        e_x_1 = e_x + 1.0f;
        eout = ein * (e_x * ((float)fin + e_x_1)) / (e_x_1 * e_x_1);
        errorOutBuffer[3] = (half)eout;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchSwish(half *dest_d, half *source_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    swish<<<gridSize, blockSize, 0, stream>>>(dest_d, source_d, numElements);
}

void launchSwishBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    swishBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
