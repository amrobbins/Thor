#include "Tanh.h"

__global__ void tanh(half *dest, half *source, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *source_half_4 = (double *)source;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = source_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    foutBuffer[0] = dest[element] = (half)(tanhf(finBuffer[0]));

    element += 1;
    if (element < numElements)
        foutBuffer[1] = dest[element] = (half)(tanhf(finBuffer[1]));

    element += 1;
    if (element < numElements)
        foutBuffer[2] = dest[element] = (half)(tanhf(finBuffer[2]));

    element += 1;
    if (element < numElements)
        foutBuffer[3] = dest[element] = (half)(tanhf(finBuffer[3]));

    double *fout_half_4 = (double *)foutBuffer;
    double *dest_half_4 = (double *)dest;
    dest_half_4[element / 4] = fout_half_4[0];
}

__global__ void tanhBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    float tanx;
    float result;

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

    tanx = tanhf(featureInBuffer[0]);
    result = (float)errorInBuffer[0] * (1.0f - tanx * tanx);
    errorOutBuffer[0] = result;

    element += 1;
    if (element < numElements) {
        tanx = tanhf(featureInBuffer[1]);
        result = (float)errorInBuffer[1] * (1.0f - tanx * tanx);
        errorOutBuffer[1] = result;
    }

    element += 1;
    if (element < numElements) {
        tanx = tanhf(featureInBuffer[2]);
        result = (float)errorInBuffer[2] * (1.0f - tanx * tanx);
        errorOutBuffer[2] = result;
    }

    element += 1;
    if (element < numElements) {
        tanx = tanhf(featureInBuffer[3]);
        result = (float)errorInBuffer[3] * (1.0f - tanx * tanx);
        errorOutBuffer[3] = result;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
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
