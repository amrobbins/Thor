#include "Exponential.h"

/**
 * exp(x)
 */
__global__ void exponential(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *featureIn_half_4 = (double *)featureIn;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    ((half2 *)foutBuffer)[0] = h2exp(((half2 *)finBuffer)[0]);
    ((half2 *)foutBuffer)[1] = h2exp(((half2 *)finBuffer)[1]);

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(exp(x)) = exp(x)
 */
__global__ void exponentialBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
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

    ((half2 *)errorOutBuffer)[0] = __hmul2(((half2 *)errorInBuffer)[0], h2exp(((half2 *)featureInBuffer)[0]));
    ((half2 *)errorOutBuffer)[1] = __hmul2(((half2 *)errorInBuffer)[1], h2exp(((half2 *)featureInBuffer)[1]));

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchExponential(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    exponential<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchExponentialBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    exponentialBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
