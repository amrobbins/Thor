#include "SoftSign.h"

/**
 * softSign(x) === x / (abs(x) + 1)
 */
__global__ void softSign(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *featureIn_half_4 = (double *)featureIn;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    const half one[2] = {(half)1.0f, (half)1.0f};
    ((half2 *)foutBuffer)[0] = __h2div(((half2 *)finBuffer)[0], __hadd2(__habs2(((half2 *)finBuffer)[0]), ((half2 *)one)[0]));
    ((half2 *)foutBuffer)[1] = __h2div(((half2 *)finBuffer)[1], __hadd2(__habs2(((half2 *)finBuffer)[1]), ((half2 *)one)[0]));

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(x/(abs(x) + 1)) = 1/(abs(x) + 1)^2
 */
__global__ void softSignBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half one[2] = {(half)1.0f, (half)1.0f};

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;

    double *errorIn_half_4 = (double *)errorIn;
    double errorInBuffer_half_4[1];
    errorInBuffer_half_4[0] = errorIn_half_4[element / 4];
    half *errorInBuffer = (half *)errorInBuffer_half_4;
    half errorOutBuffer[4];

    half absX[4];
    half absXPlusOne[4];
    half absXPlusOneSquared[4];

    ((half2 *)absX)[0] = __habs2(((half2 *)featureInBuffer)[0]);
    ((half2 *)absXPlusOne)[0] = __hadd2(((half2 *)absX)[0], ((half2 *)one)[0]);
    ((half2 *)absXPlusOneSquared)[0] = __hmul2(((half2 *)absXPlusOne)[0], ((half2 *)absXPlusOne)[0]);
    ((half2 *)errorOutBuffer)[0] = __h2div(((half2 *)errorInBuffer)[0], ((half2 *)absXPlusOneSquared)[0]);

    ((half2 *)absX)[1] = __habs2(((half2 *)featureInBuffer)[1]);
    ((half2 *)absXPlusOne)[1] = __hadd2(((half2 *)absX)[1], ((half2 *)one)[0]);
    ((half2 *)absXPlusOneSquared)[1] = __hmul2(((half2 *)absXPlusOne)[1], ((half2 *)absXPlusOne)[1]);
    ((half2 *)errorOutBuffer)[1] = __h2div(((half2 *)errorInBuffer)[1], ((half2 *)absXPlusOneSquared)[1]);

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchSoftSign(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    softSign<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchSoftSignBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    softSignBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
