#include "SoftPlus.h"

/**
 * ln(e(x) + 1)
 */
__global__ void softPlus(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half2 one = __float2half2_rn(1.0f);

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;
    half featureOutBuffer[4];

    ((half2 *)featureOutBuffer)[0] = h2log(__hadd2(h2exp(((half2 *)featureInBuffer)[0]), one));
    ((half2 *)featureOutBuffer)[1] = h2log(__hadd2(h2exp(((half2 *)featureInBuffer)[1]), one));

    double *fout_half_4 = (double *)featureOutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(ln(exp(x) + 1)) = e^x/(e^x + 1)
 */
__global__ void softPlusBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half2 one = __float2half2_rn(1.0f);

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;

    double *errorIn_half_4 = (double *)errorIn;
    double errorInBuffer_half_4[1];
    errorInBuffer_half_4[0] = errorIn_half_4[element / 4];
    half *errorInBuffer = (half *)errorInBuffer_half_4;
    half errorOutBuffer[4];

    half2 expX;

    expX = h2exp(((half2 *)featureInBuffer)[0]);
    ((half2 *)errorOutBuffer)[0] = __hmul2(((half2 *)errorInBuffer)[0], __h2div(expX, __hadd2(expX, one)));

    expX = h2exp(((half2 *)featureInBuffer)[1]);
    ((half2 *)errorOutBuffer)[1] = __hmul2(((half2 *)errorInBuffer)[1], __h2div(expX, __hadd2(expX, one)));

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchSoftPlus(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    softPlus<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchSoftPlusBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    softPlusBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
