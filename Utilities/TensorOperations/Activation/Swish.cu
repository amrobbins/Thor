#include "Swish.h"

/**
 * swish(x) === x * sigmoid(x) === x / (1 + exp(-x))
 */
__global__ void swish(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half one[2] = {(half)1.0f, (half)1.0f};
    const half negativeOne[2] = {(half)-1.0f, (half)-1.0f};

    double *featureIn_half_4 = (double *)featureIn;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    ((half2 *)foutBuffer)[0] =
        __h2div(((half2 *)finBuffer)[0], __hadd2(((half2 *)one)[0], h2exp(__hmul2(((half2 *)finBuffer)[0], ((half2 *)negativeOne)[0]))));
    ((half2 *)foutBuffer)[1] =
        __h2div(((half2 *)finBuffer)[1], __hadd2(((half2 *)one)[0], h2exp(__hmul2(((half2 *)finBuffer)[1], ((half2 *)negativeOne)[0]))));

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(x/(1 + exp(-x))) = (e^x * (x + e^x + 1))/(e^x + 1)^2
 */
__global__ void swishBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
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

    half2 ex;
    half2 ex_1;
    half2 ex_1_squared;
    half2 x_ex_1;
    half2 ex_x_ex_1;
    half2 derivative;

    ex = h2exp(((half2 *)featureInBuffer)[0]);
    ex_1 = __hadd2(one, ex);
    ex_1_squared = __hmul2(ex_1, ex_1);
    x_ex_1 = __hadd2(((half2 *)featureInBuffer)[0], ex_1);
    ex_x_ex_1 = __hmul2(ex, x_ex_1);
    derivative = __h2div(ex_x_ex_1, ex_1_squared);
    ((half2 *)errorOutBuffer)[0] = __hmul2(((half2 *)errorInBuffer)[0], derivative);

    ex = h2exp(((half2 *)featureInBuffer)[1]);
    ex_1 = __hadd2(one, ex);
    ex_1_squared = __hmul2(ex_1, ex_1);
    x_ex_1 = __hadd2(((half2 *)featureInBuffer)[1], ex_1);
    ex_x_ex_1 = __hmul2(ex, x_ex_1);
    derivative = __h2div(ex_x_ex_1, ex_1_squared);
    ((half2 *)errorOutBuffer)[1] = __hmul2(((half2 *)errorInBuffer)[1], derivative);

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchSwish(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    swish<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchSwishBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    swishBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
