#include "Sigmoid.h"

/**
 * sigmoid(x) = 1 / (1 + exp(-x))
 */
__global__ void sigmoid(half *dest, half *source, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half2 one = __float2half2_rn(1.0f);
    const half2 negativeOne = __float2half2_rn(-1.0f);

    double *source_half_4 = (double *)source;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = source_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;
    half featureOutBuffer[4];

    ((half2 *)featureOutBuffer)[0] = __h2div(one, __hadd2(one, h2exp(__hmul2(negativeOne, ((half2 *)featureInBuffer)[0]))));
    ((half2 *)featureOutBuffer)[1] = __h2div(one, __hadd2(one, h2exp(__hmul2(negativeOne, ((half2 *)featureInBuffer)[1]))));

    double *fout_half_4 = (double *)featureOutBuffer;
    double *dest_half_4 = (double *)dest;
    dest_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(1/(1 + exp(-x))) = e^x/(e^x + 1)^2
 */
__global__ void sigmoidBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
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
    half2 expX_1;
    half2 derivative;

    expX = h2exp(((half2 *)featureInBuffer)[0]);
    expX_1 = __hadd2(expX, one);
    derivative = __h2div(expX, __hmul2(expX_1, expX_1));
    ((half2 *)errorOutBuffer)[0] = __hmul2(((half2 *)errorInBuffer)[0], derivative);

    expX = h2exp(((half2 *)featureInBuffer)[1]);
    expX_1 = __hadd2(expX, one);
    derivative = __h2div(expX, __hmul2(expX_1, expX_1));
    ((half2 *)errorOutBuffer)[1] = __hmul2(((half2 *)errorInBuffer)[1], derivative);

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchSigmoid(half *dest_d, half *source_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sigmoid<<<gridSize, blockSize, 0, stream>>>(dest_d, source_d, numElements);
}

void launchSigmoidBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(min(256, numElements));
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sigmoidBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
