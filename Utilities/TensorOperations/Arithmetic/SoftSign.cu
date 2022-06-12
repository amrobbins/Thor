#include "SoftSign.h"

/**
 * x / (abs(x) + 1)
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
    ((half2 *)foutBuffer)[0] = __h2div(((half2 *)finBuffer)[0], __hadd2(__habs2(((half2 *)finBuffer)[0]), one);
    ((half2 *)foutBuffer)[1] = __h2div(((half2 *)finBuffer)[0], __hadd2(__habs2(((half2 *)finBuffer)[0]), one);

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(x/abs(x + 1)) = (abs(x + 1) - x sgn(x + 1))/(x + 1)^2
 * Since this function is discontinuous, the derivative as assigned to the values below near the discontinuity:
 * (-1.45, -1.0]: -5.0
 * (-1.0, -0.55): 5.0
 */
__global__ void softSignBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    const half[2] one = {half(1.0f), half(1.0f)};

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

    half foutBuffer[4];
    half xPlusOne[4];
    half absXPlusOne[4];
    half signXPlusOne[4];
    half xPlusOneSquared[4];
    half xSignXPlusOne[4];
    half absXplusOneSquaredMinusXSign[4];
    half fullDerivative[4];

    ((half2 *)xPlusOne)[0] = __hadd2(((half2 *)finBuffer)[0], ((half2 *)one)[0]);
    ((half2 *)absXPlusOne)[0] = __habs2(((half2 *)xPlusOne)[0]);
    ((half2 *)signXPlusOne)[0] = __hdiv2(((half2 *)xPlusOne)[0], ((half2 *)(absXPlusOne)[0]);
    ((half2 *)xPlusOneSquared)[0] = __hmul2(((half2 *)xPlusOne)[0], ((half2 *)xPlusOne)[0]);
    ((half2 *)xSignXPlusOne)[0] = __hmul2(((half2 *)finBuffer)[0], ((half2 *)signXPlusOne)[0]);
    ((half2 *)absXplusOneSquaredMinusXSign)[0] = __hsub2(((half2 *)absXPlusOne)[0], ((half2 *)xSignXPlusOne)[0]);
    ((half2 *)fullDerivative)[0] = __hdiv2(((half2 *)absXplusOneSquaredMinusXSign)[0], ((half2 *)xPlusOneSquared)[0]);
    ((half2 *)foutBuffer)[0] = __hmul2(((half2 *)errorInBuffer)[0], ((half2 *)fullDerivative)[0]);

    ((half2 *)xPlusOne)[1] = __hadd2(((half2 *)finBuffer)[1], ((half2 *)one)[1]);
    ((half2 *)absXPlusOne)[1] = __habs2(((half2 *)xPlusOne)[1]);
    ((half2 *)signXPlusOne)[1] = __hdiv2(((half2 *)xPlusOne)[1], ((half2 *)(absXPlusOne)[1]);
    ((half2 *)xPlusOneSquared)[1] = __hmul2(((half2 *)xPlusOne)[1], ((half2 *)xPlusOne)[1]);
    ((half2 *)xSignXPlusOne)[1] = __hmul2(((half2 *)finBuffer)[1], ((half2 *)signXPlusOne)[1]);
    ((half2 *)absXplusOneSquaredMinusXSign)[1] = __hsub2(((half2 *)absXPlusOne)[1], ((half2 *)xSignXPlusOne)[1]);
    ((half2 *)fullDerivative)[1] = __hdiv2(((half2 *)absXplusOneSquaredMinusXSign)[1], ((half2 *)xPlusOneSquared)[1]);
    ((half2 *)foutBuffer)[1] = __hmul2(((half2 *)errorInBuffer)[1], ((half2 *)fullDerivative)[1]);

    // Check each for nearness to the discontinuity
    half negativeFive = (half)-5.0f;
    half five = (half)5.0f;
    half negativeOnePointFourFive = (half)-1.45f;
    half negativeOne = (half)-1.0f;
    half negativePointFiveFive = (half)-0.55f;

    if(finBuffer[0] > negativeOnePointFourFive && finBuffer[0] <= negativeOne)
        foutBuffer[0] = __hmul(errorInBuffer[0], negativeFive);
    else if(finBuffer[0] > negativeOne && finBuffer[0] < negativePointFiveFive)
        foutBuffer[0] = __hmul(errorInBuffer[0], five);

    if(finBuffer[1] > negativeOnePointFourFive && finBuffer[1] <= negativeOne)
        foutBuffer[1] = __hmul(errorInBuffer[1], negativeFive);
    else if(finBuffer[1] > negativeOne && finBuffer[1] < negativePointFiveFive)
        foutBuffer[1] = __hmul(errorInBuffer[1], five);

    if(finBuffer[2] > negativeOnePointFourFive && finBuffer[2] <= negativeOne)
        foutBuffer[2] = __hmul(errorInBuffer[2], negativeFive);
    else if(finBuffer[2] > negativeOne && finBuffer[2] < negativePointFiveFive)
        foutBuffer[2] = __hmul(errorInBuffer[2], five);

    if(finBuffer[3] > negativeOnePointFourFive && finBuffer[3] <= negativeOne)
        foutBuffer[3] = __hmul(errorInBuffer[3], negativeFive);
    else if(finBuffer[3] > negativeOne && finBuffer[3] < negativePointFiveFive)
        foutBuffer[3] = __hmul(errorInBuffer[3], five);

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
