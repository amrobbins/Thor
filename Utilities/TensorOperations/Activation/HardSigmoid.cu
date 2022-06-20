#include "HardSigmoid.h"

/**
 * max( min(0.2 * x + 0.5, 1.0), 0.0)
 */
__global__ void hardSigmoid(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    double *featureIn_half_4 = (double *)featureIn;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    const half pointTwo[2] = {(half)0.2f, (half)0.2f};
    const half pointFive[2] = {(half)0.5f, (half)0.5f};
    ((half2 *)foutBuffer)[0] = __hfma2_sat(((half2 *)finBuffer)[0], ((half2 *)pointTwo)[0], ((half2 *)pointFive)[0]);
    ((half2 *)foutBuffer)[1] = __hfma2_sat(((half2 *)finBuffer)[1], ((half2 *)pointTwo)[0], ((half2 *)pointFive)[0]);

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(x) = 0.2 when 1 < x > 0; else 0
 */
__global__ void hardSigmoidBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    const half2 zero = __float2half2_rn(0.0f);
    const half2 one = __float2half2_rn(1.0f);
    const half2 pointTwo = __float2half2_rn(0.2f);

    const half singleZero = 0.0f;
    const half singleOne = 1.0f;

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;

    double *errorIn_half_4 = (double *)errorIn;
    double errorInBuffer_half_4[1];
    errorInBuffer_half_4[0] = errorIn_half_4[element / 4];
    half *errorInBuffer = (half *)errorInBuffer_half_4;
    half errorOutBuffer[4];

    half2 gteOne;
    half2 lteZero;
    half2 bothZero;
    half2 mulBuf;

    gteOne = __hge2(((half2 *)featureInBuffer)[0], one);
    lteZero = __hle2(((half2 *)featureInBuffer)[0], zero);
    errorOutBuffer[0] = singleZero;
    bothZero.x = gteOne.x || lteZero.x;
    errorOutBuffer[1] = singleZero;
    bothZero.y = gteOne.y || lteZero.y;
    if (!__hbeq2(bothZero, one)) {
        mulBuf = __hmul2(((half2 *)errorInBuffer)[0], pointTwo);
        if (!bothZero.x)
            errorOutBuffer[0] = mulBuf.x;
        if (!bothZero.y)
            errorOutBuffer[1] = mulBuf.y;
    }

    gteOne = __hge2(((half2 *)featureInBuffer)[1], one);
    lteZero = __hle2(((half2 *)featureInBuffer)[1], zero);
    errorOutBuffer[2] = singleZero;
    bothZero.x = gteOne.x || lteZero.x;
    errorOutBuffer[3] = singleZero;
    bothZero.y = gteOne.y || lteZero.y;
    if (!__hbeq2(bothZero, one)) {
        mulBuf = __hmul2(((half2 *)errorInBuffer)[1], pointTwo);
        if (!bothZero.x)
            errorOutBuffer[2] = mulBuf.x;
        if (!bothZero.y)
            errorOutBuffer[3] = mulBuf.y;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchHardSigmoid(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    hardSigmoid<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchHardSigmoidBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    hardSigmoidBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
