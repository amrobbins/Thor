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
    ((half2*)foutBuffer)[0] = __hfma2_sat(((half2*)finBuffer)[0], ((half2*)pointTwo)[0], ((half2*)pointFive)[0]);
    ((half2*)foutBuffer)[1] = __hfma2_sat(((half2*)finBuffer)[1], ((half2*)pointTwo)[0], ((half2*)pointFive)[0]);

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(x) = 0.2 when 1 < x > 0; else 0
 */
__global__ void hardSigmoidBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    const half zero = half(0.0f);
    const half one = half(1.0f);
    const half pointTwo = half(0.2f);
    half fin;
    half ein;
    half eout;

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
    if (fin >= one || fin <= zero)
        eout = zero;
    else
        eout = __hmul(ein, pointTwo);
    errorOutBuffer[0] = eout;

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[1];
        ein = errorInBuffer[1];
        if (fin >= one || fin <= zero)
            eout = zero;
        else
            eout = __hmul(ein, pointTwo);
        errorOutBuffer[1] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[2];
        ein = errorInBuffer[2];
        if (fin >= one || fin <= zero)
            eout = zero;
        else
            eout = __hmul(ein, pointTwo);
        errorOutBuffer[2] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = featureInBuffer[3];
        ein = errorInBuffer[3];
        if (fin >= one || fin <= zero)
            eout = zero;
        else
            eout = __hmul(ein, pointTwo);
        errorOutBuffer[3] = (half)eout;
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
