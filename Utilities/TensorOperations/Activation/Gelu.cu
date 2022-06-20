#include "Gelu.h"

/**
 * 0.5 * x * (1 + tanh((2/π)^0.5 * (x + 0.044715 * x^3)))
 */
__global__ void gelu(half *featureOut, half *featureIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    constexpr float sqrtTwoOverPi = 0.797884561f;
    constexpr float c = 0.044715f;

    double *featureIn_half_4 = (double *)featureIn;
    double finBuffer_half_4[1];
    finBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *finBuffer = (half *)finBuffer_half_4;
    half foutBuffer[4];

    float fin;
    float fout;

    fin = (float)finBuffer[0];
    fout = (0.5 * fin * (1.0f + tanhf(sqrtTwoOverPi * (fin + c * fin * fin * fin))));
    foutBuffer[0] = (half)fout;

    element += 1;
    if (element < numElements) {
        fin = finBuffer[1];
        fout = (0.5 * fin * (1.0f + tanhf(sqrtTwoOverPi * (fin + c * fin * fin * fin))));
        foutBuffer[1] = (half)fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[2];
        fout = (0.5 * fin * (1.0f + tanhf(sqrtTwoOverPi * (fin + c * fin * fin * fin))));
        foutBuffer[2] = (half)fout;
    }

    element += 1;
    if (element < numElements) {
        fin = finBuffer[3];
        fout = (0.5 * fin * (1.0f + tanhf(sqrtTwoOverPi * (fin + c * fin * fin * fin))));
        foutBuffer[3] = (half)fout;
    }

    double *fout_half_4 = (double *)foutBuffer;
    double *featureOut_half_4 = (double *)featureOut;
    featureOut_half_4[element / 4] = fout_half_4[0];
}

/**
 * d/dx(0.5 * x * (1 + tanh((2/π)^0.5 * (x + 0.044715 * x^3))))
 *   = 0.5 * tanh(0.797885 * (x + 0.044715 * x^3)) + (0.0535161 * x^3 + 0.398942 * x) * sech^2(0.797885 * (x + 0.044715 * x^3)) + 0.5
 */
__global__ void geluBackward(half *errorOut, half *featureIn, half *errorIn, int numElements) {
    int element = blockIdx.x * 1024 + (4 * threadIdx.x);

    if (element >= numElements)
        return;

    float fin;
    float ein;
    float eout;

    constexpr float c0 = 0.797885f;
    constexpr float c1 = 0.044715f;
    constexpr float c2 = 0.0535161f;
    constexpr float c3 = 0.398942f;

    double *featureIn_half_4 = (double *)featureIn;
    double featureInBuffer_half_4[1];
    featureInBuffer_half_4[0] = featureIn_half_4[element / 4];
    half *featureInBuffer = (half *)featureInBuffer_half_4;

    double *errorIn_half_4 = (double *)errorIn;
    double errorInBuffer_half_4[1];
    errorInBuffer_half_4[0] = errorIn_half_4[element / 4];
    half *errorInBuffer = (half *)errorInBuffer_half_4;
    half errorOutBuffer[4];

    float xCubed;
    float sechResult;

    fin = (float)featureInBuffer[0];
    ein = (float)errorInBuffer[0];
    xCubed = fin * fin * fin;
    // sech x = 1/cosh x
    sechResult = 1.0f / coshf(c0 * (fin + c1 * xCubed));
    eout = ein * (0.5f * tanhf(c0 * (fin + c1 * xCubed)) + (c2 * xCubed + c3 * fin) * sechResult * sechResult + 0.5f);
    errorOutBuffer[0] = (half)eout;

    element += 1;
    if (element < numElements) {
        fin = (float)featureInBuffer[1];
        ein = (float)errorInBuffer[1];
        xCubed = fin * fin * fin;
        sechResult = 1.0f / coshf(c0 * (fin + c1 * xCubed));
        eout = ein * (0.5f * tanhf(c0 * (fin + c1 * xCubed)) + (c2 * xCubed + c3 * fin) * sechResult * sechResult + 0.5f);
        errorOutBuffer[1] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = (float)featureInBuffer[2];
        ein = (float)errorInBuffer[2];
        xCubed = fin * fin * fin;
        sechResult = 1.0f / coshf(c0 * (fin + c1 * xCubed));
        eout = ein * (0.5f * tanhf(c0 * (fin + c1 * xCubed)) + (c2 * xCubed + c3 * fin) * sechResult * sechResult + 0.5f);
        errorOutBuffer[2] = (half)eout;
    }

    element += 1;
    if (element < numElements) {
        fin = (float)featureInBuffer[3];
        ein = (float)errorInBuffer[3];
        xCubed = fin * fin * fin;
        sechResult = 1.0f / coshf(c0 * (fin + c1 * xCubed));
        eout = ein * (0.5f * tanhf(c0 * (fin + c1 * xCubed)) + (c2 * xCubed + c3 * fin) * sechResult * sechResult + 0.5f);
        errorOutBuffer[3] = (half)eout;
    }

    double *errorOutBuffer_half_4 = (double *)errorOutBuffer;
    double *errorOut_half_4 = (double *)errorOut;
    errorOut_half_4[element / 4] = errorOutBuffer_half_4[0];
}

void launchGelu(half *featureOut_d, half *featureIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    gelu<<<gridSize, blockSize, 0, stream>>>(featureOut_d, featureIn_d, numElements);
}

void launchGeluBackward(half *errorOut_d, half *featureIn_d, half *errorIn_d, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    ScopedGpu scopedGpu(stream.getGpuNum());
    geluBackward<<<gridSize, blockSize, 0, stream>>>(errorOut_d, featureIn_d, errorIn_d, numElements);
}
