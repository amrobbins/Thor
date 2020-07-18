#include "Average.h"
#include "Utilities/Common/ScopedGpu.h"

__global__ void average(half *dest, half *source[], int numInstances, int numElements) {
    int element = blockIdx.x * 512 + threadIdx.x;

#pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        if (element >= numElements)
            return;

        half buff = source[0][element];
        float accum = (float)buff;
        for (int j = 1; j < numInstances; ++j) {
            buff = source[j][element];
            accum += (float)buff;
        }
        accum /= (float)numInstances;

        // write back
        dest[element] = (half)accum;

        element += 256;
    }
}

// source_d_pd[] is a device memory array of pointers to device memory
void launchAverage(half *dest_d, half *source_d_pd[], int numInstances, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 511) / 512);
    ScopedGpu scopedGpu(stream.getGpuNum());
    average<<<gridSize, blockSize, 0, stream.getStream()>>>(dest_d, source_d_pd, numInstances, numElements);
}
