#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

__global__ void sumScale(half *dest, half *noScaleSource, half *scaleSource, float scale, int numElements) {
    int element = blockIdx.x * 512 + threadIdx.x;

#pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        if (element >= numElements)
            return;

        dest[element] = (half)((float)noScaleSource[element] + scale * (float)scaleSource[element]);

        element += 256;
    }
}

void launchSumScale(half result_d[], half noScaleSource[], half scaleSource[], float scale, int numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 511) / 512);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sumScale<<<gridSize, blockSize, 0, stream.getStream()>>>(result_d, noScaleSource, scaleSource, scale, numElements);
}
