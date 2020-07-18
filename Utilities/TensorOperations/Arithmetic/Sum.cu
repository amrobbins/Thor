#include "Sum.h"
#include "Utilities/Common/ScopedGpu.h"

template <typename DATA_TYPE>
__global__ void sum(DATA_TYPE *dest, DATA_TYPE *source[], uint32_t numInstances, uint64_t numElements) {
    uint64_t element = blockIdx.x * 512 + threadIdx.x;

#pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        if (element >= numElements)
            return;

        DATA_TYPE buff = source[0][element];
        float accum = (float)buff;
        for (uint32_t j = 1; j < numInstances; ++j) {
            buff = source[j][element];
            accum += (float)buff;
        }

        // write back
        dest[element] = (DATA_TYPE)accum;

        element += 256;
    }
}

// FIXME: all launchXXX functions should use a ScopedGpu based on stream
// FIXME: i.e. ScopedGpu scopedGpu(stream.getGpuNum());

// source_d_pd[] is a device memory array of pointers to device memory
template <typename DATA_TYPE>
void launchSum(DATA_TYPE *dest_d, DATA_TYPE *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((numElements + 511) / 512);
    ScopedGpu scopedGpu(stream.getGpuNum());
    sum<DATA_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(dest_d, source_d_pd, numInstances, numElements);
}

template void launchSum<half>(half *dest, half *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<float>(float *dest, float *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<double>(double *dest, double *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<int8_t>(int8_t *dest, int8_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<int16_t>(int16_t *dest, int16_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<int32_t>(int32_t *dest, int32_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<int64_t>(int64_t *dest, int64_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<uint8_t>(uint8_t *dest, uint8_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<uint16_t>(uint16_t *dest, uint16_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<uint32_t>(uint32_t *dest, uint32_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
template void launchSum<uint64_t>(uint64_t *dest, uint64_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
