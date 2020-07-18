#pragma once

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

// source_d_pd[] is a device memory array of pointers to device memory
// i.e. a variable on the host that holds the address of a device memory array of pointers to each of the device memory arrays of source
// data. dest_d is a device memory array where the result will be stored.

template <typename DATA_TYPE>
void launchSum(DATA_TYPE *dest_d, DATA_TYPE *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);

extern template void launchSum<half>(half *dest, half *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<float>(float *dest, float *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<double>(double *dest, double *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<int8_t>(int8_t *dest, int8_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<int16_t>(int16_t *dest, int16_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<int32_t>(int32_t *dest, int32_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<int64_t>(int64_t *dest, int64_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<uint8_t>(uint8_t *dest, uint8_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<uint16_t>(
    uint16_t *dest, uint16_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<uint32_t>(
    uint32_t *dest, uint32_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
extern template void launchSum<uint64_t>(
    uint64_t *dest, uint64_t *source_d_pd[], uint32_t numInstances, uint64_t numElements, Stream stream);
