#pragma once

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

template <typename DATA_TYPE>
void launchAdd1dBias(DATA_TYPE *features_d, DATA_TYPE *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);

extern template void launchAdd1dBias<half>(half *features, half *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<float>(
    float *features, float *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<double>(
    double *features, double *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<int8_t>(
    int8_t *features, int8_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<int16_t>(
    int16_t *features, int16_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<int32_t>(
    int32_t *features, int32_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<int64_t>(
    int64_t *features, int64_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<uint8_t>(
    uint8_t *features, uint8_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<uint16_t>(
    uint16_t *features, uint16_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<uint32_t>(
    uint32_t *features, uint32_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
extern template void launchAdd1dBias<uint64_t>(
    uint64_t *features, uint64_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
