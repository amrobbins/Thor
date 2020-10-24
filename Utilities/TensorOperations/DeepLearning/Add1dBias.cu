#include "Add1dBias.h"
#include "Utilities/Common/ScopedGpu.h"

template <typename DATA_TYPE>
__global__ void add1dBias(DATA_TYPE *features, DATA_TYPE *biases, uint32_t batchSize, uint64_t numElementsPerBatch) {
    uint64_t element = blockIdx.x * 256 + threadIdx.x;
    if (element >= numElementsPerBatch)
        return;

    DATA_TYPE bias = biases[element];
#pragma unroll 8
    for (uint32_t batch = 0; batch < batchSize; ++batch)
        features[batch * numElementsPerBatch + element] += bias;
}

template <typename DATA_TYPE>
void launchAdd1dBias(DATA_TYPE *features_d, DATA_TYPE *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream) {
    assert(numElementsPerBatch > 0);
    assert(batchSize > 0);
    dim3 blockSize(256);
    dim3 gridSize((numElementsPerBatch + 255) / 256);
    ScopedGpu scopedGpu(stream.getGpuNum());
    add1dBias<DATA_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(features_d, biases_d, batchSize, numElementsPerBatch);
}

template void launchAdd1dBias<half>(half *features, half *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<float>(float *features, float *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<double>(double *features, double *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<int8_t>(int8_t *features, int8_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<int16_t>(
    int16_t *features, int16_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<int32_t>(
    int32_t *features, int32_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<int64_t>(
    int64_t *features, int64_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<uint8_t>(
    uint8_t *features, uint8_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<uint16_t>(
    uint16_t *features, uint16_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<uint32_t>(
    uint32_t *features, uint32_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
template void launchAdd1dBias<uint64_t>(
    uint64_t *features, uint64_t *biases_d, uint32_t batchSize, uint64_t numElementsPerBatch, Stream stream);
