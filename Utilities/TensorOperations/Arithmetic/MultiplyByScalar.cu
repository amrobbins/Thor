#include "MultiplyByScalar.h"

template <typename SOURCE_TYPE, typename SCALAR_TYPE, typename DEST_TYPE>
__global__ void multiplyByScalar(
    SOURCE_TYPE *source, SCALAR_TYPE *batchedScalars, DEST_TYPE *dest, uint32_t elementsPerBatch, uint32_t batchSize) {
    int element = blockIdx.x * 1024 + threadIdx.x;
    int batch = blockIdx.y;

    DEST_TYPE scalar = batchedScalars[batch];

    if (element >= elementsPerBatch)
        return;
    dest[element + batch * elementsPerBatch] = ((DEST_TYPE)source[element + batch * elementsPerBatch]) * scalar;
    element += 256;

    if (element >= elementsPerBatch)
        return;
    dest[element + batch * elementsPerBatch] = ((DEST_TYPE)source[element + batch * elementsPerBatch]) * scalar;
    element += 256;

    if (element >= elementsPerBatch)
        return;
    dest[element + batch * elementsPerBatch] = ((DEST_TYPE)source[element + batch * elementsPerBatch]) * scalar;
    element += 256;

    if (element >= elementsPerBatch)
        return;
    dest[element + batch * elementsPerBatch] = ((DEST_TYPE)source[element + batch * elementsPerBatch]) * scalar;
}

template <typename SOURCE_TYPE, typename SCALAR_TYPE, typename DEST_TYPE>
void launchMultiplyByScalar(
    SOURCE_TYPE *source_d, SCALAR_TYPE *batchedScalars_d, DEST_TYPE *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((elementsPerBatch + 1023) / 1024, batchSize);
    multiplyByScalar<SOURCE_TYPE, SCALAR_TYPE, DEST_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(source_d, batchedScalars_d, dest_d, elementsPerBatch, batchSize);
}

template void launchMultiplyByScalar<half, half, float>(
    half *source_d, half *batchedScalars_d, float *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
template void launchMultiplyByScalar<float, half, float>(
    float *source_d, half *batchedScalars_d, float *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
template void launchMultiplyByScalar<half, float, float>(
    half *source_d, float *batchedScalars_d, float *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
template void launchMultiplyByScalar<float, float, float>(
    float *source_d, float *batchedScalars_d, float *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);

template void launchMultiplyByScalar<half, half, half>(
    half *source_d, half *batchedScalars_d, half *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
template void launchMultiplyByScalar<float, half, half>(
    float *source_d, half *batchedScalars_d, half *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
template void launchMultiplyByScalar<half, float, half>(
    half *source_d, float *batchedScalars_d, half *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
template void launchMultiplyByScalar<float, float, half>(
    float *source_d, float *batchedScalars_d, half *dest_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);
