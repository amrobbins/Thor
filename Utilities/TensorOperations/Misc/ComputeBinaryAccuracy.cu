#include "ComputeBinaryAccuracy.h"

using namespace std;
using namespace ThorImplementation;

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
template <typename PREDICTION_TYPE_2_BYTES, typename LABEL_TYPE_1_BYTE>
__global__ void computeBinaryAccuracyPerBatchItemResult21(PREDICTION_TYPE_2_BYTES *predictions,
                                                          LABEL_TYPE_1_BYTE *labels,
                                                          uint8_t *workspace,
                                                          uint32_t batchSize) {
    PREDICTION_TYPE_2_BYTES predictionsBuffer[8];
    LABEL_TYPE_1_BYTE labelsBuffer[8];
    half resultsBuffer[8];

    uint32_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= batchSize)
        return;
    uint32_t offset8Elements = offset >> 3;
    ((double2 *)predictionsBuffer)[0] = ((double2 *)predictions)[offset8Elements];
    ((uint64_t *)labelsBuffer)[0] = ((uint64_t *)labels)[offset8Elements];

    resultsBuffer[0] = ((float)labelsBuffer[0] != 0.0f) == ((float)predictionsBuffer[0] >= 0.5f);
    resultsBuffer[1] = ((float)labelsBuffer[1] != 0.0f) == ((float)predictionsBuffer[1] >= 0.5f);
    resultsBuffer[2] = ((float)labelsBuffer[2] != 0.0f) == ((float)predictionsBuffer[2] >= 0.5f);
    resultsBuffer[3] = ((float)labelsBuffer[3] != 0.0f) == ((float)predictionsBuffer[3] >= 0.5f);
    resultsBuffer[4] = ((float)labelsBuffer[4] != 0.0f) == ((float)predictionsBuffer[4] >= 0.5f);
    resultsBuffer[5] = ((float)labelsBuffer[5] != 0.0f) == ((float)predictionsBuffer[5] >= 0.5f);
    resultsBuffer[6] = ((float)labelsBuffer[6] != 0.0f) == ((float)predictionsBuffer[6] >= 0.5f);
    resultsBuffer[7] = ((float)labelsBuffer[7] != 0.0f) == ((float)predictionsBuffer[7] >= 0.5f);

    ((double2 *)workspace)[offset8Elements] = ((double2 *)resultsBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
template <typename PREDICTION_TYPE_2_BYTES, typename LABEL_TYPE_2_BYTES>
__global__ void computeBinaryAccuracyPerBatchItemResult22(PREDICTION_TYPE_2_BYTES *predictions,
                                                          LABEL_TYPE_2_BYTES *labels,
                                                          uint8_t *workspace,
                                                          uint32_t batchSize) {
    PREDICTION_TYPE_2_BYTES predictionsBuffer[4];
    LABEL_TYPE_2_BYTES labelsBuffer[4];
    half resultsBuffer[4];

    uint32_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= batchSize)
        return;
    uint32_t offset4Elements = offset >> 2;
    ((uint64_t *)predictionsBuffer)[0] = ((uint64_t *)predictions)[offset4Elements];
    ((uint64_t *)labelsBuffer)[0] = ((uint64_t *)labels)[offset4Elements];

    resultsBuffer[0] = ((float)labelsBuffer[0] != 0.0f) == ((float)predictionsBuffer[0] >= 0.5f);
    resultsBuffer[1] = ((float)labelsBuffer[1] != 0.0f) == ((float)predictionsBuffer[1] >= 0.5f);
    resultsBuffer[2] = ((float)labelsBuffer[2] != 0.0f) == ((float)predictionsBuffer[2] >= 0.5f);
    resultsBuffer[3] = ((float)labelsBuffer[3] != 0.0f) == ((float)predictionsBuffer[3] >= 0.5f);

    ((uint64_t *)workspace)[offset4Elements] = ((uint64_t *)resultsBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
template <typename PREDICTION_TYPE_2_BYTES, typename LABEL_TYPE_4_BYTES>
__global__ void computeBinaryAccuracyPerBatchItemResult24(PREDICTION_TYPE_2_BYTES *predictions,
                                                          LABEL_TYPE_4_BYTES *labels,
                                                          uint8_t *workspace,
                                                          uint32_t batchSize) {
    PREDICTION_TYPE_2_BYTES predictionsBuffer[4];
    LABEL_TYPE_4_BYTES labelsBuffer[4];
    half resultsBuffer[4];

    uint32_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= batchSize)
        return;
    uint32_t offset4Elements = offset >> 2;
    ((uint64_t *)predictionsBuffer)[0] = ((uint64_t *)predictions)[offset4Elements];
    ((double2 *)labelsBuffer)[0] = ((double2 *)labels)[offset4Elements];

    resultsBuffer[0] = ((float)labelsBuffer[0] != 0.0f) == ((float)predictionsBuffer[0] >= 0.5f);
    resultsBuffer[1] = ((float)labelsBuffer[1] != 0.0f) == ((float)predictionsBuffer[1] >= 0.5f);
    resultsBuffer[2] = ((float)labelsBuffer[2] != 0.0f) == ((float)predictionsBuffer[2] >= 0.5f);
    resultsBuffer[3] = ((float)labelsBuffer[3] != 0.0f) == ((float)predictionsBuffer[3] >= 0.5f);

    ((uint64_t *)workspace)[offset4Elements] = ((uint64_t *)resultsBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 8 elements : 2048 elements processed per block
template <typename PREDICTION_TYPE_4_BYTES, typename LABEL_TYPE_1_BYTE>
__global__ void computeBinaryAccuracyPerBatchItemResult41(PREDICTION_TYPE_4_BYTES *predictions,
                                                          LABEL_TYPE_1_BYTE *labels,
                                                          uint8_t *workspace,
                                                          uint32_t batchSize) {
    PREDICTION_TYPE_4_BYTES predictionsBuffer[8];
    LABEL_TYPE_1_BYTE labelsBuffer[8];
    half resultsBuffer[8];

    uint32_t offset = blockIdx.x * 2048 + threadIdx.x * 8;
    if (offset >= batchSize)
        return;
    uint32_t offset8Elements = offset >> 3;
    ((double4 *)predictionsBuffer)[0] = ((double4 *)predictions)[offset8Elements];
    ((uint64_t *)labelsBuffer)[0] = ((uint64_t *)labels)[offset8Elements];

    resultsBuffer[0] = ((float)labelsBuffer[0] != 0.0f) == ((float)predictionsBuffer[0] >= 0.5f);
    resultsBuffer[1] = ((float)labelsBuffer[1] != 0.0f) == ((float)predictionsBuffer[1] >= 0.5f);
    resultsBuffer[2] = ((float)labelsBuffer[2] != 0.0f) == ((float)predictionsBuffer[2] >= 0.5f);
    resultsBuffer[3] = ((float)labelsBuffer[3] != 0.0f) == ((float)predictionsBuffer[3] >= 0.5f);
    resultsBuffer[4] = ((float)labelsBuffer[4] != 0.0f) == ((float)predictionsBuffer[4] >= 0.5f);
    resultsBuffer[5] = ((float)labelsBuffer[5] != 0.0f) == ((float)predictionsBuffer[5] >= 0.5f);
    resultsBuffer[6] = ((float)labelsBuffer[6] != 0.0f) == ((float)predictionsBuffer[6] >= 0.5f);
    resultsBuffer[7] = ((float)labelsBuffer[7] != 0.0f) == ((float)predictionsBuffer[7] >= 0.5f);

    ((double2 *)workspace)[offset8Elements] = ((double2 *)resultsBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
template <typename PREDICTION_TYPE_4_BYTES, typename LABEL_TYPE_2_BYTES>
__global__ void computeBinaryAccuracyPerBatchItemResult42(PREDICTION_TYPE_4_BYTES *predictions,
                                                          LABEL_TYPE_2_BYTES *labels,
                                                          uint8_t *workspace,
                                                          uint32_t batchSize) {
    PREDICTION_TYPE_4_BYTES predictionsBuffer[4];
    LABEL_TYPE_2_BYTES labelsBuffer[4];
    half resultsBuffer[4];

    uint32_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= batchSize)
        return;
    uint32_t offset4Elements = offset >> 2;
    ((double2 *)predictionsBuffer)[0] = ((double2 *)predictions)[offset4Elements];
    ((uint64_t *)labelsBuffer)[0] = ((uint64_t *)labels)[offset4Elements];

    resultsBuffer[0] = ((float)labelsBuffer[0] != 0.0f) == ((float)predictionsBuffer[0] >= 0.5f);
    resultsBuffer[1] = ((float)labelsBuffer[1] != 0.0f) == ((float)predictionsBuffer[1] >= 0.5f);
    resultsBuffer[2] = ((float)labelsBuffer[2] != 0.0f) == ((float)predictionsBuffer[2] >= 0.5f);
    resultsBuffer[3] = ((float)labelsBuffer[3] != 0.0f) == ((float)predictionsBuffer[3] >= 0.5f);

    ((uint64_t *)workspace)[offset4Elements] = ((uint64_t *)resultsBuffer)[0];
}

// Each block is 8 warps of 32 threads = 256 threads per block
// each thread reads 4 elements : 1024 elements processed per block
template <typename PREDICTION_TYPE_4_BYTES, typename LABEL_TYPE_4_BYTES>
__global__ void computeBinaryAccuracyPerBatchItemResult44(PREDICTION_TYPE_4_BYTES *predictions,
                                                          LABEL_TYPE_4_BYTES *labels,
                                                          uint8_t *workspace,
                                                          uint32_t batchSize) {
    PREDICTION_TYPE_4_BYTES predictionsBuffer[4];
    LABEL_TYPE_4_BYTES labelsBuffer[4];
    half resultsBuffer[4];

    uint32_t offset = blockIdx.x * 1024 + threadIdx.x * 4;
    if (offset >= batchSize)
        return;
    uint32_t offset4Elements = offset >> 2;
    ((double2 *)predictionsBuffer)[0] = ((double2 *)predictions)[offset4Elements];
    ((double2 *)labelsBuffer)[0] = ((double2 *)labels)[offset4Elements];

    resultsBuffer[0] = ((float)labelsBuffer[0] != 0.0f) == ((float)predictionsBuffer[0] >= 0.5f);
    resultsBuffer[1] = ((float)labelsBuffer[1] != 0.0f) == ((float)predictionsBuffer[1] >= 0.5f);
    resultsBuffer[2] = ((float)labelsBuffer[2] != 0.0f) == ((float)predictionsBuffer[2] >= 0.5f);
    resultsBuffer[3] = ((float)labelsBuffer[3] != 0.0f) == ((float)predictionsBuffer[3] >= 0.5f);

    ((uint64_t *)workspace)[offset4Elements] = ((uint64_t *)resultsBuffer)[0];
}

__global__ void scalarDivide(float *numerator, uint32_t denominator) {
    float numeratorBuffer = numerator[0];
    numerator[0] = numeratorBuffer / denominator;
}

shared_ptr<BatchReduce> createBinaryAccuracyBatchReduce(uint32_t batchSize, Stream stream) {
    return make_shared<BatchReduce>(batchSize,
                                    batchSize,
                                    1,
                                    true,
                                    false,
                                    ThorImplementation::TensorDescriptor::DataType::FP16,
                                    ThorImplementation::TensorDescriptor::DataType::FP32,
                                    stream);
}

template <typename PREDICTION_TYPE, typename LABEL_TYPE>
void launchComputeBinaryAccuracy(ThorImplementation::Tensor accuracy_d,
                                 PREDICTION_TYPE *predictions_d,
                                 LABEL_TYPE *labels_d,
                                 ThorImplementation::Tensor workspace_d,
                                 uint32_t batchSize,
                                 shared_ptr<BatchReduce> batchReduce,
                                 Stream stream) {
    assert(batchSize > 0);

    assert(workspace_d.getDataType() == ThorImplementation::TensorDescriptor::DataType::FP16);
    assert(accuracy_d.getDataType() == ThorImplementation::TensorDescriptor::DataType::FP32);
    float *accuracy_m = (float *)accuracy_d.getMemPtr();
    uint8_t *workspace_m = (uint8_t *)workspace_d.getMemPtr();

    dim3 blockSize(256);
    dim3 gridSize((batchSize + 1023) / 1024);
    if (sizeof(PREDICTION_TYPE) == 2) {
        if (sizeof(LABEL_TYPE) == 1) {
            gridSize = dim3((batchSize + 2047) / 2048);
            computeBinaryAccuracyPerBatchItemResult21<<<gridSize, blockSize, 0, stream>>>(predictions_d, labels_d, workspace_m, batchSize);
        } else if (sizeof(LABEL_TYPE) == 2) {
            computeBinaryAccuracyPerBatchItemResult22<<<gridSize, blockSize, 0, stream>>>(predictions_d, labels_d, workspace_m, batchSize);
        } else if (sizeof(LABEL_TYPE) == 4) {
            computeBinaryAccuracyPerBatchItemResult24<<<gridSize, blockSize, 0, stream>>>(predictions_d, labels_d, workspace_m, batchSize);
        } else {
            assert(false);
        }
    } else if (sizeof(PREDICTION_TYPE) == 4) {
        if (sizeof(LABEL_TYPE) == 1) {
            gridSize = dim3((batchSize + 2047) / 2048);
            computeBinaryAccuracyPerBatchItemResult41<<<gridSize, blockSize, 0, stream>>>(predictions_d, labels_d, workspace_m, batchSize);
        } else if (sizeof(LABEL_TYPE) == 2) {
            computeBinaryAccuracyPerBatchItemResult42<<<gridSize, blockSize, 0, stream>>>(predictions_d, labels_d, workspace_m, batchSize);
        } else if (sizeof(LABEL_TYPE) == 4) {
            computeBinaryAccuracyPerBatchItemResult44<<<gridSize, blockSize, 0, stream>>>(predictions_d, labels_d, workspace_m, batchSize);
        } else {
            assert(false);
        }
    } else {
        assert(false);
    }

    assert(batchReduce->getStream() == stream);
    // Sum and divide by batch size:
    batchReduce->reduce(workspace_d, accuracy_d);
}

template void launchComputeBinaryAccuracy<half, uint8_t>(ThorImplementation::Tensor accuracy_d,
                                                         half *predictions_d,
                                                         uint8_t *labels_d,
                                                         ThorImplementation::Tensor workspace_d,
                                                         uint32_t batchSize,
                                                         shared_ptr<BatchReduce> batchReduce,
                                                         Stream stream);
template void launchComputeBinaryAccuracy<half, uint16_t>(ThorImplementation::Tensor accuracy_d,
                                                          half *predictions_d,
                                                          uint16_t *labels_d,
                                                          ThorImplementation::Tensor workspace_d,
                                                          uint32_t batchSize,
                                                          shared_ptr<BatchReduce> batchReduce,
                                                          Stream stream);
template void launchComputeBinaryAccuracy<half, uint32_t>(ThorImplementation::Tensor accuracy_d,
                                                          half *predictions_d,
                                                          uint32_t *labels_d,
                                                          ThorImplementation::Tensor workspace_d,
                                                          uint32_t batchSize,
                                                          shared_ptr<BatchReduce> batchReduce,
                                                          Stream stream);
template void launchComputeBinaryAccuracy<half, half>(ThorImplementation::Tensor accuracy_d,
                                                      half *predictions_d,
                                                      half *labels_d,
                                                      ThorImplementation::Tensor workspace_d,
                                                      uint32_t batchSize,
                                                      shared_ptr<BatchReduce> batchReduce,
                                                      Stream stream);
template void launchComputeBinaryAccuracy<half, float>(ThorImplementation::Tensor accuracy_d,
                                                       half *predictions_d,
                                                       float *labels_d,
                                                       ThorImplementation::Tensor workspace_d,
                                                       uint32_t batchSize,
                                                       shared_ptr<BatchReduce> batchReduce,
                                                       Stream stream);

template void launchComputeBinaryAccuracy<float, uint8_t>(ThorImplementation::Tensor accuracy_d,
                                                          float *predictions_d,
                                                          uint8_t *labels_d,
                                                          ThorImplementation::Tensor workspace_d,
                                                          uint32_t batchSize,
                                                          shared_ptr<BatchReduce> batchReduce,
                                                          Stream stream);
template void launchComputeBinaryAccuracy<float, uint16_t>(ThorImplementation::Tensor accuracy_d,
                                                           float *predictions_d,
                                                           uint16_t *labels_d,
                                                           ThorImplementation::Tensor workspace_d,
                                                           uint32_t batchSize,
                                                           shared_ptr<BatchReduce> batchReduce,
                                                           Stream stream);
template void launchComputeBinaryAccuracy<float, uint32_t>(ThorImplementation::Tensor accuracy_d,
                                                           float *predictions_d,
                                                           uint32_t *labels_d,
                                                           ThorImplementation::Tensor workspace_d,
                                                           uint32_t batchSize,
                                                           shared_ptr<BatchReduce> batchReduce,
                                                           Stream stream);
template void launchComputeBinaryAccuracy<float, half>(ThorImplementation::Tensor accuracy_d,
                                                       float *predictions_d,
                                                       half *labels_d,
                                                       ThorImplementation::Tensor workspace_d,
                                                       uint32_t batchSize,
                                                       shared_ptr<BatchReduce> batchReduce,
                                                       Stream stream);
template void launchComputeBinaryAccuracy<float, float>(ThorImplementation::Tensor accuracy_d,
                                                        float *predictions_d,
                                                        float *labels_d,
                                                        ThorImplementation::Tensor workspace_d,
                                                        uint32_t batchSize,
                                                        shared_ptr<BatchReduce> batchReduce,
                                                        Stream stream);

template void launchComputeBinaryAccuracy<half, int8_t>(ThorImplementation::Tensor accuracy_d,
                                                        half *predictions_d,
                                                        int8_t *labels_d,
                                                        ThorImplementation::Tensor workspace_d,
                                                        uint32_t batchSize,
                                                        shared_ptr<BatchReduce> batchReduce,
                                                        Stream stream);
template void launchComputeBinaryAccuracy<half, int16_t>(ThorImplementation::Tensor accuracy_d,
                                                         half *predictions_d,
                                                         int16_t *labels_d,
                                                         ThorImplementation::Tensor workspace_d,
                                                         uint32_t batchSize,
                                                         shared_ptr<BatchReduce> batchReduce,
                                                         Stream stream);
template void launchComputeBinaryAccuracy<half, int32_t>(ThorImplementation::Tensor accuracy_d,
                                                         half *predictions_d,
                                                         int32_t *labels_d,
                                                         ThorImplementation::Tensor workspace_d,
                                                         uint32_t batchSize,
                                                         shared_ptr<BatchReduce> batchReduce,
                                                         Stream stream);

template void launchComputeBinaryAccuracy<float, int8_t>(ThorImplementation::Tensor accuracy_d,
                                                         float *predictions_d,
                                                         int8_t *labels_d,
                                                         ThorImplementation::Tensor workspace_d,
                                                         uint32_t batchSize,
                                                         shared_ptr<BatchReduce> batchReduce,
                                                         Stream stream);
template void launchComputeBinaryAccuracy<float, int16_t>(ThorImplementation::Tensor accuracy_d,
                                                          float *predictions_d,
                                                          int16_t *labels_d,
                                                          ThorImplementation::Tensor workspace_d,
                                                          uint32_t batchSize,
                                                          shared_ptr<BatchReduce> batchReduce,
                                                          Stream stream);
template void launchComputeBinaryAccuracy<float, int32_t>(ThorImplementation::Tensor accuracy_d,
                                                          float *predictions_d,
                                                          int32_t *labels_d,
                                                          ThorImplementation::Tensor workspace_d,
                                                          uint32_t batchSize,
                                                          shared_ptr<BatchReduce> batchReduce,
                                                          Stream stream);
