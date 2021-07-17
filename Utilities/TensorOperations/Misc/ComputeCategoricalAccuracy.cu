#include "ComputeCategoricalAccuracy.h"

using std::max;
using std::min;

// a block has blockDim.x = 32 and blockDim.y = 8
// gridDim.x = ceil(batchSize/32)
template <typename PREDICTION_TYPE, typename LABEL_TYPE>
__global__ void computeCategoricalAccuracyPhase1_perClassLabels(
    PREDICTION_TYPE *predictions, LABEL_TYPE *labels, uint8_t *workspace, uint32_t numClasses, uint32_t batchSize) {
    __shared__ PREDICTION_TYPE highestPrediction[8][32];
    __shared__ uint32_t highestPredictionClass[8][32];

    __shared__ LABEL_TYPE highestLabel[8][32];
    __shared__ uint32_t highestLabelClass[8][32];

    uint32_t batchItemsPerBlock = (batchSize + (gridDim.x - 1)) / gridDim.x;
    uint32_t batchItemBegin = batchItemsPerBlock * blockIdx.x;
    uint32_t batchItemEnd = (batchItemBegin + batchItemsPerBlock) - 1;
    if (batchItemEnd >= batchSize)
        batchItemEnd = batchSize - 1;

    for (uint32_t batchIndex = batchItemBegin + threadIdx.y; batchIndex <= batchItemEnd; batchIndex += 8) {
        uint64_t batchOffset = numClasses * batchIndex;

        PREDICTION_TYPE maxPrediction = 0.0f;
        uint32_t maxPredictionClass = 0;
        LABEL_TYPE maxLabel = 0.0f;
        uint32_t maxLabelClass = 0;
        if (threadIdx.x < numClasses) {
            maxPrediction = predictions[batchOffset + threadIdx.x];
            maxPredictionClass = threadIdx.x;
            maxLabel = labels[batchOffset + threadIdx.x];
            maxLabelClass = threadIdx.x;
        }
        for (uint32_t classIndex = threadIdx.x + 32; classIndex < numClasses; classIndex += 32) {
            if (predictions[batchOffset + classIndex] > maxPrediction) {
                maxPrediction = predictions[batchOffset + classIndex];
                maxPredictionClass = classIndex;
            }
            if (labels[batchOffset + classIndex] > maxLabel) {
                maxLabel = labels[batchOffset + classIndex];
                maxLabelClass = classIndex;
            }
        }

        highestPrediction[threadIdx.y][threadIdx.x] = maxPrediction;
        highestPredictionClass[threadIdx.y][threadIdx.x] = maxPredictionClass;
        highestLabel[threadIdx.y][threadIdx.x] = maxLabel;
        highestLabelClass[threadIdx.y][threadIdx.x] = maxLabelClass;
        if (threadIdx.x == 0) {
            for (uint32_t i = 1; i < 32; ++i) {
                if (highestPrediction[threadIdx.y][i] > maxPrediction) {
                    maxPrediction = highestPrediction[threadIdx.y][i];
                    maxPredictionClass = highestPredictionClass[threadIdx.y][i];
                }
                if (highestLabel[threadIdx.y][i] > maxLabel) {
                    maxLabel = highestLabel[threadIdx.y][i];
                    maxLabelClass = highestLabelClass[threadIdx.y][i];
                }
            }
            workspace[batchIndex] = (maxPredictionClass == maxLabelClass);
        }
    }
}

// a block has blockDim.x = 32 and blockDim.y = 8
// gridDim.x = ceil(batchSize/32)
template <typename PREDICTION_TYPE, typename LABEL_TYPE>
__global__ void computeCategoricalAccuracyPhase1_classIndexLabels(
    PREDICTION_TYPE *predictions, LABEL_TYPE *labels, uint8_t *workspace, uint32_t numClasses, uint32_t batchSize) {
    __shared__ PREDICTION_TYPE highestPrediction[8][32];
    __shared__ LABEL_TYPE highestPredictionClass[8][32];

    uint32_t batchItemsPerBlock = (batchSize + (gridDim.x - 1)) / gridDim.x;
    uint32_t batchItemBegin = batchItemsPerBlock * blockIdx.x;
    uint32_t batchItemEnd = (batchItemBegin + batchItemsPerBlock) - 1;
    if (batchItemEnd >= batchSize)
        batchItemEnd = batchSize - 1;

    for (uint32_t batchIndex = batchItemBegin + threadIdx.y; batchIndex <= batchItemEnd; batchIndex += 8) {
        uint64_t batchOffset = numClasses * batchIndex;

        PREDICTION_TYPE maxPrediction = 0.0f;
        LABEL_TYPE maxPredictionClass = 0.0f;
        if (threadIdx.x < numClasses) {
            maxPrediction = predictions[batchOffset + threadIdx.x];
            maxPredictionClass = threadIdx.x;
        }
        for (uint32_t classIndex = threadIdx.x + 32; classIndex < numClasses; classIndex += 32) {
            if (predictions[batchOffset + classIndex] > maxPrediction) {
                maxPrediction = predictions[batchOffset + classIndex];
                maxPredictionClass = classIndex;
            }
        }

        highestPrediction[threadIdx.y][threadIdx.x] = maxPrediction;
        highestPredictionClass[threadIdx.y][threadIdx.x] = maxPredictionClass;
        if (threadIdx.x == 0) {
            for (uint32_t i = 1; i < 32; ++i) {
                if (highestPrediction[threadIdx.y][i] > maxPrediction) {
                    maxPrediction = highestPrediction[threadIdx.y][i];
                    maxPredictionClass = highestPredictionClass[threadIdx.y][i];
                }
            }
            workspace[batchIndex] = (maxPredictionClass == labels[batchIndex]);
        }
    }
}

__device__ __forceinline__ float cca_warpReduce32(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0x0000ffff, val, 8);
    val += __shfl_down_sync(0x000000ff, val, 4);
    val += __shfl_down_sync(0x0000000f, val, 2);
    return val + __shfl_down_sync(0x00003, val, 1);
}

// input is an array of 1's and 0's. This is run by 1 block of up to 32 warps.
__global__ void computeCategoricalAccuracyPhase2(float *accuracy, uint8_t *workspace, uint32_t batchSize) {
    __shared__ uint32_t reduction[32];

    uint32_t warp = threadIdx.x / 32;
    if (warp * 32 >= batchSize)
        return;
    uint32_t thread = threadIdx.x % 32;

    uint32_t count = 0;
    for (uint32_t i = threadIdx.x; i < batchSize; i += blockDim.x) {
        count += workspace[i];
    }
    count = cca_warpReduce32(count);

    if (warp == 0)
        reduction[thread] = 0;
    __syncthreads();

    if (thread == 0)
        reduction[warp] = count;
    __syncthreads();

    if (warp == 0) {
        count = reduction[thread];
        count = cca_warpReduce32(count);
    }

    if (threadIdx.x == 0) {
        *accuracy = (double)count / (double)batchSize;
    }
}

template <typename PREDICTION_TYPE, typename LABEL_TYPE>
void launchComputeCategoricalAccuracy_perClassLabels(float *accuracy_d,
                                                     PREDICTION_TYPE *predictions_d,
                                                     LABEL_TYPE *labels_d,
                                                     uint8_t *workspace_d,
                                                     uint32_t numClasses,
                                                     uint32_t batchSize,
                                                     Stream stream) {
    assert(numClasses > 1);
    assert(batchSize > 0);

    uint32_t downSteps = 256 / numClasses;
    downSteps = max(1, downSteps);
    downSteps = min(8, downSteps);
    uint32_t verticalSpan = downSteps * 8;

    dim3 phase1BlockSize(32, 8);
    dim3 phase1GridSize((batchSize + (verticalSpan - 1)) / verticalSpan);
    computeCategoricalAccuracyPhase1_perClassLabels<PREDICTION_TYPE, LABEL_TYPE>
        <<<phase1GridSize, phase1BlockSize, 0, stream>>>(predictions_d, labels_d, workspace_d, numClasses, batchSize);

    dim3 phase2BlockSize(min(1024, batchSize));
    dim3 phase2GridSize(1);
    computeCategoricalAccuracyPhase2<<<phase2GridSize, phase2BlockSize, 0, stream>>>(accuracy_d, workspace_d, batchSize);
}

template <typename PREDICTION_TYPE, typename LABEL_TYPE>
void launchComputeCategoricalAccuracy_classIndexLabels(float *accuracy_d,
                                                       PREDICTION_TYPE *predictions_d,
                                                       LABEL_TYPE *labels_d,
                                                       uint8_t *workspace_d,
                                                       uint32_t numClasses,
                                                       uint32_t batchSize,
                                                       Stream stream) {
    assert(numClasses > 1);
    assert(batchSize > 0);

    uint32_t downSteps = 256 / numClasses;
    downSteps = max(1, downSteps);
    downSteps = min(8, downSteps);
    uint32_t verticalSpan = downSteps * 8;

    dim3 phase1BlockSize(32, 8);
    dim3 phase1GridSize((batchSize + (verticalSpan - 1)) / verticalSpan);
    computeCategoricalAccuracyPhase1_classIndexLabels<PREDICTION_TYPE, LABEL_TYPE>
        <<<phase1GridSize, phase1BlockSize, 0, stream>>>(predictions_d, labels_d, workspace_d, numClasses, batchSize);

    dim3 phase2BlockSize(min(1024, batchSize));
    dim3 phase2GridSize(1);
    computeCategoricalAccuracyPhase2<<<phase2GridSize, phase2BlockSize, 0, stream>>>(accuracy_d, workspace_d, batchSize);
}

template void launchComputeCategoricalAccuracy_perClassLabels<half, uint8_t>(float *accuracy_d,
                                                                             half *predictions_d,
                                                                             uint8_t *labels_d,
                                                                             uint8_t *workspace_d,
                                                                             uint32_t numClasses,
                                                                             uint32_t batchSize,
                                                                             Stream stream);
template void launchComputeCategoricalAccuracy_perClassLabels<float, uint8_t>(float *accuracy_d,
                                                                              float *predictions_d,
                                                                              uint8_t *labels_d,
                                                                              uint8_t *workspace_d,
                                                                              uint32_t numClasses,
                                                                              uint32_t batchSize,
                                                                              Stream stream);

template void launchComputeCategoricalAccuracy_perClassLabels<half, half>(
    float *accuracy_d, half *predictions_d, half *labels_d, uint8_t *workspace_d, uint32_t numClasses, uint32_t batchSize, Stream stream);
template void launchComputeCategoricalAccuracy_perClassLabels<float, half>(
    float *accuracy_d, float *predictions_d, half *labels_d, uint8_t *workspace_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchComputeCategoricalAccuracy_perClassLabels<half, float>(
    float *accuracy_d, half *predictions_d, float *labels_d, uint8_t *workspace_d, uint32_t numClasses, uint32_t batchSize, Stream stream);
template void launchComputeCategoricalAccuracy_perClassLabels<float, float>(
    float *accuracy_d, float *predictions_d, float *labels_d, uint8_t *workspace_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchComputeCategoricalAccuracy_classIndexLabels<half, uint8_t>(float *accuracy_d,
                                                                               half *predictions_d,
                                                                               uint8_t *labels_d,
                                                                               uint8_t *workspace_d,
                                                                               uint32_t numClasses,
                                                                               uint32_t batchSize,
                                                                               Stream stream);
template void launchComputeCategoricalAccuracy_classIndexLabels<float, uint8_t>(float *accuracy_d,
                                                                                float *predictions_d,
                                                                                uint8_t *labels_d,
                                                                                uint8_t *workspace_d,
                                                                                uint32_t numClasses,
                                                                                uint32_t batchSize,
                                                                                Stream stream);

template void launchComputeCategoricalAccuracy_classIndexLabels<half, uint16_t>(float *accuracy_d,
                                                                                half *predictions_d,
                                                                                uint16_t *labels_d,
                                                                                uint8_t *workspace_d,
                                                                                uint32_t numClasses,
                                                                                uint32_t batchSize,
                                                                                Stream stream);
template void launchComputeCategoricalAccuracy_classIndexLabels<float, uint16_t>(float *accuracy_d,
                                                                                 float *predictions_d,
                                                                                 uint16_t *labels_d,
                                                                                 uint8_t *workspace_d,
                                                                                 uint32_t numClasses,
                                                                                 uint32_t batchSize,
                                                                                 Stream stream);

template void launchComputeCategoricalAccuracy_classIndexLabels<half, uint32_t>(float *accuracy_d,
                                                                                half *predictions_d,
                                                                                uint32_t *labels_d,
                                                                                uint8_t *workspace_d,
                                                                                uint32_t numClasses,
                                                                                uint32_t batchSize,
                                                                                Stream stream);
template void launchComputeCategoricalAccuracy_classIndexLabels<float, uint32_t>(float *accuracy_d,
                                                                                 float *predictions_d,
                                                                                 uint32_t *labels_d,
                                                                                 uint8_t *workspace_d,
                                                                                 uint32_t numClasses,
                                                                                 uint32_t batchSize,
                                                                                 Stream stream);
