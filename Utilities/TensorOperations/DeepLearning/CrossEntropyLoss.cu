#include "CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Arithmetic/SumManyToOne.h"

// compute -label*ln(probability) elementwise
template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
__global__ void crossEntropyLossPartOne(LABEL_TYPE *labels, PROBABILITY_TYPE *probabilities, float *workspace, uint64_t numElements) {
    float label;
    float probability;
    float result;

    int element = blockIdx.x * 1024 + threadIdx.x;

    if (element >= numElements)
        return;
    label = (float)labels[element];
    if (label == 0.0f) {
        result = 0.0f;
    } else {
        probability = (float)probabilities[element];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
        result = -label * logf(probability);
    }
    workspace[element] = result;
    element += 256;

    if (element >= numElements)
        return;
    label = (float)labels[element];
    if (label == 0.0f) {
        result = 0.0f;
    } else {
        probability = (float)probabilities[element];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
        result = -label * logf(probability);
    }
    workspace[element] = result;
    element += 256;

    if (element >= numElements)
        return;
    label = (float)labels[element];
    if (label == 0.0f) {
        result = 0.0f;
    } else {
        probability = (float)probabilities[element];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
        result = -label * logf(probability);
    }
    workspace[element] = result;
    element += 256;

    if (element >= numElements)
        return;
    label = (float)labels[element];
    if (label == 0.0f) {
        result = 0.0f;
    } else {
        probability = (float)probabilities[element];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
        result = -label * logf(probability);
    }
    workspace[element] = result;
    element += 256;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
void launchCrossEntropyLoss(LABEL_TYPE *labels_d,
                            PROBABILITY_TYPE *probabilities_d,
                            float *workspace_d,
                            half *loss_d,
                            uint32_t elementsPerBatch,
                            uint32_t batchSize,
                            Stream stream) {
    uint64_t numElements = (uint64_t)elementsPerBatch * (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    crossEntropyLossPartOne<LABEL_TYPE, PROBABILITY_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(labels_d, probabilities_d, workspace_d, numElements);
    launchSumManyToOne(workspace_d, loss_d, elementsPerBatch, batchSize, false, false, stream);
}

template void launchCrossEntropyLoss<half, half>(
    half *labels_d, half *probabilities_d, float *workspace_d, half *loss_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss<float, half>(
    float *labels_d, half *probabilities_d, float *workspace_d, half *loss_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss<half, float>(
    half *labels_d, float *probabilities_d, float *workspace_d, half *loss_d, uint32_t elementsPerBatch, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss<float, float>(float *labels_d,
                                                   float *probabilities_d,
                                                   float *workspace_d,
                                                   half *loss_d,
                                                   uint32_t elementsPerBatch,
                                                   uint32_t batchSize,
                                                   Stream stream);
