#include "CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Arithmetic/SumManyToOne.h"

// compute -label*ln(probability) elementwise
template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
__global__ void crossEntropyLossPartOne_perClassLabels(LABEL_TYPE *labels,
                                                       PROBABILITY_TYPE *probabilities,
                                                       float *workspace,
                                                       uint64_t numElements) {
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

// compute -label*ln(probability) elementwise
// labels are user input and a bad label must not cause a crash.
template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
__global__ void crossEntropyLossPartOne_classIndexLabels(
    LABEL_TYPE *labels, PROBABILITY_TYPE *probabilities, float *loss, uint64_t numClasses, uint64_t batchSize) {
    uint32_t batchItem;
    float probability;
    LABEL_TYPE label;

    batchItem = blockIdx.x * 1024 + threadIdx.x;
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    if (label >= numClasses) {
        probability = 1.0E-15f;
    } else {
        probability = probabilities[batchItem * numClasses + label];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
    }
    loss[batchItem] = -logf(probability);

    batchItem += 256;
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    if (label >= numClasses) {
        probability = 1.0E-15f;
    } else {
        probability = probabilities[batchItem * numClasses + label];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
    }
    loss[batchItem] = -logf(probability);

    batchItem += 256;
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    if (label >= numClasses) {
        probability = 1.0E-15f;
    } else {
        probability = probabilities[batchItem * numClasses + label];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
    }
    loss[batchItem] = -logf(probability);

    batchItem += 256;
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    if (label >= numClasses) {
        probability = 1.0E-15f;
    } else {
        probability = probabilities[batchItem * numClasses + label];
        if (probability < 1.0E-15f || !isfinite(probability)) {
            probability = 1.0E-15f;
        }
    }
    loss[batchItem] = -logf(probability);
}

// lossGradient[b][i] == normalizedPrediction[b][i] - (label == i);
template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_GRADIENT_TYPE>
__global__ void lossGradient_classIndexLabels(
    LABEL_TYPE *labels, PROBABILITY_TYPE *probabilities, LOSS_GRADIENT_TYPE *lossGradient, uint64_t numClasses, uint64_t batchSize) {
    uint32_t elementNum;
    uint32_t batchItem;
    uint32_t thisClass;
    float probability;
    LABEL_TYPE label;
    LOSS_GRADIENT_TYPE lossGradientBuff;

    elementNum = blockIdx.x * 1024 + threadIdx.x;
    batchItem = elementNum / numClasses;
    thisClass = elementNum - (batchItem * numClasses);
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    probability = probabilities[batchItem * numClasses + thisClass];
    if (label == thisClass) {
        lossGradientBuff = (LOSS_GRADIENT_TYPE)probability - (LOSS_GRADIENT_TYPE)1.0f;
    } else {
        lossGradientBuff = probability;
    }
    lossGradient[batchItem * numClasses + thisClass] = lossGradientBuff;

    elementNum += 256;
    batchItem = elementNum / numClasses;
    thisClass = elementNum - (batchItem * numClasses);
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    probability = probabilities[batchItem * numClasses + thisClass];
    if (label == thisClass) {
        lossGradientBuff = (LOSS_GRADIENT_TYPE)probability - (LOSS_GRADIENT_TYPE)1.0f;
    } else {
        lossGradientBuff = probability;
    }
    lossGradient[batchItem * numClasses + thisClass] = lossGradientBuff;

    elementNum += 256;
    batchItem = elementNum / numClasses;
    thisClass = elementNum - (batchItem * numClasses);
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    probability = probabilities[batchItem * numClasses + thisClass];
    if (label == thisClass) {
        lossGradientBuff = (LOSS_GRADIENT_TYPE)probability - (LOSS_GRADIENT_TYPE)1.0f;
    } else {
        lossGradientBuff = probability;
    }
    lossGradient[batchItem * numClasses + thisClass] = lossGradientBuff;

    elementNum += 256;
    batchItem = elementNum / numClasses;
    thisClass = elementNum - (batchItem * numClasses);
    if (batchItem >= batchSize)
        return;
    label = labels[batchItem];
    probability = probabilities[batchItem * numClasses + thisClass];
    if (label == thisClass) {
        lossGradientBuff = (LOSS_GRADIENT_TYPE)probability - (LOSS_GRADIENT_TYPE)1.0f;
    } else {
        lossGradientBuff = probability;
    }
    lossGradient[batchItem * numClasses + thisClass] = lossGradientBuff;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
void launchCrossEntropyLoss_perClassLabels(LABEL_TYPE *labels_d,
                                           PROBABILITY_TYPE *probabilities_d,
                                           float *workspace_d,
                                           float *loss_d,
                                           uint32_t numClasses,
                                           uint32_t batchSize,
                                           Stream stream) {
    uint64_t numElements = (uint64_t)numClasses * (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);
    crossEntropyLossPartOne_perClassLabels<LABEL_TYPE, PROBABILITY_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(labels_d, probabilities_d, workspace_d, numElements);
    launchSumManyToOne(workspace_d, loss_d, numClasses, batchSize, false, false, stream);
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
void launchCrossEntropyLoss_classIndexLabels(LABEL_TYPE *labels_d,
                                             PROBABILITY_TYPE *probabilities_d,
                                             float *workspace_d,
                                             float *loss_d,
                                             uint32_t numClasses,
                                             uint32_t batchSize,
                                             Stream stream) {
    dim3 blockSize(256);
    dim3 gridSize((batchSize + 1023) / 1024);
    crossEntropyLossPartOne_classIndexLabels<LABEL_TYPE, PROBABILITY_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(labels_d, probabilities_d, loss_d, numClasses, batchSize);
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_GRADIENT_TYPE>
void launchLossGradient_classIndexLabels(LABEL_TYPE *labels_d,
                                         PROBABILITY_TYPE *probabilities_d,
                                         LOSS_GRADIENT_TYPE *lossGradient_d,
                                         uint64_t numClasses,
                                         uint64_t batchSize,
                                         Stream stream) {
    uint64_t numElements = (uint64_t)numClasses * (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);

    lossGradient_classIndexLabels<LABEL_TYPE, PROBABILITY_TYPE, LOSS_GRADIENT_TYPE>
        <<<gridSize, blockSize, 0, stream.getStream()>>>(labels_d, probabilities_d, lossGradient_d, numClasses, batchSize);
}

template void launchCrossEntropyLoss_perClassLabels<uint8_t, half>(
    uint8_t *labels_d, half *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_perClassLabels<uint8_t, float>(
    uint8_t *labels_d, float *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_perClassLabels<half, half>(
    half *labels_d, half *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_perClassLabels<half, float>(
    half *labels_d, float *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_perClassLabels<float, half>(
    float *labels_d, half *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_perClassLabels<float, float>(
    float *labels_d, float *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_classIndexLabels<uint8_t, half>(
    uint8_t *labels_d, half *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_classIndexLabels<uint8_t, float>(
    uint8_t *labels_d, float *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_classIndexLabels<uint16_t, half>(
    uint16_t *labels_d, half *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_classIndexLabels<uint16_t, float>(
    uint16_t *labels_d, float *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_classIndexLabels<uint32_t, half>(
    uint32_t *labels_d, half *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchCrossEntropyLoss_classIndexLabels<uint32_t, float>(
    uint32_t *labels_d, float *probabilities_d, float *workspace_d, float *loss_d, uint32_t numClasses, uint32_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint8_t, half, half>(
    uint8_t *labels_d, half *probabilities_d, half *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint8_t, half, float>(
    uint8_t *labels_d, half *probabilities_d, float *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint8_t, float, half>(
    uint8_t *labels_d, float *probabilities_d, half *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint8_t, float, float>(
    uint8_t *labels_d, float *probabilities_d, float *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint16_t, half, half>(
    uint16_t *labels_d, half *probabilities_d, half *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint16_t, half, float>(
    uint16_t *labels_d, half *probabilities_d, float *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint16_t, float, half>(
    uint16_t *labels_d, float *probabilities_d, half *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint16_t, float, float>(
    uint16_t *labels_d, float *probabilities_d, float *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint32_t, half, half>(
    uint32_t *labels_d, half *probabilities_d, half *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint32_t, half, float>(
    uint32_t *labels_d, half *probabilities_d, float *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint32_t, float, half>(
    uint32_t *labels_d, float *probabilities_d, half *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);

template void launchLossGradient_classIndexLabels<uint32_t, float, float>(
    uint32_t *labels_d, float *probabilities_d, float *lossGradient_d, uint64_t numClasses, uint64_t batchSize, Stream stream);
