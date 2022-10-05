#include "CategoricalCrossEntropyLoss.h"

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss(
    uint32_t numClasses, uint32_t batchSize, void *labels, void *probabilities, void *loss, void *gradient, bool computeGradient) {
    uint32_t elementwiseLossIndex = blockIdx.x * 1024 + threadIdx.x;
    uint32_t batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    const PROBABILITY_TYPE MIN_PROBABILITY = 0.001f;
    LABEL_TYPE label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    LOSS_TYPE elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    PROBABILITY_TYPE elementwiseGradient;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase(uint32_t numClasses,
                                                                         uint32_t batchSize,
                                                                         void *classOfHotLabels,
                                                                         void *probabilities,
                                                                         void *loss,
                                                                         void *gradient,
                                                                         bool computeGradient) {
    uint32_t elementwiseLossIndex = blockIdx.x * 1024 + threadIdx.x;
    uint32_t batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    const PROBABILITY_TYPE MIN_PROBABILITY = 0.001f;
    uint32_t outputClass = elementwiseLossIndex % numClasses;
    LOSS_TYPE elementwiseLoss = 0.0f;
    uint32_t classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    PROBABILITY_TYPE elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;

    elementwiseLossIndex += 256;
    uint32_t oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;

    elementwiseLossIndex += 256;
    oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;

    elementwiseLossIndex += 256;
    oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss_withScale(uint32_t numClasses,
                                                                 uint32_t batchSize,
                                                                 void *labels,
                                                                 void *probabilities,
                                                                 void *loss,
                                                                 void *gradient,
                                                                 bool computeGradient,
                                                                 float gradientScale) {
    uint32_t elementwiseLossIndex = blockIdx.x * 1024 + threadIdx.x;
    uint32_t batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    const PROBABILITY_TYPE MIN_PROBABILITY = 0.001f;
    LABEL_TYPE label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    LOSS_TYPE elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    PROBABILITY_TYPE elementwiseGradient;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;

    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase_withScale(uint32_t numClasses,
                                                                                   uint32_t batchSize,
                                                                                   void *classOfHotLabels,
                                                                                   void *probabilities,
                                                                                   void *loss,
                                                                                   void *gradient,
                                                                                   bool computeGradient,
                                                                                   float gradientScale) {
    uint32_t elementwiseLossIndex = blockIdx.x * 1024 + threadIdx.x;
    uint32_t batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    const PROBABILITY_TYPE MIN_PROBABILITY = 0.001f;
    uint32_t outputClass = elementwiseLossIndex % numClasses;
    LOSS_TYPE elementwiseLoss = 0.0f;
    uint32_t classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    PROBABILITY_TYPE elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
            elementwiseLoss *= gradientScale;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;

    elementwiseLossIndex += 256;
    uint32_t oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
            elementwiseLoss *= gradientScale;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;

    elementwiseLossIndex += 256;
    oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
            elementwiseLoss *= gradientScale;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;

    elementwiseLossIndex += 256;
    oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    elementwiseGradient = (float)probability;
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
            elementwiseLoss *= gradientScale;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCategoricalCrossEntropyLoss(void *labels_d,
                                                  void *probabilities_d,
                                                  void *loss_d,
                                                  void *gradient_d,
                                                  uint32_t numClasses,
                                                  uint32_t batchSize,
                                                  bool computeGradient,
                                                  uint32_t lossScalingFactor,
                                                  Stream stream) {
    uint64_t numElements = (uint64_t)numClasses * (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);

    if (lossScalingFactor == 1) {
        elementWiseCategoricalCrossEntropyLoss<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(
            numClasses, batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient);
    } else {
        elementWiseCategoricalCrossEntropyLoss_withScale<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient, (float)lossScalingFactor);
    }
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase(void *classOfHotLabels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    Stream stream) {
    uint64_t numElements = (uint64_t)numClasses * (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);

    if (lossScalingFactor == 1) {
        elementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, classOfHotLabels_d, probabilities_d, loss_d, gradient_d, computeGradient);
    } else {
        elementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase_withScale<INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, classOfHotLabels_d, probabilities_d, loss_d, gradient_d, computeGradient, (float)lossScalingFactor);
    }
}

template void launchElementWiseCategoricalCrossEntropyLoss<bool, half, half>(void *labels_d,
                                                                             void *probabilities_d,
                                                                             void *loss_d,
                                                                             void *gradient_d,
                                                                             uint32_t numClasses,
                                                                             uint32_t batchSize,
                                                                             bool computeGradient,
                                                                             uint32_t lossScalingFactor,
                                                                             Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<bool, half, float>(void *labels_d,
                                                                              void *probabilities_d,
                                                                              void *loss_d,
                                                                              void *gradient_d,
                                                                              uint32_t numClasses,
                                                                              uint32_t batchSize,
                                                                              bool computeGradient,
                                                                              uint32_t lossScalingFactor,
                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<bool, float, half>(void *labels_d,
                                                                              void *probabilities_d,
                                                                              void *loss_d,
                                                                              void *gradient_d,
                                                                              uint32_t numClasses,
                                                                              uint32_t batchSize,
                                                                              bool computeGradient,
                                                                              uint32_t lossScalingFactor,
                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<bool, float, float>(void *labels_d,
                                                                               void *probabilities_d,
                                                                               void *loss_d,
                                                                               void *gradient_d,
                                                                               uint32_t numClasses,
                                                                               uint32_t batchSize,
                                                                               bool computeGradient,
                                                                               uint32_t lossScalingFactor,
                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint8_t, half, half>(void *labels_d,
                                                                                void *probabilities_d,
                                                                                void *loss_d,
                                                                                void *gradient_d,
                                                                                uint32_t numClasses,
                                                                                uint32_t batchSize,
                                                                                bool computeGradient,
                                                                                uint32_t lossScalingFactor,
                                                                                Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint8_t, half, float>(void *labels_d,
                                                                                 void *probabilities_d,
                                                                                 void *loss_d,
                                                                                 void *gradient_d,
                                                                                 uint32_t numClasses,
                                                                                 uint32_t batchSize,
                                                                                 bool computeGradient,
                                                                                 uint32_t lossScalingFactor,
                                                                                 Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint8_t, float, half>(void *labels_d,
                                                                                 void *probabilities_d,
                                                                                 void *loss_d,
                                                                                 void *gradient_d,
                                                                                 uint32_t numClasses,
                                                                                 uint32_t batchSize,
                                                                                 bool computeGradient,
                                                                                 uint32_t lossScalingFactor,
                                                                                 Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint8_t, float, float>(void *labels_d,
                                                                                  void *probabilities_d,
                                                                                  void *loss_d,
                                                                                  void *gradient_d,
                                                                                  uint32_t numClasses,
                                                                                  uint32_t batchSize,
                                                                                  bool computeGradient,
                                                                                  uint32_t lossScalingFactor,
                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint16_t, half, half>(void *labels_d,
                                                                                 void *probabilities_d,
                                                                                 void *loss_d,
                                                                                 void *gradient_d,
                                                                                 uint32_t numClasses,
                                                                                 uint32_t batchSize,
                                                                                 bool computeGradient,
                                                                                 uint32_t lossScalingFactor,
                                                                                 Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint16_t, half, float>(void *labels_d,
                                                                                  void *probabilities_d,
                                                                                  void *loss_d,
                                                                                  void *gradient_d,
                                                                                  uint32_t numClasses,
                                                                                  uint32_t batchSize,
                                                                                  bool computeGradient,
                                                                                  uint32_t lossScalingFactor,
                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint16_t, float, half>(void *labels_d,
                                                                                  void *probabilities_d,
                                                                                  void *loss_d,
                                                                                  void *gradient_d,
                                                                                  uint32_t numClasses,
                                                                                  uint32_t batchSize,
                                                                                  bool computeGradient,
                                                                                  uint32_t lossScalingFactor,
                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint16_t, float, float>(void *labels_d,
                                                                                   void *probabilities_d,
                                                                                   void *loss_d,
                                                                                   void *gradient_d,
                                                                                   uint32_t numClasses,
                                                                                   uint32_t batchSize,
                                                                                   bool computeGradient,
                                                                                   uint32_t lossScalingFactor,
                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint32_t, half, half>(void *labels_d,
                                                                                 void *probabilities_d,
                                                                                 void *loss_d,
                                                                                 void *gradient_d,
                                                                                 uint32_t numClasses,
                                                                                 uint32_t batchSize,
                                                                                 bool computeGradient,
                                                                                 uint32_t lossScalingFactor,
                                                                                 Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint32_t, half, float>(void *labels_d,
                                                                                  void *probabilities_d,
                                                                                  void *loss_d,
                                                                                  void *gradient_d,
                                                                                  uint32_t numClasses,
                                                                                  uint32_t batchSize,
                                                                                  bool computeGradient,
                                                                                  uint32_t lossScalingFactor,
                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint32_t, float, half>(void *labels_d,
                                                                                  void *probabilities_d,
                                                                                  void *loss_d,
                                                                                  void *gradient_d,
                                                                                  uint32_t numClasses,
                                                                                  uint32_t batchSize,
                                                                                  bool computeGradient,
                                                                                  uint32_t lossScalingFactor,
                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<uint32_t, float, float>(void *labels_d,
                                                                                   void *probabilities_d,
                                                                                   void *loss_d,
                                                                                   void *gradient_d,
                                                                                   uint32_t numClasses,
                                                                                   uint32_t batchSize,
                                                                                   bool computeGradient,
                                                                                   uint32_t lossScalingFactor,
                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<half, half, half>(void *labels_d,
                                                                             void *probabilities_d,
                                                                             void *loss_d,
                                                                             void *gradient_d,
                                                                             uint32_t numClasses,
                                                                             uint32_t batchSize,
                                                                             bool computeGradient,
                                                                             uint32_t lossScalingFactor,
                                                                             Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<half, half, float>(void *labels_d,
                                                                              void *probabilities_d,
                                                                              void *loss_d,
                                                                              void *gradient_d,
                                                                              uint32_t numClasses,
                                                                              uint32_t batchSize,
                                                                              bool computeGradient,
                                                                              uint32_t lossScalingFactor,
                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<half, float, half>(void *labels_d,
                                                                              void *probabilities_d,
                                                                              void *loss_d,
                                                                              void *gradient_d,
                                                                              uint32_t numClasses,
                                                                              uint32_t batchSize,
                                                                              bool computeGradient,
                                                                              uint32_t lossScalingFactor,
                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<half, float, float>(void *labels_d,
                                                                               void *probabilities_d,
                                                                               void *loss_d,
                                                                               void *gradient_d,
                                                                               uint32_t numClasses,
                                                                               uint32_t batchSize,
                                                                               bool computeGradient,
                                                                               uint32_t lossScalingFactor,
                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<float, half, half>(void *labels_d,
                                                                              void *probabilities_d,
                                                                              void *loss_d,
                                                                              void *gradient_d,
                                                                              uint32_t numClasses,
                                                                              uint32_t batchSize,
                                                                              bool computeGradient,
                                                                              uint32_t lossScalingFactor,
                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<float, half, float>(void *labels_d,
                                                                               void *probabilities_d,
                                                                               void *loss_d,
                                                                               void *gradient_d,
                                                                               uint32_t numClasses,
                                                                               uint32_t batchSize,
                                                                               bool computeGradient,
                                                                               uint32_t lossScalingFactor,
                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<float, float, half>(void *labels_d,
                                                                               void *probabilities_d,
                                                                               void *loss_d,
                                                                               void *gradient_d,
                                                                               uint32_t numClasses,
                                                                               uint32_t batchSize,
                                                                               bool computeGradient,
                                                                               uint32_t lossScalingFactor,
                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss<float, float, float>(void *labels_d,
                                                                                void *probabilities_d,
                                                                                void *loss_d,
                                                                                void *gradient_d,
                                                                                uint32_t numClasses,
                                                                                uint32_t batchSize,
                                                                                bool computeGradient,
                                                                                uint32_t lossScalingFactor,
                                                                                Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint8_t, half, half>(void *labels_d,
                                                                                                  void *probabilities_d,
                                                                                                  void *loss_d,
                                                                                                  void *gradient_d,
                                                                                                  uint32_t numClasses,
                                                                                                  uint32_t batchSize,
                                                                                                  bool computeGradient,
                                                                                                  uint32_t lossScalingFactor,
                                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint8_t, half, float>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint8_t, float, half>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint8_t, float, float>(void *labels_d,
                                                                                                    void *probabilities_d,
                                                                                                    void *loss_d,
                                                                                                    void *gradient_d,
                                                                                                    uint32_t numClasses,
                                                                                                    uint32_t batchSize,
                                                                                                    bool computeGradient,
                                                                                                    uint32_t lossScalingFactor,
                                                                                                    Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint16_t, half, half>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint16_t, half, float>(void *labels_d,
                                                                                                    void *probabilities_d,
                                                                                                    void *loss_d,
                                                                                                    void *gradient_d,
                                                                                                    uint32_t numClasses,
                                                                                                    uint32_t batchSize,
                                                                                                    bool computeGradient,
                                                                                                    uint32_t lossScalingFactor,
                                                                                                    Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint16_t, float, half>(void *labels_d,
                                                                                                    void *probabilities_d,
                                                                                                    void *loss_d,
                                                                                                    void *gradient_d,
                                                                                                    uint32_t numClasses,
                                                                                                    uint32_t batchSize,
                                                                                                    bool computeGradient,
                                                                                                    uint32_t lossScalingFactor,
                                                                                                    Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint16_t, float, float>(void *labels_d,
                                                                                                     void *probabilities_d,
                                                                                                     void *loss_d,
                                                                                                     void *gradient_d,
                                                                                                     uint32_t numClasses,
                                                                                                     uint32_t batchSize,
                                                                                                     bool computeGradient,
                                                                                                     uint32_t lossScalingFactor,
                                                                                                     Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint32_t, half, half>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint32_t, half, float>(void *labels_d,
                                                                                                    void *probabilities_d,
                                                                                                    void *loss_d,
                                                                                                    void *gradient_d,
                                                                                                    uint32_t numClasses,
                                                                                                    uint32_t batchSize,
                                                                                                    bool computeGradient,
                                                                                                    uint32_t lossScalingFactor,
                                                                                                    Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint32_t, float, half>(void *labels_d,
                                                                                                    void *probabilities_d,
                                                                                                    void *loss_d,
                                                                                                    void *gradient_d,
                                                                                                    uint32_t numClasses,
                                                                                                    uint32_t batchSize,
                                                                                                    bool computeGradient,
                                                                                                    uint32_t lossScalingFactor,
                                                                                                    Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase<uint32_t, float, float>(void *labels_d,
                                                                                                     void *probabilities_d,
                                                                                                     void *loss_d,
                                                                                                     void *gradient_d,
                                                                                                     uint32_t numClasses,
                                                                                                     uint32_t batchSize,
                                                                                                     bool computeGradient,
                                                                                                     uint32_t lossScalingFactor,
                                                                                                     Stream stream);