#include "CategoricalCrossEntropyLoss.h"

using namespace std;

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss_oneHotLabels(
    uint32_t numClasses, uint32_t batchSize, void *labels, void *probabilities, void *loss, void *gradient, bool computeGradient) {
    uint32_t elementwiseLossIndex = blockIdx.x * 1024 + threadIdx.x;
    uint32_t batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    const PROBABILITY_TYPE MIN_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.000062f : 1E-36;
    LABEL_TYPE label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    PROBABILITY_TYPE elementwiseGradient;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    LOSS_TYPE elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = (float)probability - (float)label;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss_classIndexLabels(uint32_t numClasses,
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
    const PROBABILITY_TYPE MIN_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.000062f : 1E-36;
    uint32_t outputClass = elementwiseLossIndex % numClasses;
    LOSS_TYPE elementwiseLoss = 0.0f;
    uint32_t classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    PROBABILITY_TYPE elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    uint32_t oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    // Only read the class of the batch item if the batch item changed so its class was not previously loaded
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

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
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

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
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss_oneHotLabels_withScale(uint32_t numClasses,
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
    const PROBABILITY_TYPE MIN_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.000062f : 1E-36;
    LABEL_TYPE label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    PROBABILITY_TYPE elementwiseGradient;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    LOSS_TYPE elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;

    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[elementwiseLossIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = gradientScale * ((float)probability - (float)label);
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    elementWiseLoss = (float)-label * logf(probability);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementWiseLoss;
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCategoricalCrossEntropyLoss_classIndexLabels_withScale(uint32_t numClasses,
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
    const PROBABILITY_TYPE MIN_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.000062f : 1E-36;
    uint32_t outputClass = elementwiseLossIndex % numClasses;
    LOSS_TYPE elementwiseLoss = 0.0f;
    uint32_t classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    PROBABILITY_TYPE elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    uint32_t oldBatchIndex = batchIndex;
    batchIndex = elementwiseLossIndex / numClasses;
    if (batchIndex >= batchSize)
        return;
    // Only read the class of the batch item if the batch item changed so its class was not previously loaded
    if (oldBatchIndex != batchIndex)
        classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    outputClass = elementwiseLossIndex % numClasses;
    elementwiseLoss = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

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
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

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
    probability = ((PROBABILITY_TYPE *)probabilities)[elementwiseLossIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    elementwiseGradient = probability;
    if (probability < MIN_PROBABILITY)
        probability = MIN_PROBABILITY;
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient)
            elementwiseGradient -= (PROBABILITY_TYPE)1.0f;
    }
    if (computeGradient) {
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels(void *labels_d,
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

    ScopedGpu scopedGpu(stream.getGpuNum());

    if (lossScalingFactor == 1 || !computeGradient) {
        elementWiseCategoricalCrossEntropyLoss_oneHotLabels<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient);
    } else {
        elementWiseCategoricalCrossEntropyLoss_oneHotLabels_withScale<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient, (float)lossScalingFactor);
    }
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels(void *classOfHotLabels_d,
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

    // Do not use the unsafe variants
    if (is_same<INDEX_TYPE, half>::value) {
        assert(false);
        return;
    }
    if (is_same<INDEX_TYPE, float>::value) {
        assert(false);
        return;
    }
    if (is_same<INDEX_TYPE, bool>::value) {
        assert(false);
        return;
    }

    ScopedGpu scopedGpu(stream.getGpuNum());

    if (lossScalingFactor == 1 || !computeGradient) {
        elementWiseCategoricalCrossEntropyLoss_classIndexLabels<INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, classOfHotLabels_d, probabilities_d, loss_d, gradient_d, computeGradient);
    } else {
        elementWiseCategoricalCrossEntropyLoss_classIndexLabels_withScale<INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, classOfHotLabels_d, probabilities_d, loss_d, gradient_d, computeGradient, (float)lossScalingFactor);
    }
}

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<bool, half, half>(void *labels_d,
                                                                                          void *probabilities_d,
                                                                                          void *loss_d,
                                                                                          void *gradient_d,
                                                                                          uint32_t numClasses,
                                                                                          uint32_t batchSize,
                                                                                          bool computeGradient,
                                                                                          uint32_t lossScalingFactor,
                                                                                          Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<bool, half, float>(void *labels_d,
                                                                                           void *probabilities_d,
                                                                                           void *loss_d,
                                                                                           void *gradient_d,
                                                                                           uint32_t numClasses,
                                                                                           uint32_t batchSize,
                                                                                           bool computeGradient,
                                                                                           uint32_t lossScalingFactor,
                                                                                           Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<bool, float, half>(void *labels_d,
                                                                                           void *probabilities_d,
                                                                                           void *loss_d,
                                                                                           void *gradient_d,
                                                                                           uint32_t numClasses,
                                                                                           uint32_t batchSize,
                                                                                           bool computeGradient,
                                                                                           uint32_t lossScalingFactor,
                                                                                           Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<bool, float, float>(void *labels_d,
                                                                                            void *probabilities_d,
                                                                                            void *loss_d,
                                                                                            void *gradient_d,
                                                                                            uint32_t numClasses,
                                                                                            uint32_t batchSize,
                                                                                            bool computeGradient,
                                                                                            uint32_t lossScalingFactor,
                                                                                            Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint8_t, half, half>(void *labels_d,
                                                                                             void *probabilities_d,
                                                                                             void *loss_d,
                                                                                             void *gradient_d,
                                                                                             uint32_t numClasses,
                                                                                             uint32_t batchSize,
                                                                                             bool computeGradient,
                                                                                             uint32_t lossScalingFactor,
                                                                                             Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint8_t, half, float>(void *labels_d,
                                                                                              void *probabilities_d,
                                                                                              void *loss_d,
                                                                                              void *gradient_d,
                                                                                              uint32_t numClasses,
                                                                                              uint32_t batchSize,
                                                                                              bool computeGradient,
                                                                                              uint32_t lossScalingFactor,
                                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint8_t, float, half>(void *labels_d,
                                                                                              void *probabilities_d,
                                                                                              void *loss_d,
                                                                                              void *gradient_d,
                                                                                              uint32_t numClasses,
                                                                                              uint32_t batchSize,
                                                                                              bool computeGradient,
                                                                                              uint32_t lossScalingFactor,
                                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint8_t, float, float>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint16_t, half, half>(void *labels_d,
                                                                                              void *probabilities_d,
                                                                                              void *loss_d,
                                                                                              void *gradient_d,
                                                                                              uint32_t numClasses,
                                                                                              uint32_t batchSize,
                                                                                              bool computeGradient,
                                                                                              uint32_t lossScalingFactor,
                                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint16_t, half, float>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint16_t, float, half>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint16_t, float, float>(void *labels_d,
                                                                                                void *probabilities_d,
                                                                                                void *loss_d,
                                                                                                void *gradient_d,
                                                                                                uint32_t numClasses,
                                                                                                uint32_t batchSize,
                                                                                                bool computeGradient,
                                                                                                uint32_t lossScalingFactor,
                                                                                                Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint32_t, half, half>(void *labels_d,
                                                                                              void *probabilities_d,
                                                                                              void *loss_d,
                                                                                              void *gradient_d,
                                                                                              uint32_t numClasses,
                                                                                              uint32_t batchSize,
                                                                                              bool computeGradient,
                                                                                              uint32_t lossScalingFactor,
                                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint32_t, half, float>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint32_t, float, half>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<uint32_t, float, float>(void *labels_d,
                                                                                                void *probabilities_d,
                                                                                                void *loss_d,
                                                                                                void *gradient_d,
                                                                                                uint32_t numClasses,
                                                                                                uint32_t batchSize,
                                                                                                bool computeGradient,
                                                                                                uint32_t lossScalingFactor,
                                                                                                Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<half, half, half>(void *labels_d,
                                                                                          void *probabilities_d,
                                                                                          void *loss_d,
                                                                                          void *gradient_d,
                                                                                          uint32_t numClasses,
                                                                                          uint32_t batchSize,
                                                                                          bool computeGradient,
                                                                                          uint32_t lossScalingFactor,
                                                                                          Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<half, half, float>(void *labels_d,
                                                                                           void *probabilities_d,
                                                                                           void *loss_d,
                                                                                           void *gradient_d,
                                                                                           uint32_t numClasses,
                                                                                           uint32_t batchSize,
                                                                                           bool computeGradient,
                                                                                           uint32_t lossScalingFactor,
                                                                                           Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<half, float, half>(void *labels_d,
                                                                                           void *probabilities_d,
                                                                                           void *loss_d,
                                                                                           void *gradient_d,
                                                                                           uint32_t numClasses,
                                                                                           uint32_t batchSize,
                                                                                           bool computeGradient,
                                                                                           uint32_t lossScalingFactor,
                                                                                           Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<half, float, float>(void *labels_d,
                                                                                            void *probabilities_d,
                                                                                            void *loss_d,
                                                                                            void *gradient_d,
                                                                                            uint32_t numClasses,
                                                                                            uint32_t batchSize,
                                                                                            bool computeGradient,
                                                                                            uint32_t lossScalingFactor,
                                                                                            Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<float, half, half>(void *labels_d,
                                                                                           void *probabilities_d,
                                                                                           void *loss_d,
                                                                                           void *gradient_d,
                                                                                           uint32_t numClasses,
                                                                                           uint32_t batchSize,
                                                                                           bool computeGradient,
                                                                                           uint32_t lossScalingFactor,
                                                                                           Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<float, half, float>(void *labels_d,
                                                                                            void *probabilities_d,
                                                                                            void *loss_d,
                                                                                            void *gradient_d,
                                                                                            uint32_t numClasses,
                                                                                            uint32_t batchSize,
                                                                                            bool computeGradient,
                                                                                            uint32_t lossScalingFactor,
                                                                                            Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<float, float, half>(void *labels_d,
                                                                                            void *probabilities_d,
                                                                                            void *loss_d,
                                                                                            void *gradient_d,
                                                                                            uint32_t numClasses,
                                                                                            uint32_t batchSize,
                                                                                            bool computeGradient,
                                                                                            uint32_t lossScalingFactor,
                                                                                            Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<float, float, float>(void *labels_d,
                                                                                             void *probabilities_d,
                                                                                             void *loss_d,
                                                                                             void *gradient_d,
                                                                                             uint32_t numClasses,
                                                                                             uint32_t batchSize,
                                                                                             bool computeGradient,
                                                                                             uint32_t lossScalingFactor,
                                                                                             Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint8_t, half, half>(void *labels_d,
                                                                                                 void *probabilities_d,
                                                                                                 void *loss_d,
                                                                                                 void *gradient_d,
                                                                                                 uint32_t numClasses,
                                                                                                 uint32_t batchSize,
                                                                                                 bool computeGradient,
                                                                                                 uint32_t lossScalingFactor,
                                                                                                 Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint8_t, half, float>(void *labels_d,
                                                                                                  void *probabilities_d,
                                                                                                  void *loss_d,
                                                                                                  void *gradient_d,
                                                                                                  uint32_t numClasses,
                                                                                                  uint32_t batchSize,
                                                                                                  bool computeGradient,
                                                                                                  uint32_t lossScalingFactor,
                                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint8_t, float, half>(void *labels_d,
                                                                                                  void *probabilities_d,
                                                                                                  void *loss_d,
                                                                                                  void *gradient_d,
                                                                                                  uint32_t numClasses,
                                                                                                  uint32_t batchSize,
                                                                                                  bool computeGradient,
                                                                                                  uint32_t lossScalingFactor,
                                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint8_t, float, float>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint16_t, half, half>(void *labels_d,
                                                                                                  void *probabilities_d,
                                                                                                  void *loss_d,
                                                                                                  void *gradient_d,
                                                                                                  uint32_t numClasses,
                                                                                                  uint32_t batchSize,
                                                                                                  bool computeGradient,
                                                                                                  uint32_t lossScalingFactor,
                                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint16_t, half, float>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint16_t, float, half>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint16_t, float, float>(void *labels_d,
                                                                                                    void *probabilities_d,
                                                                                                    void *loss_d,
                                                                                                    void *gradient_d,
                                                                                                    uint32_t numClasses,
                                                                                                    uint32_t batchSize,
                                                                                                    bool computeGradient,
                                                                                                    uint32_t lossScalingFactor,
                                                                                                    Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint32_t, half, half>(void *labels_d,
                                                                                                  void *probabilities_d,
                                                                                                  void *loss_d,
                                                                                                  void *gradient_d,
                                                                                                  uint32_t numClasses,
                                                                                                  uint32_t batchSize,
                                                                                                  bool computeGradient,
                                                                                                  uint32_t lossScalingFactor,
                                                                                                  Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint32_t, half, float>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint32_t, float, half>(void *labels_d,
                                                                                                   void *probabilities_d,
                                                                                                   void *loss_d,
                                                                                                   void *gradient_d,
                                                                                                   uint32_t numClasses,
                                                                                                   uint32_t batchSize,
                                                                                                   bool computeGradient,
                                                                                                   uint32_t lossScalingFactor,
                                                                                                   Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<uint32_t, float, float>(void *labels_d,
                                                                                                    void *probabilities_d,
                                                                                                    void *loss_d,
                                                                                                    void *gradient_d,
                                                                                                    uint32_t numClasses,
                                                                                                    uint32_t batchSize,
                                                                                                    bool computeGradient,
                                                                                                    uint32_t lossScalingFactor,
                                                                                                    Stream stream);

// The following functions will not be instantiated (they are unsafe as the class in not exact) and are guarded by assertions

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<half, half, half>(void *labels_d,
                                                                                              void *probabilities_d,
                                                                                              void *loss_d,
                                                                                              void *gradient_d,
                                                                                              uint32_t numClasses,
                                                                                              uint32_t batchSize,
                                                                                              bool computeGradient,
                                                                                              uint32_t lossScalingFactor,
                                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<half, half, float>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<half, float, half>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<half, float, float>(void *labels_d,
                                                                                                void *probabilities_d,
                                                                                                void *loss_d,
                                                                                                void *gradient_d,
                                                                                                uint32_t numClasses,
                                                                                                uint32_t batchSize,
                                                                                                bool computeGradient,
                                                                                                uint32_t lossScalingFactor,
                                                                                                Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<float, half, half>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<float, half, float>(void *labels_d,
                                                                                                void *probabilities_d,
                                                                                                void *loss_d,
                                                                                                void *gradient_d,
                                                                                                uint32_t numClasses,
                                                                                                uint32_t batchSize,
                                                                                                bool computeGradient,
                                                                                                uint32_t lossScalingFactor,
                                                                                                Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<float, float, half>(void *labels_d,
                                                                                                void *probabilities_d,
                                                                                                void *loss_d,
                                                                                                void *gradient_d,
                                                                                                uint32_t numClasses,
                                                                                                uint32_t batchSize,
                                                                                                bool computeGradient,
                                                                                                uint32_t lossScalingFactor,
                                                                                                Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<float, float, float>(void *labels_d,
                                                                                                 void *probabilities_d,
                                                                                                 void *loss_d,
                                                                                                 void *gradient_d,
                                                                                                 uint32_t numClasses,
                                                                                                 uint32_t batchSize,
                                                                                                 bool computeGradient,
                                                                                                 uint32_t lossScalingFactor,
                                                                                                 Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<bool, half, half>(void *labels_d,
                                                                                              void *probabilities_d,
                                                                                              void *loss_d,
                                                                                              void *gradient_d,
                                                                                              uint32_t numClasses,
                                                                                              uint32_t batchSize,
                                                                                              bool computeGradient,
                                                                                              uint32_t lossScalingFactor,
                                                                                              Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<bool, half, float>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<bool, float, half>(void *labels_d,
                                                                                               void *probabilities_d,
                                                                                               void *loss_d,
                                                                                               void *gradient_d,
                                                                                               uint32_t numClasses,
                                                                                               uint32_t batchSize,
                                                                                               bool computeGradient,
                                                                                               uint32_t lossScalingFactor,
                                                                                               Stream stream);

template void launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<bool, float, float>(void *labels_d,
                                                                                                void *probabilities_d,
                                                                                                void *loss_d,
                                                                                                void *gradient_d,
                                                                                                uint32_t numClasses,
                                                                                                uint32_t batchSize,
                                                                                                bool computeGradient,
                                                                                                uint32_t lossScalingFactor,
                                                                                                Stream stream);