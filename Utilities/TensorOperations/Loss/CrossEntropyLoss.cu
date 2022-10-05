#include "CrossEntropyLoss.h"

/**
 * CrossEntropy(predictionProbabilites, labels) = -labels_i * ln(predictionProbabilites_i), for i in numClasses
 *
 * In the special and usual case where labels are one hot, let t be the index of the single hot label then:
 * CrossEntropy(predictionProbabilites, labels) = -ln(predictionProbabilites_t) for class == t
 * CrossEntropy(predictionProbabilites, labels) = 0 for class != t
 * This is a bit faster to compute and since this is the common case it is directly implemented.
 * Also it allows sending labels as an integer array that indicate the true class, rather than a larger array of one-hot vectors.
 *
 * Gradient of CrossEntropy(predictionProbabilites, labels):
 * d/dx(-labels_i * log(predictionProbabilites_i)) = -labels_i/predictionProbabilites_i
 * One hot case:
 * d/dx(-labels_i * log(predictionProbabilites_i)) = -1/predictionProbabilites_t for class == t
 * d/dx(-labels_i * log(predictionProbabilites_i)) = 0 for class != t
 */
template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCrossEntropyLoss(
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
        elementwiseGradient = (float)-label / (float)probability;
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
        elementwiseGradient = (float)-label / (float)probability;
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
        elementwiseGradient = (float)-label / (float)probability;
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
        elementwiseGradient = (float)-label / (float)probability;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCrossEntropyLoss_oneHotSpecialCase(uint32_t numClasses,
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
    PROBABILITY_TYPE elementwiseGradient = 0.0f;
    uint32_t classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = ((PROBABILITY_TYPE)-1.0f) / probability;
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
    elementwiseGradient = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = ((PROBABILITY_TYPE)-1.0f) / probability;
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
    elementwiseGradient = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = ((PROBABILITY_TYPE)-1.0f) / probability;
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
    elementwiseGradient = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = ((PROBABILITY_TYPE)-1.0f) / probability;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCrossEntropyLoss_withScale(uint32_t numClasses,
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
        elementwiseGradient = -(gradientScale * (float)label) / (float)probability;
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
        elementwiseGradient = -(gradientScale * (float)label) / (float)probability;
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
        elementwiseGradient = -(gradientScale * (float)label) / (float)probability;
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
        elementwiseGradient = -(gradientScale * (float)label) / (float)probability;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseCrossEntropyLoss_oneHotSpecialCase_withScale(uint32_t numClasses,
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
    PROBABILITY_TYPE elementwiseGradient = 0.0f;
    uint32_t classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    PROBABILITY_TYPE probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = -((PROBABILITY_TYPE)gradientScale) / probability;
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
    elementwiseGradient = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = -((PROBABILITY_TYPE)gradientScale) / probability;
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
    elementwiseGradient = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = -((PROBABILITY_TYPE)gradientScale) / probability;
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
    elementwiseGradient = 0.0f;
    classOfHotLabel = ((INDEX_TYPE *)classOfHotLabels)[batchIndex];
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex + classOfHotLabel];
    if (probability < MIN_PROBABILITY || !isfinite((float)probability)) {
        probability = MIN_PROBABILITY;
    }
    if (outputClass == classOfHotLabel) {
        elementwiseLoss = -logf(probability);
        if (computeGradient) {
            elementwiseGradient = -((PROBABILITY_TYPE)gradientScale) / probability;
        }
    }
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
    if (computeGradient)
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCrossEntropyLoss(void *labels_d,
                                       void *probabilities_d,
                                       void *loss_d,
                                       void *gradient_d,
                                       uint32_t numClasses,
                                       uint32_t batchSize,
                                       bool computeGradient,
                                       uint32_t lossScalingFactor,
                                       bool computeCategoricalCrossEntropyGradient,
                                       bool computeBinaryCrossEntropyGradient,
                                       Stream stream) {
    assert(!(computeCategoricalCrossEntropyGradient && computeBinaryCrossEntropyGradient));

    if (computeGradient && computeCategoricalCrossEntropyGradient) {
        launchElementWiseCategoricalCrossEntropyLoss<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE>(
            labels_d, probabilities_d, loss_d, gradient_d, numClasses, batchSize, computeGradient, lossScalingFactor, stream);
        return;
    }

    if (computeGradient && computeBinaryCrossEntropyGradient) {
        /*
        launchElementWiseBinaryCrossEntropyLoss<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE>(labels_d,
                                                                                              probabilities_d,
                                                                                              loss_d,
                                                                                              gradient_d,
                                                                                              numClasses,
                                                                                              batchSize,
                                                                                              computeGradient,
                                                                                              lossScalingFactor,
                                                                                              stream);
        */
        return;
    }

    uint64_t numElements = (uint64_t)numClasses * (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);

    if (lossScalingFactor == 1) {
        elementWiseCrossEntropyLoss<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(
            numClasses, batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient);
    } else {
        elementWiseCrossEntropyLoss_withScale<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE><<<gridSize, blockSize, 0, stream.getStream()>>>(
            numClasses, batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient, (float)lossScalingFactor);
    }
}

template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCrossEntropyLoss_oneHotSpecialCase(void *classOfHotLabels_d,
                                                         void *probabilities_d,
                                                         void *loss_d,
                                                         void *gradient_d,
                                                         uint32_t numClasses,
                                                         uint32_t batchSize,
                                                         bool computeGradient,
                                                         uint32_t lossScalingFactor,
                                                         bool computeCategoricalCrossEntropyGradient,
                                                         bool computeBinaryCrossEntropyGradient,
                                                         Stream stream) {
    uint64_t numElements = (uint64_t)numClasses * (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);

    if (lossScalingFactor == 1) {
        elementWiseCrossEntropyLoss_oneHotSpecialCase<INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, classOfHotLabels_d, probabilities_d, loss_d, gradient_d, computeGradient);
    } else {
        elementWiseCrossEntropyLoss_oneHotSpecialCase_withScale<INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                numClasses, batchSize, classOfHotLabels_d, probabilities_d, loss_d, gradient_d, computeGradient, (float)lossScalingFactor);
    }
}

template void launchElementWiseCrossEntropyLoss<bool, half, half>(void *labels_d,
                                                                  void *probabilities_d,
                                                                  void *loss_d,
                                                                  void *gradient_d,
                                                                  uint32_t numClasses,
                                                                  uint32_t batchSize,
                                                                  bool computeGradient,
                                                                  uint32_t lossScalingFactor,
                                                                  bool computeCategoricalCrossEntropyGradient,
                                                                  bool computeBinaryCrossEntropyGradient,
                                                                  Stream stream);

template void launchElementWiseCrossEntropyLoss<bool, half, float>(void *labels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   bool computeCategoricalCrossEntropyGradient,
                                                                   bool computeBinaryCrossEntropyGradient,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<bool, float, half>(void *labels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   bool computeCategoricalCrossEntropyGradient,
                                                                   bool computeBinaryCrossEntropyGradient,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<bool, float, float>(void *labels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    bool computeCategoricalCrossEntropyGradient,
                                                                    bool computeBinaryCrossEntropyGradient,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, half, half>(void *labels_d,
                                                                     void *probabilities_d,
                                                                     void *loss_d,
                                                                     void *gradient_d,
                                                                     uint32_t numClasses,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     uint32_t lossScalingFactor,
                                                                     bool computeCategoricalCrossEntropyGradient,
                                                                     bool computeBinaryCrossEntropyGradient,
                                                                     Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, half, float>(void *labels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      bool computeCategoricalCrossEntropyGradient,
                                                                      bool computeBinaryCrossEntropyGradient,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, float, half>(void *labels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      bool computeCategoricalCrossEntropyGradient,
                                                                      bool computeBinaryCrossEntropyGradient,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, float, float>(void *labels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       bool computeCategoricalCrossEntropyGradient,
                                                                       bool computeBinaryCrossEntropyGradient,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, half, half>(void *labels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      bool computeCategoricalCrossEntropyGradient,
                                                                      bool computeBinaryCrossEntropyGradient,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, half, float>(void *labels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       bool computeCategoricalCrossEntropyGradient,
                                                                       bool computeBinaryCrossEntropyGradient,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, float, half>(void *labels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       bool computeCategoricalCrossEntropyGradient,
                                                                       bool computeBinaryCrossEntropyGradient,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, float, float>(void *labels_d,
                                                                        void *probabilities_d,
                                                                        void *loss_d,
                                                                        void *gradient_d,
                                                                        uint32_t numClasses,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        uint32_t lossScalingFactor,
                                                                        bool computeCategoricalCrossEntropyGradient,
                                                                        bool computeBinaryCrossEntropyGradient,
                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, half, half>(void *labels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      bool computeCategoricalCrossEntropyGradient,
                                                                      bool computeBinaryCrossEntropyGradient,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, half, float>(void *labels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       bool computeCategoricalCrossEntropyGradient,
                                                                       bool computeBinaryCrossEntropyGradient,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, float, half>(void *labels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       bool computeCategoricalCrossEntropyGradient,
                                                                       bool computeBinaryCrossEntropyGradient,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, float, float>(void *labels_d,
                                                                        void *probabilities_d,
                                                                        void *loss_d,
                                                                        void *gradient_d,
                                                                        uint32_t numClasses,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        uint32_t lossScalingFactor,
                                                                        bool computeCategoricalCrossEntropyGradient,
                                                                        bool computeBinaryCrossEntropyGradient,
                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss<half, half, half>(void *labels_d,
                                                                  void *probabilities_d,
                                                                  void *loss_d,
                                                                  void *gradient_d,
                                                                  uint32_t numClasses,
                                                                  uint32_t batchSize,
                                                                  bool computeGradient,
                                                                  uint32_t lossScalingFactor,
                                                                  bool computeCategoricalCrossEntropyGradient,
                                                                  bool computeBinaryCrossEntropyGradient,
                                                                  Stream stream);

template void launchElementWiseCrossEntropyLoss<half, half, float>(void *labels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   bool computeCategoricalCrossEntropyGradient,
                                                                   bool computeBinaryCrossEntropyGradient,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<half, float, half>(void *labels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   bool computeCategoricalCrossEntropyGradient,
                                                                   bool computeBinaryCrossEntropyGradient,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<half, float, float>(void *labels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    bool computeCategoricalCrossEntropyGradient,
                                                                    bool computeBinaryCrossEntropyGradient,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<float, half, half>(void *labels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   bool computeCategoricalCrossEntropyGradient,
                                                                   bool computeBinaryCrossEntropyGradient,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<float, half, float>(void *labels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    bool computeCategoricalCrossEntropyGradient,
                                                                    bool computeBinaryCrossEntropyGradient,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<float, float, half>(void *labels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    bool computeCategoricalCrossEntropyGradient,
                                                                    bool computeBinaryCrossEntropyGradient,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<float, float, float>(void *labels_d,
                                                                     void *probabilities_d,
                                                                     void *loss_d,
                                                                     void *gradient_d,
                                                                     uint32_t numClasses,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     uint32_t lossScalingFactor,
                                                                     bool computeCategoricalCrossEntropyGradient,
                                                                     bool computeBinaryCrossEntropyGradient,
                                                                     Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, half, half>(void *labels_d,
                                                                                       void *probabilities_d,
                                                                                       void *loss_d,
                                                                                       void *gradient_d,
                                                                                       uint32_t numClasses,
                                                                                       uint32_t batchSize,
                                                                                       bool computeGradient,
                                                                                       uint32_t lossScalingFactor,
                                                                                       bool computeCategoricalCrossEntropyGradient,
                                                                                       bool computeBinaryCrossEntropyGradient,
                                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, half, float>(void *labels_d,
                                                                                        void *probabilities_d,
                                                                                        void *loss_d,
                                                                                        void *gradient_d,
                                                                                        uint32_t numClasses,
                                                                                        uint32_t batchSize,
                                                                                        bool computeGradient,
                                                                                        uint32_t lossScalingFactor,
                                                                                        bool computeCategoricalCrossEntropyGradient,
                                                                                        bool computeBinaryCrossEntropyGradient,
                                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, float, half>(void *labels_d,
                                                                                        void *probabilities_d,
                                                                                        void *loss_d,
                                                                                        void *gradient_d,
                                                                                        uint32_t numClasses,
                                                                                        uint32_t batchSize,
                                                                                        bool computeGradient,
                                                                                        uint32_t lossScalingFactor,
                                                                                        bool computeCategoricalCrossEntropyGradient,
                                                                                        bool computeBinaryCrossEntropyGradient,
                                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint8_t, float, float>(void *labels_d,
                                                                                         void *probabilities_d,
                                                                                         void *loss_d,
                                                                                         void *gradient_d,
                                                                                         uint32_t numClasses,
                                                                                         uint32_t batchSize,
                                                                                         bool computeGradient,
                                                                                         uint32_t lossScalingFactor,
                                                                                         bool computeCategoricalCrossEntropyGradient,
                                                                                         bool computeBinaryCrossEntropyGradient,
                                                                                         Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, half, half>(void *labels_d,
                                                                                        void *probabilities_d,
                                                                                        void *loss_d,
                                                                                        void *gradient_d,
                                                                                        uint32_t numClasses,
                                                                                        uint32_t batchSize,
                                                                                        bool computeGradient,
                                                                                        uint32_t lossScalingFactor,
                                                                                        bool computeCategoricalCrossEntropyGradient,
                                                                                        bool computeBinaryCrossEntropyGradient,
                                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, half, float>(void *labels_d,
                                                                                         void *probabilities_d,
                                                                                         void *loss_d,
                                                                                         void *gradient_d,
                                                                                         uint32_t numClasses,
                                                                                         uint32_t batchSize,
                                                                                         bool computeGradient,
                                                                                         uint32_t lossScalingFactor,
                                                                                         bool computeCategoricalCrossEntropyGradient,
                                                                                         bool computeBinaryCrossEntropyGradient,
                                                                                         Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, float, half>(void *labels_d,
                                                                                         void *probabilities_d,
                                                                                         void *loss_d,
                                                                                         void *gradient_d,
                                                                                         uint32_t numClasses,
                                                                                         uint32_t batchSize,
                                                                                         bool computeGradient,
                                                                                         uint32_t lossScalingFactor,
                                                                                         bool computeCategoricalCrossEntropyGradient,
                                                                                         bool computeBinaryCrossEntropyGradient,
                                                                                         Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint16_t, float, float>(void *labels_d,
                                                                                          void *probabilities_d,
                                                                                          void *loss_d,
                                                                                          void *gradient_d,
                                                                                          uint32_t numClasses,
                                                                                          uint32_t batchSize,
                                                                                          bool computeGradient,
                                                                                          uint32_t lossScalingFactor,
                                                                                          bool computeCategoricalCrossEntropyGradient,
                                                                                          bool computeBinaryCrossEntropyGradient,
                                                                                          Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, half, half>(void *labels_d,
                                                                                        void *probabilities_d,
                                                                                        void *loss_d,
                                                                                        void *gradient_d,
                                                                                        uint32_t numClasses,
                                                                                        uint32_t batchSize,
                                                                                        bool computeGradient,
                                                                                        uint32_t lossScalingFactor,
                                                                                        bool computeCategoricalCrossEntropyGradient,
                                                                                        bool computeBinaryCrossEntropyGradient,
                                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, half, float>(void *labels_d,
                                                                                         void *probabilities_d,
                                                                                         void *loss_d,
                                                                                         void *gradient_d,
                                                                                         uint32_t numClasses,
                                                                                         uint32_t batchSize,
                                                                                         bool computeGradient,
                                                                                         uint32_t lossScalingFactor,
                                                                                         bool computeCategoricalCrossEntropyGradient,
                                                                                         bool computeBinaryCrossEntropyGradient,
                                                                                         Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, float, half>(void *labels_d,
                                                                                         void *probabilities_d,
                                                                                         void *loss_d,
                                                                                         void *gradient_d,
                                                                                         uint32_t numClasses,
                                                                                         uint32_t batchSize,
                                                                                         bool computeGradient,
                                                                                         uint32_t lossScalingFactor,
                                                                                         bool computeCategoricalCrossEntropyGradient,
                                                                                         bool computeBinaryCrossEntropyGradient,
                                                                                         Stream stream);

template void launchElementWiseCrossEntropyLoss_oneHotSpecialCase<uint32_t, float, float>(void *labels_d,
                                                                                          void *probabilities_d,
                                                                                          void *loss_d,
                                                                                          void *gradient_d,
                                                                                          uint32_t numClasses,
                                                                                          uint32_t batchSize,
                                                                                          bool computeGradient,
                                                                                          uint32_t lossScalingFactor,
                                                                                          bool computeCategoricalCrossEntropyGradient,
                                                                                          bool computeBinaryCrossEntropyGradient,
                                                                                          Stream stream);
