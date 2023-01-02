#include "BinaryCrossEntropyLoss.h"

using namespace std;

/**
 * Binary Cross Entropy Loss (i.e. sigmoid then cross entropy loss):
 * loss = -( label * log(probability) + (1 - label) * (log(1 - probability)) )
 * where label is 0 or 1
 * Gradient of Binary Cross Entropy with respect to the predicted probability:
 * gradient = probability - label
 *
 * In the loss function, log(0) is avoided by choosing a minimum value close to the minimum positive value of fp16 or fp32 respectively.
 * Note that this minimum does not affect the gradient.
 */
template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseBinaryCrossEntropyLoss(
    uint32_t batchSize, void *labels, void *probabilities, void *loss, void *gradient, bool computeGradient) {
    uint32_t elementwiseLossIndex = blockIdx.x * 1024 + threadIdx.x;
    uint32_t batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;

    const PROBABILITY_TYPE ONE = 1.0f;
    const PROBABILITY_TYPE MIN_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.000062f : 1E-36;
    const PROBABILITY_TYPE MAX_TRUE_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.9995f : 0.9999999f;

    PROBABILITY_TYPE label;
    PROBABILITY_TYPE probability;
    PROBABILITY_TYPE elementwiseGradient;
    PROBABILITY_TYPE logTerm;
    LOSS_TYPE elementwiseLoss;

    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
__global__ void elementWiseBinaryCrossEntropyLoss_withScale(
    uint32_t batchSize, void *labels, void *probabilities, void *loss, void *gradient, bool computeGradient, float gradientScale) {
    uint32_t elementwiseLossIndex = blockIdx.x * 1024 + threadIdx.x;
    uint32_t batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;

    const PROBABILITY_TYPE ONE = 1.0f;
    const PROBABILITY_TYPE MIN_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.000062f : 1E-36;
    const PROBABILITY_TYPE MAX_TRUE_PROBABILITY = is_same<PROBABILITY_TYPE, half>::value ? 0.9995f : 0.9999999f;

    PROBABILITY_TYPE label;
    PROBABILITY_TYPE probability;
    PROBABILITY_TYPE elementwiseGradient;
    PROBABILITY_TYPE logTerm;
    LOSS_TYPE elementwiseLoss;

    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;

    elementwiseLossIndex += 256;
    batchIndex = elementwiseLossIndex;
    if (batchIndex >= batchSize)
        return;
    label = ((LABEL_TYPE *)labels)[batchIndex] ? 1.0f : 0.0f;
    probability = ((PROBABILITY_TYPE *)probabilities)[batchIndex];
    probability = !isfinite((float)probability) || isnan((float)probability) ? 0.0f : (float)probability;
    if (computeGradient) {
        elementwiseGradient = probability;
        if (label)
            elementwiseGradient -= ONE;
        elementwiseGradient *= (PROBABILITY_TYPE)gradientScale;
        ((PROBABILITY_TYPE *)gradient)[elementwiseLossIndex] = elementwiseGradient;
    }
    if (label) {
        if (probability < MIN_PROBABILITY)
            probability = MIN_PROBABILITY;
    } else {
        if (probability > MAX_TRUE_PROBABILITY)
            probability = MAX_TRUE_PROBABILITY;
    }
    logTerm = label ? probability : ONE - probability;
    elementwiseLoss = -logf(logTerm);
    ((LOSS_TYPE *)loss)[elementwiseLossIndex] = elementwiseLoss;
}

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseBinaryCrossEntropyLoss(void *labels_d,
                                             void *probabilities_d,
                                             void *loss_d,
                                             void *gradient_d,
                                             uint32_t batchSize,
                                             bool computeGradient,
                                             uint32_t lossScalingFactor,
                                             Stream stream) {
    uint64_t numElements = (uint64_t)batchSize;
    dim3 blockSize(256);
    dim3 gridSize((numElements + 1023) / 1024);

    if (lossScalingFactor == 1 || !computeGradient) {
        elementWiseBinaryCrossEntropyLoss<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient);
    } else {
        elementWiseBinaryCrossEntropyLoss_withScale<LABEL_TYPE, PROBABILITY_TYPE, LOSS_TYPE>
            <<<gridSize, blockSize, 0, stream.getStream()>>>(
                batchSize, labels_d, probabilities_d, loss_d, gradient_d, computeGradient, (float)lossScalingFactor);
    }
}

template void launchElementWiseBinaryCrossEntropyLoss<bool, half, half>(void *labels_d,
                                                                        void *probabilities_d,
                                                                        void *loss_d,
                                                                        void *gradient_d,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        uint32_t lossScalingFactor,
                                                                        Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<bool, half, float>(void *labels_d,
                                                                         void *probabilities_d,
                                                                         void *loss_d,
                                                                         void *gradient_d,
                                                                         uint32_t batchSize,
                                                                         bool computeGradient,
                                                                         uint32_t lossScalingFactor,
                                                                         Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<bool, float, half>(void *labels_d,
                                                                         void *probabilities_d,
                                                                         void *loss_d,
                                                                         void *gradient_d,
                                                                         uint32_t batchSize,
                                                                         bool computeGradient,
                                                                         uint32_t lossScalingFactor,
                                                                         Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<bool, float, float>(void *labels_d,
                                                                          void *probabilities_d,
                                                                          void *loss_d,
                                                                          void *gradient_d,
                                                                          uint32_t batchSize,
                                                                          bool computeGradient,
                                                                          uint32_t lossScalingFactor,
                                                                          Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint8_t, half, half>(void *labels_d,
                                                                           void *probabilities_d,
                                                                           void *loss_d,
                                                                           void *gradient_d,
                                                                           uint32_t batchSize,
                                                                           bool computeGradient,
                                                                           uint32_t lossScalingFactor,
                                                                           Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint8_t, half, float>(void *labels_d,
                                                                            void *probabilities_d,
                                                                            void *loss_d,
                                                                            void *gradient_d,
                                                                            uint32_t batchSize,
                                                                            bool computeGradient,
                                                                            uint32_t lossScalingFactor,
                                                                            Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint8_t, float, half>(void *labels_d,
                                                                            void *probabilities_d,
                                                                            void *loss_d,
                                                                            void *gradient_d,
                                                                            uint32_t batchSize,
                                                                            bool computeGradient,
                                                                            uint32_t lossScalingFactor,
                                                                            Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint8_t, float, float>(void *labels_d,
                                                                             void *probabilities_d,
                                                                             void *loss_d,
                                                                             void *gradient_d,
                                                                             uint32_t batchSize,
                                                                             bool computeGradient,
                                                                             uint32_t lossScalingFactor,
                                                                             Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint16_t, half, half>(void *labels_d,
                                                                            void *probabilities_d,
                                                                            void *loss_d,
                                                                            void *gradient_d,
                                                                            uint32_t batchSize,
                                                                            bool computeGradient,
                                                                            uint32_t lossScalingFactor,
                                                                            Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint16_t, half, float>(void *labels_d,
                                                                             void *probabilities_d,
                                                                             void *loss_d,
                                                                             void *gradient_d,
                                                                             uint32_t batchSize,
                                                                             bool computeGradient,
                                                                             uint32_t lossScalingFactor,
                                                                             Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint16_t, float, half>(void *labels_d,
                                                                             void *probabilities_d,
                                                                             void *loss_d,
                                                                             void *gradient_d,
                                                                             uint32_t batchSize,
                                                                             bool computeGradient,
                                                                             uint32_t lossScalingFactor,
                                                                             Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint16_t, float, float>(void *labels_d,
                                                                              void *probabilities_d,
                                                                              void *loss_d,
                                                                              void *gradient_d,
                                                                              uint32_t batchSize,
                                                                              bool computeGradient,
                                                                              uint32_t lossScalingFactor,
                                                                              Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint32_t, half, half>(void *labels_d,
                                                                            void *probabilities_d,
                                                                            void *loss_d,
                                                                            void *gradient_d,
                                                                            uint32_t batchSize,
                                                                            bool computeGradient,
                                                                            uint32_t lossScalingFactor,
                                                                            Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint32_t, half, float>(void *labels_d,
                                                                             void *probabilities_d,
                                                                             void *loss_d,
                                                                             void *gradient_d,
                                                                             uint32_t batchSize,
                                                                             bool computeGradient,
                                                                             uint32_t lossScalingFactor,
                                                                             Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint32_t, float, half>(void *labels_d,
                                                                             void *probabilities_d,
                                                                             void *loss_d,
                                                                             void *gradient_d,
                                                                             uint32_t batchSize,
                                                                             bool computeGradient,
                                                                             uint32_t lossScalingFactor,
                                                                             Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<uint32_t, float, float>(void *labels_d,
                                                                              void *probabilities_d,
                                                                              void *loss_d,
                                                                              void *gradient_d,
                                                                              uint32_t batchSize,
                                                                              bool computeGradient,
                                                                              uint32_t lossScalingFactor,
                                                                              Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<half, half, half>(void *labels_d,
                                                                        void *probabilities_d,
                                                                        void *loss_d,
                                                                        void *gradient_d,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        uint32_t lossScalingFactor,
                                                                        Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<half, half, float>(void *labels_d,
                                                                         void *probabilities_d,
                                                                         void *loss_d,
                                                                         void *gradient_d,
                                                                         uint32_t batchSize,
                                                                         bool computeGradient,
                                                                         uint32_t lossScalingFactor,
                                                                         Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<half, float, half>(void *labels_d,
                                                                         void *probabilities_d,
                                                                         void *loss_d,
                                                                         void *gradient_d,
                                                                         uint32_t batchSize,
                                                                         bool computeGradient,
                                                                         uint32_t lossScalingFactor,
                                                                         Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<half, float, float>(void *labels_d,
                                                                          void *probabilities_d,
                                                                          void *loss_d,
                                                                          void *gradient_d,
                                                                          uint32_t batchSize,
                                                                          bool computeGradient,
                                                                          uint32_t lossScalingFactor,
                                                                          Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<float, half, half>(void *labels_d,
                                                                         void *probabilities_d,
                                                                         void *loss_d,
                                                                         void *gradient_d,
                                                                         uint32_t batchSize,
                                                                         bool computeGradient,
                                                                         uint32_t lossScalingFactor,
                                                                         Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<float, half, float>(void *labels_d,
                                                                          void *probabilities_d,
                                                                          void *loss_d,
                                                                          void *gradient_d,
                                                                          uint32_t batchSize,
                                                                          bool computeGradient,
                                                                          uint32_t lossScalingFactor,
                                                                          Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<float, float, half>(void *labels_d,
                                                                          void *probabilities_d,
                                                                          void *loss_d,
                                                                          void *gradient_d,
                                                                          uint32_t batchSize,
                                                                          bool computeGradient,
                                                                          uint32_t lossScalingFactor,
                                                                          Stream stream);

template void launchElementWiseBinaryCrossEntropyLoss<float, float, float>(void *labels_d,
                                                                           void *probabilities_d,
                                                                           void *loss_d,
                                                                           void *gradient_d,
                                                                           uint32_t batchSize,
                                                                           bool computeGradient,
                                                                           uint32_t lossScalingFactor,
                                                                           Stream stream);
