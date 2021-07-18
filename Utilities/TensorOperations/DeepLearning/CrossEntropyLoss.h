#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cmath>
#include <limits>

template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
void launchCrossEntropyLoss_perClassLabels(LABEL_TYPE *labels_d,
                                           PROBABILITY_TYPE *probabilities_d,
                                           float *workspace_d,
                                           float *loss_d,
                                           uint32_t numClasses,
                                           uint32_t batchSize,
                                           Stream stream);

template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
void launchCrossEntropyLoss_classIndexLabels(LABEL_TYPE *labels_d,
                                             PROBABILITY_TYPE *probabilities_d,
                                             float *workspace_d,
                                             float *loss_d,
                                             uint32_t numClasses,
                                             uint32_t batchSize,
                                             Stream stream);

template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_GRADIENT_TYPE>
void launchLossGradient_classIndexLabels(LABEL_TYPE *labels_d,
                                         PROBABILITY_TYPE *probabilities_d,
                                         LOSS_GRADIENT_TYPE *lossGradient_d,
                                         uint64_t numClasses,
                                         uint64_t batchSize,
                                         Stream stream);
