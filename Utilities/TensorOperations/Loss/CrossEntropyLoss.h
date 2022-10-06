#pragma once

#include "BinaryCrossEntropyLoss.h"
#include "CategoricalCrossEntropyLoss.h"
#include "Utilities/Common/Stream.h"

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cmath>
#include <cstdint>
#include <limits>

enum class CrossEntropyLossType { BINARY = 104, CATEGORICAL, UNINITIALIZED };

template <typename LABEL_OR_INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCrossEntropyLoss(void *labelsOrClassOfHotLabels_d,
                                       void *probabilities_d,
                                       void *loss_d,
                                       void *gradient_d,
                                       uint32_t numClasses,
                                       uint32_t batchSize,
                                       bool computeGradient,
                                       uint32_t lossScalingFactor,
                                       CrossEntropyLossType crossEntropyLossType,
                                       bool indexLabels,
                                       Stream stream);
