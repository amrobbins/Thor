#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

// This version takes in a scalar whose value is considered to be 0 when scalar is either 0 or false, otherwise it is considered to be 1.
template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseBinaryCrossEntropyLoss(void *labels_d,
                                             void *probabilities_d,
                                             void *loss_d,
                                             void *gradient_d,
                                             uint32_t batchSize,
                                             bool computeGradient,
                                             uint32_t lossScalingFactor,
                                             Stream stream);