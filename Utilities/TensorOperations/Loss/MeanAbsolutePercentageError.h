#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cmath>
#include <limits>
#include <type_traits>

// Computes mean absolute error, which is populated in the loss_d memory
// When computeGradient is true, the gradient will be populated in the gradient_d memory,
// when computeGradient is false, gradient_d can be null as it is not used.
template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
void launchMeanAbsolutePercentageError(void *labels_d,
                                       void *predictions_d,
                                       void *elementLoss_d,
                                       void *gradient_d,
                                       uint32_t numPredictions,
                                       uint32_t batchSize,
                                       bool computeGradient,
                                       Stream stream,
                                       float epsilon = 0.0001f,
                                       float maxMagnitude = 1000.0f);
