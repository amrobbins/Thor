#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>

template <typename PREDICTION_TYPE, typename LABEL_TYPE>
void launchComputeCategoricalAccuracy_perClassLabels(float *accuracy_d,
                                                     PREDICTION_TYPE *predictions_d,
                                                     LABEL_TYPE *labels_d,
                                                     uint8_t *workspace_d,
                                                     uint32_t numClasses,
                                                     uint32_t batchSize,
                                                     Stream stream);

template <typename PREDICTION_TYPE, typename LABEL_TYPE>
void launchComputeCategoricalAccuracy_classIndexLabels(float *accuracy_d,
                                                       PREDICTION_TYPE *predictions_d,
                                                       LABEL_TYPE *labels_d,
                                                       uint8_t *workspace_d,
                                                       uint32_t numClasses,
                                                       uint32_t batchSize,
                                                       Stream stream);
