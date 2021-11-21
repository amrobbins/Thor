#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>

/**
 * Note:
 * perClass labels mean that there is one variable sent per class per batch item that indicates whether the batch item belongs to the class.
 * classIndex labels mean that there is one variable sent per batch item that carries the numerical value of the class of that batch item.
 *
 * Note:
 * workspace required is TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}) for both cases.
 */

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
