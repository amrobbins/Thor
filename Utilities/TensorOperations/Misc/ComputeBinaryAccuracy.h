#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>

std::shared_ptr<ThorImplementation::StampedCubReduction> createBinaryAccuracyReduction(
    ThorImplementation::Tensor workspace_d,
    ThorImplementation::Tensor accuracy_d,
    uint32_t batchSize,
    Stream stream);

// Note: accuracy_d is an FP32 tensor. workspace_d is a FP16 tensor. Both are on the same device as the other arrays.
//       workspace is FP16 because it stores one 0/1 result per batch item before the central FP32 CUB reduction.

template <typename PREDICTION_TYPE, typename LABEL_TYPE>
void launchComputeBinaryAccuracy(ThorImplementation::Tensor accuracy_d,
                                 PREDICTION_TYPE *predictions_d,
                                 LABEL_TYPE *labels_d,
                                 ThorImplementation::Tensor workspace_d,
                                 uint32_t batchSize,
                                 std::shared_ptr<ThorImplementation::StampedCubReduction> reduction,
                                 Stream stream);
