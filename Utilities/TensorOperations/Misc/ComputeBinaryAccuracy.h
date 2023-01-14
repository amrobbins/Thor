#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>

std::shared_ptr<ThorImplementation::BatchReduce> createBinaryAccuracyBatchReduce(uint32_t batchSize, Stream stream);

// Note: accuracy_d is an FP32 tensor. workspace_d is a FP16 tensor. Both are on the same device as the other arrays.
//       tensors are used to that Tensor::divide can be used by BatchReduce
//       workspace is an FP16 tensor because cudnn reduce does not support uint8 -> fp32

template <typename PREDICTION_TYPE, typename LABEL_TYPE>
void launchComputeBinaryAccuracy(ThorImplementation::Tensor accuracy_d,
                                 PREDICTION_TYPE *predictions_d,
                                 LABEL_TYPE *labels_d,
                                 ThorImplementation::Tensor workspace_d,
                                 uint32_t batchSize,
                                 std::shared_ptr<ThorImplementation::BatchReduce> batchReduce,
                                 Stream stream);