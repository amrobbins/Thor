#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cmath>
#include <limits>
#include <type_traits>

template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
void launchMeanSquaredError(LABEL_TYPE *labels_d,
                            PREDICTION_TYPE *predictions_d,
                            LOSS_TYPE *loss_d,
                            LOSS_TYPE *workspace_d,
                            uint32_t numPredictions,
                            uint32_t batchSize,
                            Stream stream,
                            BatchReduce *batchReduce);
