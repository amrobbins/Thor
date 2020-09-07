#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cmath>
#include <limits>

template <typename LABEL_TYPE, typename PROBABILITY_TYPE>
void launchCrossEntropyLoss(LABEL_TYPE *labels_d,
                            PROBABILITY_TYPE *probabilities_d,
                            float *workspace_d,
                            half *loss_d,
                            uint32_t elementsPerBatch,
                            uint32_t batchSize,
                            Stream stream);
