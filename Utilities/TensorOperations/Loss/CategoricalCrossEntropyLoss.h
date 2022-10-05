#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cmath>
#include <cstdint>
#include <limits>

// This version takes in a one-hot vector of labels per class per item in the batch
// The distinction here about "CategoricalCrossEntropy" is that the gradient is computed
// as if a cross entropy loss gradient was sent through a softmax backward layer,
// same numerical result but much faster and more stable.
// The forward pass takes the output of the forward pass of softmax.
template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCategoricalCrossEntropyLoss(void *labels_d,
                                                  void *probabilities_d,
                                                  void *loss_d,
                                                  void *gradient_d,
                                                  uint32_t numClasses,
                                                  uint32_t batchSize,
                                                  bool computeGradient,
                                                  uint32_t lossScalingFactor,
                                                  Stream stream);

// This version takes in an integer per item in the batch that specifies the true class of the example.
template <typename INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCategoricalCrossEntropyLoss_oneHotSpecialCase(void *classOfHotLabels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    Stream stream);