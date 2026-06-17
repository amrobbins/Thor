#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda_fp16.h>
#include <cstdint>
#include <optional>

// Logits-native sparse categorical cross entropy.
//
// logits:   [numRows, numClasses]
// labels:   [numRows] integer class ids
// mask:     optional [numRows], valid when mask > 0.5
// loss:     [numRows]
// gradient: [numRows, numClasses], dense dL/dlogits
//
// The kernel computes logsumexp(logits[row]) - logits[row, label[row]] and the
// dense gradient softmax(logits[row]) - one_hot(label[row]) without materializing
// an intermediate softmax/probability tensor or a per-class raw loss tensor.
template <typename LABEL_TYPE, typename LOGIT_TYPE, typename LOSS_TYPE, typename MASK_TYPE>
void launchSparseCategoricalCrossEntropyWithLogits(void *labels_d,
                                                   void *logits_d,
                                                   void *mask_d,
                                                   void *loss_d,
                                                   void *gradient_d,
                                                   uint32_t numClasses,
                                                   uint32_t numRows,
                                                   bool computeGradient,
                                                   uint32_t lossScalingFactor,
                                                   float lossWeight,
                                                   bool hasIgnoreIndex,
                                                   uint32_t ignoreIndex,
                                                   bool hasMask,
                                                   Stream stream);
