#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cuda_fp16.h>
#include <cstdint>
#include <optional>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kRaggedSparseCategoricalCrossEntropyWithLogitsPartitionRequirement =
    RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT;

}  // namespace ThorImplementation

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

// Ragged packed-prefix variant.
//
// rowCapacity is the physical packed capacity. activeCount_d points to Thor's
// managed [1] DEVICE_ACTIVE_COUNT carrier (UINT32 or UINT64). The kernel still
// launches rowCapacity blocks, but every block checks the device scalar before
// reading labels, mask, or logits. Rows in [active_count, rowCapacity) are
// therefore neither read nor written.
template <typename LABEL_TYPE, typename LOGIT_TYPE, typename LOSS_TYPE, typename MASK_TYPE>
void launchRaggedSparseCategoricalCrossEntropyWithLogits(void *labels_d,
                                                         void *logits_d,
                                                         void *mask_d,
                                                         void *loss_d,
                                                         void *gradient_d,
                                                         void *activeCount_d,
                                                         ThorImplementation::DataType activeCountDataType,
                                                         uint32_t numClasses,
                                                         uint32_t rowCapacity,
                                                         bool computeGradient,
                                                         uint32_t lossScalingFactor,
                                                         float lossWeight,
                                                         bool hasIgnoreIndex,
                                                         uint32_t ignoreIndex,
                                                         bool hasMask,
                                                         Stream stream);
