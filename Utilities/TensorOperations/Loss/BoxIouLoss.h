#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

// Kind values intentionally mirror ThorImplementation::BoxIouLoss::Kind.
// Keeping the kernel ABI integer-based avoids including implementation-layer
// headers from Utilities.
enum class BoxIouLossKind : uint32_t { IOU = 0, GIOU = 1, DIOU = 2, CIOU = 3 };

template <typename LABEL_TYPE, typename PREDICTION_TYPE, typename LOSS_TYPE>
void launchBoxIouLoss(void* labels_d,
                      void* predictions_d,
                      void* loss_d,
                      void* gradient_d,
                      uint32_t numBoxes,
                      BoxIouLossKind kind,
                      float eps,
                      bool computeGradient,
                      float lossScalingFactor,
                      Stream stream);
