#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

void launchBfloat16DropOutForward(const void *input,
                                  void *output,
                                  uint8_t *keepMask,
                                  uint64_t numElements,
                                  float probabilityOfDroppingOut,
                                  uint64_t randomSeed,
                                  uint64_t forwardSequence,
                                  Stream stream);

void launchBfloat16DropOutBackward(const void *errorInput,
                                   void *errorOutput,
                                   const uint8_t *keepMask,
                                   uint64_t numElements,
                                   float probabilityOfDroppingOut,
                                   Stream stream);

}  // namespace ThorImplementation
