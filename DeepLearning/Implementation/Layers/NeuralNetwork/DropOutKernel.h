#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

void launchDropOutForward(const void *input,
                          void *output,
                          DataType dataType,
                          uint64_t numElements,
                          float probabilityOfDroppingOut,
                          uint64_t randomSeed,
                          uint64_t forwardSequence,
                          Stream stream);

void launchDropOutBackward(const void *errorInput,
                           void *errorOutput,
                           DataType dataType,
                           uint64_t numElements,
                           float probabilityOfDroppingOut,
                           uint64_t randomSeed,
                           uint64_t forwardSequence,
                           Stream stream);

// Compatibility wrappers retained for callers that select BF16 explicitly.
void launchBfloat16DropOutForward(const void *input,
                                  void *output,
                                  uint64_t numElements,
                                  float probabilityOfDroppingOut,
                                  uint64_t randomSeed,
                                  uint64_t forwardSequence,
                                  Stream stream);

void launchBfloat16DropOutBackward(const void *errorInput,
                                   void *errorOutput,
                                   uint64_t numElements,
                                   float probabilityOfDroppingOut,
                                   uint64_t randomSeed,
                                   uint64_t forwardSequence,
                                   Stream stream);

}  // namespace ThorImplementation
