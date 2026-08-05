#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

void launchScaleGradient(const void *source,
                         void *destination,
                         DataType dataType,
                         float scale,
                         uint64_t numElements,
                         Stream stream);

}  // namespace ThorImplementation
