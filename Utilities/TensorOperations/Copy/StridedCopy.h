#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

namespace ThorImplementation {

// Materialize a tensor view into dense storage without changing dtype or shape.
// Source and destination must be on the same GPU, and destination must be dense.
void materializeTensorViewAsync(const Tensor& source, Tensor& destination, Stream stream);

}  // namespace ThorImplementation
