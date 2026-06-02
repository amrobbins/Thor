#pragma once

#include <cuda_runtime_api.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

namespace ThorImplementation {

void launchSetCudaGraphConditionalFromBool(cudaGraphConditionalHandle handle, const Tensor& predicate, Stream stream);

}  // namespace ThorImplementation
