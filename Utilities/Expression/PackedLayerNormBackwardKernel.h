#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

// Packed LayerNorm backward over the *authoritative active prefix*, not a
// capacity bucket. The caller obtains active_rows from the existing row
// partition HOST_EXTENT carrier and links mean/inv_variance to the matching
// real training forward. All three physical gradients are written; only the
// requested logical routes are exposed by the expression planner.
//
// Inactive x/dY/statistics/dX storage is never read or written. For zero
// active rows, dScale and dBias are explicitly zero; dX is untouched.
void launchPackedLayerNormBackwardActivePrefix(const Tensor& x,
                                                const Tensor& scale,
                                                const Tensor& dY,
                                                const Tensor& mean,
                                                const Tensor& inv_variance,
                                                Tensor dX,
                                                Tensor dScale,
                                                Tensor dBias,
                                                uint64_t active_rows,
                                                cudaStream_t stream);

}  // namespace ThorImplementation
