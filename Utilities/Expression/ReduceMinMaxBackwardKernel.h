#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

namespace ThorImplementation {

/** Rank-sized device metadata prepared once while stamping a min/max backward scatter. */
struct ReduceMinMaxBackwardScatterPlan {
    Tensor metadata;
    uint32_t input_rank = 0;
    uint32_t reduction_rank = 0;
    uint64_t output_numel = 0;
};

[[nodiscard]] ReduceMinMaxBackwardScatterPlan prepareReduceMinMaxBackwardScatter(
    const std::vector<uint64_t>& input_dims,
    const std::vector<uint64_t>& reduction_axes,
    const std::vector<uint64_t>& squeeze_axes,
    const TensorPlacement& placement,
    const Stream& stream);

void launchReduceMinMaxBackwardScatter(const void* grad_output,
                                       const uint32_t* arg_indices,
                                       void* grad_input,
                                       const ReduceMinMaxBackwardScatterPlan& plan,
                                       DataType grad_output_dtype,
                                       DataType grad_input_dtype,
                                       cudaStream_t stream);

}  // namespace ThorImplementation
