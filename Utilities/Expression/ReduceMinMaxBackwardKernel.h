#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

void launchReduceMinMaxBackwardScatter(const void* grad_output,
                                       const uint32_t* arg_indices,
                                       void* grad_input,
                                       const std::vector<uint64_t>& input_dims,
                                       const std::vector<uint64_t>& reduction_axes,
                                       const std::vector<uint64_t>& squeeze_axes,
                                       TensorDescriptor::DataType grad_output_dtype,
                                       TensorDescriptor::DataType grad_input_dtype,
                                       cudaStream_t stream);

}  // namespace ThorImplementation
