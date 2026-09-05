#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kRaggedSequenceSlicePartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

// Compact the selected row windows into output_values according to
// output_offsets. Only selected active values are read and written; inactive
// input/output packed capacity is untouched.
void launchRaggedSequenceSliceValues(const Tensor& input_values,
                                     const Tensor& input_offsets,
                                     const Tensor& output_offsets,
                                     Tensor& output_values,
                                     uint64_t start,
                                     uint64_t length,
                                     uint64_t batch_size,
                                     Stream& stream);

// Backward for sequence-axis slice. Every active input-gradient value is
// initialized to exact zero, then the selected window receives the compact
// upstream gradient. Inactive input-gradient capacity is untouched.
void launchRaggedSequenceSliceBackward(const Tensor& input_offsets,
                                       const Tensor& output_offsets,
                                       const Tensor& output_gradient,
                                       Tensor& input_gradient,
                                       uint64_t start,
                                       uint64_t length,
                                       uint64_t batch_size,
                                       Stream& stream);

}  // namespace ThorImplementation
