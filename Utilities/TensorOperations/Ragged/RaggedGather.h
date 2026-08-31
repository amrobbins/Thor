#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

// Gather source values with row-local UINT32/UINT64 indices. source_offsets
// defines source partition P and indices_offsets defines destination partition
// Q. output_values has Q's packed capacity and source trailing geometry. Only
// active indices in [0, Q[B]) are read. An index is relative to its source row.
void launchRaggedGather(const Tensor& source_values,
                        const Tensor& source_offsets,
                        const Tensor& indices_values,
                        const Tensor& indices_offsets,
                        Tensor& output_values,
                        uint64_t batch_size,
                        Stream& stream);

// Backward for row-local gather. Active source-gradient positions are first
// initialized to exact zero, then every active output gradient is accumulated
// into its selected source token. Duplicate row-local indices therefore sum.
// Inactive source-gradient capacity is untouched. Backward supports FP16,
// BF16, and FP32 feature gradients.
void launchRaggedGatherBackward(const Tensor& source_offsets,
                                const Tensor& indices_values,
                                const Tensor& indices_offsets,
                                const Tensor& output_gradient,
                                Tensor& source_gradient,
                                uint64_t batch_size,
                                Stream& stream);

}  // namespace ThorImplementation
