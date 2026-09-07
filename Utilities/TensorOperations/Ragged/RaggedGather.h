#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kRaggedGatherPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

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

// Backward for row-local gather. Each source row is zeroed cooperatively, then
// its active output gradients scatter-add across the trailing value width. A
// single-index row writes directly; rows with multiple indices use atomics
// because row-local indices may repeat. Inactive source-gradient capacity is
// untouched. Backward supports FP16, BF16, and FP32 gradients.
void launchRaggedGatherBackward(const Tensor& source_offsets,
                                const Tensor& indices_values,
                                const Tensor& indices_offsets,
                                const Tensor& output_gradient,
                                Tensor& source_gradient,
                                uint64_t batch_size,
                                Stream& stream);

}  // namespace ThorImplementation
