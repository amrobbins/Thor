#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

// Count retained tokens independently in every logical row. mask_values is a
// scalar BOOLEAN packed tensor sharing input_offsets; inactive mask capacity is
// never read. output_lengths has shape [B] and the same UINT32/UINT64 dtype as
// input_offsets.
void launchRaggedFilterRowLengths(const Tensor& mask_values,
                                  const Tensor& input_offsets,
                                  Tensor& output_lengths,
                                  uint64_t batch_size,
                                  Stream& stream);

// Stable row-local compaction. Selected active input tokens preserve order and
// are packed according to output_offsets. Inactive input/output capacity is not
// read or written.
void launchRaggedFilterValues(const Tensor& input_values,
                              const Tensor& mask_values,
                              const Tensor& input_offsets,
                              const Tensor& output_offsets,
                              Tensor& output_values,
                              uint64_t batch_size,
                              Stream& stream);

// Backward for stable ragged filtering. Every active input-gradient token is
// initialized to exact zero; retained positions then receive the corresponding
// compact upstream gradient. Inactive input-gradient capacity is untouched.
void launchRaggedFilterBackward(const Tensor& mask_values,
                                const Tensor& input_offsets,
                                const Tensor& output_offsets,
                                const Tensor& output_gradient,
                                Tensor& input_gradient,
                                uint64_t batch_size,
                                Stream& stream);

}  // namespace ThorImplementation
