#pragma once

#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>

// Sequence-axis concatenate for canonical rank-1 ragged tensors.
//
// Every input has the same logical batch size and trailing value shape, but may
// have a different row partition and packed capacity. The output row partition
// is produced explicitly:
//
//     output_offsets[row] = sum_i input_offsets_i[row]
//
// Packed values are then copied row-by-row in input order. Only logical active
// values are read/written; inactive source and destination capacity is left
// untouched. `input_values` and `input_offsets` are device arrays containing one
// pointer per logical sequence input. Repeated offsets pointers are allowed.
void launchRaggedSequenceConcatenate(void *output_values,
                                     void *output_offsets,
                                     void *input_values[],
                                     void *input_offsets[],
                                     uint32_t num_inputs,
                                     std::size_t value_element_size_bytes,
                                     uint64_t elements_per_value,
                                     std::size_t offsets_element_size_bytes,
                                     uint64_t batch_size,
                                     Stream stream);

// Backward split for sequence-axis concatenate. `input_gradients` is a device
// pointer table parallel to the logical inputs; null entries are skipped. The
// authoritative input partitions determine exactly which gradient elements are
// written, so inactive gradient capacity remains undefined/untouched.
void launchRaggedSequenceConcatenateBackward(void *input_gradients[],
                                             const void *output_gradient,
                                             void *input_offsets[],
                                             uint32_t num_inputs,
                                             std::size_t value_element_size_bytes,
                                             uint64_t elements_per_value,
                                             std::size_t offsets_element_size_bytes,
                                             uint64_t batch_size,
                                             Stream stream);
