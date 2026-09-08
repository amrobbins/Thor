#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstddef>
#include <cstdint>

namespace ThorImplementation {

// Sequence concatenate consumes authoritative host partition structure but no
// device offsets representation. StampedNetwork derives a compact copy plan
// from that host state before physical execution.
inline constexpr RaggedPartitionRequirement kRaggedSequenceConcatenatePartitionRequirement =
    RaggedPartitionRequirement::HOST_EXTENT;

struct RaggedSequenceCopySpan32 {
    uint32_t inputIndex;
    uint32_t sourceBegin;
    uint32_t destinationBegin;
    uint32_t valueCount;
};
static_assert(sizeof(RaggedSequenceCopySpan32) == 16);

struct RaggedSequenceCopySpan64 {
    uint32_t inputIndex;
    uint32_t padding;
    uint64_t sourceBegin;
    uint64_t destinationBegin;
    uint64_t valueCount;
};
static_assert(sizeof(RaggedSequenceCopySpan64) == 32);

}  // namespace ThorImplementation

// Sequence-axis concatenate for canonical rank-1 ragged tensors.
//
// StampedNetwork derives the authoritative output partition and the non-empty
// copy spans from authoritative host input partitions in one O(batch * inputs)
// pass. CUDA consumes only packed-value pointers plus that compact span table;
// it never reads or reconstructs row offsets. `copy_spans` points to either
// RaggedSequenceCopySpan32 or RaggedSequenceCopySpan64 according to
// `offsets_element_size_bytes`. `active_output_values` is used only to size the
// payload-aware launch.
void launchRaggedSequenceConcatenate(void *output_values,
                                     void *input_values[],
                                     const void *copy_spans,
                                     uint64_t span_count,
                                     std::size_t value_element_size_bytes,
                                     uint64_t elements_per_value,
                                     std::size_t offsets_element_size_bytes,
                                     uint64_t active_output_values,
                                     Stream stream);

// Backward split reuses the exact same copy-span table. `input_gradients` is a
// device pointer table parallel to logical inputs; null entries are skipped.
// Only values named by non-empty spans are written, so inactive gradient
// capacity remains untouched.
void launchRaggedSequenceConcatenateBackward(void *input_gradients[],
                                             const void *output_gradient,
                                             const void *copy_spans,
                                             uint64_t span_count,
                                             std::size_t value_element_size_bytes,
                                             uint64_t elements_per_value,
                                             std::size_t offsets_element_size_bytes,
                                             uint64_t active_output_values,
                                             Stream stream);
