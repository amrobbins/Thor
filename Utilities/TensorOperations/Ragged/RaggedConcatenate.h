#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstddef>
#include <cstdint>

namespace ThorImplementation {
// Trailing-axis concatenate/split needs only the active packed-value count, not individual row boundaries.
inline constexpr RaggedPartitionRequirement kRaggedTrailingConcatenatePartitionRequirement = RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT;
}  // namespace ThorImplementation

// Concatenate/split contiguous packed ragged values along a trailing feature
// axis while touching only the authoritative active prefix named by the managed
// [1] active-count carrier. Inactive packed capacity is deliberately neither
// read nor canonicalized.
//
// axisOffsets is a device array of numArrays + 1 cumulative axis offsets:
//   {0, axisElements[0], axisElements[0] + axisElements[1], ...}
// outerSlicesPerValue is the product of packed-value dimensions before the
// concatenated axis; innerElements is the product of dimensions after it.
void launchRaggedConcatenate(void *dest,
                             void *source[],
                             std::size_t elementSizeBytes,
                             uint64_t capacityRows,
                             uint64_t elementsPerOutputValue,
                             uint64_t outerSlicesPerValue,
                             uint64_t innerElements,
                             uint32_t numSourceArrays,
                             const uint64_t axisOffsets[],
                             const void *activeCount,
                             std::size_t activeCountElementSizeBytes,
                             Stream stream);

void launchRaggedSplit(void *dest[],
                       void *source,
                       std::size_t elementSizeBytes,
                       uint64_t capacityRows,
                       uint64_t elementsPerSourceValue,
                       uint64_t outerSlicesPerValue,
                       uint64_t innerElements,
                       uint32_t numDestArrays,
                       const uint64_t axisOffsets[],
                       const void *activeCount,
                       std::size_t activeCountElementSizeBytes,
                       Stream stream);
