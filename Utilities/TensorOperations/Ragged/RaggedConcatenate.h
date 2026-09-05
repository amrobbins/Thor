#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstddef>
#include <cstdint>

namespace ThorImplementation {
// Trailing-axis concatenate/split needs only the active packed-value count, not individual row boundaries.
inline constexpr RaggedPartitionRequirement kRaggedTrailingConcatenatePartitionRequirement = RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT;
}  // namespace ThorImplementation

// Concatenate/split packed ragged values along a trailing feature axis while
// touching only the authoritative active prefix named by the managed [1]
// active-count carrier. Inactive packed capacity is deliberately neither read
// nor canonicalized.
void launchRaggedConcatenate(void *dest,
                             void *source[],
                             std::size_t elementSizeBytes,
                             long fullCapacityNumElements,
                             uint64_t elementsPerOutputValue,
                             int numDimensions,
                             int numSourceArrays,
                             int axisDimension,
                             long axisElementsPerSourceArray[],
                             long stridePerDestDimension[],
                             long stridePerSourceDimension[],
                             const void *activeCount,
                             std::size_t activeCountElementSizeBytes,
                             Stream stream);

void launchRaggedSplit(void *dest[],
                       void *source,
                       std::size_t elementSizeBytes,
                       long fullCapacityNumElements,
                       uint64_t elementsPerSourceValue,
                       int numDimensions,
                       int numDestArrays,
                       int axisDimension,
                       long axisElementsPerDestArray[],
                       long stridePerSourceDimension[],
                       long stridePerDestDimension[],
                       const void *activeCount,
                       std::size_t activeCountElementSizeBytes,
                       Stream stream);
