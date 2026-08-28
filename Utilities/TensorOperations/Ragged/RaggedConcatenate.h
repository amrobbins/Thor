#pragma once

#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>

// Concatenate/split packed ragged values along a trailing feature axis while
// touching only the authoritative active prefix offsets[batch_size]. Inactive
// packed capacity is deliberately neither read nor canonicalized.
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
                             const void *offsets,
                             std::size_t offsetsElementSizeBytes,
                             uint64_t batchSize,
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
                       const void *offsets,
                       std::size_t offsetsElementSizeBytes,
                       uint64_t batchSize,
                       Stream stream);
