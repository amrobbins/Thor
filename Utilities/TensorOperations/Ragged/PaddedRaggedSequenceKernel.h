#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

void launchPackedToPaddedRaggedSequence(const Tensor& packedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& paddedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream);

// Consumer-owned sanitation for a retained padded representation. Active row
// prefixes are copied byte-for-byte while inactive positions in the selected
// [B,C,1,W] prefix are written as zero. Source storage is never modified.
void launchSanitizedPaddedRaggedSequenceCopy(const Tensor& sourcePaddedValues,
                                             const Tensor& rowOffsets,
                                             Tensor& destinationPaddedValues,
                                             uint64_t batchSize,
                                             uint64_t channels,
                                             uint64_t widthCapacity,
                                             Stream& stream);

void launchPaddedToPackedRaggedSequence(const Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& packedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream);

}  // namespace ThorImplementation
