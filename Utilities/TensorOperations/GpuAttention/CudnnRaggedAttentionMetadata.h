#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

/**
 * Converts Thor's canonical rank-1 row partition into the metadata expected by
 * cuDNN's ragged SDPA ABI.
 *
 * Canonical Thor offsets are token offsets stored as UINT32/UINT64. cuDNN
 * consumes INT32 sequence lengths plus INT32 element offsets. The two output
 * offset tensors may use different elements-per-token scales (for example Q
 * and O when Dqk != Dv).
 *
 * The canonical row partition is already validated by the RaggedTensor
 * producer/consumer contract. This adapter therefore performs only static host
 * validation and an asynchronous conversion kernel; it does not add device
 * asserts, traps, validation flags, or host readbacks to the attention path.
 */
void convertCanonicalRowPartitionForCudnnAttention(const Tensor& canonicalOffsets,
                                                    uint64_t batchSize,
                                                    uint64_t maxTotalValues,
                                                    uint64_t firstElementsPerToken,
                                                    uint64_t secondElementsPerToken,
                                                    Tensor sequenceLengths,
                                                    Tensor firstElementOffsets,
                                                    Tensor secondElementOffsets,
                                                    Stream stream);

}  // namespace ThorImplementation
