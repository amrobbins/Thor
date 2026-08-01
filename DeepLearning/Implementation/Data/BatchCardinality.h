#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include <cstdint>
#include <limits>

namespace ThorImplementation {

/** Returns the valid-example count for a physically full batch. */
inline uint32_t fullBatchValidExampleCount(uint64_t batchCapacity) {
    THOR_THROW_IF_FALSE(batchCapacity >= 1);
    THOR_THROW_IF_FALSE(batchCapacity <= std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(batchCapacity);
}

/**
 * Returns the number of logical examples represented by batchNum in one
 * fixed-capacity traversal of a split. The final batch may contain a smaller
 * valid prefix; every earlier batch is full capacity.
 */
inline uint32_t validExamplesForBatch(uint64_t batchNum,
                                      uint64_t numExamples,
                                      uint64_t batchCapacity) {
    THOR_THROW_IF_FALSE(numExamples >= 1);
    THOR_THROW_IF_FALSE(batchCapacity >= 1);
    THOR_THROW_IF_FALSE(batchCapacity <= std::numeric_limits<uint32_t>::max());
    THOR_THROW_IF_FALSE(batchNum <= (numExamples - 1) / batchCapacity);

    const uint64_t firstExample = batchNum * batchCapacity;
    const uint64_t remainingExamples = numExamples - firstExample;
    const uint64_t validExamples =
        remainingExamples < batchCapacity ? remainingExamples : batchCapacity;
    THOR_THROW_IF_FALSE(validExamples >= 1);
    THOR_THROW_IF_FALSE(validExamples <= std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(validExamples);
}

}  // namespace ThorImplementation
