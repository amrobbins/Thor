#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

// Thor's run-length encode of keys: the runs of equal adjacent keys (all of a key's when the keys are sorted), each
// with its first position and its length. One kernel: a block takes a tile of keys, finds the run heads (a key that
// differs from the one before it), learns how many heads the tiles before it hold and where the last of them is by
// looking back through what those tiles publish (one word a tile), and writes its runs. The item count may be on the
// device (activeCount): the kernel reads it, tiles past it do nothing, and the grid is sized for the capacity, so a
// launch can be captured in a CUDA graph and still do only a batch's work.
struct RunLengthEncodePlan {
    DataType keyDataType = DataType::UINT32;
    uint64_t capacity = 0;      // the most keys a launch reads
    uint32_t tiles = 0;         // the kernel's grid
    uint64_t scratchWords = 0;  // UINT64 words of scratch a launch needs (8-byte aligned)

    bool operator==(const RunLengthEncodePlan&) const = default;
};

/**
 * The plan for up to `capacity` keys (1 to 2^30 - 1) of type UINT16, UINT32 or UINT64.
 * Throws std::invalid_argument otherwise.
 */
[[nodiscard]] RunLengthEncodePlan planRunLengthEncode(DataType keyDataType, uint64_t capacity);

/**
 * Encodes the first n keys, n = the plan's capacity when activeCount is null, else min(*activeCount, capacity) read
 * on the device (activeCountDataType UINT32 or UINT64). With R runs: *numRuns = R; for r < R, uniqueKeys[r] is run r's
 * key, runOffsets[r] its first position (the exclusive sum of the counts before it) and runCounts[r] its length.
 * Nothing past R is written; n = 0 gives R = 0. Stream-ordered, no allocation, no synchronization. `scratch` holds
 * plan.scratchWords UINT64 words, which the launch clears first; with clearScratch false it does not, and they must
 * already be 0 (cleared by an earlier kernel of the same stream or graph, which saves a memset node). Every array is
 * on the stream's GPU; uniqueKeys does not overlap keys.
 */
void launchRunLengthEncode(const RunLengthEncodePlan& plan,
                           const void* keys,
                           void* uniqueKeys,
                           uint32_t* runCounts,
                           uint32_t* runOffsets,
                           uint32_t* numRuns,
                           uint64_t* scratch,
                           const void* activeCount,
                           DataType activeCountDataType,
                           Stream stream,
                           bool clearScratch = true);

}  // namespace ThorImplementation
