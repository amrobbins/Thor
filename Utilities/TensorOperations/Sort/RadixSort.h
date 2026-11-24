#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

// Thor's stable least-significant-digit radix sort of (key, value) pairs: unsigned keys of 16, 32 or 64 bits whose
// set bits all lie below keyBits, UINT32 values. Pairs of equal keys keep their input order, so the result is the
// same on every run.
//
// A launch is one memset of the scratch, one histogram kernel (every pass's digit counts) and one kernel per pass of
// up to 8 bits. A pass block takes a tile of 512 or 4096 pairs, ranks them among the tile's pairs of the same
// digit, learns how many pairs of each digit the tiles before it hold by looking back through the counts those tiles
// publish (one row of the scratch per tile), puts the tile in digit order in shared memory and writes each digit's run
// of it to its place in the pass's output, so the stores are contiguous. The item count may be on the device
// (activeCount): every kernel reads it, tiles past it do nothing, and the grids are sized for the capacity, so a launch
// can be captured in a CUDA graph and still do only a batch's work.
//
// Passes alternate between the two pairs of arrays, starting from (keys, values) into (keysAlternate,
// valuesAlternate): the sorted pairs are in the alternates when sortedInAlternate, else back in (keys, values). Both
// pairs are overwritten up to the item count; nothing past it is read or written.
struct RadixSortPairsPlan {
    static constexpr uint32_t kMaxPasses = 8;
    static constexpr uint32_t kMaxDigitBits = 8;

    DataType keyDataType = DataType::UINT32;
    uint64_t capacity = 0;  // the most pairs a launch sorts
    uint32_t keyBits = 0;
    uint32_t passes = 0;
    uint8_t digitBits[kMaxPasses] = {};
    uint8_t digitShift[kMaxPasses] = {};
    uint32_t tilePairs = 0;       // 512 or 4096: the pairs of a pass block
    uint32_t tiles = 0;           // pass kernel grid
    uint32_t histogramTiles = 0;  // histogram kernel grid
    uint64_t scratchWords = 0;    // UINT32 words of scratch a launch needs (4-byte aligned)
    bool sortedInAlternate = false;

    bool operator==(const RadixSortPairsPlan&) const = default;
};

/**
 * The plan for sorting up to `capacity` pairs (1 to 2^30 - 1) whose keys are below 2^keyBits (1 to the key type's
 * bits). tilePairs 0 chooses: 512 up to 262144 pairs (131072 when a digit is 8 bits), else 4096.
 * Throws std::invalid_argument otherwise.
 */
[[nodiscard]] RadixSortPairsPlan planRadixSortPairs(DataType keyDataType,
                                                    uint64_t capacity,
                                                    uint32_t keyBits,
                                                    uint32_t tilePairs = 0);

/**
 * Sorts the first n pairs, n = the plan's capacity when activeCount is null, else min(*activeCount, capacity) read on
 * the device (activeCountDataType UINT32 or UINT64). Stream-ordered, no allocation, no synchronization. `scratch`
 * holds plan.scratchWords UINT32 words. Every array is on the stream's GPU; the pairs' arrays do not overlap.
 */
void launchRadixSortPairs(const RadixSortPairsPlan& plan,
                          void* keys,
                          void* keysAlternate,
                          uint32_t* values,
                          uint32_t* valuesAlternate,
                          uint32_t* scratch,
                          const void* activeCount,
                          DataType activeCountDataType,
                          Stream stream);

}  // namespace ThorImplementation
