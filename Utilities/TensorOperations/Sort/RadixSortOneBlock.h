#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

// Thor's stable radix sort of a few thousand (key, value) pairs in one kernel of one block: unsigned keys of 16, 32 or
// 64 bits whose set bits all lie below keyBits, UINT32 values, sorted in place. Pairs of equal keys keep their input
// order, so the result is the same on every run.
//
// The block loads every pair into registers, makes each pass of up to 8 bits in shared memory (the pass of
// RadixSort.h's kernel without its global reads and writes, its histogram and its lookback) and stores the sorted pairs
// once. No scratch and no alternate arrays. The item count may be on the device (activeCount): nothing past it is read
// or written, so a launch can be captured in a CUDA graph and still sort only a batch's pairs.
//
// This is the path for inputs too small to fill more than a block, where RadixSort.h's launch is a chain of one
// histogram kernel and a kernel a pass, each of a few blocks.
struct RadixSortOneBlockPlan {
    static constexpr uint32_t kMaxPasses = 8;
    static constexpr uint32_t kMaxDigitBits = 8;
    static constexpr uint32_t kWarpPairs = 128;  // a warp holds 32 x 4 consecutive pairs

    DataType keyDataType = DataType::UINT32;
    uint32_t capacity = 0;  // the most pairs a launch sorts
    uint32_t keyBits = 0;
    uint32_t passes = 0;
    uint8_t digitBits[kMaxPasses] = {};
    uint8_t digitShift[kMaxPasses] = {};
    uint32_t warps = 0;  // the block is warps x 32 threads: one warp per 128 pairs of capacity

    bool operator==(const RadixSortOneBlockPlan&) const = default;
};

/**
 * The most pairs one block sorts: 4096 with UINT16 or UINT32 keys, 2048 with UINT64 keys (the block's shared memory).
 * Throws std::invalid_argument for any other key type.
 */
[[nodiscard]] uint32_t radixSortOneBlockMaxPairs(DataType keyDataType);

/**
 * The plan for sorting up to `capacity` pairs (1 to radixSortOneBlockMaxPairs) whose keys are below 2^keyBits (1 to
 * the key type's bits). The digits are those of planRadixSortPairs. Throws std::invalid_argument otherwise.
 */
[[nodiscard]] RadixSortOneBlockPlan planRadixSortPairsOneBlock(DataType keyDataType,
                                                               uint64_t capacity,
                                                               uint32_t keyBits);

/**
 * Sorts the first n pairs of (keys, values) in place, n = the plan's capacity when activeCount is null, else
 * min(*activeCount, capacity) read on the device (activeCountDataType UINT32 or UINT64). Stream-ordered, no
 * allocation, no synchronization. Both arrays are on the stream's GPU.
 */
void launchRadixSortPairsOneBlock(const RadixSortOneBlockPlan& plan,
                                  void* keys,
                                  uint32_t* values,
                                  const void* activeCount,
                                  DataType activeCountDataType,
                                  Stream stream);

}  // namespace ThorImplementation
