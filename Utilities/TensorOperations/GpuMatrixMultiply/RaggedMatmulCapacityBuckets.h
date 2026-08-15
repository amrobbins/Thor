#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ThorImplementation {

/**
 * Build the finite packed-row capacity classes used by a ragged matmul.
 *
 * Capacities below 16 use only their full row capacity.  Larger capacities
 * keep the full row capacity plus descending powers of two beginning at the
 * power of two nearest to half of the full capacity and ending at 8.
 * Exact nearest-power ties choose the larger power of two.
 *
 * Examples:
 *   15  -> {15}
 *   16  -> {8, 16}
 *   66  -> {8, 16, 32, 66}
 *   100 -> {8, 16, 32, 64, 100}
 */
[[nodiscard]] std::vector<uint64_t> makeRaggedMatmulCapacityBuckets(uint64_t fullCapacityRows);

/**
 * Select the smallest cached capacity that can contain activeRows.
 *
 * activeRows must be non-zero.  A future ragged execution path should handle
 * the zero-active-row case as a no-op before asking for a GEMM capacity.
 */
[[nodiscard]] uint64_t chooseRaggedMatmulCapacityBucket(uint64_t activeRows, std::span<const uint64_t> capacityBuckets);

}  // namespace ThorImplementation
